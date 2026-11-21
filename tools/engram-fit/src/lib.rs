//! Core logic for `engram-fit` — kept in a lib so the math is unit-testable
//! without a network. The binary is a thin clap wrapper.
//!
//! Why this tool exists: contrastively-trained embedders are anisotropic —
//! all vectors share a dominant component, which biases the fly-hash k-WTA
//! toward the same winning bits for every input and collapses jaccard
//! discrimination. The fix is subtracting a **model-level** mean (never a
//! corpus-level one: a per-corpus mean has a cold-start hole, drifts at
//! t+100, and subtracts the user's topical signature — which is signal).
//! This tool computes that frozen artifact once per model+engine, checks the
//! space is healthy after centering, and calibrates the jaccard bands.

use anyhow::{bail, Context, Result};
use engram::SparseCode;
use serde::{Deserialize, Serialize};

pub const BITS_WORDS: usize = 32; // 2048-bit codes, matching Engram<S>
pub type Code = SparseCode<BITS_WORDS>;

// ---------------------------------------------------------------------------
// Artifact
// ---------------------------------------------------------------------------

/// The frozen per-model centering artifact. Everything an app needs to wire
/// engram to this encoder: both role centers, the prefixes used, and the
/// exact tag string to hash with `engram::model_tag`.
#[derive(Serialize, Deserialize, Debug)]
pub struct CenterArtifact {
    /// Free-form engine identity ("mistralrs-0.6-metal-isq8", …). Part of the
    /// tag: a different engine/quant is a different space.
    pub engine: String,
    pub model: String,
    pub dim: usize,
    pub samples: usize,
    pub doc_prefix: String,
    pub query_prefix: String,
    /// Mean of doc-prefixed embeddings over the sentence pack (L2-normalized inputs).
    pub doc_center: Vec<f32>,
    /// Mean of query-prefixed embeddings over the sentence pack.
    pub query_center: Vec<f32>,
    /// The string to feed `engram::model_tag(...)` — includes engine, model,
    /// prefixes, and a content hash of the centers themselves.
    pub tag: String,
    /// `engram::model_tag(tag)`, precomputed for convenience.
    pub tag_u64: u64,
}

pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 && norm.is_finite() {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// ---------------------------------------------------------------------------
// Storage-diet transforms: MRL truncation + f16 round-trip.
//
// Matryoshka-trained models (EmbeddingGemma) concentrate meaning in the
// leading dims: keep the first `dims`, L2-renormalize, and you have a valid
// smaller embedding. f16 storage halves bytes again; the round-trip here
// SIMULATES it exactly, so fits/tests measure the same values production
// stores. Order in production must match the fit: truncate → renorm →
// f16 round-trip → center → gate.
// ---------------------------------------------------------------------------

/// MRL truncation in place: keep the first `dims` components, renormalize.
pub fn truncate_normalize(vectors: &mut [Vec<f32>], dims: usize) {
    for v in vectors.iter_mut() {
        v.truncate(dims);
        l2_normalize(v);
    }
}

/// Round-trip every component through f16 (via the engram crate's converters
/// — the SAME bit math the packing shards use) — the exact values production
/// reads back from an f16-packed store.
pub fn f16_round_trip(vectors: &mut [Vec<f32>]) {
    for v in vectors.iter_mut() {
        for x in v.iter_mut() {
            *x = engram::f16_bits_to_f32(engram::f32_to_f16_bits(*x));
        }
    }
}

/// Round-trip through sqlite-vec's int8 "unit" quantization
/// (x → round(clamp(x,−1,1)×127)): the values a `vec_int8` store yields.
/// Smaller than f16 (1 byte/component) AND `vec_distance_cosine`-queryable.
pub fn int8_unit_round_trip(vectors: &mut [Vec<f32>]) {
    for v in vectors.iter_mut() {
        for x in v.iter_mut() {
            *x = (x.clamp(-1.0, 1.0) * 127.0).round() / 127.0;
        }
    }
}

pub fn mean_vector(vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
    let Some(first) = vectors.first() else {
        bail!("no vectors to average")
    };
    let dim = first.len();
    let mut mean = vec![0.0f64; dim];
    for v in vectors {
        if v.len() != dim {
            bail!("inconsistent embedding dims: {} vs {}", v.len(), dim);
        }
        for (m, x) in mean.iter_mut().zip(v) {
            *m += *x as f64;
        }
    }
    let n = vectors.len() as f64;
    Ok(mean.iter().map(|m| (*m / n) as f32).collect())
}

pub fn build_artifact(
    engine: &str,
    model: &str,
    doc_prefix: &str,
    query_prefix: &str,
    doc_embs: &[Vec<f32>],
    query_embs: &[Vec<f32>],
) -> Result<CenterArtifact> {
    let doc_center = mean_vector(doc_embs)?;
    let query_center = mean_vector(query_embs)?;
    if doc_center.len() != query_center.len() {
        bail!("doc/query centers disagree on dim");
    }
    // Content-hash the centers so the tag changes if the pack or math ever
    // does — the centering vector is part of the encoder identity.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in doc_center.iter().chain(query_center.iter()) {
        for b in v.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    let tag = format!("{engine}/{model}+doc:{doc_prefix}+q:{query_prefix}+center:{h:016x}");
    let tag_u64 = engram::model_tag(&tag);
    Ok(CenterArtifact {
        engine: engine.into(),
        model: model.into(),
        dim: doc_center.len(),
        samples: doc_embs.len(),
        doc_prefix: doc_prefix.into(),
        query_prefix: query_prefix.into(),
        doc_center,
        query_center,
        tag,
        tag_u64,
    })
}

// ---------------------------------------------------------------------------
// Health: pairwise-jaccard statistics over codes of UNRELATED sentences
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct JaccardStats {
    pub pairs: usize,
    pub mean: f32,
    pub p50: f32,
    pub p95: f32,
    pub max: f32,
}

pub fn codes_from(embs: &[Vec<f32>], k: usize) -> Vec<Code> {
    embs.iter().map(|v| Code::from_dense_fly(v, k)).collect()
}

pub fn pairwise_jaccard_stats(codes: &[Code]) -> JaccardStats {
    let mut js: Vec<f32> = Vec::with_capacity(codes.len() * (codes.len() - 1) / 2);
    for i in 0..codes.len() {
        for j in (i + 1)..codes.len() {
            js.push(codes[i].jaccard(&codes[j]));
        }
    }
    stats_of(&mut js)
}

pub fn stats_of(js: &mut [f32]) -> JaccardStats {
    js.sort_unstable_by(f32::total_cmp);
    let n = js.len();
    let mean = if n == 0 {
        0.0
    } else {
        js.iter().sum::<f32>() / n as f32
    };
    let pick = |q: f32| -> f32 {
        if n == 0 {
            0.0
        } else {
            js[((n - 1) as f32 * q) as usize]
        }
    };
    JaccardStats {
        pairs: n,
        mean,
        p50: pick(0.5),
        p95: pick(0.95),
        max: js.last().copied().unwrap_or(0.0),
    }
}

/// Per-cue MAX jaccard against all stored codes (skipping the aligned index,
/// for same-set/self comparisons): the coincidence ceiling a deliberate
/// retrieval must clear at this corpus size. `recall_anchor` calibrated from
/// PAIRWISE unrelated stats under-shoots badly — a query faces the max over N
/// stored episodes, an order statistic that grows with N. Recalibrate when
/// the corpus grows by ~an order of magnitude.
pub fn max_overlap_stats(cues: &[Code], stored: &[Code]) -> JaccardStats {
    let mut maxes: Vec<f32> = cues
        .iter()
        .enumerate()
        .map(|(i, c)| {
            stored
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, s)| c.jaccard(s))
                .fold(0.0f32, f32::max)
        })
        .collect();
    stats_of(&mut maxes)
}

/// The one-number verdict: mean pairwise jaccard between unrelated codes.
/// Healthy sparse codes barely collide; a collapsed (anisotropic) space
/// shares bits everywhere and the DUP/LINK bands can't breathe.
pub fn verdict(mean_unrelated: f32) -> &'static str {
    if mean_unrelated < 0.10 {
        "HEALTHY — bands have room; wire it up"
    } else if mean_unrelated < 0.20 {
        "MARGINAL — usable, but recalibrate the bands (run `calibrate`) and prefer centered vectors"
    } else {
        "COLLAPSED — do not use this encoder config without centering; if this IS centered, pick a different model"
    }
}

// ---------------------------------------------------------------------------
// Calibrate: labeled pairs -> per-label jaccard percentiles -> suggested bands
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PairLabel {
    /// Same meaning restated (paraphrase / near-duplicate).
    Dup,
    /// Same topic, different content — should Link, not Strengthen.
    Related,
    /// Nothing in common.
    Unrelated,
}

pub struct LabeledPair {
    pub label: PairLabel,
    pub a: String,
    pub b: String,
}

/// Parse a pairs file: one pair per line, `label<TAB>text a<TAB>text b`.
pub fn parse_pairs(content: &str) -> Result<Vec<LabeledPair>> {
    let mut out = Vec::new();
    for (i, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.splitn(3, '\t');
        let (label, a, b) = (
            parts.next().unwrap_or(""),
            parts.next().unwrap_or(""),
            parts.next().unwrap_or(""),
        );
        if a.is_empty() || b.is_empty() {
            bail!("line {}: expected `label<TAB>text a<TAB>text b`", i + 1);
        }
        let label = match label {
            "dup" => PairLabel::Dup,
            "related" => PairLabel::Related,
            "unrelated" => PairLabel::Unrelated,
            other => bail!(
                "line {}: unknown label {other:?} (dup|related|unrelated)",
                i + 1
            ),
        };
        out.push(LabeledPair {
            label,
            a: a.to_string(),
            b: b.to_string(),
        });
    }
    if out.is_empty() {
        bail!("no pairs found");
    }
    Ok(out)
}

#[derive(Debug, Serialize)]
pub struct Calibration {
    pub dup: JaccardStats,
    pub related: JaccardStats,
    pub unrelated: JaccardStats,
    /// Suggested `dup_jaccard`. The loss is ASYMMETRIC: a false Strengthen
    /// silently drops a payload (costly); a missed Strengthen becomes a Link
    /// (harmless — the memory is just less compact). So the band sits above
    /// the related tail (p95 × 1.25), accepting that hard paraphrases
    /// (e.g. cross-lingual) Link instead of Strengthen.
    pub suggested_dup: f32,
    /// Suggested `link_jaccard`: midpoint of related-p25 and unrelated-p95
    /// (both link errors are mild, so split the difference).
    pub suggested_link: f32,
    /// Suggested `recall_anchor`: above unrelated-p95 with margin. May sit
    /// ABOVE the link band — write-time linking is cheap and can afford to be
    /// permissive, while deliberate injection poisons context and cannot.
    pub suggested_anchor: f32,
}

pub fn calibrate(mut dup: Vec<f32>, mut related: Vec<f32>, mut unrelated: Vec<f32>) -> Calibration {
    let d = stats_of(&mut dup);
    let r = stats_of(&mut related);
    let u = stats_of(&mut unrelated);
    let q = |sorted: &[f32], quant: f32| -> f32 {
        if sorted.is_empty() {
            0.0
        } else {
            sorted[((sorted.len() - 1) as f32 * quant) as usize]
        }
    };
    let suggested_dup = (q(&related, 0.95) * 1.25).max(q(&unrelated, 0.95) * 2.0);
    let suggested_link = ((q(&related, 0.25) + q(&unrelated, 0.95)) / 2.0).min(suggested_dup);
    // Anchor: clear the ENTIRE observed coincidence distribution with margin
    // (max × 1.5), floor 0.05. Not q95: with ~20 labeled unrelated pairs the
    // quantile index discards the tail, and authored pairs under-sample true
    // coincidence anyway — measured on the lifecycle stream, real coincidence
    // tops out ~0.07–0.08 at k=128 for BOTH fitted models while q95×1.5 let
    // it through for one of them. Deliberately NOT capped by the link band
    // (injection is gated harder than linking).
    let suggested_anchor = (q(&unrelated, 1.0) * 1.5).max(0.05);
    Calibration {
        dup: d,
        related: r,
        unrelated: u,
        suggested_dup,
        suggested_link,
        suggested_anchor,
    }
}

// ---------------------------------------------------------------------------
// Tune: sweep fly-hash k over captured pair embeddings — the last free
// parameter. Sparse k resolves near-identical content sharply but compresses
// moderate-cosine pairs (real paraphrases); denser k trades some contrast for
// resolution. Entirely offline: consumes fixtures, no engine needed.
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct KSweepRow {
    pub k: usize,
    pub dup_mean: f32,
    pub related_mean: f32,
    pub unrelated_mean: f32,
    /// q25(dup) − q75(related): headroom for a clean Strengthen band.
    pub strengthen_margin: f32,
    /// q25(related) − q95(unrelated): headroom for a clean Link band.
    pub link_margin: f32,
    pub calibration: Calibration,
}

pub fn sweep_k(
    a: &[Vec<f32>],
    b: &[Vec<f32>],
    labels: &[PairLabel],
    ks: &[usize],
) -> Vec<KSweepRow> {
    let dim = a.first().map_or(0, |v| v.len());
    let mut rows = Vec::with_capacity(ks.len());
    for &k in ks {
        let mut hasher = engram::FlyHasher::<BITS_WORDS>::new(dim, k);
        let (mut dup, mut related, mut unrelated) = (Vec::new(), Vec::new(), Vec::new());
        for ((va, vb), label) in a.iter().zip(b).zip(labels) {
            let j = hasher.sparsify(va).jaccard(&hasher.sparsify(vb));
            match label {
                PairLabel::Dup => dup.push(j),
                PairLabel::Related => related.push(j),
                PairLabel::Unrelated => unrelated.push(j),
            }
        }
        let q = |v: &Vec<f32>, quant: f32| -> f32 {
            let mut s = v.clone();
            s.sort_unstable_by(f32::total_cmp);
            if s.is_empty() {
                0.0
            } else {
                s[((s.len() - 1) as f32 * quant) as usize]
            }
        };
        let strengthen_margin = q(&dup, 0.25) - q(&related, 0.75);
        let link_margin = q(&related, 0.25) - q(&unrelated, 0.95);
        let cal = calibrate(dup.clone(), related.clone(), unrelated.clone());
        rows.push(KSweepRow {
            k,
            dup_mean: cal.dup.mean,
            related_mean: cal.related.mean,
            unrelated_mean: cal.unrelated.mean,
            strengthen_margin,
            link_margin,
            calibration: cal,
        });
    }
    rows
}

// ---------------------------------------------------------------------------
// OpenAI-compatible /v1/embeddings client (mistralrs serve, and friends)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Deserialize)]
struct EmbeddingItem {
    index: usize,
    embedding: Vec<f32>,
}

/// Embed `texts` (already prefixed by the caller) against an OpenAI-compatible
/// endpoint. Batched; embeddings are L2-normalized on return so downstream
/// math never depends on the engine's normalization choice.
pub fn embed_all(endpoint: &str, model: &str, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    let url = format!("{}/v1/embeddings", endpoint.trim_end_matches('/'));
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(32) {
        let resp: EmbeddingsResponse = ureq::post(&url)
            .send_json(serde_json::json!({ "model": model, "input": chunk }))
            .with_context(|| format!("POST {url} failed — is the engine serving?"))?
            .into_json()
            .context("malformed /v1/embeddings response")?;
        let mut items = resp.data;
        if items.len() != chunk.len() {
            bail!(
                "endpoint returned {} embeddings for {} inputs",
                items.len(),
                chunk.len()
            );
        }
        items.sort_by_key(|i| i.index);
        for mut item in items {
            l2_normalize(&mut item.embedding);
            out.push(item.embedding);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Embeddings capture file: real model output frozen as a test fixture, so the
// anisotropy/centering behavior of an actual encoder is a plain `cargo test`
// (no engine, no network). Format: magic, dim, n, then n*dim f32 LE doc
// embeddings followed by n*dim f32 LE query embeddings.
// ---------------------------------------------------------------------------

const EMBS_MAGIC: u32 = 0x454D_4253; // "EMBS"

/// Two embedding sets of equal length: (doc, query) for pack dumps,
/// (a-sides, b-sides) for pair dumps.
pub type EmbeddingSets = (Vec<Vec<f32>>, Vec<Vec<f32>>);

pub fn write_embeddings_file(
    path: &std::path::Path,
    doc: &[Vec<f32>],
    query: &[Vec<f32>],
) -> Result<()> {
    anyhow::ensure!(
        !doc.is_empty() && doc.len() == query.len(),
        "doc/query sets must match"
    );
    let dim = doc[0].len();
    let mut out: Vec<u8> = Vec::with_capacity(12 + doc.len() * dim * 8);
    out.extend_from_slice(&EMBS_MAGIC.to_le_bytes());
    out.extend_from_slice(&(dim as u32).to_le_bytes());
    out.extend_from_slice(&(doc.len() as u32).to_le_bytes());
    for v in doc.iter().chain(query.iter()) {
        anyhow::ensure!(v.len() == dim, "inconsistent dims");
        for x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    std::fs::write(path, out).with_context(|| format!("writing {}", path.display()))
}

pub fn read_embeddings_file(path: &std::path::Path) -> Result<EmbeddingSets> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    anyhow::ensure!(bytes.len() >= 12, "truncated embeddings file");
    let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    anyhow::ensure!(u32_at(0) == EMBS_MAGIC, "not an EMBS file");
    let dim = u32_at(4) as usize;
    let n = u32_at(8) as usize;
    anyhow::ensure!(bytes.len() == 12 + 2 * n * dim * 4, "size mismatch");
    let read_set = |offset: usize| -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                let base = offset + i * dim * 4;
                (0..dim)
                    .map(|j| {
                        f32::from_le_bytes(
                            bytes[base + j * 4..base + j * 4 + 4].try_into().unwrap(),
                        )
                    })
                    .collect()
            })
            .collect()
    };
    let doc = read_set(12);
    let query = read_set(12 + n * dim * 4);
    Ok((doc, query))
}

// ---------------------------------------------------------------------------
// Lifecycle stream: a labeled message/query/sleep sequence for end-to-end
// evaluation (see assets/stream.tsv and tests/lifecycle.rs).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum StreamItem {
    /// A message to observe (doc role).
    Message(String),
    /// A query (query role). `expected` holds substrings resolving to the
    /// answer messages; empty = the correct behavior is SILENCE.
    Query { text: String, expected: Vec<String> },
    /// Weekly consolidation.
    Sleep,
}

/// Parse a stream file: `m<TAB>text`, `q<TAB>text<TAB>substr|substr` (or `-`),
/// `s`. Comments/# and blank lines ignored.
pub fn parse_stream(content: &str) -> Result<Vec<StreamItem>> {
    let mut out = Vec::new();
    for (i, line) in content.lines().enumerate() {
        let line = line.trim_end();
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.splitn(3, '\t');
        match parts.next().unwrap_or("") {
            "m" => {
                let text = parts.next().unwrap_or("").to_string();
                anyhow::ensure!(!text.is_empty(), "line {}: empty message", i + 1);
                out.push(StreamItem::Message(text));
            }
            "q" => {
                let text = parts.next().unwrap_or("").to_string();
                let exp = parts.next().unwrap_or("");
                anyhow::ensure!(
                    !text.is_empty() && !exp.is_empty(),
                    "line {}: query needs text and expected",
                    i + 1
                );
                let expected = if exp == "-" {
                    Vec::new()
                } else {
                    exp.split('|').map(str::to_string).collect()
                };
                out.push(StreamItem::Query { text, expected });
            }
            "s" => out.push(StreamItem::Sleep),
            other => anyhow::bail!("line {}: unknown kind {other:?}", i + 1),
        }
    }
    anyhow::ensure!(!out.is_empty(), "empty stream");
    Ok(out)
}

/// Resolve one query's expected substrings to message indices; errors if a
/// substring matches nothing (catches authoring mistakes at dump time).
pub fn resolve_expected(expected: &[String], messages: &[String]) -> Result<Vec<usize>> {
    let mut ids = Vec::new();
    for sub in expected {
        let matched: Vec<usize> = messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.contains(sub.as_str()))
            .map(|(i, _)| i)
            .collect();
        anyhow::ensure!(
            !matched.is_empty(),
            "expected substring {sub:?} matches no message"
        );
        ids.extend(matched);
    }
    ids.sort_unstable();
    ids.dedup();
    Ok(ids)
}

const STREAM_MAGIC: u32 = 0x454D_4251; // "EMBQ"

/// Stream fixture: doc-role message embeddings + query-role query embeddings
/// (different counts, unlike the paired EMBS format).
pub fn write_stream_embeddings(
    path: &std::path::Path,
    messages: &[Vec<f32>],
    queries: &[Vec<f32>],
) -> Result<()> {
    anyhow::ensure!(!messages.is_empty(), "no message embeddings");
    let dim = messages[0].len();
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(&STREAM_MAGIC.to_le_bytes());
    out.extend_from_slice(&(dim as u32).to_le_bytes());
    out.extend_from_slice(&(messages.len() as u32).to_le_bytes());
    out.extend_from_slice(&(queries.len() as u32).to_le_bytes());
    for v in messages.iter().chain(queries.iter()) {
        anyhow::ensure!(v.len() == dim, "inconsistent dims");
        for x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    std::fs::write(path, out).with_context(|| format!("writing {}", path.display()))
}

pub fn read_stream_embeddings(path: &std::path::Path) -> Result<EmbeddingSets> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    anyhow::ensure!(bytes.len() >= 16, "truncated stream fixture");
    let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    anyhow::ensure!(u32_at(0) == STREAM_MAGIC, "not an EMBQ file");
    let dim = u32_at(4) as usize;
    let n_m = u32_at(8) as usize;
    let n_q = u32_at(12) as usize;
    anyhow::ensure!(bytes.len() == 16 + (n_m + n_q) * dim * 4, "size mismatch");
    let read_at = |offset: usize, n: usize| -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                let base = offset + i * dim * 4;
                (0..dim)
                    .map(|j| {
                        f32::from_le_bytes(
                            bytes[base + j * 4..base + j * 4 + 4].try_into().unwrap(),
                        )
                    })
                    .collect()
            })
            .collect()
    };
    let messages = read_at(16, n_m);
    let queries = read_at(16 + n_m * dim * 4, n_q);
    Ok((messages, queries))
}

/// The bundled lifecycle stream.
pub fn default_stream() -> &'static str {
    include_str!("../assets/stream.tsv")
}

/// The bundled multilingual sentence pack (diverse domains, EN/IT/JA).
pub fn default_pack() -> Vec<String> {
    include_str!("../assets/pack.txt")
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(String::from)
        .collect()
}

pub fn apply_center(embs: &mut [Vec<f32>], center: &[f32]) {
    for v in embs.iter_mut() {
        engram::center_in_place(v, center);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_and_normalize() {
        let vs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let m = mean_vector(&vs).unwrap();
        assert_eq!(m, vec![0.5, 0.5]);
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn artifact_tag_changes_with_centers() {
        let a = build_artifact(
            "e",
            "m",
            "passage: ",
            "query: ",
            &[vec![1.0, 0.0]],
            &[vec![0.0, 1.0]],
        )
        .unwrap();
        let b = build_artifact(
            "e",
            "m",
            "passage: ",
            "query: ",
            &[vec![0.9, 0.1]],
            &[vec![0.0, 1.0]],
        )
        .unwrap();
        assert_ne!(a.tag, b.tag);
        assert_ne!(a.tag_u64, b.tag_u64);
        assert_eq!(a.tag_u64, engram::model_tag(&a.tag));
    }

    #[test]
    fn stats_and_verdict() {
        let mut js = vec![0.0, 0.02, 0.04, 0.06, 0.08];
        let s = stats_of(&mut js);
        assert_eq!(s.pairs, 5);
        assert!((s.mean - 0.04).abs() < 1e-6);
        assert_eq!(s.max, 0.08);
        assert!(verdict(0.05).starts_with("HEALTHY"));
        assert!(verdict(0.15).starts_with("MARGINAL"));
        assert!(verdict(0.30).starts_with("COLLAPSED"));
    }

    #[test]
    fn centering_restores_code_diversity() {
        // Synthetic anisotropy: every vector = big shared mu + small unique
        // deviation. Uncentered codes collide heavily; centered ones don't.
        // This is the whole thesis of the tool, in miniature.
        let dim = 384;
        let mut seed = 7u64;
        let mut rand = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let mu: Vec<f32> = (0..dim).map(|_| rand()).collect();
        let mut embs: Vec<Vec<f32>> = (0..40)
            .map(|_| {
                let mut v: Vec<f32> = mu
                    .iter()
                    .map(|m| m + 0.25 * rand()) // deviation ~4x smaller than mu
                    .collect();
                l2_normalize(&mut v);
                v
            })
            .collect();

        let uncentered = pairwise_jaccard_stats(&codes_from(&embs, 40));
        let center = mean_vector(&embs).unwrap();
        apply_center(&mut embs, &center);
        let centered = pairwise_jaccard_stats(&codes_from(&embs, 40));

        assert!(
            uncentered.mean > 0.4,
            "premise: anisotropy must collapse codes (got {})",
            uncentered.mean
        );
        assert!(
            centered.mean < 0.10,
            "centering must restore diversity (got {})",
            centered.mean
        );
    }

    #[test]
    fn parse_pairs_and_calibrate() {
        let pairs = parse_pairs(
            "dup\tthe cat sat\ta cat was sitting\nrelated\tcats\tdogs\nunrelated\tcats\tgdp figures\n",
        )
        .unwrap();
        assert_eq!(pairs.len(), 3);
        assert!(parse_pairs("badlabel\ta\tb\n").is_err());

        let c = calibrate(
            vec![0.85, 0.9, 0.95],
            vec![0.4, 0.5, 0.6],
            vec![0.01, 0.03, 0.05],
        );
        assert!(c.suggested_dup > c.suggested_link);
        assert!(c.suggested_link > c.suggested_anchor - 1e-6);
    }
}
