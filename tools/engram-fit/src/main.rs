//! engram-fit — fit engram to an embedding model.
//!
//! Talks to any OpenAI-compatible `/v1/embeddings` endpoint, so the artifact
//! is computed by the SAME engine binary production uses:
//!
//! ```text
//! mistralrs serve -m google/embeddinggemma-300m --port 8080   # or your engine
//!
//! engram-fit center    --endpoint http://localhost:8080 --model embeddinggemma-300m \
//!                      --engine mistralrs-metal --out center.json
//! engram-fit health    --endpoint http://localhost:8080 --model embeddinggemma-300m \
//!                      --center center.json
//! engram-fit calibrate --endpoint http://localhost:8080 --model embeddinggemma-300m \
//!                      --center center.json --pairs pairs.tsv
//! ```

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use engram_fit::*;
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "engram-fit",
    about = "Fit engram to an embedding model: center, health, calibrate"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args)]
struct EngineArgs {
    /// OpenAI-compatible endpoint base URL (e.g. mistralrs serve)
    #[arg(long)]
    endpoint: String,
    /// Model name as the endpoint knows it
    #[arg(long)]
    model: String,
    /// Prefix for stored/document texts (E5: "passage: "; EmbeddingGemma:
    /// "title: none | text: "; Qwen3-Embedding: "")
    #[arg(long, default_value = "passage: ")]
    doc_prefix: String,
    /// Prefix for query texts (E5: "query: "; EmbeddingGemma:
    /// "task: search result | query: "; Qwen3: your instruct template)
    #[arg(long, default_value = "query: ")]
    query_prefix: String,
    /// Sentence pack (one sentence per line); bundled multilingual pack by default
    #[arg(long)]
    pack: Option<PathBuf>,
    /// Fly-hash active bits (must match production Params::k)
    #[arg(long, default_value_t = 40)]
    k: usize,
}

#[derive(Subcommand)]
enum Cmd {
    /// Compute the frozen per-model centering artifact (doc + query centers)
    Center {
        #[command(flatten)]
        engine: EngineArgs,
        /// Free-form engine identity recorded in the tag (engine+quant+platform)
        #[arg(long, default_value = "openai-compat")]
        engine_id: String,
        /// Output path for the artifact JSON
        #[arg(long, default_value = "center.json")]
        out: PathBuf,
        /// OFFLINE: derive from a pack fixture (`dump` output) instead of
        /// hitting the endpoint — for storage-transform variants of an
        /// already-captured model
        #[arg(long)]
        pack_embs: Option<PathBuf>,
        /// Round-trip every embedding through f16 before computing centers —
        /// use when production stores embeddings f16-packed (the tag then
        /// fingerprints the f16 space; append e.g. "-f16" to --model too)
        #[arg(long)]
        f16: bool,
        /// Round-trip through sqlite-vec int8 "unit" quantization instead —
        /// use when production stores embeddings as int8 vectors (append
        /// e.g. "-int8" to --model)
        #[arg(long, conflicts_with = "f16")]
        int8: bool,
    },
    /// One-number anisotropy check: mean pairwise jaccard of unrelated codes,
    /// raw vs centered
    Health {
        #[command(flatten)]
        engine: EngineArgs,
        /// Centering artifact from `center` (omit to see the raw space only)
        #[arg(long)]
        center: Option<PathBuf>,
    },
    /// Capture real model embeddings (doc + query prefixed pack) into a
    /// fixture file, so encoder behavior becomes an offline `cargo test`
    Dump {
        #[command(flatten)]
        engine: EngineArgs,
        /// Output path (e.g. tests/fixtures/<model>.embs)
        #[arg(long)]
        out: PathBuf,
        /// Dump a labeled pairs file instead of the pack: the two sets in the
        /// fixture become the a-sides and b-sides (both doc-prefixed)
        #[arg(long)]
        pairs: Option<PathBuf>,
    },
    /// Capture the lifecycle stream's embeddings (messages doc-role, queries
    /// query-role) into a fixture for the end-to-end evaluation
    DumpStream {
        #[command(flatten)]
        engine: EngineArgs,
        /// Stream file (defaults to the bundled assets/stream.tsv)
        #[arg(long)]
        stream: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Offline: sweep fly-hash k over captured fixtures and report band
    /// margins per k — closes the loop from model to tuned Params
    Tune {
        /// Pack fixture from `dump` (used to derive the centering vector)
        #[arg(long)]
        pack_embs: PathBuf,
        /// Pairs fixture from `dump --pairs`
        #[arg(long)]
        pairs_embs: PathBuf,
        /// The pairs file the fixture was dumped from (for the labels)
        #[arg(long)]
        pairs: PathBuf,
        /// Comma-separated k values to sweep
        #[arg(long, default_value = "20,40,64,96,128,192,256")]
        ks: String,
        /// Sweep the raw (uncentered) space instead
        #[arg(long)]
        raw: bool,
    },
    /// Jaccard percentiles over labeled pairs -> suggested DUP/LINK/anchor
    Calibrate {
        #[command(flatten)]
        engine: EngineArgs,
        /// Pairs file: `label<TAB>text a<TAB>text b`, label = dup|related|unrelated
        #[arg(long)]
        pairs: PathBuf,
        /// Centering artifact (recommended: calibrate the space you will run)
        #[arg(long)]
        center: Option<PathBuf>,
    },
}

fn load_pack(engine: &EngineArgs) -> Result<Vec<String>> {
    let pack = match &engine.pack {
        Some(path) => fs::read_to_string(path)
            .with_context(|| format!("reading pack {}", path.display()))?
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(String::from)
            .collect(),
        None => default_pack(),
    };
    anyhow::ensure!(
        pack.len() >= 50,
        "pack too small ({} sentences) — the center would be noise",
        pack.len()
    );
    Ok(pack)
}

fn load_center(path: &PathBuf) -> Result<CenterArtifact> {
    let art: CenterArtifact = serde_json::from_str(
        &fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?,
    )
    .context("parsing center artifact")?;
    Ok(art)
}

fn embed_prefixed(engine: &EngineArgs, prefix: &str, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    let prefixed: Vec<String> = texts.iter().map(|t| format!("{prefix}{t}")).collect();
    embed_all(&engine.endpoint, &engine.model, &prefixed)
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Center {
            engine,
            engine_id,
            out,
            pack_embs,
            f16,
            int8,
        } => {
            let (mut doc, mut query) = if let Some(embs) = pack_embs {
                eprintln!("offline: deriving centers from {}", embs.display());
                read_embeddings_file(&embs)?
            } else {
                let pack = load_pack(&engine)?;
                eprintln!(
                    "embedding {} sentences twice (doc + query prefixes)…",
                    pack.len()
                );
                let doc = embed_prefixed(&engine, &engine.doc_prefix, &pack)?;
                let query = embed_prefixed(&engine, &engine.query_prefix, &pack)?;
                (doc, query)
            };
            if f16 {
                f16_round_trip(&mut doc);
                f16_round_trip(&mut query);
            }
            if int8 {
                int8_unit_round_trip(&mut doc);
                int8_unit_round_trip(&mut query);
            }
            let art = build_artifact(
                &engine_id,
                &engine.model,
                &engine.doc_prefix,
                &engine.query_prefix,
                &doc,
                &query,
            )?;
            fs::write(&out, serde_json::to_string(&art)?)
                .with_context(|| format!("writing {}", out.display()))?;
            println!(
                "wrote {} (dim {}, {} samples)",
                out.display(),
                art.dim,
                art.samples
            );
            println!("tag: {}", art.tag);
            println!(
                "tag_u64: {:#018x}  <- use with Memory::snapshot/restore",
                art.tag_u64
            );
            println!("\nThe artifact is FROZEN: version it with the model; changing it is a wipe-and-reimport event.");
        }

        Cmd::Health { engine, center } => {
            let pack = load_pack(&engine)?;
            eprintln!("embedding {} sentences (doc prefix)…", pack.len());
            let mut doc = embed_prefixed(&engine, &engine.doc_prefix, &pack)?;

            let raw = pairwise_jaccard_stats(&codes_from(&doc, engine.k));
            println!(
                "raw space:      mean {:.3}  p50 {:.3}  p95 {:.3}  max {:.3}  ({} pairs)",
                raw.mean, raw.p50, raw.p95, raw.max, raw.pairs
            );

            let mean_for_verdict = if let Some(path) = center {
                let art = load_center(&path)?;
                anyhow::ensure!(
                    art.dim == doc[0].len(),
                    "center dim {} != embedding dim {}",
                    art.dim,
                    doc[0].len()
                );
                apply_center(&mut doc, &art.doc_center);
                let centered = pairwise_jaccard_stats(&codes_from(&doc, engine.k));
                println!(
                    "centered space: mean {:.3}  p50 {:.3}  p95 {:.3}  max {:.3}",
                    centered.mean, centered.p50, centered.p95, centered.max
                );
                centered.mean
            } else {
                println!("(no --center given — judging the raw space)");
                raw.mean
            };
            println!("\nverdict: {}", verdict(mean_for_verdict));
        }

        Cmd::Dump { engine, out, pairs } => {
            let (a, b) = match &pairs {
                None => {
                    let pack = load_pack(&engine)?;
                    eprintln!(
                        "embedding {} sentences twice (doc + query prefixes)…",
                        pack.len()
                    );
                    let doc = embed_prefixed(&engine, &engine.doc_prefix, &pack)?;
                    let query = embed_prefixed(&engine, &engine.query_prefix, &pack)?;
                    (doc, query)
                }
                Some(path) => {
                    let labeled = parse_pairs(&fs::read_to_string(path)?)?;
                    eprintln!("embedding {} pairs (doc prefix both sides)…", labeled.len());
                    let a_texts: Vec<String> = labeled.iter().map(|p| p.a.clone()).collect();
                    let b_texts: Vec<String> = labeled.iter().map(|p| p.b.clone()).collect();
                    let a = embed_prefixed(&engine, &engine.doc_prefix, &a_texts)?;
                    let b = embed_prefixed(&engine, &engine.doc_prefix, &b_texts)?;
                    (a, b)
                }
            };
            write_embeddings_file(&out, &a, &b)?;
            println!(
                "wrote {} ({} rows, dim {}, {} KiB)",
                out.display(),
                a.len(),
                a[0].len(),
                (12 + 2 * a.len() * a[0].len() * 4) / 1024
            );
        }

        Cmd::DumpStream {
            engine,
            stream,
            out,
        } => {
            let content = match &stream {
                Some(p) => fs::read_to_string(p)?,
                None => default_stream().to_string(),
            };
            let items = parse_stream(&content)?;
            let messages: Vec<String> = items
                .iter()
                .filter_map(|i| match i {
                    StreamItem::Message(m) => Some(m.clone()),
                    _ => None,
                })
                .collect();
            let mut queries = Vec::new();
            for item in &items {
                if let StreamItem::Query { text, expected } = item {
                    // Validate expectations at dump time — catches typos early.
                    resolve_expected(expected, &messages)?;
                    queries.push(text.clone());
                }
            }
            eprintln!(
                "embedding {} messages (doc role) + {} queries (query role)…",
                messages.len(),
                queries.len()
            );
            let m_embs = embed_prefixed(&engine, &engine.doc_prefix, &messages)?;
            let q_embs = embed_prefixed(&engine, &engine.query_prefix, &queries)?;
            write_stream_embeddings(&out, &m_embs, &q_embs)?;
            println!(
                "wrote {} ({} msgs, {} queries)",
                out.display(),
                m_embs.len(),
                q_embs.len()
            );
        }

        Cmd::Tune {
            pack_embs,
            pairs_embs,
            pairs,
            ks,
            raw,
        } => {
            let (pack_doc, _) = read_embeddings_file(&pack_embs)?;
            let (mut a, mut b) = read_embeddings_file(&pairs_embs)?;
            let labeled = parse_pairs(&fs::read_to_string(&pairs)?)?;
            anyhow::ensure!(
                labeled.len() == a.len(),
                "pairs file has {} pairs but fixture has {} rows",
                labeled.len(),
                a.len()
            );
            let labels: Vec<PairLabel> = labeled.iter().map(|p| p.label).collect();

            if !raw {
                let center = mean_vector(&pack_doc)?;
                apply_center(&mut a, &center);
                apply_center(&mut b, &center);
            }
            let ks: Vec<usize> = ks
                .split(',')
                .map(|s| s.trim().parse::<usize>().context("bad k"))
                .collect::<Result<_>>()?;

            println!("{} space:", if raw { "raw" } else { "centered" });
            println!(
                "{:>5} {:>9} {:>9} {:>9} {:>12} {:>12} {:>8} {:>8}",
                "k", "dup", "related", "unrel", "str.margin", "link.margin", "DUP", "LINK"
            );
            let rows = sweep_k(&a, &b, &labels, &ks);
            let mut best: Option<&engram_fit::KSweepRow> = None;
            for row in &rows {
                println!(
                    "{:>5} {:>9.3} {:>9.3} {:>9.3} {:>12.3} {:>12.3} {:>8.2} {:>8.2}",
                    row.k,
                    row.dup_mean,
                    row.related_mean,
                    row.unrelated_mean,
                    row.strengthen_margin,
                    row.link_margin,
                    row.calibration.suggested_dup,
                    row.calibration.suggested_link
                );
                if best.is_none_or(|b| {
                    row.strengthen_margin + row.link_margin > b.strengthen_margin + b.link_margin
                }) {
                    best = Some(row);
                }
            }
            if let Some(bst) = best {
                println!("\nbest k by combined margin: {}", bst.k);
                println!("  set Params.k = {}", bst.k);
                println!(
                    "  set gate.dup_jaccard  = {:.2}",
                    bst.calibration.suggested_dup
                );
                println!(
                    "  set gate.link_jaccard = {:.2}",
                    bst.calibration.suggested_link
                );
                // The anchor blocks pure COINCIDENCE, not same-topic cousins:
                // real question→answer jaccard overlaps the cousin band, so a
                // cousin-proof anchor would refuse real answers too. Cousins
                // surfacing on a high-demand query are context, not junk.
                println!(
                    "  set params.recall_anchor = {:.2}   (blocks coincidence; same-topic memories above it WILL surface — that's context)",
                    bst.calibration.suggested_anchor
                );

                // Informative: where same-topic cousins live at this k, from
                // the pack's cross-role max-overlap (an upper "cousin ceiling").
                let (mut doc_side, mut cue_side) = read_embeddings_file(&pack_embs)?;
                if !raw {
                    let dc = mean_vector(&doc_side)?;
                    let qc = mean_vector(&cue_side)?;
                    apply_center(&mut doc_side, &dc);
                    apply_center(&mut cue_side, &qc);
                }
                let ceiling =
                    max_overlap_stats(&codes_from(&cue_side, bst.k), &codes_from(&doc_side, bst.k));
                println!(
                    "  cousin band (same-topic, N≈{}): p95 {:.3}, max {:.3} — question→answer scores land in this band too",
                    doc_side.len(), ceiling.p95, ceiling.max
                );
            }
        }

        Cmd::Calibrate {
            engine,
            pairs,
            center,
        } => {
            let content = fs::read_to_string(&pairs)
                .with_context(|| format!("reading {}", pairs.display()))?;
            let labeled = parse_pairs(&content)?;
            eprintln!("embedding {} pairs (doc prefix both sides)…", labeled.len());

            let mut texts: Vec<String> = Vec::with_capacity(labeled.len() * 2);
            for p in &labeled {
                texts.push(p.a.clone());
                texts.push(p.b.clone());
            }
            let mut embs = embed_prefixed(&engine, &engine.doc_prefix, &texts)?;
            if let Some(path) = &center {
                let art = load_center(path)?;
                apply_center(&mut embs, &art.doc_center);
            } else {
                eprintln!("warning: calibrating the RAW space; pass --center to calibrate what production will run");
            }
            let codes = codes_from(&embs, engine.k);

            let (mut dup, mut related, mut unrelated) = (Vec::new(), Vec::new(), Vec::new());
            for (i, p) in labeled.iter().enumerate() {
                let j = codes[2 * i].jaccard(&codes[2 * i + 1]);
                match p.label {
                    PairLabel::Dup => dup.push(j),
                    PairLabel::Related => related.push(j),
                    PairLabel::Unrelated => unrelated.push(j),
                }
            }
            let cal = calibrate(dup, related, unrelated);
            println!(
                "dup:       mean {:.3}  p50 {:.3}  p95 {:.3}  ({} pairs)",
                cal.dup.mean, cal.dup.p50, cal.dup.p95, cal.dup.pairs
            );
            println!(
                "related:   mean {:.3}  p50 {:.3}  p95 {:.3}  ({} pairs)",
                cal.related.mean, cal.related.p50, cal.related.p95, cal.related.pairs
            );
            println!(
                "unrelated: mean {:.3}  p50 {:.3}  p95 {:.3}  ({} pairs)",
                cal.unrelated.mean, cal.unrelated.p50, cal.unrelated.p95, cal.unrelated.pairs
            );
            println!("\nsuggested DUP_JACCARD:   {:.2}", cal.suggested_dup);
            println!("suggested LINK_JACCARD:  {:.2}", cal.suggested_link);
            println!("suggested recall_anchor: {:.2}", cal.suggested_anchor);
            println!("\nSanity: dup and related distributions should not overlap much; if they do, the model can't separate paraphrase from topic-mate and Strengthen-vs-Link will be noisy.");
        }
    }
    Ok(())
}
