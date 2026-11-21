//! Full-circle tests against REAL model output, fully offline: the fixtures
//! under `tests/fixtures/` were captured from models served by llama.cpp
//! (see the runbook in this crate's README to regenerate). No engine or
//! network needed at test time.
//!
//! The circle: model → captured embeddings → centering artifact → tuned
//! bands → engram's actual gate decisions on real language.
//!
//! Fitted models covered (each behavior gets a per-model #[test]):
//! - `me5-small-q8`      — multilingual-e5-small q8_0, 384-dim, HEAVILY
//!   anisotropic raw (the case centering exists for).
//! - `gemma300m-qat-q4`  — EmbeddingGemma-300m QAT q4_0, 768-dim, nearly
//!   isotropic raw (centering still helps, but isn't load-bearing).

use engram::{Decision, Engram, InMemoryStore, PayloadStore};
use engram_fit::*;
use std::path::PathBuf;

const K: usize = 128;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(name)
}

fn pack(model: &str) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    read_embeddings_file(&fixture(&format!("tests/fixtures/{model}.embs"))).expect("pack fixture")
}

fn pairs(model: &str) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<PairLabel>) {
    let (a, b) = read_embeddings_file(&fixture(&format!("tests/fixtures/{model}.pairs.embs")))
        .expect("pairs fixture");
    let labels: Vec<PairLabel> =
        parse_pairs(&std::fs::read_to_string(fixture("assets/pairs.tsv")).unwrap())
            .unwrap()
            .iter()
            .map(|p| p.label)
            .collect();
    assert_eq!(
        labels.len(),
        a.len(),
        "pairs.tsv and fixture out of sync — re-dump"
    );
    (a, b, labels)
}

// ---------------------------------------------------------------------------
// Space health / anisotropy
// ---------------------------------------------------------------------------

#[test]
fn me5_anisotropy_is_real_and_centering_fixes_it() {
    // The reason engram-fit exists, measured on a real encoder: the raw
    // mE5-small space is collapsed (unrelated sentences share bits well into
    // the Link band); model-level centering restores an order of magnitude
    // of headroom.
    let (mut doc, _) = pack("me5-small-q8");
    let raw = pairwise_jaccard_stats(&codes_from(&doc, 40));
    assert!(
        raw.mean > 0.15,
        "premise changed: raw space no longer collapsed (mean {})",
        raw.mean
    );

    let center = mean_vector(&doc).unwrap();
    apply_center(&mut doc, &center);
    let centered = pairwise_jaccard_stats(&codes_from(&doc, 40));
    assert!(
        centered.mean < 0.05,
        "centering must restore diversity (mean {})",
        centered.mean
    );
    assert!(verdict(centered.mean).starts_with("HEALTHY"));
}

#[test]
fn gemma_raw_space_already_healthy_centering_still_helps() {
    // 2025-era training: the shared-cone anisotropy that cripples mE5 is
    // mostly gone. Centering is no longer load-bearing here — but it must
    // still help (never hurt), and the centered space must be HEALTHY.
    let (mut doc, _) = pack("gemma300m-qat-q4");
    let raw = pairwise_jaccard_stats(&codes_from(&doc, 40));
    assert!(
        verdict(raw.mean).starts_with("HEALTHY"),
        "gemma raw space regressed (mean {})",
        raw.mean
    );

    let center = mean_vector(&doc).unwrap();
    apply_center(&mut doc, &center);
    let centered = pairwise_jaccard_stats(&codes_from(&doc, 40));
    assert!(
        centered.mean <= raw.mean,
        "centering must not hurt (raw {} → centered {})",
        raw.mean,
        centered.mean
    );
    assert!(verdict(centered.mean).starts_with("HEALTHY"));
}

// ---------------------------------------------------------------------------
// Per-model behaviors
// ---------------------------------------------------------------------------

#[test]
fn me5_doc_and_query_prefixes_produce_distinct_means() {
    prefixes_produce_distinct_means("me5-small-q8");
}

#[test]
fn gemma_doc_and_query_prefixes_produce_distinct_means() {
    prefixes_produce_distinct_means("gemma300m-qat-q4");
}

fn prefixes_produce_distinct_means(model: &str) {
    // Asymmetric models put each role in its own sub-cone; per-role centers
    // are not interchangeable.
    let (doc, query) = pack(model);
    let dc = mean_vector(&doc).unwrap();
    let qc = mean_vector(&query).unwrap();
    let dist: f32 = dc
        .iter()
        .zip(&qc)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt();
    assert!(
        dist > 0.01,
        "[{model}] role means unexpectedly identical (dist {dist})"
    );
}

#[test]
fn artifact_is_deterministic() {
    let (doc, query) = pack("me5-small-q8");
    let a = build_artifact(
        "llama.cpp",
        "me5-small-q8",
        "passage: ",
        "query: ",
        &doc,
        &query,
    )
    .unwrap();
    let b = build_artifact(
        "llama.cpp",
        "me5-small-q8",
        "passage: ",
        "query: ",
        &doc,
        &query,
    )
    .unwrap();
    assert_eq!(a.tag, b.tag);
    assert_eq!(a.tag_u64, b.tag_u64);
    assert_eq!(a.dim, 384);
}

#[test]
fn me5_label_ordering_holds_on_real_pairs() {
    label_ordering_holds("me5-small-q8");
}

#[test]
fn gemma_label_ordering_holds_on_real_pairs() {
    label_ordering_holds("gemma300m-qat-q4");
}

fn label_ordering_holds(model: &str) {
    // dup > related > unrelated must hold with real margins, in the space
    // engram would actually run (centered).
    let (mut a, mut b, labels) = pairs(model);
    let (pack_doc, _) = pack(model);
    let center = mean_vector(&pack_doc).unwrap();
    apply_center(&mut a, &center);
    apply_center(&mut b, &center);

    let rows = sweep_k(&a, &b, &labels, &[40, 128]);
    for row in &rows {
        assert!(
            row.dup_mean > 2.0 * row.related_mean,
            "[{model}] k={}: dup ({}) should clearly exceed related ({})",
            row.k,
            row.dup_mean,
            row.related_mean
        );
        assert!(
            row.related_mean > 2.0 * row.unrelated_mean,
            "[{model}] k={}: related ({}) should clearly exceed unrelated ({})",
            row.k,
            row.related_mean,
            row.unrelated_mean
        );
    }
}

#[test]
fn me5_full_circle_gate_decisions_on_real_language() {
    full_circle_gate_decisions("me5-small-q8", "me5-small-q8", |_| {});
}

#[test]
fn gemma_full_circle_gate_decisions_on_real_language() {
    full_circle_gate_decisions("gemma300m-qat-q4", "gemma300m-qat-q4", |_| {});
}

#[test]
fn gemma_f16_full_circle_gate_decisions() {
    // Storage diet (f16 packing — see lifecycle.rs; MRL-512 was measured and
    // rejected) must not change gate behavior.
    full_circle_gate_decisions("gemma300m-qat-q4", "gemma-f16", |v| f16_round_trip(v));
}

#[test]
fn gemma_int8_full_circle_gate_decisions() {
    // sqlite-vec int8 "unit" storage — the production pick (SQL-queryable AND
    // smallest); must not change gate behavior either.
    full_circle_gate_decisions("gemma300m-qat-q4", "gemma-int8", |v| {
        int8_unit_round_trip(v)
    });
}

fn full_circle_gate_decisions(fixture: &str, model: &str, transform: impl Fn(&mut Vec<Vec<f32>>)) {
    // Tune from the fixture, configure a real Engram with the tuned params,
    // and check the gate's DECISIONS on real sentence pairs. Tolerances are
    // deliberate: the asymmetric-loss design guarantees "never destructive"
    // (no false Strengthen), not "always optimal".
    let (mut a, mut b, labels) = pairs(fixture);
    let (mut pack_doc, _) = pack(fixture);
    transform(&mut a);
    transform(&mut b);
    transform(&mut pack_doc);
    let center = mean_vector(&pack_doc).unwrap();
    apply_center(&mut a, &center);
    apply_center(&mut b, &center);
    let mut warm: Vec<Vec<f32>> = pack_doc[..30].to_vec();
    apply_center(&mut warm, &center);

    let cal = &sweep_k(&a, &b, &labels, &[K])[0].calibration;

    // NOTE: pairs.tsv deliberately echoes pack themes (and a few verbatim
    // pack sentences), so a b-side can legitimately Strengthen/Link a WARM
    // episode. The gate metric is therefore pair-partner attribution: what
    // did b decide about its own partner a?
    let mut recognized_dups = 0usize; // dup b bound (Strengthen|Link) to anything stored
    let mut partner_false_strengthens = 0usize; // non-dup b Strengthened ITS PARTNER
    let mut unrelated_partner_binds = 0usize; // unrelated b bound to its partner at all
    let mut related_partner_links = 0usize; // related b LINKED to its partner (positive case)
    let (mut n_dup, mut n_rel, mut n_unrel) = (0usize, 0usize, 0usize);

    for i in 0..a.len() {
        let mut m: Engram<InMemoryStore<String>> = Engram::new(InMemoryStore::new());
        m.params.k = K;
        m.gate.dup_jaccard = cal.suggested_dup;
        m.gate.link_jaccard = cal.suggested_link;
        m.params.recall_anchor = cal.suggested_anchor;

        // Warm the frequency counts so bit-novelty behaves like a real,
        // non-empty memory rather than a cold one.
        for (j, w) in warm.iter().enumerate() {
            let code = m.sparsify(w);
            m.import(code, format!("warm {j}"));
        }

        // a's "home" episode: the one it wrote, or the warm one it merged into.
        let a_home = m.observe(&a[i], "side a".to_string(), 0.0).episode;
        let obs = m.observe(&b[i], "side b".to_string(), 0.0);
        let target = match obs.decision {
            Decision::Strengthen(t) | Decision::Link(t) => Some(t),
            _ => None,
        };
        let bound_to_partner = target.is_some() && target == a_home;

        match labels[i] {
            PairLabel::Dup => {
                n_dup += 1;
                if target.is_some() {
                    recognized_dups += 1;
                }
            }
            PairLabel::Related | PairLabel::Unrelated => {
                if bound_to_partner && matches!(obs.decision, Decision::Strengthen(_)) {
                    partner_false_strengthens += 1;
                }
                if labels[i] == PairLabel::Related {
                    n_rel += 1;
                    if bound_to_partner && matches!(obs.decision, Decision::Link(_)) {
                        related_partner_links += 1;
                    }
                }
                if labels[i] == PairLabel::Unrelated {
                    n_unrel += 1;
                    if bound_to_partner {
                        unrelated_partner_binds += 1;
                    }
                }
            }
        }
    }

    eprintln!(
        "[{model}] dups recognized {recognized_dups}/{n_dup}, related linked \
         {related_partner_links}/{n_rel}, unrelated binds {unrelated_partner_binds}/{n_unrel}, \
         false strengthens {partner_false_strengthens}"
    );

    // Never destructive: Strengthen (which drops the payload) must not fire
    // on a non-duplicate pair partner.
    assert!(
        partner_false_strengthens <= 1,
        "[{model}] {partner_false_strengthens} non-dup partners Strengthened — payload loss"
    );
    // Paraphrases are recognized as same-or-related, not written as strangers.
    assert!(
        recognized_dups * 10 >= n_dup * 6,
        "[{model}] only {recognized_dups}/{n_dup} dups recognized"
    );
    // Unrelated partners stay strangers.
    assert!(
        unrelated_partner_binds <= 2,
        "[{model}] {unrelated_partner_binds}/{n_unrel} unrelated pairs bound to their partner"
    );
    // The Link band does POSITIVE work on real topic-mates, not just
    // avoid-harm: a meaningful share of related pairs Hebbian-link.
    assert!(
        related_partner_links * 10 >= n_rel * 3,
        "[{model}] only {related_partner_links}/{n_rel} related pairs linked to their partner"
    );
}

#[test]
fn me5_full_circle_query_retrieval_on_real_language() {
    full_circle_query_retrieval("me5-small-q8", "me5-small-q8", |_| {});
}

#[test]
fn gemma_full_circle_query_retrieval_on_real_language() {
    full_circle_query_retrieval("gemma300m-qat-q4", "gemma300m-qat-q4", |_| {});
}

#[test]
fn gemma_f16_full_circle_query_retrieval() {
    full_circle_query_retrieval("gemma300m-qat-q4", "gemma-f16", |v| f16_round_trip(v));
}

#[test]
fn gemma_int8_full_circle_query_retrieval() {
    full_circle_query_retrieval("gemma300m-qat-q4", "gemma-int8", |v| {
        int8_unit_round_trip(v)
    });
}

fn full_circle_query_retrieval(fixture: &str, model: &str, transform: impl Fn(&mut Vec<Vec<f32>>)) {
    // The READ half of the circle, on the asymmetric roles these models
    // exist for: store passages (doc prefix, doc center), query with the same
    // sentences' QUERY-role embeddings (query prefix, query center). Known
    // content must come back as the top hit; held-out (never stored) queries
    // must get SILENCE — the injection gate refusing at capacity is the whole
    // point of the read design.
    let (mut doc, mut query) = pack(fixture);
    transform(&mut doc);
    transform(&mut query);
    let doc_center = mean_vector(&doc).unwrap();
    let query_center = mean_vector(&query).unwrap();
    apply_center(&mut doc, &doc_center);
    apply_center(&mut query, &query_center);

    const STORED: usize = 150; // rest are held-out unknowns

    // Anchor from the unrelated-pairs calibration: it blocks pure
    // COINCIDENCE. It deliberately does NOT block same-topic cousins —
    // real question→answer jaccard overlaps the cousin band at these model
    // scales, so a cousin-proof anchor would refuse real answers too
    // (measured; see the lifecycle test).
    let (mut pa, mut pb, plabels) = pairs(fixture);
    transform(&mut pa);
    transform(&mut pb);
    apply_center(&mut pa, &doc_center);
    apply_center(&mut pb, &doc_center);
    let cal = &sweep_k(&pa, &pb, &plabels, &[K])[0].calibration;

    let mut m: Engram<InMemoryStore<String>> = Engram::new(InMemoryStore::new());
    m.params.k = K;
    m.params.recall_anchor = cal.suggested_anchor;
    m.params.recall_floor = 0.05;

    for (i, d) in doc[..STORED].iter().enumerate() {
        let code = m.sparsify(d);
        m.import(code, format!("sentence {i}"));
    }

    // Stored content, queried by its query-role embedding → top hit is itself.
    let mut top1 = 0usize;
    for (i, q) in query[..STORED].iter().enumerate() {
        let hits = m.recall_with_demand(q, 0.9, 4);
        if hits
            .first()
            .and_then(|h| m.store.get(*h))
            .is_some_and(|p| p == format!("sentence {i}"))
        {
            top1 += 1;
        }
    }

    // Held-out queries: NOT stored, but the pack is a topically clustered
    // little world, so many held-outs have stored same-topic cousins the
    // coincidence anchor rightly lets through (they're context on a
    // high-demand query, not junk). The assertion is therefore modest here;
    // the strict silence test — genuinely unrelated question-style queries —
    // lives in the lifecycle eval.
    let mut silent = 0usize;
    let held_out = &query[STORED..];
    for q in held_out {
        if m.recall_with_demand(q, 0.9, 4).is_empty() {
            silent += 1;
        }
    }

    eprintln!(
        "[{model}] cross-role top-1 {top1}/{STORED}, held-out silence {silent}/{}",
        held_out.len()
    );

    assert!(
        top1 * 100 >= STORED * 90,
        "[{model}] cross-role retrieval too weak: {top1}/{STORED} top-1"
    );
    assert!(
        silent * 100 >= held_out.len() * 30,
        "[{model}] even cousin-tolerant retrieval should refuse some held-outs: {silent}/{}",
        held_out.len()
    );

    // Close the loop end-to-end: feedback on a real hit trains the gates,
    // and sleep afterwards keeps the used episode alive.
    let hits = m.recall_with_demand(&query[0], 0.9, 4);
    assert!(!hits.is_empty());
    let (w0, r0) = (m.gate.threshold, m.read_gate.threshold);
    m.feedback(&hits[..1]);
    assert!(
        m.gate.threshold < w0,
        "[{model}] useful hit must reward the write gate"
    );
    assert!(
        m.read_gate.threshold < r0,
        "[{model}] useful hit must reward the read gate"
    );
    let used_payload = hits[0];
    for _ in 0..3 {
        m.sleep();
    }
    assert!(
        m.graph()
            .iter()
            .any(|(_, ep)| ep.payload == used_payload && ep.used),
        "[{model}] the used episode must survive consolidation"
    );
}
