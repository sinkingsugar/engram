//! The "does the system actually work" test: six simulated weeks of a
//! fictional person's messages (synthetic text, real embeddings,
//! fixture-frozen), run through the FULL engram loop — observe → passive
//! channel → deliberate recall → feedback → weekly sleep — and scored
//! against a brute-force cosine-RAG baseline that stores everything and
//! always answers.
//!
//! Parameterized over fitted models; each gets its own #[test] so a
//! regression names the model that broke.
//!
//! What "works" means here:
//!   1. Answerable queries hit (any expected message in the returned handles).
//!   2. Silence queries (never-discussed topics) inject NOTHING — the failure
//!      mode RAG cannot avoid by construction.
//!   3. The memory is COMPACT: restatements strengthen/link instead of piling
//!      up; sleep prunes unused trivia; used facts survive.
//!   4. The reward loop moves the learned thresholds sensibly.

use engram::{Decision, IdentityStore, Memory};
use engram_fit::*;
use std::path::PathBuf;

fn asset(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(name)
}

struct Eval {
    week: usize,
    query: String,
    expected: Vec<usize>,
    engram_hits: Vec<usize>,
    rag_hits: Vec<usize>,
}

#[test]
fn lifecycle_six_weeks_me5_small_q8() {
    run_lifecycle("me5-small-q8", "me5-small-q8", |_| {});
}

#[test]
fn lifecycle_six_weeks_gemma300m_qat_q4() {
    run_lifecycle("gemma300m-qat-q4", "gemma300m-qat-q4", |_| {});
}

#[test]
fn lifecycle_six_weeks_gemma_f16() {
    // The storage diet (f16 packing, 3KB → 1.5KB per message) must not change
    // the verdict. Transform order matches production: f16 round-trip →
    // center → gate. (MRL-512 was measured too and REJECTED: it sank the
    // cross-lingual answer below the coincidence tail — see the runbook.)
    run_lifecycle("gemma300m-qat-q4", "gemma-f16", |v| {
        f16_round_trip(v);
    });
}

#[test]
fn lifecycle_six_weeks_gemma_int8() {
    // sqlite-vec int8 "unit" storage (768B per message, 4× smaller than f32,
    // and vec_distance_cosine-queryable) — the coarsest diet that keeps the
    // vectors SQL-searchable. Must hold the verdict like f16 does.
    run_lifecycle("gemma300m-qat-q4", "gemma-int8", |v| {
        int8_unit_round_trip(v);
    });
}

fn run_lifecycle(fixture: &str, model: &str, transform: impl Fn(&mut Vec<Vec<f32>>)) {
    // ---- load fixtures, center per role (as production would) ----
    let (mut pack_doc, mut pack_query) =
        read_embeddings_file(&asset(&format!("tests/fixtures/{fixture}.embs"))).unwrap();
    let (mut msg_embs, mut query_embs) =
        read_stream_embeddings(&asset(&format!("tests/fixtures/{fixture}.stream.embs"))).unwrap();
    transform(&mut pack_doc);
    transform(&mut pack_query);
    transform(&mut msg_embs);
    transform(&mut query_embs);
    let doc_center = mean_vector(&pack_doc).unwrap();
    let query_center = mean_vector(&pack_query).unwrap();
    apply_center(&mut msg_embs, &doc_center);
    apply_center(&mut query_embs, &query_center);

    let items = parse_stream(default_stream()).unwrap();
    let messages: Vec<String> = items
        .iter()
        .filter_map(|i| match i {
            StreamItem::Message(m) => Some(m.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        messages.len(),
        msg_embs.len(),
        "stream.tsv changed — re-dump the fixture"
    );

    // ---- engram configured exactly as the runbook prescribes: k=128 plus
    // the pairs-derived calibration for THIS model (dup/link structure bands,
    // coincidence anchor). Real question→answer jaccard lands INSIDE the
    // same-topic cousin band at these model scales — the anchor blocks
    // coincidence, not cousins (a cousin-proof anchor would refuse real
    // answers too; measured, the hard way, on mE5).
    const K: usize = 128;
    let (mut pa, mut pb, plabels) = {
        let (a, b) =
            read_embeddings_file(&asset(&format!("tests/fixtures/{fixture}.pairs.embs"))).unwrap();
        let labels: Vec<PairLabel> =
            parse_pairs(&std::fs::read_to_string(asset("assets/pairs.tsv")).unwrap())
                .unwrap()
                .iter()
                .map(|p| p.label)
                .collect();
        (a, b, labels)
    };
    transform(&mut pa);
    transform(&mut pb);
    apply_center(&mut pa, &doc_center);
    apply_center(&mut pb, &doc_center);
    let cal = &sweep_k(&pa, &pb, &plabels, &[K])[0].calibration;

    let mut m: Memory<32, IdentityStore<u64>> = Memory::new(IdentityStore::new());
    m.params.k = K;
    m.gate.dup_jaccard = cal.suggested_dup;
    m.gate.link_jaccard = cal.suggested_link;
    m.params.recall_anchor = cal.suggested_anchor;
    m.params.recall_floor = 0.05;
    // Passive-channel threshold scaled to the measured activation range at
    // k=128 (defaults are synthetic-geometry calibrated).
    m.params.surface_threshold = 0.12;

    let (w0, r0) = (m.gate.threshold, m.read_gate.threshold);
    let mut evals: Vec<Eval> = Vec::new();
    let (mut msg_i, mut query_i, mut week) = (0usize, 0usize, 1usize);
    let mut decisions = [0usize; 4]; // WriteNew, Link, Strengthen, Skip
    let mut passive_surfaced = 0usize;
    let mut total_pruned = 0usize;

    for item in &items {
        match item {
            StreamItem::Message(text) => {
                let demand = engram::heuristic_demand(text);
                let obs = m.observe(&msg_embs[msg_i], msg_i as u64, demand);
                passive_surfaced += obs.surfaced.len();
                match obs.decision {
                    Decision::WriteNew => decisions[0] += 1,
                    Decision::Link(_) => decisions[1] += 1,
                    Decision::Strengthen(_) => decisions[2] += 1,
                    Decision::Skip => decisions[3] += 1,
                }
                msg_i += 1;
            }
            StreamItem::Query { text, expected } => {
                let expected = resolve_expected(expected, &messages).unwrap();
                let hits = m.recall_with_demand(&query_embs[query_i], 0.9, 4);
                let engram_hits: Vec<usize> = hits.iter().map(|&h| h as usize).collect();

                // Feedback: the caller "used" a hit iff it answered the question.
                let used: Vec<u64> = hits
                    .iter()
                    .copied()
                    .filter(|h| expected.contains(&(*h as usize)))
                    .take(1)
                    .collect();
                if !hits.is_empty() {
                    m.feedback(&used);
                }

                // RAG baseline: cosine top-4 over EVERYTHING seen so far,
                // no gate, always answers.
                let mut scored: Vec<(usize, f32)> = (0..msg_i)
                    .map(|j| {
                        let dot: f32 = query_embs[query_i]
                            .iter()
                            .zip(&msg_embs[j])
                            .map(|(a, b)| a * b)
                            .sum();
                        (j, dot)
                    })
                    .collect();
                scored.sort_by(|a, b| b.1.total_cmp(&a.1));
                let rag_hits: Vec<usize> = scored.iter().take(4).map(|&(j, _)| j).collect();

                evals.push(Eval {
                    week,
                    query: text.clone(),
                    expected,
                    engram_hits,
                    rag_hits,
                });
                query_i += 1;
            }
            StreamItem::Sleep => {
                total_pruned += m.sleep().pruned;
                week += 1;
            }
        }
    }

    // ---- score ----
    let mut table = String::new();
    let (mut ans_total, mut ans_engram, mut ans_rag) = (0, 0, 0);
    let (mut sil_total, mut sil_engram_ok, mut sil_rag_ok) = (0, 0, 0);
    for e in &evals {
        let hit = |hits: &[usize]| hits.iter().any(|h| e.expected.contains(h));
        let (verdict_e, verdict_r);
        if e.expected.is_empty() {
            sil_total += 1;
            let ok_e = e.engram_hits.is_empty();
            let ok_r = e.rag_hits.is_empty(); // never true: RAG has no gate
            sil_engram_ok += ok_e as usize;
            sil_rag_ok += ok_r as usize;
            verdict_e = if ok_e { "SILENT✓" } else { "JUNK✗" };
            verdict_r = if ok_r { "SILENT✓" } else { "JUNK✗" };
        } else {
            ans_total += 1;
            let ok_e = hit(&e.engram_hits);
            let ok_r = hit(&e.rag_hits);
            ans_engram += ok_e as usize;
            ans_rag += ok_r as usize;
            verdict_e = if ok_e {
                "HIT✓"
            } else if e.engram_hits.is_empty() {
                "silent✗"
            } else {
                "MISS✗"
            };
            verdict_r = if ok_r { "HIT✓" } else { "MISS✗" };
        }
        table.push_str(&format!(
            "  w{} {:60} engram: {:8} rag: {:8}\n",
            e.week,
            &e.query[..e.query.len().min(60)],
            verdict_e,
            verdict_r
        ));
    }

    let alive = m.graph().len();
    eprintln!("\n==== lifecycle report [{model}] ====");
    eprint!("{table}");
    eprintln!("  answerable: engram {ans_engram}/{ans_total}  rag {ans_rag}/{ans_total}");
    eprintln!("  silence:    engram {sil_engram_ok}/{sil_total}  rag {sil_rag_ok}/{sil_total}");
    eprintln!(
        "  memory: {alive} episodes for {} messages ({}% compression), {} pruned",
        messages.len(),
        100 - alive * 100 / messages.len(),
        total_pruned
    );
    eprintln!(
        "  decisions: {} new, {} link, {} strengthen, {} skip; {} passive surfacings",
        decisions[0], decisions[1], decisions[2], decisions[3], passive_surfaced
    );
    eprintln!(
        "  thresholds: write {:.3}→{:.3}, read {:.3}→{:.3}",
        w0, m.gate.threshold, r0, m.read_gate.threshold
    );

    // ---- the verdict, as assertions ----
    // Engram answers at least as well as it must (recurring + once-stated
    // facts), and matches RAG within one hit.
    assert!(
        ans_engram * 10 >= ans_total * 6,
        "[{model}] answerable hit rate too low: {ans_engram}/{ans_total}"
    );
    assert!(
        ans_engram + 1 >= ans_rag,
        "[{model}] engram ({ans_engram}) lags RAG ({ans_rag}) by more than one hit"
    );
    // The gate does what RAG cannot: silence on never-discussed topics.
    // One boundary leak is tolerated and known (a true cousin at the anchor
    // boundary, injected as weak context, not fabrication).
    assert!(
        sil_engram_ok + 1 >= sil_total,
        "[{model}] engram injected junk on {}/{sil_total} silence queries",
        sil_total - sil_engram_ok
    );
    assert!(
        sil_rag_ok == 0,
        "baseline sanity: ungated RAG always injects"
    );
    // The memory is compact: restatements merged, not piled up.
    assert!(
        decisions[1] + decisions[2] + decisions[3] >= 8,
        "[{model}] dedup machinery barely fired: {decisions:?}"
    );
    assert!(
        alive < messages.len(),
        "[{model}] no forgetting happened at all"
    );
    assert!(
        total_pruned > 0,
        "[{model}] the quiet-weeks tail should push unreinforced trivia past the floor"
    );
    // The reward loop actually learned something.
    assert!(
        (m.gate.threshold - w0).abs() > 1e-6 || (m.read_gate.threshold - r0).abs() > 1e-6,
        "[{model}] thresholds never moved"
    );
}
