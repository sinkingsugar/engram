//! Diagnostics (print-only, run with `-- --nocapture`): the measured jaccard
//! regimes behind the anchor calibration. For every stream query, the best
//! jaccard of a CORRECT answer vs the best cousin; plus the labeled
//! unrelated-pairs tail. This is the data that set `suggested_anchor` to
//! unrelated-MAX × 1.5 (q95 at n=20 discards the tail — it let real
//! coincidence through for gemma) and that shows NO threshold separates
//! answers from cousins on either model (gemma min answer 0.089 < gemma max
//! cousin 0.138): the anchor blocks coincidence, cousins are context.

use engram_fit::*;
use std::path::PathBuf;

fn asset(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(name)
}

#[test]
fn probe_unrelated_pairs_tail() {
    for model in ["me5-small-q8", "gemma300m-qat-q4"] {
        let (mut a, mut b) =
            read_embeddings_file(&asset(&format!("tests/fixtures/{model}.pairs.embs"))).unwrap();
        let labels: Vec<PairLabel> =
            parse_pairs(&std::fs::read_to_string(asset("assets/pairs.tsv")).unwrap())
                .unwrap()
                .iter()
                .map(|p| p.label)
                .collect();
        let (pack_doc, _) =
            read_embeddings_file(&asset(&format!("tests/fixtures/{model}.embs"))).unwrap();
        let center = mean_vector(&pack_doc).unwrap();
        apply_center(&mut a, &center);
        apply_center(&mut b, &center);
        let ca = codes_from(&a, 128);
        let cb = codes_from(&b, 128);
        let mut unrel: Vec<f32> = labels
            .iter()
            .enumerate()
            .filter(|(_, l)| **l == PairLabel::Unrelated)
            .map(|(i, _)| ca[i].jaccard(&cb[i]))
            .collect();
        unrel.sort_by(f32::total_cmp);
        eprintln!(
            "{model}: unrelated pairs n={} median={:.3} q95={:.3} max={:.3}",
            unrel.len(),
            unrel[unrel.len() / 2],
            unrel[(unrel.len() as f32 * 0.95) as usize],
            unrel.last().unwrap()
        );
    }
}

#[test]
fn probe_mrl512_f16_vs_768() {
    // The storage-diet question: what do truncation+f16 cost in band
    // separation? Prints the k=128 sweep row for both variants.
    let (mut a768, mut b768) =
        read_embeddings_file(&asset("tests/fixtures/gemma300m-qat-q4.pairs.embs")).unwrap();
    let labels: Vec<PairLabel> =
        parse_pairs(&std::fs::read_to_string(asset("assets/pairs.tsv")).unwrap())
            .unwrap()
            .iter()
            .map(|p| p.label)
            .collect();
    let (mut pack768, _) =
        read_embeddings_file(&asset("tests/fixtures/gemma300m-qat-q4.embs")).unwrap();

    for (label, diet) in [("768-f32", false), ("mrl512-f16", true)] {
        let (mut a, mut b, mut pack) = (a768.clone(), b768.clone(), pack768.clone());
        if diet {
            for v in [&mut a, &mut b, &mut pack] {
                truncate_normalize(v, 512);
                f16_round_trip(v);
            }
        }
        let center = mean_vector(&pack).unwrap();
        apply_center(&mut a, &center);
        apply_center(&mut b, &center);
        apply_center(&mut pack, &center);
        let row = &sweep_k(&a, &b, &labels, &[128])[0];
        let health = pairwise_jaccard_stats(&codes_from(&pack, 40));
        eprintln!(
            "{label:>11}: dup {:.3} related {:.3} unrel {:.3} | anchor {:.2} dup-band {:.2} | space mean {:.3} ({})",
            row.dup_mean,
            row.related_mean,
            row.unrelated_mean,
            row.calibration.suggested_anchor,
            row.calibration.suggested_dup,
            health.mean,
            verdict(health.mean)
        );
    }
    // keep borrows alive for clarity of the clone-per-variant pattern
    let _ = (&mut a768, &mut b768, &mut pack768);
}

#[test]
fn probe_answer_vs_cousin_regimes() {
    for (model, diet) in [
        ("me5-small-q8", false),
        ("gemma300m-qat-q4", false),
        ("gemma300m-qat-q4", true),
    ] {
        let (mut pack_doc, mut pack_query) =
            read_embeddings_file(&asset(&format!("tests/fixtures/{model}.embs"))).unwrap();
        let (mut msg_embs, mut query_embs) =
            read_stream_embeddings(&asset(&format!("tests/fixtures/{model}.stream.embs"))).unwrap();
        if diet {
            for v in [
                &mut pack_doc,
                &mut pack_query,
                &mut msg_embs,
                &mut query_embs,
            ] {
                truncate_normalize(v, 512);
                f16_round_trip(v);
            }
        }
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

        let msg_codes = codes_from(&msg_embs, 128);
        let query_codes = codes_from(&query_embs, 128);

        eprintln!("\n== {model} diet={diet} (k=128, centered) ==");
        let (mut msg_i, mut query_i) = (0usize, 0usize);
        for item in &items {
            match item {
                StreamItem::Message(_) => msg_i += 1,
                StreamItem::Query { text, expected } => {
                    let expected = resolve_expected(expected, &messages).unwrap();
                    let mut best_ans = 0f32;
                    let mut best_other = 0f32;
                    for (j, code) in msg_codes.iter().take(msg_i).enumerate() {
                        let jac = query_codes[query_i].jaccard(code);
                        if expected.contains(&j) {
                            best_ans = best_ans.max(jac);
                        } else {
                            best_other = best_other.max(jac);
                        }
                    }
                    let kind = if expected.is_empty() {
                        "SILENCE"
                    } else {
                        "answer "
                    };
                    eprintln!(
                        "  {kind} ans={best_ans:.3} cousin={best_other:.3}  {}",
                        &text[..text.len().min(52)]
                    );
                    query_i += 1;
                }
                StreamItem::Sleep => {}
            }
        }
    }
}
