//! End-to-end behavioral spec (§8 of the brief), driven through the `Memory`
//! orchestrator with hand-built codes (full control over overlap) and with
//! dense vectors through the fly-hash path.

use engram::{
    Decision, Engram, IdentityStore, InMemoryStore, Memory, PayloadStore, RecallAction, SparseCode,
};

type Code = SparseCode<32>; // 2048 bits

fn code(bits: impl IntoIterator<Item = usize>) -> Code {
    Code::from_bits(&bits.into_iter().collect::<Vec<_>>())
}

fn memory() -> Engram<InMemoryStore<String>> {
    Engram::new(InMemoryStore::new())
}

/// Payload sugar: the InMemoryStore keeps owned Strings.
fn p(s: &str) -> String {
    s.to_string()
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 16
    }

    fn code(&mut self) -> Code {
        let bits: Vec<usize> = (0..40).map(|_| (self.next() % 2048) as usize).collect();
        Code::from_bits(&bits)
    }

    fn dense(&mut self, dim: usize) -> Vec<f32> {
        (0..dim)
            .map(|_| (self.next() as f32 / (u64::MAX >> 16) as f32) * 2.0 - 1.0)
            .collect()
    }
}

#[test]
fn gate_behavior_dup_related_unrelated() {
    let mut m = memory();

    // First sight: novel → write.
    let a = m.observe_code(code(0..40), p("misting runs every 15 minutes"), 0.0);
    assert_eq!(a.decision, Decision::WriteNew);

    // Exact duplicate → Strengthen, the write you DON'T do.
    let dup = m.observe_code(code(0..40), p("misting runs every 15 minutes"), 0.0);
    assert!(matches!(dup.decision, Decision::Strengthen(_)));
    assert_eq!(m.graph().len(), 1);

    // Related (jaccard ≈ 0.43) → write + Hebbian link.
    let related_code = code((0..24).chain(100..116));
    let related = m.observe_code(related_code, p("fogger interval note"), 0.0);
    assert!(matches!(related.decision, Decision::Link(_)));
    assert_eq!(m.graph().len(), 2);

    // Unrelated → plain write.
    let unrelated = m.observe_code(code(1000..1040), p("cacti need draining soil"), 0.0);
    assert_eq!(unrelated.decision, Decision::WriteNew);
    assert_eq!(m.graph().len(), 3);
}

#[test]
fn gate_behavior_through_dense_vectors() {
    let mut m = memory();
    let mut rng = Lcg(7);
    let base = rng.dense(384);

    let first = m.observe(&base, p("first sighting"), 0.0);
    assert_eq!(first.decision, Decision::WriteNew);

    // Identical embedding → near-duplicate.
    let again = m.observe(&base, p("same thing again"), 0.0);
    assert!(matches!(again.decision, Decision::Strengthen(_)));

    // Small perturbation (a paraphrase-grade embedding shift) must NOT create
    // a fresh unlinked episode.
    let noise = rng.dense(384);
    let similar: Vec<f32> = base.iter().zip(&noise).map(|(b, n)| b + 0.1 * n).collect();
    let close = m.observe(&similar, p("slightly reworded"), 0.0);
    assert!(
        !matches!(close.decision, Decision::WriteNew),
        "near-identical embedding wrote a disconnected episode: {:?}",
        close.decision
    );

    // A fresh random embedding is unrelated → new episode.
    let other = m.observe(&rng.dense(384), p("different topic"), 0.0);
    assert_eq!(other.decision, Decision::WriteNew);
}

#[test]
fn saturation_regression_novelty_survives_scale() {
    // The decision-9 test: after hundreds of diverse episodes, a brand-new
    // code must STILL look novel. A binary Bloom union fails this.
    let mut m = memory();
    let mut rng = Lcg(42);
    for i in 0..500 {
        m.observe_code(rng.code(), p(&format!("episode {i}")), 0.0);
    }
    let fresh = rng.code();
    let novelty = m.novelty_estimate(&fresh);
    assert!(
        novelty > 0.35,
        "novelty saturated after 500 episodes: {novelty}"
    );
}

#[test]
fn consolidation_prunes_unused_keeps_strengthened() {
    let mut m = memory();
    let hot = code(0..40);
    m.observe_code(hot.clone(), p("recurring fact"), 0.0);
    m.observe_code(code(500..540), p("one-off trivia"), 0.0);

    let threshold_before = m.gate.threshold;
    for _ in 0..6 {
        m.observe_code(hot.clone(), p("recurring fact"), 0.0);
    }
    let mut pruned = 0;
    for _ in 0..10 {
        pruned += m.sleep().pruned;
    }
    assert_eq!(pruned, 1, "exactly the one-off should fade");
    assert_eq!(m.graph().len(), 1);
    // Forgetting reclaims durable storage too (store.remove was called).
    assert_eq!(m.store.len(), 1, "pruned payload should be evicted");
    // The recurring fact is still recognized after consolidation.
    let after = m.observe_code(hot, p("recurring fact"), 0.0);
    assert!(matches!(after.decision, Decision::Strengthen(_)));
    // The dead write charged the write gate: the bar went up.
    assert!(m.gate.threshold > threshold_before);
}

#[test]
fn passive_surfacing_and_refractory_brake() {
    let mut m = memory();
    let a = code(0..40);
    m.observe_code(a, p("the penthouse has a rooftop terrace"), 0.0);

    // A related-but-distinct cue (Link band) should get "reminded" of A...
    let cue = code((0..24).chain(200..216));
    let obs = m.observe_code(cue, p("terrace planning note"), 0.0);
    assert!(matches!(obs.decision, Decision::Link(_)));
    assert_eq!(obs.surfaced.len(), 1, "should surface the associate");
    let surfaced_payload = obs.surfaced[0].payload;
    assert_eq!(
        m.store.get(surfaced_payload).unwrap(),
        "the penthouse has a rooftop terrace"
    );

    // ...but re-observing the same cue immediately must NOT re-surface it:
    // it's already in context (refractory / repetition suppression).
    let again = m.observe_code(code((0..24).chain(200..216)), p("terrace again"), 0.0);
    assert!(
        again.surfaced.is_empty(),
        "refractory brake failed: {:?}",
        again.surfaced
    );
}

#[test]
fn surfacing_budget_is_respected() {
    let mut m = memory();
    // Ten episodes all sharing bits with the cue.
    for i in 0..10 {
        let c = code((0..30).chain(300 + i * 40..300 + i * 40 + 10));
        m.observe_code(c, p(&format!("variant {i}")), 0.0);
    }
    let obs = m.observe_code(code((0..30).chain(1500..1510)), p("probe"), 0.0);
    assert!(
        obs.surfaced.len() <= m.params.surface_budget,
        "budget exceeded: {} surfaced",
        obs.surfaced.len()
    );
}

#[test]
fn lowering_inhibition_widens_surfacing() {
    let build = |inhibition: f32| {
        let mut m = memory();
        for i in 0..10 {
            let c = code((0..18).chain(300 + i * 40..300 + i * 40 + 22));
            m.observe_code(c, p(&format!("remote associate {i}")), 0.0);
        }
        m.set_inhibition(inhibition);
        let obs = m.observe_code(code((0..18).chain(1500..1522)), p("probe"), 0.0);
        obs.surfaced.len()
    };
    let focused = build(1.0);
    let loose = build(0.2);
    assert!(
        loose > focused,
        "mescaline mode should surface more: focused={focused}, loose={loose}"
    );
}

#[test]
fn read_gate_suppresses_and_retrieves() {
    let mut m = memory();
    let a = code(0..40);
    m.observe_code(a.clone(), p("the fact"), 0.0);
    m.observe_code(a.clone(), p("the fact"), 0.0); // strengthen → well known

    // Known content, explicit query → retrieval fires.
    let hits = m.recall_code(&a, 4);
    assert!(!hits.is_empty(), "known content should retrieve");
    assert_eq!(m.store.get(hits[0]).unwrap(), "the fact");

    // Unknown content, explicit query → knowledge gap, nothing fabricated.
    let unknown = code(1500..1540);
    let hits = m.recall_code(&unknown, 4);
    assert!(hits.is_empty(), "unknown content must not inject anything");

    // Low demand suppresses even known content (the Suppress cell).
    assert_eq!(m.should_recall_code(&a, 0.1), RecallAction::Skip);
}

#[test]
fn knowledge_gap_triggers_eager_write() {
    let mut m = memory();
    // Ask about something we don't have → gap flagged.
    let unknown = code(900..940);
    let hits = m.recall_code(&unknown, 4);
    assert!(hits.is_empty());
    // The answer arrives next; even if the write gate would normally be shy,
    // the pending gap forces the write.
    let obs = m.observe_code(code(900..940), p("the answer that arrived"), 0.9);
    assert!(matches!(
        obs.decision,
        Decision::WriteNew | Decision::Link(_) | Decision::Strengthen(_)
    ));
    assert!(!m.graph().is_empty());
}

#[test]
fn knowledge_gap_ttl_and_relatedness() {
    // P1.1 acceptance: the gap has a fuse and a subject — it force-writes the
    // answer to the question we asked, not the next thing through the door.
    let mut m = memory();
    // Familiar background so a familiar-mix probe sits in the Skip band.
    let a1 = code(0..40);
    let a2 = code(40..80);
    for _ in 0..6 {
        m.observe_code(a1.clone(), p("bg1"), 0.0);
        m.observe_code(a2.clone(), p("bg2"), 0.0);
    }
    m.gate.threshold = 0.6; // shy write gate (max clamp)

    // Premise: half-familiar, unrelated-to-any-episode content gets Skipped.
    let d = code((1000..1016).chain(0..12).chain(40..52));
    let obs = m.observe_code(d.clone(), p("no gap armed"), 0.0);
    assert_eq!(
        obs.decision,
        Decision::Skip,
        "premise: D sits in the Skip band"
    );

    // Arm the gap with an unknown query Q; D shares 16/40 bits with Q.
    let q = code(1000..1040);
    assert!(m.recall_code(&q, 4).is_empty());
    let obs = m.observe_code(d, p("the answer"), 0.0);
    assert_eq!(
        obs.decision,
        Decision::WriteNew,
        "gap must force the RELATED write"
    );

    // An unrelated Skip while a fresh gap is armed must NOT be force-written.
    let q3 = code(1500..1540);
    assert!(m.recall_code(&q3, 4).is_empty());
    let d3 = code((1700..1716).chain(24..36).chain(64..76));
    let obs = m.observe_code(d3, p("unrelated to the gap"), 0.0);
    assert_eq!(obs.decision, Decision::Skip, "the gap is not a blank check");

    // TTL: a gap left unanswered past its fuse stops forcing anything.
    let q2 = code(1300..1340);
    assert!(m.recall_code(&q2, 4).is_empty());
    for _ in 0..40 {
        // Low-demand ticks (below the demand floor: no retrieval, no re-arm).
        let _ = m.recall_code_with_demand(&code(1900..1940), 0.1, 1);
    }
    let d2 = code((1300..1316).chain(12..24).chain(52..64));
    let obs = m.observe_code(d2, p("answer, but too late"), 0.0);
    assert_eq!(
        obs.decision,
        Decision::Skip,
        "expired gap must not force-write"
    );
}

#[test]
fn feedback_trains_both_gates() {
    let mut m = memory();
    let a = code(0..40);
    m.observe_code(a.clone(), p("useful fact"), 0.0);
    m.observe_code(a.clone(), p("useful fact"), 0.0);

    let write_t = m.gate.threshold;
    let read_t = m.read_gate.threshold;
    let hits = m.recall_code(&a, 4);
    assert!(!hits.is_empty());
    m.feedback(&hits);
    assert!(
        m.gate.threshold < write_t,
        "useful memory should lower the write bar"
    );
    assert!(
        m.read_gate.threshold < read_t,
        "helpful retrieval should lower the read bar"
    );

    // A wasted retrieval trains the read gate the other way.
    let read_t = m.read_gate.threshold;
    let hits = m.recall_code(&a, 4);
    assert!(!hits.is_empty());
    m.feedback(&[]);
    assert!(
        m.read_gate.threshold > read_t,
        "wasted retrieval should raise the read bar"
    );
}

#[test]
fn passive_surface_can_be_rewarded() {
    // P1.5a: the proactive channel is trainable — feedback on a passively
    // surfaced item marks/strengthens it and moves both gates.
    let mut m = memory();
    let a_obs = m.observe_code(code(0..40), p("the penthouse fact"), 0.0);
    let a_id = a_obs.episode.unwrap();

    let obs = m.observe_code(code((0..24).chain(200..216)), p("related note"), 0.0);
    assert_eq!(obs.surfaced.len(), 1, "premise: A surfaces passively");
    assert_eq!(obs.surfaced[0].episode, a_id);

    let count_before = m.graph().get(a_id).unwrap().count;
    let w0 = m.gate.threshold;
    let r0 = m.read_gate.threshold;
    m.feedback(&[obs.surfaced[0].payload]);
    let a = m.graph().get(a_id).unwrap();
    assert!(a.used, "surfaced episode must be marked used");
    assert!(
        a.count > count_before,
        "surfaced episode must be strengthened"
    );
    assert!(m.gate.threshold < w0);
    assert!(m.read_gate.threshold < r0);
}

#[test]
fn feedback_after_sleep_is_inert() {
    // P1.5b: consolidation invalidates slot-based feedback targets; delayed
    // feedback across a sleep must do nothing (never mark a reused slot).
    let mut m = memory();
    let a = code(0..40);
    m.observe_code(a.clone(), p("fact"), 0.0);
    m.observe_code(a.clone(), p("fact"), 0.0);
    let hits = m.recall_code(&a, 4);
    assert!(!hits.is_empty());
    m.sleep();
    let w = m.gate.threshold;
    let r = m.read_gate.threshold;
    m.feedback(&hits);
    assert_eq!(
        m.gate.threshold, w,
        "stale feedback must not train the write gate"
    );
    assert_eq!(
        m.read_gate.threshold, r,
        "stale feedback must not train the read gate"
    );
}

#[test]
fn resolving_high_demand_answer_does_not_rearm_gap() {
    // P1.6: a knowledge gap is "wanted it, don't have it" — an observation
    // that just wrote the answer must not immediately re-flag it as missing.
    let mut m = memory();
    let a1 = code(0..40);
    let a2 = code(40..80);
    for _ in 0..6 {
        m.observe_code(a1.clone(), p("bg1"), 0.0);
        m.observe_code(a2.clone(), p("bg2"), 0.0);
    }
    m.gate.threshold = 0.6; // shy gate so the follow-up probe sits in Skip band

    let q = code(1000..1040);
    assert!(m.recall_code_with_demand(&q, 0.9, 4).is_empty()); // arms the gap
    let obs = m.observe_code(q.clone(), p("the answer"), 0.9); // resolves it
    assert!(obs.episode.is_some());
    assert!(
        !obs.knowledge_gap,
        "a resolving write must not flag a new gap"
    );

    // Related Skip-band probe: force-written ONLY if the gap re-armed.
    let d = code((1000..1016).chain(0..12).chain(40..52));
    let obs = m.observe_code(d, p("related trivia"), 0.0);
    assert_eq!(
        obs.decision,
        Decision::Skip,
        "resolved gap must stay resolved"
    );
}

#[test]
fn feedback_reward_is_one_step_per_call() {
    // P1.3 acceptance: four used hits move each threshold exactly as far as
    // one used hit — the reward is per-call, not per-hit.
    let run = |use_all: bool| {
        let mut m = memory();
        let a = code(0..40);
        let b = code((0..24).chain(100..116)); // Link-band relative of a
        m.observe_code(a.clone(), p("a"), 0.0);
        m.observe_code(b, p("b"), 0.0);
        m.observe_code(a.clone(), p("a"), 0.0);
        let hits = m.recall_code(&a, 4);
        assert!(hits.len() >= 2, "need multiple hits, got {}", hits.len());
        let w0 = m.gate.threshold;
        let r0 = m.read_gate.threshold;
        if use_all {
            m.feedback(&hits);
        } else {
            m.feedback(&hits[..1]);
        }
        (w0 - m.gate.threshold, r0 - m.read_gate.threshold)
    };
    let (w_many, r_many) = run(true);
    let (w_one, r_one) = run(false);
    assert!(w_many > 0.0, "useful feedback must move the write gate");
    assert!(
        (w_many - w_one).abs() < 1e-6,
        "write reward must be one step: {w_many} vs {w_one}"
    );
    assert!(
        (r_many - r_one).abs() < 1e-6,
        "read reward must be one step: {r_many} vs {r_one}"
    );
}

#[test]
fn capacity_10k_recall_stays_sane() {
    // Graph-level (no per-write gate scan) so 10k episodes stay fast in debug.
    let mut g: engram::MemoryGraph<32, u64> = engram::MemoryGraph::new();
    let mut rng = Lcg(1234);
    let mut probes = Vec::new();
    for i in 0..10_000u64 {
        let c = rng.code();
        if i % 1000 == 0 {
            probes.push(c.clone());
        }
        g.advance_tick();
        g.write(c, i);
    }
    // Every probe must come back as the top hit for its own code.
    for probe in &probes {
        let hits = g.recall(probe, 2, 3);
        assert!(!hits.is_empty());
        let top = g.get(hits[0].0).unwrap();
        assert!(
            (probe.jaccard(&top.code) - 1.0).abs() < 1e-6,
            "stored code must be its own best match"
        );
    }
}

#[test]
fn unknown_high_demand_at_10k_retrieves_nothing() {
    // P0.3 acceptance: at capacity the bit-frequency availability estimate of
    // a fresh unknown code sits ~0.5 (that is the anti-saturation design), so
    // the read gate opens — the ANCHOR gate must then refuse: no stored
    // episode actually relates to the cue, so nothing may reach context.
    let mut m = memory();
    let mut rng = Lcg(1234);
    for i in 0..10_000 {
        m.import(rng.code(), p(&format!("e{i}")));
    }
    let unknown = rng.code();
    assert!(
        m.novelty_estimate(&unknown) < 0.85,
        "premise: availability estimate must look 'familiar' at capacity"
    );
    let hits = m.recall_code_with_demand(&unknown, 0.9, 8);
    assert!(
        hits.is_empty(),
        "unknown high-demand query must not inject random overlap at capacity"
    );
}

#[test]
fn exact_match_at_10k_still_retrieves() {
    // P0.3 counterpart: the anchor/floor must not overfilter — an old exact
    // match at capacity (max recency penalty) still comes back on top.
    let mut m = memory();
    let mut rng = Lcg(777);
    let target = rng.code();
    m.import(target.clone(), p("needle"));
    for i in 0..10_000 {
        m.import(rng.code(), p(&format!("hay{i}")));
    }
    let hits = m.recall_code_with_demand(&target, 0.9, 4);
    assert!(!hits.is_empty(), "old exact match must still retrieve");
    assert_eq!(m.store.get(hits[0]).as_deref(), Some("needle"));
}

#[test]
fn associate_above_floor_still_retrieves() {
    // Link-band associate rides along with the direct anchor — the floor must
    // not kill useful associative recall.
    let mut m = memory();
    let a = code(0..40);
    let b = code((0..24).chain(100..116));
    m.observe_code(a.clone(), p("a"), 0.0);
    m.observe_code(b, p("b"), 0.0);
    let hits = m.recall_code(&a, 4);
    assert_eq!(hits.len(), 2, "associate must not be overfiltered");
}

#[test]
fn observe_costs_at_most_one_scan() {
    // P0.1 acceptance: the fused path never pays for two full scans, and the
    // cheap-reject path pays for none.
    let mut m = memory();
    let mut rng = Lcg(99);
    for i in 0..50 {
        m.observe_code(rng.code(), p(&format!("w{i}")), 0.0);
    }
    let before = m.graph().scan_count();
    m.observe_code(rng.code(), p("probe"), 0.0);
    let cost = m.graph().scan_count() - before;
    assert!(cost <= 1, "observe cost {cost} scans, budget is 1");

    // Cold start: max novelty → cheap-reject, no surfacing possible → 0 scans.
    let mut m2 = memory();
    let before = m2.graph().scan_count();
    m2.observe_code(rng.code(), p("first"), 0.0);
    assert_eq!(
        m2.graph().scan_count() - before,
        0,
        "cheap-reject must skip the scan entirely"
    );
}

#[test]
fn read_gate_2x2_through_public_api() {
    // P0.2 acceptance: all four cells reachable with real demand values.
    let mut m = memory();
    let known = code(0..40);
    m.observe_code(known.clone(), p("known fact"), 0.0);
    m.observe_code(known.clone(), p("known fact"), 0.0);

    // Suppress: high availability, low demand → nothing injected.
    assert!(m.recall_code_with_demand(&known, 0.1, 4).is_empty());
    // Surface: mid demand + availability → gentle top-1 injection.
    let surf = m.recall_code_with_demand(&known, 0.4, 4);
    assert_eq!(surf.len(), 1, "surface must inject exactly one");
    // Retrieve: high demand + availability → full retrieval.
    let hits = m.recall_code_with_demand(&known, 0.9, 4);
    assert!(!hits.is_empty(), "retrieve cell must fire");
    // Skip: low demand, low availability → nothing.
    assert!(m
        .recall_code_with_demand(&code(1500..1540), 0.1, 4)
        .is_empty());
}

#[test]
fn snapshot_restore_roundtrip() {
    let tag = engram::model_tag("test-encoder-v1");
    let mut m = memory();
    let a = code(0..40);
    m.observe_code(a.clone(), p("durable fact"), 0.0);
    m.observe_code(a.clone(), p("durable fact"), 0.0);
    m.observe_code(code(700..740), p("other fact"), 0.0);
    m.gate.reward(false); // move a threshold so we can see it travel

    let bytes = m.snapshot(tag);

    // P1.4: a different embedding model must be refused — the codes would be
    // semantic garbage with intact structure.
    let refused: Option<Engram<InMemoryStore<String>>> = Engram::restore(
        &bytes,
        engram::model_tag("some-other-encoder"),
        InMemoryStore::new(),
    );
    assert!(
        refused.is_none(),
        "model-tag mismatch must refuse to restore"
    );

    // Payload durability is the store's job — hand the same store back.
    let store = std::mem::replace(&mut m.store, InMemoryStore::new());
    let mut r = Engram::restore(&bytes, tag, store).expect("restore should parse");

    assert_eq!(r.graph().len(), m.graph().len());
    assert_eq!(r.tick(), m.tick());
    assert!((r.gate.threshold - m.gate.threshold).abs() < 1e-6);
    assert!((r.novelty_estimate(&a) - m.novelty_estimate(&a)).abs() < 1e-4);

    // Behavior survives the roundtrip: the fact is still recognized & recallable.
    let hits = r.recall_code(&a, 4);
    assert!(!hits.is_empty());
    assert_eq!(r.store.get(hits[0]).unwrap(), "durable fact");
    let obs = r.observe_code(a, p("durable fact"), 0.0);
    assert!(matches!(obs.decision, Decision::Strengthen(_)));
}

#[test]
fn identity_store_carries_uuids_no_content_duplication() {
    // The "payload is a UUID / your own id" case: engram carries the id and
    // hands it straight back — no engram-side content store, no second id.
    let mut m: Memory<32, IdentityStore<u128>> = Memory::new(IdentityStore::new());
    let uuid_a: u128 = 0x1111_2222_3333_4444_5555_6666_7777_8888;
    let uuid_b: u128 = 0xAAAA_BBBB_CCCC_DDDD_EEEE_FFFF_0000_9999;

    m.observe_code(code(0..40), uuid_a, 0.0);
    m.observe_code(code(0..40), uuid_a, 0.0); // strengthen (dup)
    m.observe_code(code(1000..1040), uuid_b, 0.0);

    let hits = m.recall_code(&code(0..40), 4);
    assert_eq!(
        hits.first().copied(),
        Some(uuid_a),
        "recall returns your own id"
    );

    // Snapshot roundtrips a 16-byte handle (the header records the width).
    let tag = engram::model_tag("test-encoder-v1");
    let bytes = m.snapshot(tag);
    let r: Memory<32, IdentityStore<u128>> =
        Memory::restore(&bytes, tag, IdentityStore::new()).expect("restore should parse");
    assert_eq!(r.graph().len(), 2);

    // A snapshot written with a 16-byte handle must be rejected by an 8-byte store.
    let mismatch: Option<Memory<32, IdentityStore<u64>>> =
        Memory::restore(&bytes, tag, IdentityStore::new());
    assert!(
        mismatch.is_none(),
        "handle-width mismatch must fail to restore"
    );
}

#[test]
fn heuristic_demand_orders_sensibly() {
    let recall_q = engram::heuristic_demand("what did we decide about the fogger schedule?");
    let statement = engram::heuristic_demand("nice weather today");
    assert!(recall_q > 0.6);
    assert!(statement < 0.25);
}

/// Relational state (cross-device sync): the graph must be reconstructible
/// from drained rows + the original dense vectors (codes re-derive through
/// the deterministic fly-hash — they never travel). One user = one graph:
/// a second device folding the same rows must recall identically.
#[test]
fn relational_state_roundtrip_rebuilds_equivalent_graph() {
    let mut lcg = Lcg(42);
    let dim = 256;

    // Device A: observe a small corpus (payload = caller-owned u64 handles),
    // strengthen one episode via a duplicate, reward it via feedback.
    let mut a: Memory<32, IdentityStore<u64>> = Memory::new(IdentityStore::new());
    let denses: Vec<(u64, Vec<f32>)> = (1..=8u64).map(|p| (p, lcg.dense(dim))).collect();
    for (p, d) in &denses {
        a.observe(d, *p, 0.0);
    }
    a.observe(&denses[0].1, 1, 0.0); // dup → strengthen payload 1
    let obs = a.observe(&denses[0].1, 1, 0.0);
    if !obs.surfaced.is_empty() {
        a.feedback(&[obs.surfaced[0].payload]);
    }

    // Drain rows; a second drain must be empty (dirty is a delta, not a dump).
    let report = a.drain_dirty();
    assert!(!report.episodes.is_empty());
    assert!(report.removed.is_empty());
    let again = a.drain_dirty();
    assert!(again.episodes.is_empty() && again.removed.is_empty());

    // Device B: fold the rows (two passes: episodes, then edges).
    let mut b: Memory<32, IdentityStore<u64>> = Memory::new(IdentityStore::new());
    let dense_of = |p: u64| denses.iter().find(|(q, _)| *q == p).unwrap().1.clone();
    for ep in &report.episodes {
        b.upsert_episode(
            &dense_of(ep.payload),
            ep.payload,
            ep.strength,
            ep.count,
            ep.used,
        );
    }
    for ep in &report.episodes {
        b.set_episode_edges(ep.payload, &ep.edges);
    }
    assert_eq!(a.graph().len(), b.graph().len());

    // Same cue, same top hit, on both devices.
    let cue = denses[0].1.clone();
    let ha = a.recall(&cue, 4);
    let hb = b.recall(&cue, 4);
    assert_eq!(ha.first(), hb.first(), "one graph: devices must agree");
    assert_eq!(hb.first().copied(), Some(1u64));

    // Importing must NOT mark dirty — the state came FROM the rows.
    let echo = b.drain_dirty();
    assert!(
        echo.episodes.is_empty(),
        "imports must not echo back as dirty"
    );

    // Synced row delete: the episode is gone from recall on the receiver.
    b.remove_episode(1u64);
    let hb2 = b.recall(&cue, 4);
    assert_ne!(hb2.first().copied(), Some(1u64));

    // Sleep dirties every survivor (decay touched them) and reports prunes
    // through the same drain (removed = row deletes).
    for _ in 0..10 {
        a.sleep();
    }
    let after_sleep = a.drain_dirty();
    assert!(
        !after_sleep.removed.is_empty(),
        "prunes must become row deletes"
    );
}

/// A remote sleep's prunes apply at SYNC time: reconciling with the synced
/// row set drops episodes whose rows are gone — but locally-dirty (pending
/// flush) episodes survive, they're newer than any row.
#[test]
fn retain_episodes_applies_remote_prunes_but_spares_dirty() {
    let mut lcg = Lcg(7);
    let dim = 256;
    let mut m: Memory<32, IdentityStore<u64>> = Memory::new(IdentityStore::new());
    let denses: Vec<(u64, Vec<f32>)> = (1..=5u64).map(|p| (p, lcg.dense(dim))).collect();
    for (p, d) in &denses {
        m.observe(d, *p, 0.0);
    }
    // Age 4 and 5 out of the temporal-binding window (writes Hebbian-bind to
    // RECENT episodes and link() dirties both ends — a fresh write right
    // after would re-dirty them and retain would correctly spare them).
    for _ in 0..4 {
        m.observe(&denses[0].1, 1, 0.0); // dup → strengthen 1, advance ticks
    }
    m.drain_dirty(); // flushed: nothing dirty now

    // A fresh local observation that has NOT been flushed yet (binds only
    // to episode 1, still recent — 1 stays in the keep set anyway).
    let fresh_dense = lcg.dense(dim);
    m.observe(&fresh_dense, 99u64, 0.0);

    // Remote sleep pruned payloads 4 and 5: the synced row set keeps 1..3
    // (99's row doesn't exist yet either — dirty must protect it).
    let removed = m.retain_episodes([1u64, 2, 3]);
    assert_eq!(removed, 2, "4 and 5 drop; dirty 99 survives");
    let hits = m.recall(&fresh_dense, 4);
    assert_eq!(
        hits.first().copied(),
        Some(99u64),
        "pending-flush episode survives"
    );
    let gone = m.recall(&denses[3].1, 4);
    assert_ne!(
        gone.first().copied(),
        Some(4u64),
        "pruned row is gone from recall"
    );
}

/// Flush-failure recovery and dirty-wins-over-rows: (a) `undrain` restores a
/// consumed drain so a rolled-back flush transaction loses nothing; (b) an
/// upsert (sync-time re-fold) must NOT overwrite a locally-dirty episode —
/// pending-flush state is newer than any row.
#[test]
fn undrain_restores_and_upsert_spares_dirty() {
    let mut lcg = Lcg(11);
    let dim = 256;
    let mut m: Memory<32, IdentityStore<u64>> = Memory::new(IdentityStore::new());
    let d1 = lcg.dense(dim);
    m.observe(&d1, 1u64, 0.0);

    // Failed flush: drain, then undrain — the next drain must see it again.
    let r1 = m.drain_dirty();
    assert!(!r1.episodes.is_empty());
    m.undrain();
    let r2 = m.drain_dirty();
    assert_eq!(
        r1.episodes.len(),
        r2.episodes.len(),
        "undrain must restore the failed flush's material"
    );

    // Dirty protection: strengthen (dirty, unflushed), then a re-fold row
    // with stale state must not revert it.
    m.observe(&d1, 1u64, 0.0); // dup → strengthen: count 2, dirty
    let before = m
        .graph()
        .iter()
        .find(|(_, ep)| ep.payload == 1)
        .unwrap()
        .1
        .count;
    m.upsert_episode(&d1, 1u64, 1.0, 1, false); // stale row: count 1
    let after = m
        .graph()
        .iter()
        .find(|(_, ep)| ep.payload == 1)
        .unwrap()
        .1
        .count;
    assert_eq!(
        before, after,
        "dirty episode must not be clobbered by a row"
    );
    assert!(before >= 2);
}

#[test]
fn penalize_knocks_strength_and_folds() {
    let mut lcg = Lcg(13);
    let dim = 256;
    let mut m: Memory<32, IdentityStore<u64>> = Memory::new(IdentityStore::new());
    let d1 = lcg.dense(dim);
    m.observe(&d1, 1u64, 0.0);
    m.observe(&d1, 1u64, 0.0); // dup → strengthen: strength 2
    let _ = m.drain_dirty(); // flushed: clean slate

    // The host judged it "shown repeatedly, never helped": knock it.
    assert!(m.penalize_episode(1u64, 0.5));
    let s = m
        .graph()
        .iter()
        .find(|(_, ep)| ep.payload == 1)
        .unwrap()
        .1
        .strength;
    assert!((s - 1.0).abs() < 1e-6, "strength 2.0 × 0.5 = 1.0, got {s}");

    // The knock is durable-row material (folds fleet-wide like any mutation).
    let r = m.drain_dirty();
    assert_eq!(r.episodes.len(), 1, "penalty must mark the episode dirty");
    assert!((r.episodes[0].strength - 1.0).abs() < 1e-6);

    // Unknown payloads are a polite no-op.
    assert!(!m.penalize_episode(999u64, 0.5));
}

#[test]
fn penalize_factor_is_clamped() {
    // A knock can only knock: factor > 1 must not amplify strength past the
    // [0, STRENGTH_MAX] range the recall floor's score math relies on, and a
    // negative factor must floor at zero, not go negative.
    let mut m = memory();
    let a = code(0..40);
    m.observe_code(a.clone(), p("fact"), 0.0);
    let obs = m.observe_code(a.clone(), p("fact"), 0.0); // strengthen: strength 2
    let id = obs.episode.unwrap();
    let r = m.graph().get(id).unwrap().payload;

    let before = m.graph().get(id).unwrap().strength;
    assert!(m.penalize_episode(r, 3.0));
    assert!(
        m.graph().get(id).unwrap().strength <= before + 1e-6,
        "factor > 1 must not amplify"
    );
    assert!(m.penalize_episode(r, -2.0));
    assert_eq!(
        m.graph().get(id).unwrap().strength,
        0.0,
        "negative factor floors at 0"
    );
}

#[test]
fn penalized_episode_prunes_as_row_delete_not_dead() {
    // The documented follow-through: consolidation's prune floor finishes the
    // job a penalty started — and the prune flows to drain as a row delete.
    // Since the episode HAD surfaced (that's why the host penalized it), it
    // must not count as "sat dead" and unfairly charge the write gate.
    let mut m = memory();
    let a = code(0..40);
    m.observe_code(a.clone(), p("shown but never helps"), 0.0);
    let hits = m.recall_code(&a, 1); // surfaces it: last_surfaced is set
    assert_eq!(hits.len(), 1);
    m.drain_dirty(); // flushed: the prune below is the only new row material

    m.penalize_episode(hits[0], 0.05); // strength 1.0 → 0.05, below prune floor
    let w0 = m.gate.threshold;
    let report = m.sleep();
    assert_eq!(
        report.pruned, 1,
        "the penalty must let consolidation finish the job"
    );
    assert_eq!(report.dead, 0, "a surfaced episode is not 'sat dead'");
    assert_eq!(m.gate.threshold, w0, "no unfair write-gate charge");
    assert_eq!(m.store.len(), 0, "payload evicted from the store");

    let r = m.drain_dirty();
    assert!(
        r.removed.contains(&hits[0]),
        "the penalty-driven prune must emit a row delete"
    );
}

#[test]
fn edge_healing_on_second_pass() {
    // set_episode_edges skips targets whose rows haven't arrived yet; the
    // promise is that a later pass heals. Prove the healing actually happens
    // and the healed edge carries activation.
    let mut lcg = Lcg(23);
    let dim = 256;
    let mut b: Memory<32, IdentityStore<u64>> = Memory::new(IdentityStore::new());
    let d1 = lcg.dense(dim);
    let d2 = lcg.dense(dim);

    b.upsert_episode(&d1, 1u64, 1.0, 1, false);
    b.set_episode_edges(1u64, &[(2u64, 0.5)]); // 2's row not here yet → skipped
    let edges_of_1 = |m: &Memory<32, IdentityStore<u64>>| {
        m.graph()
            .iter()
            .find(|(_, ep)| ep.payload == 1)
            .unwrap()
            .1
            .edges
            .len()
    };
    assert_eq!(
        edges_of_1(&b),
        0,
        "unknown target must be skipped, not fabricated"
    );

    b.upsert_episode(&d2, 2u64, 1.0, 1, false);
    b.set_episode_edges(1u64, &[(2u64, 0.5)]); // the healing pass
    assert_eq!(edges_of_1(&b), 1, "second pass must heal the edge");

    let hits = b.recall(&d1, 4);
    assert!(hits.contains(&2u64), "healed edge must carry activation");
}

#[test]
fn snapshot_carries_pending_flush() {
    // A restore that dropped dirty state would orphan those mutations from
    // durability forever (the rows would never see them). Snapshot v4 must
    // carry both pending upserts and pending removals.
    let tag = engram::model_tag("test-encoder-v1");
    let mut lcg = Lcg(31);
    let dim = 256;
    let mut m: Memory<32, IdentityStore<u64>> = Memory::new(IdentityStore::new());
    let d1 = lcg.dense(dim);
    let d2 = lcg.dense(dim);
    m.observe(&d1, 1u64, 0.0);
    m.observe(&d2, 2u64, 0.0);
    m.drain_dirty(); // flushed: clean slate

    for _ in 0..6 {
        m.observe(&d1, 1u64, 0.0); // strengthen 1 → dirty again
    }
    let mut pruned = 0;
    for _ in 0..10 {
        pruned += m.sleep().pruned; // 2 fades → pending row delete
    }
    assert_eq!(pruned, 1);

    let bytes = m.snapshot(tag);
    let mut r: Memory<32, IdentityStore<u64>> =
        Memory::restore(&bytes, tag, IdentityStore::new()).expect("restore should parse");

    let orig = m.drain_dirty();
    let rest = r.drain_dirty();
    assert_eq!(
        orig.episodes.len(),
        rest.episodes.len(),
        "pending upserts must survive the snapshot"
    );
    assert_eq!(
        orig.removed, rest.removed,
        "pending removals must survive the snapshot"
    );
    assert_eq!(rest.removed, vec![2u64]);
}

#[test]
fn undrain_edge_cases() {
    let mut lcg = Lcg(17);
    let dim = 256;
    let mut m: Memory<32, IdentityStore<u64>> = Memory::new(IdentityStore::new());
    let d1 = lcg.dense(dim);
    let d2 = lcg.dense(dim);

    // Undrain with no prior drain: a no-op, nothing conjured.
    m.undrain();
    assert!(m.drain_dirty().episodes.is_empty());

    m.observe(&d1, 1u64, 0.0);
    let r1 = m.drain_dirty();
    assert_eq!(r1.episodes.len(), 1);

    // Age episode 1 out of the temporal-binding window so the next write
    // doesn't re-dirty it by linking (that would mask the merge assertion).
    let far = lcg.dense(dim);
    for _ in 0..4 {
        let _ = m.recall_with_demand(&far, 0.1, 1); // tick advances, no write
    }

    // Drain → new mutation → undrain: the next drain sees the UNION.
    m.observe(&d2, 2u64, 0.0);
    m.undrain();
    let r2 = m.drain_dirty();
    assert_eq!(
        r2.episodes.len(),
        2,
        "undrain must merge with new dirt, not replace it"
    );

    // Double undrain: the second has nothing left to restore.
    m.undrain();
    m.undrain();
    let r3 = m.drain_dirty();
    assert_eq!(r3.episodes.len(), 2);
    assert!(m.drain_dirty().episodes.is_empty());
}

#[test]
fn feedback_after_relational_removal_is_inert() {
    // A synced row delete landing between a recall and its feedback must
    // invalidate that pending target — NOT read as "the retrieval was wasted"
    // (which would wrongly train the read gate down).
    let mut m = memory();
    let a = code(0..40);
    m.observe_code(a.clone(), p("fact"), 0.0);
    m.observe_code(a.clone(), p("fact"), 0.0);
    let hits = m.recall_code(&a, 4);
    assert!(!hits.is_empty());

    m.remove_episode(hits[0]); // sync-time delete lands mid-flight

    let w = m.gate.threshold;
    let r = m.read_gate.threshold;
    m.feedback(&hits);
    assert_eq!(
        m.gate.threshold, w,
        "invalidated feedback must not train the write gate"
    );
    assert_eq!(
        m.read_gate.threshold, r,
        "invalidated feedback must not train the read gate"
    );
}

/// Feedback recorded as facts replays by payload, and a clone answers a
/// query without touching the original.
#[test]
fn credit_by_payload_and_query_on_a_clone() {
    let mut mem: Memory<32, IdentityStore<u64>> = Memory::new(IdentityStore::new());
    let mut rng = Lcg(7);
    let mut codes = Vec::new();
    for h in 0..5u64 {
        let c = rng.code();
        mem.import(c.clone(), h);
        codes.push(c);
    }
    let eid = mem
        .graph()
        .episode_of(&2)
        .expect("payload 2 has an episode");
    assert!(!mem.graph().get(eid).unwrap().used);
    mem.credit(&[2, 99], true);
    assert!(mem.graph().get(eid).unwrap().used, "credited by payload");
    // Nothing shown: no reward step at all.
    let before = mem.read_gate.threshold;
    mem.credit(&[], false);
    assert_eq!(mem.read_gate.threshold, before);

    let tick = mem.tick();
    let mut copy = mem.clone();
    copy.recall_code(&codes[3], 3);
    assert_eq!(mem.tick(), tick, "the original did not move");
    assert!(copy.tick() > tick);
}

// ---- inverted index ≡ full scan, end to end through the orchestrator ----

fn assert_scan_equivalent<S: PayloadStore>(m: &Memory<32, S>, cues: &[Code]) {
    let g = m.graph();
    for cue in cues {
        let bm = g.best_match(cue).map(|(i, j)| (i.0, j.to_bits()));
        let bm_ref = g.best_match_scan(cue).map(|(i, j)| (i.0, j.to_bits()));
        assert_eq!(bm, bm_ref, "best_match diverged from the full scan");
        for hops in [0u8, 2, 3] {
            for top in [0usize, 1, 4, 64] {
                let (h, b) = g.recall_with_best(cue, hops, top);
                let (hr, br) = g.recall_with_best_scan(cue, hops, top);
                let bits = |h: &Vec<(engram::EpisodeId, f32)>| {
                    h.iter()
                        .map(|&(i, s)| (i.0, s.to_bits()))
                        .collect::<Vec<_>>()
                };
                assert_eq!(
                    bits(&h),
                    bits(&hr),
                    "hits diverged (hops {hops}, top {top})"
                );
                assert_eq!(
                    b.map(|(i, j)| (i.0, j.to_bits())),
                    br.map(|(i, j)| (i.0, j.to_bits())),
                    "best diverged (hops {hops}, top {top})"
                );
            }
        }
    }
}

/// Real fly-hashed codes (clustered dense vectors), the full verb set —
/// observe, recall, feedback, sleep, snapshot/restore, relational fold ops —
/// and after every step the indexed scan must equal the reference full scan
/// bit for bit.
#[test]
fn indexed_scan_matches_full_scan_through_memory() {
    let tag = engram::model_tag("index-equivalence");
    for seed in 1..=4u64 {
        let mut rng = Lcg(seed);
        let dim = 32;
        let centroids: Vec<Vec<f32>> = (0..6).map(|_| rng.dense(dim)).collect();
        let sample = |rng: &mut Lcg| {
            let c = &centroids[(rng.next() % 6) as usize];
            let noise = rng.dense(dim);
            let spread = (rng.next() % 4) as f32 * 0.15;
            c.iter()
                .zip(&noise)
                .map(|(a, b)| a + spread * b)
                .collect::<Vec<f32>>()
        };
        let mut m: Memory<32, IdentityStore<u64>> = Memory::new(IdentityStore::new());
        let mut next = 0u64;
        let mut seen: Vec<Vec<f32>> = Vec::new();
        for step in 0..400 {
            let v = sample(&mut rng);
            match rng.next() % 20 {
                0..=9 => {
                    m.observe(&v, next, (rng.next() % 3) as f32 * 0.5);
                    next += 1;
                    seen.push(v);
                }
                10..=11 => {
                    let hits = m.recall(&v, 4);
                    m.feedback(&hits[..hits.len().min(1)]);
                }
                12 => {
                    m.sleep();
                }
                13 => {
                    let bytes = m.snapshot(tag);
                    m = Memory::restore(&bytes, tag, IdentityStore::new()).unwrap();
                }
                14 if !seen.is_empty() => {
                    // Fold overwrite with a different embedding (code change).
                    let p = rng.next() % next.max(1);
                    m.drain_dirty();
                    m.upsert_episode(&v, p, 1.5, 2, false);
                }
                15 if next > 0 => {
                    m.remove_episode(rng.next() % next);
                }
                16 if next > 0 => {
                    m.drain_dirty();
                    let keep: Vec<u64> = (0..next).filter(|_| rng.next() % 10 < 7).collect();
                    m.retain_episodes(keep);
                }
                17 => {
                    let c = m.sparsify(&v);
                    m.import(c, next);
                    next += 1;
                }
                _ => {
                    m.observe(&v, next, 1.0);
                    next += 1;
                }
            }
            if step % 4 == 0 {
                let mut cues: Vec<Code> = (0..3)
                    .map(|_| {
                        let v = sample(&mut rng);
                        m.sparsify(&v)
                    })
                    .collect();
                cues.push(Code::empty());
                cues.push(rng.code());
                assert_scan_equivalent(&m, &cues);
            }
        }
    }
}
