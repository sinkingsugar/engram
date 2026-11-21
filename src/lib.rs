//! # engram — a low-power, self-consolidating memory gate
//!
//! The hippocampus in front of your vector store: takes an embedding the
//! caller already computed (the API boundary is a dense vector — engram owns
//! no text encoder), sparsifies it with a fly-hash, gates writes by novelty,
//! surfaces memories by spreading activation, and forgets on purpose during
//! `sleep()`.
//!
//! One loop: an explicit `recall()` is just a synthetic high-demand
//! observation; the passive channel surfaces associations on every `observe()`
//! by riding the same single scan the write gate pays for. **Activation is
//! continuous; injection is gated** — the control problem is inhibition, not
//! triggering: refractory periods, a surfacing budget, and graded injection
//! hang off a single `inhibition` dial.
//!
//! Recall today is the hippocampal path only (spreading activation over the
//! episode graph). Cortex fusion — ANN over a vector-capable [`PayloadStore`],
//! fused with the associative hits — is pending, not implemented.
//!
//! ## The payload boundary
//!
//! Two things go in on a write and they are *not* the same: the **embedding**
//! (the key engram gates and searches on — sparsified, then discarded) and the
//! **payload** (an opaque handle engram stores and hands back on recall — a
//! store index, a rowid, a UUID). The embedding can't be reconstructed to text
//! and engram owns no encoder, so both are supplied. Pick the store that fits:
//! [`InMemoryStore`] (engram assigns an index and holds your content),
//! [`IdentityStore`] (the handle *is* your own id — engram carries it, you
//! resolve content), or your own [`PayloadStore`] impl (e.g. SQLite).

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod gate;
pub mod graph;
mod index;
pub mod sdr;
pub mod store;

use alloc::vec::Vec;

pub use gate::{BitFreq, Decision, NoveltyGate, ReadGate, RecallAction, DUP_JACCARD, LINK_JACCARD};
pub use graph::{ConsolidationReport, DirtyReport, Episode, EpisodeId, EpisodeState, MemoryGraph};
pub use sdr::{FlyHasher, SparseCode};
pub use store::{IdentityStore, InMemoryStore, PayloadRef, PayloadStore};

/// Tunables. Defaults come from [`Params::for_bits`].
#[derive(Clone, Debug)]
pub struct Params {
    /// Active bits per code (fly-hash k). ~2% of the bit-space.
    pub k: usize,
    /// Bit-novelty at or above this skips the episode scan entirely → WriteNew
    /// (cheap-reject; mostly fires cold or on genuinely alien input).
    pub fast_novel: f32,
    /// Spreading-activation hops.
    pub hops: u8,
    /// Max passively surfaced items per tick (before inhibition scaling).
    pub surface_budget: usize,
    /// Min activation for passive surfacing (before inhibition scaling).
    pub surface_threshold: f32,
    /// Cheap pre-gate for the passive channel: below this availability
    /// estimate (1 − bit-novelty, 30ns) nothing could surface anyway, so the
    /// scan is skipped entirely (before inhibition scaling).
    pub surface_availability_floor: f32,
    /// Ticks an episode stays refractory after surfacing (before scaling).
    pub refractory: u64,
    /// Availability below this + demand above `gap_demand` = knowledge gap.
    pub gap_availability: f32,
    pub gap_demand: f32,
    /// Ticks a flagged knowledge gap stays armed waiting for its answer.
    pub gap_ttl: u64,
    /// Min jaccard between the gap cue and a later observation for the eager
    /// write to fire — the answer to the question we asked, not the next
    /// thing through the door.
    pub gap_match: f32,
    /// Deliberate retrieval needs a direct ANCHOR: the scan's best jaccard
    /// must clear this or nothing is injected (the bit-frequency availability
    /// estimate said "familiar bits", but no stored episode actually anchors
    /// the cue → knowledge gap, not a retrieval). Random 40-bit collisions
    /// sit at ~0.05–0.10; the Link band starts at 0.34.
    pub recall_anchor: f32,
    /// Min final recall score (activation × strength × recency, all ≤ 1) for
    /// a hit to be injected into context. Keeps high-demand queries from
    /// returning coincidental low-overlap junk at capacity.
    pub recall_floor: f32,
}

impl Params {
    pub fn for_bits(bits: usize) -> Self {
        Self {
            k: (bits / 50).max(8),
            fast_novel: 0.85,
            hops: 2,
            surface_budget: 3,
            surface_threshold: 0.25,
            surface_availability_floor: 0.15,
            refractory: 8,
            gap_availability: 0.25,
            gap_demand: 0.6,
            gap_ttl: 32,
            gap_match: 0.25,
            recall_anchor: 0.15,
            recall_floor: 0.12,
        }
    }
}

/// An armed knowledge gap: the cue we lacked an answer for, with a fuse.
#[derive(Clone)]
struct PendingGap<const W: usize> {
    code: SparseCode<W>,
    expires: u64,
}

/// A passively or actively surfaced memory. `R` is the store's handle type.
#[derive(Clone, Copy, Debug)]
pub struct Surfaced<R> {
    pub payload: R,
    pub episode: EpisodeId,
    pub activation: f32,
}

/// What one observation did — the decision, the scalars behind it (so callers
/// know *why*), and anything the passive channel surfaced.
#[derive(Debug)]
pub struct Observation<R> {
    pub decision: Decision,
    /// Effective novelty: `1 − best_jaccard` when an episode scan ran (ground
    /// truth), bit-frequency novelty otherwise.
    pub novelty: f32,
    /// `1 − novelty` — the feeling-of-knowing estimate.
    pub availability: f32,
    /// High demand + low availability: need it, don't have it. The next
    /// observation is eagerly written (the read gate tells the write gate to
    /// pay attention).
    pub knowledge_gap: bool,
    /// Episode written, linked, or strengthened by this observation.
    pub episode: Option<EpisodeId>,
    /// Proactive channel: what this observation was "reminded of".
    pub surfaced: Vec<Surfaced<R>>,
}

/// The orchestrator: fly-hasher + write gate + read gate + graph + store,
/// plus the one inhibition dial. Cloneable, so a caller can answer a
/// one-off query on a copy without touching shared, folded state.
#[derive(Clone)]
pub struct Memory<const W: usize, S: PayloadStore> {
    pub params: Params,
    pub gate: NoveltyGate,
    pub read_gate: ReadGate,
    pub store: S,
    freq: BitFreq,
    graph: MemoryGraph<W, S::Ref>,
    hasher: Option<FlyHasher<W>>,
    /// 1.0 = focused … 0.0 = association flood. Scales all three brakes.
    inhibition: f32,
    pending_gap: Option<PendingGap<W>>,
    last_retrieval: Vec<(EpisodeId, S::Ref)>,
}

/// 2048-bit default configuration.
pub type Engram<S> = Memory<32, S>;

impl<const W: usize, S: PayloadStore> Memory<W, S> {
    pub fn new(store: S) -> Self {
        Self {
            params: Params::for_bits(W * 64),
            gate: NoveltyGate::default(),
            read_gate: ReadGate::default(),
            store,
            freq: BitFreq::new(W * 64),
            graph: MemoryGraph::new(),
            hasher: None,
            inhibition: 1.0,
            pending_gap: None,
            last_retrieval: Vec::new(),
        }
    }

    pub fn graph(&self) -> &MemoryGraph<W, S::Ref> {
        &self.graph
    }

    pub fn tick(&self) -> u64 {
        self.graph.tick()
    }

    /// §6b inhibition dial, clamped to [0, 1].
    pub fn set_inhibition(&mut self, i: f32) {
        self.inhibition = i.clamp(0.0, 1.0);
    }

    pub fn inhibition(&self) -> f32 {
        self.inhibition
    }

    /// Cheap novelty estimate for a code (no scan, no side effects).
    pub fn novelty_estimate(&self, code: &SparseCode<W>) -> f32 {
        self.freq.novelty(code)
    }

    // ---- relational state (cross-device sync) ----
    // The graph is a deterministic fold of synced facts (embeddings +
    // observations + these state rows); the snapshot blob is only a local
    // cache. `drain_dirty` after every mutation batch → upsert/delete rows in
    // ONE transaction with the observation log; `upsert_episode` /
    // `set_episode_edges` / `remove_episode` rebuild a graph FROM rows (codes
    // re-derived from the synced embeddings — they never travel).

    /// Everything mutated since the last drain — flush to durable rows.
    pub fn drain_dirty(&mut self) -> DirtyReport<S::Ref> {
        self.graph.drain_dirty()
    }

    /// Restore the last drain after a FAILED flush transaction — the drain's
    /// clear is not transactional with the caller's row writes; without this
    /// a rolled-back flush silently orphans those mutations from durability.
    pub fn undrain(&mut self) {
        self.graph.undrain();
    }

    /// Apply one episode's synced edge list (targets not yet imported are
    /// skipped; a later pass heals).
    pub fn set_episode_edges(&mut self, payload: S::Ref, edges: &[(S::Ref, f32)]) {
        self.graph.set_edges_by_payload(payload, edges);
    }

    /// Per-episode negative feedback: a memory that was shown repeatedly
    /// (the host counts injections) and never earned `feedback` credit.
    /// Strength × `factor`; dirty-marked so the knock syncs via the fold.
    pub fn penalize_episode(&mut self, payload: S::Ref, factor: f32) -> bool {
        self.graph.penalize_by_payload(payload, factor)
    }

    /// Apply a synced row delete. Frequency counts rebuild at the next sleep
    /// (same convention as prune). Pending feedback targeting the removed
    /// episode is invalidated — a sync-time delete landing between a recall
    /// and its `feedback` must not read as "the retrieval was wasted".
    pub fn remove_episode(&mut self, payload: S::Ref) {
        if self.graph.remove_by_payload(payload) {
            self.store.remove(payload);
            self.last_retrieval.retain(|&(_, r)| r != payload);
        }
    }

    /// Reconcile with the synced row set: drop live episodes whose rows are
    /// gone (a remote sleep's prunes apply at SYNC time, not next boot).
    /// Locally-dirty (pending-flush) episodes survive; pending feedback
    /// targets among the removed are invalidated (see [`Memory::remove_episode`]).
    /// Returns removed count.
    pub fn retain_episodes<I: IntoIterator<Item = S::Ref>>(&mut self, keep: I) -> usize {
        let keep: alloc::collections::BTreeSet<S::Ref> = keep.into_iter().collect();
        let removed = self.graph.retain_payloads(&keep);
        for &r in &removed {
            self.store.remove(r);
        }
        self.last_retrieval.retain(|&(_, r)| !removed.contains(&r));
        removed.len()
    }

    /// Dense → sparse via the (lazily built, dimension-keyed) fly-hasher.
    pub fn sparsify(&mut self, dense: &[f32]) -> SparseCode<W> {
        let k = self.params.k;
        let need_rebuild = self
            .hasher
            .as_ref()
            .is_none_or(|h| h.dim() != dense.len() || h.k() != k);
        if need_rebuild {
            self.hasher = Some(FlyHasher::new(dense.len(), k));
        }
        self.hasher.as_mut().unwrap().sparsify(dense)
    }

    /// ONE observation: embedding in, decision + surfaced memories out.
    /// `payload` is the opaque handle the store keys on (content for
    /// [`InMemoryStore`], your own id for [`IdentityStore`]). `demand` ∈ [0,1]
    /// is the caller's estimate of "does the current task want memory?" — see
    /// [`heuristic_demand`] for a cheap default.
    pub fn observe(
        &mut self,
        dense: &[f32],
        payload: S::Payload,
        demand: f32,
    ) -> Observation<S::Ref> {
        let code = self.sparsify(dense);
        self.observe_code(code, payload, demand)
    }

    /// Same, for callers that bring their own sparse codes.
    ///
    /// Cost discipline (settled decision 2): the 30ns bit-novelty check runs
    /// first and decides whether the full episode scan is needed at all; when
    /// it is, ONE fused scan serves both the write gate (best match) and the
    /// passive channel (seed activations). Never two scans per observation.
    pub fn observe_code(
        &mut self,
        code: SparseCode<W>,
        payload: S::Payload,
        demand: f32,
    ) -> Observation<S::Ref> {
        let tick = self.graph.advance_tick();
        // Feedback refers to the immediately previous interaction: this
        // observation's surfaced items become the new feedback targets.
        self.last_retrieval.clear();
        if code.is_empty() {
            return Observation {
                decision: Decision::Skip,
                novelty: 0.0,
                availability: 0.0,
                knowledge_gap: false,
                episode: None,
                surfaced: Vec::new(),
            };
        }

        // Expire a stale knowledge gap; check relatedness of a live one.
        if self.pending_gap.as_ref().is_some_and(|g| tick > g.expires) {
            self.pending_gap = None;
        }
        let gap_related = self
            .pending_gap
            .as_ref()
            .is_some_and(|g| code.jaccard(&g.code) >= self.params.gap_match);

        // The cheap bouncer, and the pre-gates it buys us.
        let bit_novelty = self.freq.novelty(&code);
        let availability_est = 1.0 - bit_novelty;
        let inh_scale = 0.25 + 0.75 * self.inhibition;
        let budget = self.params.surface_budget + ((1.0 - self.inhibition) * 8.0) as usize;
        let need_gate_scan = bit_novelty < self.params.fast_novel;
        let want_surface = availability_est >= self.params.surface_availability_floor * inh_scale;

        // At most ONE full scan, and only when something needs it.
        let (candidates, best) = if need_gate_scan || want_surface {
            let top = if want_surface { budget * 3 } else { 0 };
            self.graph.recall_with_best(&code, self.params.hops, top)
        } else {
            (Vec::new(), None)
        };

        let (mut decision, novelty) = if !need_gate_scan {
            // Cheap-reject: so novel there is nothing worth deciding against.
            (Decision::WriteNew, bit_novelty)
        } else {
            match best {
                // Structure from jaccard, write-worthiness from bit-novelty
                // (global familiarity) — see NoveltyGate::decide.
                Some((id, j)) => (self.gate.decide(bit_novelty, j, Some(id)), 1.0 - j),
                None => (self.gate.decide(bit_novelty, 0.0, None), bit_novelty),
            }
        };

        // Eager write after a knowledge gap: the read gate flagged that THIS
        // kind of content is wanted (cue-related, within TTL); don't let the
        // write gate skip it.
        if gap_related && decision == Decision::Skip {
            decision = Decision::WriteNew;
        }

        let episode = match decision {
            Decision::WriteNew => {
                let r = self.store.put(payload);
                let id = self.graph.write(code.clone(), r);
                self.freq.record(&code);
                Some(id)
            }
            Decision::Link(b) => {
                let r = self.store.put(payload);
                let id = self.graph.write(code.clone(), r);
                self.graph.link(id, b, 0.5);
                self.freq.record(&code);
                Some(id)
            }
            Decision::Strengthen(b) => {
                // The write we DON'T do: `payload` is dropped, not stored.
                self.graph.strengthen(b);
                self.freq.record(&code);
                Some(b)
            }
            Decision::Skip => None,
        };

        // Any related trace laid down (or reinforced) resolves the gap.
        if gap_related && episode.is_some() {
            self.pending_gap = None;
        }

        let availability = 1.0 - novelty;
        // A knowledge gap is "wanted it, DON'T have it": if this observation
        // just wrote or reinforced something, the want was met — arming a new
        // gap here would immediately re-flag every freshly-answered question
        // as missing again.
        let knowledge_gap = episode.is_none()
            && demand >= self.params.gap_demand
            && availability < self.params.gap_availability;
        if knowledge_gap {
            self.pending_gap = Some(PendingGap {
                code: code.clone(),
                expires: tick + self.params.gap_ttl,
            });
        }

        let surfaced = self.filter_surfaced(candidates, episode, budget, inh_scale);

        Observation {
            decision,
            novelty,
            availability,
            knowledge_gap,
            episode,
            surfaced,
        }
    }

    /// Bulk import: write an episode directly, bypassing the gate (no scan,
    /// no dedup — temporal binding still applies). For backfill/migration of
    /// content that is already known-worth-keeping; normal traffic goes
    /// through `observe`.
    pub fn import(&mut self, code: SparseCode<W>, payload: S::Payload) -> Option<EpisodeId> {
        self.graph.advance_tick();
        if code.is_empty() {
            return None;
        }
        let r = self.store.put(payload);
        let id = self.graph.write(code.clone(), r);
        self.freq.record(&code);
        Some(id)
    }

    /// Explicit query with **forced high demand** — the "I am being asked
    /// directly" special case. For the general path, where the caller has a
    /// real demand estimate, use [`Memory::recall_with_demand`]: the
    /// availability × demand 2×2 only exists if demand actually flows in.
    /// Never writes. Returns payload handles only when the read gate fires.
    pub fn recall(&mut self, dense: &[f32], top: usize) -> Vec<S::Ref> {
        self.recall_with_demand(dense, 1.0, top)
    }

    /// The general read path: caller-supplied demand gates retrieval through
    /// the full 2×2. Low demand → Skip/Suppress (nothing injected, no matter
    /// how available); mid demand + availability → gentle Surface (top-1);
    /// high demand + availability → full Retrieve; high demand without
    /// availability → knowledge gap (armed for eager write, nothing
    /// fabricated).
    pub fn recall_with_demand(&mut self, dense: &[f32], demand: f32, top: usize) -> Vec<S::Ref> {
        let code = self.sparsify(dense);
        self.recall_code_with_demand(&code, demand, top)
    }

    /// See [`Memory::recall`].
    pub fn recall_code(&mut self, code: &SparseCode<W>, top: usize) -> Vec<S::Ref> {
        self.recall_code_with_demand(code, 1.0, top)
    }

    /// See [`Memory::recall_with_demand`].
    pub fn recall_code_with_demand(
        &mut self,
        code: &SparseCode<W>,
        demand: f32,
        top: usize,
    ) -> Vec<S::Ref> {
        let tick = self.graph.advance_tick();
        self.last_retrieval.clear();
        if code.is_empty() {
            return Vec::new();
        }
        let availability = 1.0 - self.freq.novelty(code);
        match self.read_gate.decide(availability, demand) {
            RecallAction::Skip => {
                // Knowledge-gap cell only: wanted it, don't have it.
                if demand >= self.params.gap_demand && availability < self.params.gap_availability {
                    self.pending_gap = Some(PendingGap {
                        code: code.clone(),
                        expires: tick + self.params.gap_ttl,
                    });
                }
                Vec::new()
            }
            RecallAction::Surface => self.retrieve(code, 1, demand, tick),
            RecallAction::Retrieve => self.retrieve(code, top, demand, tick),
        }
    }

    /// Expose the read decision so callers know *why* nothing came back.
    pub fn should_recall_code(&self, code: &SparseCode<W>, demand: f32) -> RecallAction {
        let availability = 1.0 - self.freq.novelty(code);
        self.read_gate.decide(availability, demand)
    }

    pub fn should_recall(&mut self, dense: &[f32], demand: f32) -> RecallAction {
        let code = self.sparsify(dense);
        self.should_recall_code(&code, demand)
    }

    /// Close the reward loop: which injected handles actually helped?
    /// Targets the immediately previous interaction — a `recall()`'s hits OR
    /// an `observe()`'s passively surfaced items (both channels are
    /// trainable). An empty slice means the injection was wasted (trains the
    /// read gate down).
    ///
    /// Each gate takes at most ONE reward step per feedback call, regardless
    /// of how many hits were used — the reward signal is sparse and gentle by
    /// design; a good retrieval session must not slam a threshold to its
    /// clamp. (Episodes are still individually marked and strengthened.)
    pub fn feedback(&mut self, used: &[S::Ref]) {
        if self.last_retrieval.is_empty() {
            return;
        }
        let mut any = false;
        let retrieval: Vec<(EpisodeId, S::Ref)> = core::mem::take(&mut self.last_retrieval);
        for (eid, r) in retrieval {
            // Validate the slot still holds what we injected — a stale id
            // must never mark/strengthen a reused slot (see also sleep()).
            if used.contains(&r) && self.graph.get(eid).is_some_and(|ep| ep.payload == r) {
                any = true;
                self.graph.mark_used(eid);
                self.graph.strengthen(eid);
            }
        }
        if any {
            self.gate.reward(true);
        }
        self.read_gate.reward(any);
    }

    /// Credit by payload: the replayable form of [`Memory::feedback`], for
    /// callers that record feedback as facts and fold them on every device
    /// (where "the previous interaction" is not reproducible). Marks used
    /// and strengthens the episodes holding `used`; `injected` says whether
    /// anything was shown at all — shown but none used trains the read gate
    /// down, exactly as `feedback(&[])` does. One reward step per call.
    pub fn credit(&mut self, used: &[S::Ref], injected: bool) {
        if !injected {
            return;
        }
        let mut any = false;
        for r in used {
            if let Some(eid) = self.graph.episode_of(r) {
                any = true;
                self.graph.mark_used(eid);
                self.graph.strengthen(eid);
            }
        }
        if any {
            self.gate.reward(true);
        }
        self.read_gate.reward(any);
    }

    /// Background consolidation: decay + prune, rebuild the frequency counts
    /// from survivors (forgetting actually forgets), charge the write gate for
    /// memories that sat dead, and evict pruned payloads from the durable store.
    pub fn sleep(&mut self) -> ConsolidationReport<S::Ref> {
        // Pending feedback targets slot-based ids; consolidation can prune
        // and later reuse those slots. Delayed feedback across a sleep is
        // dropped rather than risk marking the wrong episode.
        self.last_retrieval.clear();
        let report = self.graph.consolidate();
        for _ in 0..report.dead.min(4) {
            self.gate.reward(false);
        }
        // Forgetting reclaims durable storage too — `remove` is a no-op for
        // stores whose handles point at caller-owned data (see IdentityStore).
        for &r in &report.forgotten {
            self.store.remove(r);
        }
        self.freq.reset();
        // Borrow dance: collect (code, count) without holding &self.graph.
        let mut codes: Vec<(SparseCode<W>, u32)> = Vec::with_capacity(self.graph.len());
        for (_, ep) in self.graph.iter() {
            codes.push((ep.code.clone(), ep.count));
        }
        for (code, count) in &codes {
            self.freq.record_n(code, *count);
        }
        report
    }

    /// Deliberate retrieval with the two-stage injection confidence gate:
    /// 1. ANCHOR — the scan's best direct jaccard must clear `recall_anchor`,
    ///    or the availability estimate was a mirage (familiar bits, no actual
    ///    episode): treat as a knowledge gap, inject nothing.
    /// 2. FLOOR — each hit's final score must clear `recall_floor`, so
    ///    coincidental low-overlap matches never reach context at capacity.
    fn retrieve(
        &mut self,
        code: &SparseCode<W>,
        top: usize,
        demand: f32,
        tick: u64,
    ) -> Vec<S::Ref> {
        let (hits, best) = self.graph.recall_with_best(code, self.params.hops, top);
        if best.is_none_or(|(_, j)| j < self.params.recall_anchor) {
            // High-demand miss the bit-counts couldn't see: arm the gap.
            if demand >= self.params.gap_demand {
                self.pending_gap = Some(PendingGap {
                    code: code.clone(),
                    expires: tick + self.params.gap_ttl,
                });
            }
            return Vec::new();
        }
        let mut out = Vec::with_capacity(hits.len());
        for (id, score) in hits {
            if score < self.params.recall_floor {
                continue;
            }
            // Deliberate retrieval bypasses the refractory brake but still
            // arms it — what you just recalled won't also nag passively.
            self.graph.mark_surfaced(id);
            if let Some(ep) = self.graph.get(id) {
                self.last_retrieval.push((id, ep.payload));
                out.push(ep.payload);
            }
        }
        out
    }

    /// The passive/associative channel with all three brakes applied:
    /// refractory, budget, threshold — each scaled by the inhibition dial.
    /// Candidates come from the SAME scan the write gate already paid for
    /// (see `observe_code`); this function only filters — it never re-scans.
    fn filter_surfaced(
        &mut self,
        candidates: Vec<(EpisodeId, f32)>,
        exclude: Option<EpisodeId>,
        budget: usize,
        inh_scale: f32,
    ) -> Vec<Surfaced<S::Ref>> {
        if candidates.is_empty() {
            return Vec::new();
        }
        let threshold = self.params.surface_threshold * inh_scale;
        let refractory = (self.params.refractory as f32 * self.inhibition) as u64;

        let tick = self.graph.tick();
        let mut out = Vec::new();
        for (id, activation) in candidates {
            if out.len() >= budget {
                break;
            }
            if Some(id) == exclude || activation < threshold {
                continue;
            }
            let Some(ep) = self.graph.get(id) else {
                continue;
            };
            if ep.last_surfaced != 0 && tick.saturating_sub(ep.last_surfaced) <= refractory {
                continue;
            }
            out.push(Surfaced {
                payload: ep.payload,
                episode: id,
                activation,
            });
            // Passive injections are feedback targets too — the proactive
            // "this reminds me of…" channel must be trainable, not just the
            // deliberate one.
            self.last_retrieval.push((id, ep.payload));
            self.graph.mark_surfaced(id);
        }
        out
    }

    // ---- persistence (settled decision 11) ----
    // Serializes codes, structure, thresholds, tick, the armed gap — NOT
    // payloads (durability is the PayloadStore's job) and NOT Params/gate
    // learning rates & clamps (those reset to defaults on restore; only the
    // two learned thresholds travel). Frequency counts are rebuilt from
    // surviving episodes, i.e. a restore lands in the same state a sleep
    // would produce.

    const MAGIC: u32 = 0x454E_474D; // "ENGM"
    const VERSION: u16 = 4; // v4: pending flush material (dirty + removals) travels

    /// `model_tag` fingerprints the embedding model that produced every code
    /// in this memory (see [`model_tag`]). A snapshot restored under a
    /// different encoder would be semantic garbage with intact structure —
    /// silently. The tag makes that a hard error instead.
    pub fn snapshot(&self, model_tag: u64) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&Self::MAGIC.to_le_bytes());
        out.extend_from_slice(&Self::VERSION.to_le_bytes());
        out.extend_from_slice(&(W as u16).to_le_bytes());
        out.push(<S::Ref as PayloadRef>::SIZE as u8);
        out.extend_from_slice(&model_tag.to_le_bytes());
        out.extend_from_slice(&(self.params.k as u32).to_le_bytes());
        let dim = self.hasher.as_ref().map_or(0, |h| h.dim()) as u32;
        out.extend_from_slice(&dim.to_le_bytes());
        out.extend_from_slice(&self.gate.threshold.to_le_bytes());
        out.extend_from_slice(&self.read_gate.threshold.to_le_bytes());
        out.extend_from_slice(&self.inhibition.to_le_bytes());
        match &self.pending_gap {
            None => out.push(0),
            Some(g) => {
                out.push(1);
                for w in g.code.as_words() {
                    out.extend_from_slice(&w.to_le_bytes());
                }
                out.extend_from_slice(&g.expires.to_le_bytes());
            }
        }
        self.graph.write_bytes(&mut out);
        out
    }

    /// Refuses (returns `None`) on magic/version/width mismatch AND on a
    /// `model_tag` mismatch — an engram is downstream of its encoder; if the
    /// encoder changed, the hippocampus must be wiped, not reinterpreted.
    pub fn restore(bytes: &[u8], model_tag: u64, store: S) -> Option<Self> {
        let mut cur = Cursor::new(bytes);
        if cur.u32()? != Self::MAGIC || cur.u16()? != Self::VERSION || cur.u16()? != W as u16 {
            return None;
        }
        if cur.u8()? as usize != <S::Ref as PayloadRef>::SIZE {
            return None; // snapshot was written with a different handle width
        }
        if cur.u64()? != model_tag {
            return None; // different embedding model: codes are meaningless
        }
        let k = cur.u32()? as usize;
        let dim = cur.u32()? as usize;
        let gate_t = cur.f32()?;
        let read_t = cur.f32()?;
        let inhibition = cur.f32()?;
        let pending_gap = match cur.u8()? {
            0 => None,
            _ => {
                let mut words = [0u64; W];
                for w in words.iter_mut() {
                    *w = cur.u64()?;
                }
                let expires = cur.u64()?;
                Some(PendingGap {
                    code: SparseCode::from_words(words),
                    expires,
                })
            }
        };
        let graph = MemoryGraph::read_bytes(&mut cur)?;

        let mut params = Params::for_bits(W * 64);
        params.k = k;
        let mut gate = NoveltyGate::default();
        gate.threshold = gate_t;
        let mut read_gate = ReadGate::default();
        read_gate.threshold = read_t;

        let mut freq = BitFreq::new(W * 64);
        let mut codes: Vec<(SparseCode<W>, u32)> = Vec::new();
        for (_, ep) in graph.iter() {
            codes.push((ep.code.clone(), ep.count));
        }
        for (code, count) in &codes {
            freq.record_n(code, *count);
        }

        Some(Self {
            params,
            gate,
            read_gate,
            store,
            freq,
            graph,
            hasher: if dim > 0 {
                Some(FlyHasher::new(dim, k))
            } else {
                None
            },
            inhibition,
            pending_gap,
            last_retrieval: Vec::new(),
        })
    }
}

/// Relational fold: rebuilding a graph FROM synced rows. Only available when
/// the store is identity-shaped (`Payload = Ref`, the [`IdentityStore`] /
/// edge-talk shape): in a fold the payload handle IS the cross-device-stable
/// episode identity, so it must arrive from the rows verbatim. A store that
/// mints fresh handles on `put` (like [`InMemoryStore`]) cannot round-trip
/// that identity — every re-fold would mint a new handle and duplicate the
/// episode instead of updating it, which is why this impl never touches the
/// store: content durability already lives in the caller's rows.
impl<const W: usize, R: PayloadRef, S: PayloadStore<Ref = R, Payload = R>> Memory<W, S> {
    /// Import/overwrite one episode from its synced row. Two-pass load:
    /// upsert every episode first, then apply edges. Frequency (availability)
    /// statistics record only on CREATE — overwriting an existing episode is
    /// frequency-neutral, so sync-time re-folds can run repeatedly without
    /// skewing availability (sleep rebuilds the counts exactly anyway).
    pub fn upsert_episode(
        &mut self,
        dense: &[f32],
        payload: R,
        strength: f32,
        count: u32,
        used: bool,
    ) -> EpisodeId {
        let code = self.sparsify(dense);
        let (id, created) = self
            .graph
            .upsert_state(code.clone(), payload, strength, count, used);
        if created {
            self.freq.record_n(&code, count.max(1));
        }
        id
    }
}

/// Apply a frozen per-model centering vector: subtract, then L2-renormalize.
/// Contrastively-trained embedders (E5, EmbeddingGemma, Qwen3-Embedding, …)
/// produce anisotropic spaces — a dominant shared component that biases the
/// fly-hash's k-WTA toward the same winners for every input. Removing the
/// *model-level* mean (computed once per model from generic text — see the
/// `engram-fit` tool — never from user data, never drifting) restores code
/// diversity. The centering vector is part of the encoder identity: version
/// it in [`model_tag`], and use the role-appropriate center for prefixed
/// models (document center for stored content, query center for cues).
pub fn center_in_place(dense: &mut [f32], center: &[f32]) {
    debug_assert_eq!(dense.len(), center.len());
    let mut norm = 0.0f32;
    for (d, c) in dense.iter_mut().zip(center) {
        *d -= c;
        norm += *d * *d;
    }
    if norm > 0.0 && norm.is_finite() {
        let inv = inv_sqrt(norm);
        for d in dense.iter_mut() {
            *d *= inv;
        }
    }
}

/// 1/sqrt(x) without `std` float math (core has none): bit-level seed + three
/// Newton–Raphson steps — converged to f32 precision, identical on all
/// feature sets.
#[inline]
fn inv_sqrt(x: f32) -> f32 {
    let mut y = f32::from_bits(0x5f37_59df - (x.to_bits() >> 1));
    for _ in 0..3 {
        y *= 1.5 - 0.5 * x * y * y;
    }
    y
}

/// f32 → IEEE 754 binary16 bits, round-to-nearest-even. Pure bit math
/// (no_std-friendly). For f16 embedding storage: engram gates on codes, not
/// dense values, and f16's ~1e-3 relative error is far below the centered
/// component scale — but store and fit must see the SAME values, so fit
/// tooling round-trips through these exact functions.
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x007f_ffff;
    if exp == 0xff {
        // inf / nan (preserve nan-ness)
        return sign | 0x7c00 | if man != 0 { 0x0200 } else { 0 };
    }
    let unbiased = exp - 127;
    if unbiased > 15 {
        return sign | 0x7c00; // overflow → inf
    }
    if unbiased >= -14 {
        // normal half
        let half_exp = (unbiased + 15) as u32;
        let mut half = (half_exp << 10) | (man >> 13);
        let round = man & 0x1fff;
        if round > 0x1000 || (round == 0x1000 && (half & 1) == 1) {
            half += 1; // carry may ripple into the exponent — that's correct
        }
        if half >= 0x7c00 {
            return sign | 0x7c00;
        }
        return sign | half as u16;
    }
    // subnormal half (or underflow to zero)
    if unbiased < -25 {
        return sign;
    }
    let man = man | 0x0080_0000; // implicit leading 1: 24-bit significand
                                 // value = man24 × 2^(unbiased−23); half subnormal unit is 2^−24, so the
                                 // integer half mantissa is man24 × 2^(unbiased+1) → shift right by
                                 // −unbiased−1 (14 at the subnormal boundary, 24 at underflow).
    let shift = (-unbiased - 1) as u32;
    let mut half = man >> shift;
    let rem = man & ((1u32 << shift) - 1);
    let halfway = 1u32 << (shift - 1);
    if rem > halfway || (rem == halfway && (half & 1) == 1) {
        half += 1;
    }
    sign | half as u16
}

/// IEEE 754 binary16 bits → f32 (exact — every f16 value is representable).
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign // ±0
        } else {
            // subnormal: value = man/2^10 × 2^−14 — renormalize to 1.m × 2^e
            let mut e = 0i32;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            sign | (((113 + e) as u32) << 23) | (m << 13) // 113 = 127−14
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 112) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// FNV-1a fingerprint of an embedding-model identifier, for
/// [`Memory::snapshot`] / [`Memory::restore`]. Use the most specific name you
/// have (e.g. `"all-MiniLM-L6-v2-int8"`); any model/quantization change must
/// produce a different tag.
pub fn model_tag(name: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Cheap caller-side demand estimator for text queries: references to prior
/// context (anaphora), questions, explicit recall verbs. A convenience — the
/// caller usually has better task signals; engram itself only sees vectors.
pub fn heuristic_demand(text: &str) -> f32 {
    let mut score = 0.0f32;
    let lower: alloc::string::String = text.chars().flat_map(|c| c.to_lowercase()).collect();
    if lower.contains('?') {
        score += 0.35;
    }
    const RECALL_MARKERS: [&str; 8] = [
        "remember",
        "recall",
        "what did",
        "when did",
        "where did",
        "last time",
        "we decided",
        "you said",
    ];
    if RECALL_MARKERS.iter().any(|m| lower.contains(m)) {
        score += 0.45;
    }
    const ANAPHORA: [&str; 6] = [" that ", " it ", " this ", " those ", "earlier", "before"];
    let hits = ANAPHORA.iter().filter(|m| lower.contains(*m)).count();
    score += 0.1 * hits.min(3) as f32;
    score.clamp(0.0, 1.0)
}

/// Little-endian byte cursor for snapshot parsing.
pub(crate) struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        if end > self.data.len() {
            return None;
        }
        let s = &self.data[self.pos..end];
        self.pos = end;
        Some(s)
    }

    pub(crate) fn take_bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        self.take(n)
    }

    pub(crate) fn u8(&mut self) -> Option<u8> {
        self.take(1).map(|s| s[0])
    }

    pub(crate) fn u16(&mut self) -> Option<u16> {
        self.take(2)
            .map(|s| u16::from_le_bytes(s.try_into().unwrap()))
    }

    pub(crate) fn u32(&mut self) -> Option<u32> {
        self.take(4)
            .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
    }

    pub(crate) fn u64(&mut self) -> Option<u64> {
        self.take(8)
            .map(|s| u64::from_le_bytes(s.try_into().unwrap()))
    }

    pub(crate) fn f32(&mut self) -> Option<f32> {
        self.take(4)
            .map(|s| f32::from_le_bytes(s.try_into().unwrap()))
    }
}

#[cfg(test)]
mod f16_tests {
    use super::{f16_bits_to_f32, f32_to_f16_bits};

    #[test]
    fn known_constants() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(-2.0), 0xc000);
        assert_eq!(f32_to_f16_bits(0.5), 0x3800);
        assert_eq!(f32_to_f16_bits(65504.0), 0x7bff); // f16 max
        assert_eq!(f32_to_f16_bits(1e6), 0x7c00); // overflow → inf
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f16_bits_to_f32(0x3c00), 1.0);
        assert_eq!(f16_bits_to_f32(0x0001), 5.960_464_5e-8); // min subnormal
    }

    #[test]
    fn exhaustive_roundtrip_identity() {
        // Every one of the 65536 half values must survive f16→f32→f16 exactly
        // (NaNs keep NaN-ness; payloads may canonicalize).
        for bits in 0..=u16::MAX {
            let f = f16_bits_to_f32(bits);
            if f.is_nan() {
                assert!(f16_bits_to_f32(f32_to_f16_bits(f)).is_nan());
                continue;
            }
            assert_eq!(
                f32_to_f16_bits(f),
                bits,
                "roundtrip failed for {bits:#06x} ({f})"
            );
        }
    }

    #[test]
    fn rounding_is_nearest_even() {
        // 1.0 + 2^-11 is exactly halfway between 1.0 and the next half up —
        // must round to EVEN mantissa (1.0).
        assert_eq!(f32_to_f16_bits(1.0 + 0.000_488_281_25), 0x3c00);
        // Just above halfway rounds up.
        assert_eq!(f32_to_f16_bits(1.0 + 0.000_489), 0x3c01);
        // Embedding-scale values keep ~3 decimal digits.
        let x = 0.0123f32;
        let rt = f16_bits_to_f32(f32_to_f16_bits(x));
        assert!((rt - x).abs() / x < 1e-3, "{x} → {rt}");
    }
}
