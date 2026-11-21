//! The episodic memory graph: slot-stable episodes, Hebbian edges, spreading
//! activation recall, and consolidation (decay + prune + report).
//!
//! Generic over the payload handle `R` ([`PayloadRef`]): the episode carries a
//! small, fixed-size id (a store index, a rowid, a UUID) — never the content.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;

use core::cell::RefCell;

use crate::index::{BitIndex, ScanScratch, BULK_REMOVE};
use crate::sdr::SparseCode;
use crate::store::PayloadRef;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EpisodeId(pub u32);

/// `(episode, score)` pairs as returned by recall, plus the best raw-jaccard
/// match from the same scan (the write gate's ambiguous-band ground truth).
pub type RecallHits = Vec<(EpisodeId, f32)>;
pub type BestMatch = Option<(EpisodeId, f32)>;

#[derive(Clone, Debug)]
pub struct Episode<const W: usize, R> {
    pub code: SparseCode<W>,
    pub payload: R,
    pub strength: f32,
    pub count: u32,
    pub last_seen: u64,
    /// Refractory bookkeeping: tick of last surfacing (0 = never surfaced).
    pub last_surfaced: u64,
    /// Set by feedback when this episode's payload actually helped downstream.
    pub used: bool,
    /// Hebbian links: co-occurrence + temporal adjacency.
    pub edges: Vec<(EpisodeId, f32)>,
}

pub const STRENGTH_INIT: f32 = 1.0;
pub const STRENGTH_MAX: f32 = 8.0;
pub const STRENGTH_DECAY: f32 = 0.85;
pub const PRUNE_FLOOR: f32 = 0.3;
pub const EDGE_DECAY: f32 = 0.9;
pub const EDGE_FLOOR: f32 = 0.05;
/// Edge weight for temporal adjacency binding at write time.
pub const TEMPORAL_W: f32 = 0.25;
/// How many ticks back an episode counts as "recently active" for binding.
pub const RECENT_TICKS: u64 = 3;
const RECENT_MAX: usize = 8;
/// Minimum cue jaccard to seed spreading activation.
const SEED_MIN: f32 = 0.05;
/// Per-hop activation transfer factor.
const SPREAD: f32 = 0.5;

pub struct ConsolidationReport<R> {
    pub pruned: usize,
    /// Of the pruned: never surfaced, never used, seen once — "sat dead".
    /// Feeds `NoveltyGate::reward(false)`.
    pub dead: usize,
    /// Payload handles of the pruned episodes — the orchestrator hands these to
    /// `PayloadStore::remove` so forgetting reclaims durable storage too.
    pub forgotten: Vec<R>,
}

/// Cross-device episode state: exactly what syncs. Codes are NOT here — they
/// derive from the (separately synced) embeddings; ticks (`last_seen`,
/// `last_surfaced`) are session-local politeness, not memory. Edges ride the
/// episode (one row per episode, edges embedded): symmetric duplication means
/// either side's row alone reconstructs its outgoing links.
#[derive(Clone, Debug, PartialEq)]
pub struct EpisodeState<R> {
    pub payload: R,
    pub strength: f32,
    pub count: u32,
    pub used: bool,
    /// Outgoing Hebbian edges as (target payload, weight).
    pub edges: Vec<(R, f32)>,
}

/// Everything that changed since the last drain — the flush unit the
/// orchestrator writes to durable rows (upserts + deletes) in one transaction.
#[derive(Debug, Default)]
pub struct DirtyReport<R> {
    pub episodes: Vec<EpisodeState<R>>,
    /// Payloads of episodes removed (pruned or explicitly removed) — delete
    /// their rows.
    pub removed: Vec<R>,
}

#[derive(Clone)]
pub struct MemoryGraph<const W: usize, R: PayloadRef = u64> {
    slots: Vec<Option<Episode<W, R>>>,
    free: Vec<u32>,
    tick: u64,
    /// (slot, tick) of recently written/strengthened episodes, for Hebbian binding.
    recent: Vec<(u32, u64)>,
    /// Full-scan counter (debug/acceptance: `observe` must cost ≤ 1 scan).
    scans: core::cell::Cell<u64>,
    /// payload → slot. Payloads are the only cross-device-stable identity;
    /// relational state import/export resolves through this.
    index: BTreeMap<R, u32>,
    /// Slots mutated since the last `drain_dirty` (write/strengthen/link/
    /// mark_used/decay — NOT mark_surfaced, refractory is session-local).
    dirty: BTreeSet<u32>,
    /// Payloads removed since the last drain (prune or explicit removal).
    removed_dirty: Vec<R>,
    /// What the last `drain_dirty` took, for `undrain` — the drain's clear is
    /// not transactional with the caller's row writes; on a failed flush the
    /// caller restores the dirty state so nothing is silently lost.
    last_drain: BTreeSet<u32>,
    last_drain_removed: Vec<R>,
    /// Exact inverted index (bit → slots), so the fused scan only visits
    /// episodes sharing a bit with the cue. Derived: rebuilt on restore, never
    /// serialized. Every code mutation goes through `place`/`upsert_state`/
    /// the removal paths, which keep it in step (that is why `get_mut` is
    /// crate-private: a caller rewriting `ep.code` would desync it).
    inv: BitIndex,
    /// Reused per-scan working memory (no O(slots) alloc per observation).
    scratch: RefCell<ScanScratch>,
}

impl<const W: usize, R: PayloadRef> Default for MemoryGraph<W, R> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            tick: 0,
            recent: Vec::new(),
            scans: core::cell::Cell::new(0),
            index: BTreeMap::new(),
            dirty: BTreeSet::new(),
            removed_dirty: Vec::new(),
            last_drain: BTreeSet::new(),
            last_drain_removed: Vec::new(),
            inv: BitIndex::new(W * 64),
            scratch: RefCell::new(ScanScratch::default()),
        }
    }
}

impl<const W: usize, R: PayloadRef> MemoryGraph<W, R> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn advance_tick(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    pub fn len(&self) -> usize {
        self.slots.len() - self.free.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The episode holding `payload`, if any.
    pub fn episode_of(&self, payload: &R) -> Option<EpisodeId> {
        self.index.get(payload).map(|&s| EpisodeId(s))
    }

    pub fn get(&self, id: EpisodeId) -> Option<&Episode<W, R>> {
        self.slots.get(id.0 as usize).and_then(|s| s.as_ref())
    }

    /// Crate-private: episodes' codes are indexed, so outside mutation of
    /// `code` would desync the inverted index.
    pub(crate) fn get_mut(&mut self, id: EpisodeId) -> Option<&mut Episode<W, R>> {
        self.slots.get_mut(id.0 as usize).and_then(|s| s.as_mut())
    }

    pub fn iter(&self) -> impl Iterator<Item = (EpisodeId, &Episode<W, R>)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|e| (EpisodeId(i as u32), e)))
    }

    fn prune_recent(&mut self) {
        let tick = self.tick;
        self.recent
            .retain(|&(_, t)| tick.saturating_sub(t) <= RECENT_TICKS);
        if self.recent.len() > RECENT_MAX {
            let excess = self.recent.len() - RECENT_MAX;
            self.recent.drain(..excess);
        }
    }

    fn push_recent(&mut self, slot: u32) {
        self.recent.retain(|&(s, _)| s != slot);
        self.recent.push((slot, self.tick));
        self.prune_recent();
    }

    fn place(&mut self, ep: Episode<W, R>) -> u32 {
        let payload = ep.payload;
        let slot = match self.free.pop() {
            Some(s) => {
                self.slots[s as usize] = Some(ep);
                s
            }
            None => {
                self.slots.push(Some(ep));
                (self.slots.len() - 1) as u32
            }
        };
        self.index.insert(payload, slot);
        let code = &self.slots[slot as usize].as_ref().unwrap().code;
        self.inv.insert(slot, code);
        slot
    }

    /// Write a new episode; Hebbian-bind it to episodes active in the last few
    /// ticks (temporal + co-occurrence).
    pub fn write(&mut self, code: SparseCode<W>, payload: R) -> EpisodeId {
        let ep = Episode {
            code,
            payload,
            strength: STRENGTH_INIT,
            count: 1,
            last_seen: self.tick,
            last_surfaced: 0,
            used: false,
            edges: Vec::new(),
        };
        let slot = self.place(ep);
        let id = EpisodeId(slot);
        self.dirty.insert(slot);
        self.prune_recent();
        let recent: Vec<u32> = self.recent.iter().map(|&(s, _)| s).collect();
        for other in recent {
            self.link(id, EpisodeId(other), TEMPORAL_W);
        }
        self.push_recent(slot);
        id
    }

    pub fn strengthen(&mut self, id: EpisodeId) {
        let tick = self.tick;
        if let Some(ep) = self.get_mut(id) {
            ep.strength = (ep.strength + 1.0).min(STRENGTH_MAX);
            ep.count += 1;
            ep.last_seen = tick;
            self.dirty.insert(id.0);
            self.push_recent(id.0);
        }
    }

    /// Upsert a symmetric Hebbian edge; weights accumulate, capped at 1.0.
    pub fn link(&mut self, a: EpisodeId, b: EpisodeId, w: f32) {
        if a == b || self.get(a).is_none() || self.get(b).is_none() {
            return;
        }
        for (from, to) in [(a, b), (b, a)] {
            let ep = self.get_mut(from).unwrap();
            if let Some(e) = ep.edges.iter_mut().find(|(t, _)| *t == to) {
                e.1 = (e.1 + w).min(1.0);
            } else {
                ep.edges.push((to, w.min(1.0)));
            }
            self.dirty.insert(from.0);
        }
    }

    /// Total cue scans performed (best_match + recall), indexed or not.
    /// Debug / acceptance counter: one `observe()` must cost at most ONE scan.
    pub fn scan_count(&self) -> u64 {
        self.scans.get()
    }

    /// Heap bytes held by the inverted index and the reusable scan scratch —
    /// the memory the index costs on top of the episodes themselves.
    pub fn index_heap_bytes(&self) -> usize {
        let s = self.scratch.borrow();
        self.inv.heap_bytes()
            + s.counts.capacity() * 4
            + s.act.capacity() * 4
            + s.touched.capacity() * 4
            + s.active.capacity() * 8
            + s.sources.capacity() * 8
    }

    /// The lowest live slot — the full scan's initial `best` when no episode
    /// shares a bit with the cue (every jaccard is then exactly `+0.0`).
    fn first_live(&self) -> Option<EpisodeId> {
        self.slots
            .iter()
            .position(|s| s.is_some())
            .map(|i| EpisodeId(i as u32))
    }

    /// Phase 1 of the indexed scan: overlap counts via postings, then exact
    /// jaccard for every episode sharing ≥ 1 bit. Returns the best match with
    /// the full scan's semantics (max jaccard, ties to the LOWEST slot —
    /// what a strict `>` over ascending slots keeps; an all-zero graph yields
    /// the first live slot at 0.0). With `seed`, writes `act[slot] = j` for
    /// `j ≥ SEED_MIN` and marks those slots in the `active` bitmap. Leaves
    /// `counts` all zero and `touched` empty.
    fn scan_indexed(&self, cue: &SparseCode<W>, s: &mut ScanScratch, seed: bool) -> BestMatch {
        s.ensure(self.slots.len());
        let ScanScratch {
            counts,
            act,
            touched,
            active,
            ..
        } = s;
        let mut cue_pop = 0u32;
        for b in cue.iter_ones() {
            cue_pop += 1;
            for &slot in self.inv.postings(b) {
                let c = &mut counts[slot as usize];
                if *c == 0 {
                    touched.push(slot);
                }
                *c += 1;
            }
        }
        let mut best: Option<(u32, f32)> = None;
        for &slot in touched.iter() {
            let inter = core::mem::take(&mut counts[slot as usize]);
            // Same integers `SparseCode::jaccard` divides: popcnt(a|b) =
            // |a| + |b| − |a&b|, and inter ≥ 1 so the union is never 0.
            let union = cue_pop + self.inv.pop(slot) - inter;
            let j = inter as f32 / union as f32;
            if best.is_none_or(|(bs, bj)| j > bj || (j == bj && slot < bs)) {
                best = Some((slot, j));
            }
            if seed && j >= SEED_MIN {
                act[slot as usize] = j;
                active[slot as usize / 64] |= 1u64 << (slot % 64);
            }
        }
        touched.clear();
        match best {
            Some((slot, j)) => Some((EpisodeId(slot), j)),
            // Nothing shares a bit: every jaccard is +0.0 (0/|ep|, or the
            // empty-union 0.0), so the scan's first candidate stands.
            None => self.first_live().map(|id| (id, 0.0)),
        }
    }

    /// The most similar stored episode (ties: lowest slot). Ground truth for
    /// the write gate's ambiguous band. Indexed: visits only episodes that
    /// share a bit with `code`.
    pub fn best_match(&self, code: &SparseCode<W>) -> Option<(EpisodeId, f32)> {
        self.scans.set(self.scans.get() + 1);
        let mut s = self.scratch.borrow_mut();
        self.scan_indexed(code, &mut s, false)
    }

    /// Retrieval: seed activation from cue similarity, spread it `hops` times
    /// along Hebbian edges, score by activation × strength × recency.
    pub fn recall(&self, cue: &SparseCode<W>, hops: u8, top: usize) -> Vec<(EpisodeId, f32)> {
        self.recall_with_best(cue, hops, top).0
    }

    /// The fused scan: ONE pass yields both the best match (the write gate's
    /// ambiguous-band ground truth) and the seed activations for spreading —
    /// `observe()` needs both views of the same data and must not pay for two
    /// scans (settled decision 2).
    ///
    /// Indexed and sparse throughout: candidates come from the inverted
    /// index, spreading walks only active episodes, scoring only activated
    /// ones. Bit-for-bit identical to [`Self::recall_with_best_scan`] (the
    /// reference full scan; property-tested): sources are visited in
    /// ascending slot order so every float accumulates in the same order, and
    /// the scored list is built in ascending slot order before the same sort.
    pub fn recall_with_best(
        &self,
        cue: &SparseCode<W>,
        hops: u8,
        top: usize,
    ) -> (RecallHits, BestMatch) {
        if self.slots.is_empty() {
            return (Vec::new(), None);
        }
        self.scans.set(self.scans.get() + 1);
        let mut guard = self.scratch.borrow_mut();
        let s = &mut *guard;
        let best = self.scan_indexed(cue, s, top > 0);
        if top == 0 {
            return (Vec::new(), best);
        }
        let ScanScratch {
            act,
            active,
            sources,
            ..
        } = s;
        // The active set is a bitmap: walking it word by word visits slots
        // in ascending order with no sort, which is what keeps every float
        // accumulating in the full scan's order.
        for _ in 0..hops {
            // The pre-hop activations of the sources (the full scan's
            // `snapshot`), in ascending slot order.
            sources.clear();
            for (wi, &word) in active.iter().enumerate() {
                let mut m = word;
                while m != 0 {
                    let slot = (wi * 64) as u32 + m.trailing_zeros();
                    m &= m - 1;
                    let a = act[slot as usize];
                    // NaN passes, as the scan's `a <= 0.01 → skip` lets it.
                    if a > 0.01 || a.is_nan() {
                        sources.push((slot, a));
                    }
                }
            }
            for &(src, a) in sources.iter() {
                let ep = self.slots[src as usize].as_ref().unwrap();
                for &(tgt, w) in &ep.edges {
                    if let Some(slot) = self.slots.get(tgt.0 as usize) {
                        if slot.is_some() {
                            let t = tgt.0 as usize;
                            active[t / 64] |= 1u64 << (t % 64);
                            act[t] += a * w * SPREAD;
                        }
                    }
                }
            }
            // Clamp AFTER each full hop (not per-edge, to avoid edge-order
            // artifacts): a hub receiving many weak inputs must not
            // accumulate unbounded activation and outscore a direct match.
            // Raw activation lives in [0, 1] — a perfect direct seed is the
            // ceiling; association can reach it but never exceed it.
            for (wi, &word) in active.iter().enumerate() {
                let mut m = word;
                while m != 0 {
                    let a = &mut act[wi * 64 + m.trailing_zeros() as usize];
                    m &= m - 1;
                    if *a > 1.0 {
                        *a = 1.0;
                    }
                }
            }
        }
        let mut scored: Vec<(EpisodeId, f32)> = Vec::new();
        for (wi, word) in active.iter_mut().enumerate() {
            let mut m = core::mem::take(word);
            while m != 0 {
                let slot = (wi * 64) as u32 + m.trailing_zeros();
                m &= m - 1;
                let a = core::mem::take(&mut act[slot as usize]);
                if a <= 0.0 {
                    continue;
                }
                let ep = self.slots[slot as usize].as_ref().unwrap();
                scored.push((EpisodeId(slot), self.score(a, ep)));
            }
        }
        sources.clear();
        scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(top);
        (scored, best)
    }

    /// Final score: activation × strength × recency (same expression, same
    /// evaluation order as the reference scan).
    #[inline]
    fn score(&self, a: f32, ep: &Episode<W, R>) -> f32 {
        let strength_w = 0.6 + 0.4 * (ep.strength / STRENGTH_MAX);
        let age = self.tick.saturating_sub(ep.last_seen) as f32;
        // Recency is a modulator (≤2×), never a dominance term: an old
        // exact match must still beat a fresh coincidental overlap.
        let recency_w = 0.5 + 0.5 / (1.0 + 0.01 * age);
        a * strength_w * recency_w
    }

    /// REFERENCE full scan (the pre-index implementation, kept verbatim in
    /// behavior): one jaccard pass over every live episode, dense spreading.
    /// Not for production — it exists so tests and benches can prove the
    /// indexed [`Self::recall_with_best`] is bit-identical and measure it.
    #[doc(hidden)]
    pub fn recall_with_best_scan(
        &self,
        cue: &SparseCode<W>,
        hops: u8,
        top: usize,
    ) -> (RecallHits, BestMatch) {
        if self.slots.is_empty() {
            return (Vec::new(), None);
        }
        let n = self.slots.len();
        let mut act = vec![0.0f32; n];
        let mut best: Option<(EpisodeId, f32)> = None;
        for (id, ep) in self.iter() {
            let j = cue.jaccard(&ep.code);
            if best.is_none_or(|(_, bj)| j > bj) {
                best = Some((id, j));
            }
            if j >= SEED_MIN {
                act[id.0 as usize] = j;
            }
        }
        if top == 0 {
            return (Vec::new(), best);
        }
        for _ in 0..hops {
            let snapshot = act.clone();
            for (id, ep) in self.iter() {
                let a = snapshot[id.0 as usize];
                if a <= 0.01 {
                    continue;
                }
                for &(tgt, w) in &ep.edges {
                    if let Some(slot) = self.slots.get(tgt.0 as usize) {
                        if slot.is_some() {
                            act[tgt.0 as usize] += a * w * SPREAD;
                        }
                    }
                }
            }
            for a in &mut act {
                if *a > 1.0 {
                    *a = 1.0;
                }
            }
        }
        let mut scored: Vec<(EpisodeId, f32)> = self
            .iter()
            .filter_map(|(id, ep)| {
                let a = act[id.0 as usize];
                if a <= 0.0 {
                    return None;
                }
                let strength_w = 0.6 + 0.4 * (ep.strength / STRENGTH_MAX);
                let age = self.tick.saturating_sub(ep.last_seen) as f32;
                let recency_w = 0.5 + 0.5 / (1.0 + 0.01 * age);
                Some((id, a * strength_w * recency_w))
            })
            .collect();
        scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(top);
        (scored, best)
    }

    /// REFERENCE linear best-match scan (pre-index). See
    /// [`Self::recall_with_best_scan`].
    #[doc(hidden)]
    pub fn best_match_scan(&self, code: &SparseCode<W>) -> Option<(EpisodeId, f32)> {
        let mut best: Option<(EpisodeId, f32)> = None;
        for (id, ep) in self.iter() {
            let j = code.jaccard(&ep.code);
            if best.is_none_or(|(_, bj)| j > bj) {
                best = Some((id, j));
            }
        }
        best
    }

    pub fn mark_surfaced(&mut self, id: EpisodeId) {
        let tick = self.tick;
        if let Some(ep) = self.get_mut(id) {
            ep.last_surfaced = tick;
        }
    }

    pub fn mark_used(&mut self, id: EpisodeId) {
        if let Some(ep) = self.get_mut(id) {
            ep.used = true;
            self.dirty.insert(id.0);
        }
    }

    // ---- relational state (cross-device sync) ----

    /// Everything mutated since the last drain, as durable-row material.
    /// Episodes still alive → upserts; `removed` → row deletes. Edge targets
    /// that died since the mutation are dropped here (their rows are gone).
    /// The drained set is stashed for `undrain`: the caller writes the rows
    /// in a transaction, and on FAILURE must undrain — otherwise the clear
    /// below silently orphans these mutations from durability.
    pub fn drain_dirty(&mut self) -> DirtyReport<R> {
        let mut episodes = Vec::with_capacity(self.dirty.len());
        let dirty = core::mem::take(&mut self.dirty);
        self.last_drain = dirty.clone();
        self.last_drain_removed = self.removed_dirty.clone();
        for slot in dirty {
            let Some(ep) = self.slots.get(slot as usize).and_then(|s| s.as_ref()) else {
                continue; // pruned after being dirtied — it's in `removed`
            };
            let edges = ep
                .edges
                .iter()
                .filter_map(|&(t, w)| {
                    self.slots
                        .get(t.0 as usize)
                        .and_then(|s| s.as_ref())
                        .map(|te| (te.payload, w))
                })
                .collect();
            episodes.push(EpisodeState {
                payload: ep.payload,
                strength: ep.strength,
                count: ep.count,
                used: ep.used,
                edges,
            });
        }
        DirtyReport {
            episodes,
            removed: core::mem::take(&mut self.removed_dirty),
        }
    }

    /// Restore the last drain after a FAILED flush: the dirty set and pending
    /// removals go back so the next flush retries them. Call only when the
    /// transaction that consumed the drain rolled back.
    pub fn undrain(&mut self) {
        let restored = core::mem::take(&mut self.last_drain);
        self.dirty.extend(restored);
        let removed = core::mem::take(&mut self.last_drain_removed);
        for r in removed {
            if !self.removed_dirty.contains(&r) {
                self.removed_dirty.push(r);
            }
        }
    }

    /// Import/overwrite one episode's synced state. Does NOT mark dirty (the
    /// state came FROM the rows), does not temporal-bind, does not touch
    /// `recent` — imported history is knowledge, not present experience.
    /// Ticks start neutral (0). Edges are applied separately (two-pass load:
    /// all episodes first, then `set_edges_by_payload` — targets must exist).
    /// LOCALLY-DIRTY episodes are NOT overwritten (same rule as
    /// `retain_payloads`): pending-flush state is newer than any row — a
    /// sync-time re-fold racing a mid-batch sweep must not revert its
    /// strengthens. Returns `(id, created)` — `created` lets the orchestrator
    /// keep repeated re-folds frequency-neutral (only NEW episodes record bits).
    pub fn upsert_state(
        &mut self,
        code: SparseCode<W>,
        payload: R,
        strength: f32,
        count: u32,
        used: bool,
    ) -> (EpisodeId, bool) {
        if let Some(&slot) = self.index.get(&payload) {
            if self.dirty.contains(&slot) {
                return (EpisodeId(slot), false); // pending flush wins over rows
            }
            if let Some(ep) = self.slots.get_mut(slot as usize).and_then(|s| s.as_mut()) {
                if ep.code != code {
                    self.inv.remove(slot, &ep.code);
                    self.inv.insert(slot, &code);
                }
                ep.code = code;
                ep.strength = strength;
                ep.count = count;
                ep.used = used;
                return (EpisodeId(slot), false);
            }
        }
        let ep = Episode {
            code,
            payload,
            strength,
            count,
            last_seen: 0,
            last_surfaced: 0,
            used,
            edges: Vec::new(),
        };
        (EpisodeId(self.place(ep)), true)
    }

    /// Replace an episode's outgoing edges from synced rows. Unknown targets
    /// (not yet imported/synced) are skipped — the next import pass heals.
    pub fn set_edges_by_payload(&mut self, payload: R, edges: &[(R, f32)]) {
        let Some(&slot) = self.index.get(&payload) else {
            return;
        };
        let resolved: Vec<(EpisodeId, f32)> = edges
            .iter()
            .filter_map(|&(t, w)| self.index.get(&t).map(|&s| (EpisodeId(s), w)))
            .collect();
        if let Some(ep) = self.slots.get_mut(slot as usize).and_then(|s| s.as_mut()) {
            ep.edges = resolved;
        }
    }

    /// Reconcile with the synced row set: drop every live episode whose
    /// payload is NOT in `keep` — a remote sleep's prunes take effect at
    /// sync time instead of the next boot. LOCALLY-DIRTY episodes survive
    /// (they're pending flush, newer than any row — e.g. a provisional
    /// pre-persist observation). Removals are NOT re-dirtied (their rows are
    /// already gone). Returns the removed payloads (store eviction).
    pub fn retain_payloads(&mut self, keep: &BTreeSet<R>) -> Vec<R> {
        let dirty = self.dirty.clone(); // split the borrow from slots below
        let mut removed_slots: Vec<u32> = Vec::new();
        let mut removed: Vec<R> = Vec::new();
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if let Some(ep) = slot {
                if keep.contains(&ep.payload) || dirty.contains(&(i as u32)) {
                    continue;
                }
                removed.push(ep.payload);
                removed_slots.push(i as u32);
                if removed_slots.len() <= BULK_REMOVE {
                    self.inv.remove(i as u32, &ep.code);
                }
                *slot = None;
            }
        }
        if !removed_slots.is_empty() {
            let mut gone = vec![false; self.slots.len()];
            for &r in &removed_slots {
                gone[r as usize] = true;
            }
            for slot in self.slots.iter_mut().flatten() {
                slot.edges.retain(|(t, _)| !gone[t.0 as usize]);
            }
            self.recent.retain(|&(s, _)| !gone[s as usize]);
            self.index.retain(|_, &mut s| !gone[s as usize]);
            if removed_slots.len() > BULK_REMOVE {
                self.inv.retain_live(&gone);
            }
            self.free.extend_from_slice(&removed_slots);
        }
        removed
    }

    /// Targeted negative feedback, payload-addressed like the other
    /// relational ops: a memory that keeps surfacing without ever helping
    /// gets its strength multiplied by `factor`, clamped to [0, 1] — a knock
    /// can only knock, never amplify (the recall floor relies on strength
    /// staying within [0, STRENGTH_MAX]). Gentler than forgetting;
    /// consolidation's prune floor finishes the job if it keeps failing.
    /// Dirty-marked so the knock folds fleet-wide.
    pub fn penalize_by_payload(&mut self, payload: R, factor: f32) -> bool {
        let factor = factor.clamp(0.0, 1.0);
        let Some(&slot) = self.index.get(&payload) else {
            return false;
        };
        if let Some(Some(ep)) = self.slots.get_mut(slot as usize) {
            ep.strength *= factor;
            self.dirty.insert(slot);
            true
        } else {
            false
        }
    }

    /// Remove an episode by payload (synced row delete). Scrubs incoming
    /// edges so the freed slot can't be re-targeted by stale links.
    pub fn remove_by_payload(&mut self, payload: R) -> bool {
        let Some(slot) = self.index.remove(&payload) else {
            return false;
        };
        if let Some(s) = self.slots.get_mut(slot as usize) {
            if let Some(ep) = s.take() {
                self.inv.remove(slot, &ep.code);
            }
        }
        let id = EpisodeId(slot);
        for other in self.slots.iter_mut().flatten() {
            other.edges.retain(|(t, _)| *t != id);
        }
        self.recent.retain(|&(s, _)| s != slot);
        self.dirty.remove(&slot);
        self.free.push(slot);
        true
    }

    /// Sleep: decay strengths and edges, prune below floor, drop dangling
    /// edges. Reports prunes (counts + payload handles) so the orchestrator can
    /// close the reward loop and evict from the durable store.
    pub fn consolidate(&mut self) -> ConsolidationReport<R> {
        let mut pruned = 0usize;
        let mut dead = 0usize;
        let mut forgotten: Vec<R> = Vec::new();
        let mut removed: Vec<u32> = Vec::new();

        for (i, slot) in self.slots.iter_mut().enumerate() {
            if let Some(ep) = slot {
                ep.strength *= STRENGTH_DECAY;
                ep.edges.retain_mut(|(_, w)| {
                    *w *= EDGE_DECAY;
                    *w >= EDGE_FLOOR
                });
                if ep.strength < PRUNE_FLOOR {
                    pruned += 1;
                    if !ep.used && ep.count <= 1 && ep.last_surfaced == 0 {
                        dead += 1;
                    }
                    forgotten.push(ep.payload);
                    removed.push(i as u32);
                    if removed.len() <= BULK_REMOVE {
                        self.inv.remove(i as u32, &ep.code);
                    }
                    *slot = None;
                } else {
                    // Decay touched every survivor — all are flush material.
                    self.dirty.insert(i as u32);
                }
            }
        }
        if !removed.is_empty() {
            // Bitmap instead of `removed.contains()` — a mass-prune after a
            // long run must not go O(pruned × edges).
            let mut gone = vec![false; self.slots.len()];
            for &r in &removed {
                gone[r as usize] = true;
            }
            for slot in self.slots.iter_mut().flatten() {
                slot.edges.retain(|(t, _)| !gone[t.0 as usize]);
            }
            self.recent.retain(|&(s, _)| !gone[s as usize]);
            if removed.len() > BULK_REMOVE {
                self.inv.retain_live(&gone);
            }
            for &r in &removed {
                self.dirty.remove(&r);
            }
            self.index.retain(|_, &mut s| !gone[s as usize]);
            self.removed_dirty.extend_from_slice(&forgotten);
            self.free.extend_from_slice(&removed);
        }
        ConsolidationReport {
            pruned,
            dead,
            forgotten,
        }
    }

    // ---- persistence helpers (used by Memory::snapshot/restore) ----

    pub(crate) fn write_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.tick.to_le_bytes());
        out.extend_from_slice(&(self.slots.len() as u32).to_le_bytes());
        for slot in &self.slots {
            match slot {
                None => out.push(0),
                Some(ep) => {
                    out.push(1);
                    for w in ep.code.as_words() {
                        out.extend_from_slice(&w.to_le_bytes());
                    }
                    ep.payload.write_bytes(out);
                    out.extend_from_slice(&ep.strength.to_le_bytes());
                    out.extend_from_slice(&ep.count.to_le_bytes());
                    out.extend_from_slice(&ep.last_seen.to_le_bytes());
                    out.extend_from_slice(&ep.last_surfaced.to_le_bytes());
                    out.push(ep.used as u8);
                    out.extend_from_slice(&(ep.edges.len() as u32).to_le_bytes());
                    for (t, w) in &ep.edges {
                        out.extend_from_slice(&t.0.to_le_bytes());
                        out.extend_from_slice(&w.to_le_bytes());
                    }
                }
            }
        }
        // Pending flush material travels with the snapshot: a restore that
        // silently dropped `dirty`/`removed_dirty` would orphan those
        // mutations from durability forever — the rows would never see them.
        // (`last_drain` does NOT travel: drain → tx → commit/undrain is a
        // synchronous host-side contract, not cross-restore state.)
        out.extend_from_slice(&(self.dirty.len() as u32).to_le_bytes());
        for &s in &self.dirty {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out.extend_from_slice(&(self.removed_dirty.len() as u32).to_le_bytes());
        for r in &self.removed_dirty {
            r.write_bytes(out);
        }
    }

    pub(crate) fn read_bytes(cur: &mut crate::Cursor<'_>) -> Option<Self> {
        let tick = cur.u64()?;
        let n = cur.u32()? as usize;
        let mut slots: Vec<Option<Episode<W, R>>> = Vec::with_capacity(n);
        let mut free = Vec::new();
        for i in 0..n {
            match cur.u8()? {
                0 => {
                    slots.push(None);
                    free.push(i as u32);
                }
                1 => {
                    let mut words = [0u64; W];
                    for w in words.iter_mut() {
                        *w = cur.u64()?;
                    }
                    let payload = R::from_bytes(cur.take_bytes(R::SIZE)?)?;
                    let strength = cur.f32()?;
                    let count = cur.u32()?;
                    let last_seen = cur.u64()?;
                    let last_surfaced = cur.u64()?;
                    let used = cur.u8()? != 0;
                    let edge_n = cur.u32()? as usize;
                    let mut edges = Vec::with_capacity(edge_n);
                    for _ in 0..edge_n {
                        let t = cur.u32()?;
                        let w = cur.f32()?;
                        edges.push((EpisodeId(t), w));
                    }
                    slots.push(Some(Episode {
                        code: SparseCode::from_words(words),
                        payload,
                        strength,
                        count,
                        last_seen,
                        last_surfaced,
                        used,
                        edges,
                    }));
                }
                _ => return None,
            }
        }
        let mut index = BTreeMap::new();
        let mut inv = BitIndex::new(W * 64);
        for (i, slot) in slots.iter().enumerate() {
            if let Some(ep) = slot {
                index.insert(ep.payload, i as u32);
                inv.insert(i as u32, &ep.code);
            }
        }
        inv.shrink_to_fit();
        let dirty_n = cur.u32()? as usize;
        let mut dirty = BTreeSet::new();
        for _ in 0..dirty_n {
            dirty.insert(cur.u32()?);
        }
        let removed_n = cur.u32()? as usize;
        let mut removed_dirty = Vec::with_capacity(removed_n);
        for _ in 0..removed_n {
            removed_dirty.push(R::from_bytes(cur.take_bytes(R::SIZE)?)?);
        }
        Some(Self {
            slots,
            free,
            tick,
            recent: Vec::new(),
            scans: core::cell::Cell::new(0),
            index,
            dirty,
            removed_dirty,
            last_drain: BTreeSet::new(),
            last_drain_removed: Vec::new(),
            inv,
            scratch: RefCell::new(ScanScratch::default()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Graph = MemoryGraph<32>; // R defaults to u64
    type Code = SparseCode<32>;

    fn code(range: core::ops::Range<usize>) -> Code {
        Code::from_bits(&range.collect::<Vec<_>>())
    }

    #[test]
    fn write_strengthen_prune_lifecycle() {
        let mut g = Graph::new();
        g.advance_tick();
        let a = g.write(code(0..40), 1);
        g.advance_tick();
        let b = g.write(code(100..140), 2);

        for _ in 0..6 {
            g.advance_tick();
            g.strengthen(a);
        }
        let mut total_pruned = 0;
        let mut forgotten = Vec::new();
        for _ in 0..10 {
            let r = g.consolidate();
            total_pruned += r.pruned;
            forgotten.extend(r.forgotten);
        }
        assert!(g.get(a).is_some(), "strengthened episode must survive");
        assert!(g.get(b).is_none(), "unused episode must be forgotten");
        assert_eq!(total_pruned, 1);
        assert_eq!(
            forgotten,
            vec![2],
            "the forgotten payload handle is reported"
        );
    }

    #[test]
    fn temporal_binding_creates_edges() {
        let mut g = Graph::new();
        g.advance_tick();
        let a = g.write(code(0..40), 1);
        g.advance_tick();
        let b = g.write(code(100..140), 2);
        assert!(g.get(a).unwrap().edges.iter().any(|(t, _)| *t == b));
        assert!(g.get(b).unwrap().edges.iter().any(|(t, _)| *t == a));
    }

    #[test]
    fn spreading_activation_reaches_associates() {
        let mut g = Graph::new();
        // a and b written adjacently (linked); c far in time and code space.
        g.advance_tick();
        let a = g.write(code(0..40), 1);
        g.advance_tick();
        let b = g.write(code(100..140), 2);
        for _ in 0..10 {
            g.advance_tick();
        }
        let c = g.write(code(1000..1040), 3);

        // Cue matches only a; b should still activate through the edge.
        let hits = g.recall(&code(0..40), 2, 10);
        let ids: Vec<EpisodeId> = hits.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids[0], a);
        assert!(ids.contains(&b), "associate should surface via edge");
        assert!(!ids.contains(&c), "unrelated episode must stay silent");
    }

    #[test]
    fn hub_activation_stays_bounded_and_direct_match_wins() {
        // P0.4 regression: a hub fed by many weakly-activated spokes must not
        // go supernova and outrank a direct cue match. Without the per-hop
        // clamp the hub accumulates ~4+ activation here and wins.
        let mut g = Graph::new();
        g.advance_tick();
        let hub = g.write(code(0..40), 0);
        for i in 0..10usize {
            g.advance_tick();
            let spoke = g.write(
                Code::from_bits(
                    &(0..20)
                        .chain(200 + i * 40..220 + i * 40)
                        .collect::<Vec<_>>(),
                ),
                (i + 1) as u64,
            );
            g.link(spoke, hub, 1.0);
        }
        g.advance_tick();
        let cue = Code::from_bits(&(0..20).chain(1000..1020).collect::<Vec<_>>());
        let direct = g.write(cue.clone(), 99);

        let hits = g.recall(&cue, 2, 5);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].0, direct, "direct match must outrank the hub");
        // Final scores are activation(≤1) × strength_w(≤1) × recency(≤1).
        assert!(hits.iter().all(|&(_, s)| s.is_finite() && s <= 1.0 + 1e-6));
    }

    #[test]
    fn slot_reuse_after_prune() {
        let mut g = Graph::new();
        g.advance_tick();
        let a = g.write(code(0..40), 1);
        for _ in 0..10 {
            g.consolidate();
        }
        assert!(g.get(a).is_none());
        let b = g.write(code(50..90), 2);
        assert_eq!(a.0, b.0, "freed slot should be reused");
        assert_eq!(g.len(), 1);
    }

    // ---- inverted index ≡ full scan (property test) ----

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            // splitmix64
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n.max(1)
        }
        fn chance(&mut self, pct: u64) -> bool {
            self.below(100) < pct
        }
        fn f32(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    /// Codes from a mix of regimes: a tiny bit universe (dense ties, many
    /// equal jaccards), mid-size, the full 2048 bits at real density, and the
    /// empty code (j = 0 against everything).
    fn rand_code(rng: &mut Rng) -> Code {
        match rng.below(10) {
            0 => Code::empty(),
            1..=4 => {
                let k = 1 + rng.below(6) as usize;
                Code::from_bits(&(0..k).map(|_| rng.below(12) as usize).collect::<Vec<_>>())
            }
            5..=6 => {
                let k = 1 + rng.below(20) as usize;
                Code::from_bits(&(0..k).map(|_| rng.below(128) as usize).collect::<Vec<_>>())
            }
            _ => Code::from_bits(
                &(0..40)
                    .map(|_| rng.below(2048) as usize)
                    .collect::<Vec<_>>(),
            ),
        }
    }

    fn live_payloads(g: &Graph) -> Vec<u64> {
        g.iter().map(|(_, e)| e.payload).collect()
    }

    type ScanBits = (Vec<(u32, u32)>, Option<(u32, u32)>);

    fn bits_of(r: &(RecallHits, BestMatch)) -> ScanBits {
        (
            r.0.iter().map(|&(id, s)| (id.0, s.to_bits())).collect(),
            r.1.map(|(id, j)| (id.0, j.to_bits())),
        )
    }

    /// The index holds exactly the live codes' bits, nothing else.
    fn check_index(g: &Graph) {
        let mut fresh = BitIndex::new(32 * 64);
        for (id, ep) in g.iter() {
            fresh.insert(id.0, &ep.code);
        }
        for b in 0..32 * 64 {
            let mut have = g.inv.postings(b).to_vec();
            let mut want = fresh.postings(b).to_vec();
            have.sort_unstable();
            want.sort_unstable();
            assert_eq!(have, want, "postings of bit {b} out of step");
        }
        for (id, ep) in g.iter() {
            assert_eq!(g.inv.pop(id.0), ep.code.count_ones());
        }
        let s = g.scratch.borrow();
        assert!(s.counts.iter().all(|&c| c == 0), "scratch counts leaked");
        assert!(s.act.iter().all(|&a| a == 0.0), "scratch act leaked");
        assert!(s.active.iter().all(|&w| w == 0), "scratch active leaked");
        assert!(s.touched.is_empty() && s.sources.is_empty());
    }

    fn check_equivalent(g: &Graph, rng: &mut Rng, codes: &[Code]) {
        check_index(g);
        let mut cues: Vec<Code> = (0..4).map(|_| rand_code(rng)).collect();
        cues.push(Code::empty());
        for _ in 0..2 {
            if !codes.is_empty() {
                cues.push(codes[rng.below(codes.len() as u64) as usize].clone());
            }
        }
        for cue in &cues {
            let bm = g.best_match(cue).map(|(i, j)| (i.0, j.to_bits()));
            let bm_ref = g.best_match_scan(cue).map(|(i, j)| (i.0, j.to_bits()));
            assert_eq!(bm, bm_ref, "best_match diverged");
            for hops in 0..=3u8 {
                for top in [0usize, 1, 3, 8, 1000] {
                    let got = bits_of(&g.recall_with_best(cue, hops, top));
                    let want = bits_of(&g.recall_with_best_scan(cue, hops, top));
                    assert_eq!(
                        got, want,
                        "recall_with_best diverged (hops {hops}, top {top})"
                    );
                }
            }
        }
        check_index(g);
    }

    #[test]
    fn indexed_scan_is_bit_identical_to_full_scan() {
        for seed in 0..60u64 {
            let mut rng = Rng(seed);
            let mut g = Graph::new();
            let mut codes: Vec<Code> = Vec::new();
            let mut next_payload = 0u64;
            for step in 0..300 {
                let ids: Vec<EpisodeId> = g.iter().map(|(id, _)| id).collect();
                let pick = |rng: &mut Rng| ids[rng.below(ids.len() as u64) as usize];
                match rng.below(100) {
                    0..=34 => {
                        // Write — sometimes an exact duplicate of a stored code.
                        let c = if !codes.is_empty() && rng.chance(20) {
                            codes[rng.below(codes.len() as u64) as usize].clone()
                        } else {
                            rand_code(&mut rng)
                        };
                        codes.push(c.clone());
                        g.write(c, next_payload);
                        next_payload += 1;
                    }
                    35..=44 => {
                        // Burst of writes, so a later sleep mass-prunes
                        // past the bulk-removal threshold.
                        for _ in 0..(10 + rng.below(40)) {
                            let c = rand_code(&mut rng);
                            codes.push(c.clone());
                            g.write(c, next_payload);
                            next_payload += 1;
                        }
                    }
                    45..=54 if !ids.is_empty() => {
                        let id = pick(&mut rng);
                        g.strengthen(id);
                    }
                    55..=61 if !ids.is_empty() => {
                        // Links, including heavy ones (hub clamp territory).
                        let (a, b) = (pick(&mut rng), pick(&mut rng));
                        g.link(a, b, rng.f32() * 1.5);
                    }
                    62..=67 if !ids.is_empty() => {
                        // Merge / sync overwrite: an existing payload's code
                        // and state replaced from rows (or a new one created).
                        let c = rand_code(&mut rng);
                        codes.push(c.clone());
                        let payload = if rng.chance(70) {
                            g.get(pick(&mut rng)).unwrap().payload
                        } else {
                            next_payload += 1;
                            next_payload - 1
                        };
                        if rng.chance(50) {
                            g.drain_dirty(); // so the overwrite isn't shielded
                        }
                        g.upsert_state(c, payload, rng.f32() * 3.0, 1 + rng.below(4) as u32, false);
                        if rng.chance(30) {
                            let tgt = g.get(pick(&mut rng)).unwrap().payload;
                            g.set_edges_by_payload(payload, &[(tgt, rng.f32())]);
                        }
                    }
                    68..=72 if !ids.is_empty() => {
                        let p = g.get(pick(&mut rng)).unwrap().payload;
                        g.remove_by_payload(p);
                    }
                    73..=76 if !ids.is_empty() => {
                        // Sync reconcile: keep a random subset (sometimes
                        // dropping many → bulk path).
                        g.drain_dirty();
                        let keep_pct = if rng.chance(50) { 90 } else { 30 };
                        let keep: BTreeSet<u64> = live_payloads(&g)
                            .into_iter()
                            .filter(|_| rng.chance(keep_pct))
                            .collect();
                        g.retain_payloads(&keep);
                    }
                    77..=80 if !ids.is_empty() => {
                        let p = g.get(pick(&mut rng)).unwrap().payload;
                        g.penalize_by_payload(p, rng.f32());
                    }
                    81..=86 => {
                        for _ in 0..(1 + rng.below(4)) {
                            g.consolidate();
                        }
                    }
                    87..=90 => {
                        // Snapshot round trip: the index is rebuilt, not read.
                        let mut bytes = Vec::new();
                        g.write_bytes(&mut bytes);
                        let mut cur = crate::Cursor::new(&bytes);
                        g = Graph::read_bytes(&mut cur).expect("round trip");
                    }
                    91..=92 => {
                        g = g.clone();
                    }
                    93..=95 if !ids.is_empty() => {
                        g.mark_surfaced(pick(&mut rng));
                    }
                    _ => {
                        for _ in 0..(1 + rng.below(30)) {
                            g.advance_tick();
                        }
                    }
                }
                if step % 3 == 0 {
                    check_equivalent(&g, &mut rng, &codes);
                }
            }
            check_equivalent(&g, &mut rng, &codes);
        }
    }
}
