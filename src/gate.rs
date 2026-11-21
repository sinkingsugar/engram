//! The two gates and the shared novelty source.
//!
//! `BitFreq` holds per-bit observation counts over the SDR bit-space (settled
//! decision 9 — a binary Bloom union saturates after ~150 diverse episodes;
//! counts, normalized against the expected background frequency, never do).
//! `NoveltyGate` decides writes; `ReadGate` decides retrieval. They read the
//! same scalar in opposite directions and learn from the same reward channel.

use alloc::vec;
use alloc::vec::Vec;

use crate::graph::EpisodeId;
use crate::sdr::SparseCode;

/// Per-bit observation counts over the SDR bit-space. THE novelty /
/// availability source. Flat, direct-indexed (the bit-space is small — no
/// sketch needed); rebuilt from surviving episodes at sleep, so forgetting
/// actually forgets.
#[derive(Clone, Debug)]
pub struct BitFreq {
    counts: Vec<u32>,
    records: u64,
}

impl BitFreq {
    pub fn new(bits: usize) -> Self {
        Self {
            counts: vec![0; bits],
            records: 0,
        }
    }

    pub fn bits(&self) -> usize {
        self.counts.len()
    }

    pub fn records(&self) -> u64 {
        self.records
    }

    /// Novelty ∈ [0,1]: how surprising this code's active bits are given what
    /// is stored, normalized against the expected background count per bit.
    /// 1.0 = never seen anything like it; → 0 only for genuinely over-familiar
    /// codes, regardless of how full the memory is (no saturation).
    ///
    /// Hot path: `active_bits` count-array reads, no allocation.
    pub fn novelty<const W: usize>(&self, code: &SparseCode<W>) -> f32 {
        let active = code.count_ones();
        if active == 0 {
            return 0.0;
        }
        // Expected count per bit if all records used bits uniformly.
        let bg = 1.0 + (self.records as f32) * (active as f32) / (self.counts.len() as f32);
        let mut sum = 0.0f32;
        for b in code.iter_ones() {
            sum += bg / (bg + self.counts[b] as f32);
        }
        sum / active as f32
    }

    pub fn record<const W: usize>(&mut self, code: &SparseCode<W>) {
        self.record_n(code, 1);
    }

    pub fn record_n<const W: usize>(&mut self, code: &SparseCode<W>, n: u32) {
        for b in code.iter_ones() {
            self.counts[b] = self.counts[b].saturating_add(n);
        }
        self.records += n as u64;
    }

    pub fn reset(&mut self) {
        self.counts.fill(0);
        self.records = 0;
    }
}

/// Default jaccard at or above which an observation is a near-duplicate →
/// Strengthen, don't write. With exact-k codes, `J = overlap / (2k − overlap)`;
/// for k=40 this is ≈ 36/40 bits shared. These defaults come from synthetic
/// code geometry — **calibrate per model** with `engram-fit calibrate` and set
/// the tuned values on [`NoveltyGate`] (they're fields, not laws).
pub const DUP_JACCARD: f32 = 0.82;
/// Default jaccard at or above which (but below dup) → related → write +
/// Hebbian link. For k=40 this is ≈ 20/40 bits shared.
pub const LINK_JACCARD: f32 = 0.34;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    WriteNew,
    Link(EpisodeId),
    Strengthen(EpisodeId),
    Skip,
}

/// The write gate. `threshold` is one of the crate's two learned scalars —
/// reward-modulated, hard-clamped (settled decision 6).
#[derive(Clone, Debug)]
pub struct NoveltyGate {
    pub threshold: f32,
    lr: f32,
    min: f32,
    max: f32,
    /// Strengthen band floor. Default [`DUP_JACCARD`]; tune per model.
    pub dup_jaccard: f32,
    /// Link band floor. Default [`LINK_JACCARD`]; tune per model.
    pub link_jaccard: f32,
}

impl Default for NoveltyGate {
    fn default() -> Self {
        Self {
            threshold: 0.15,
            lr: 0.05,
            min: 0.05,
            max: 0.6,
            dup_jaccard: DUP_JACCARD,
            link_jaccard: LINK_JACCARD,
        }
    }
}

impl NoveltyGate {
    /// Hot path. Two signals, two jobs: `best_jaccard` (from the episode scan)
    /// picks the *structure* — Strengthen a near-duplicate, Link a relative —
    /// and `novelty` (the bit-frequency scalar: global familiarity) gates
    /// *write-worthiness* for the unrelated remainder. The Link check runs
    /// before the threshold Skip on purpose: a raised bar means "write less
    /// trivia," not "stop forming associations" — and keeping the threshold on
    /// the global scalar (not `1 − jaccard`, which never dips below ~0.18 for
    /// unmatched codes) keeps the learned threshold meaningful across its
    /// whole clamp range.
    pub fn decide(&self, novelty: f32, best_jaccard: f32, best: Option<EpisodeId>) -> Decision {
        if let Some(b) = best {
            if best_jaccard >= self.dup_jaccard {
                return Decision::Strengthen(b);
            }
            if best_jaccard >= self.link_jaccard {
                return Decision::Link(b);
            }
        }
        if novelty < self.threshold {
            return Decision::Skip;
        }
        Decision::WriteNew
    }

    /// Reward-modulated update: written memory later used → lower the bar
    /// (keep writing similar); sat dead → raise it. Clamped by construction.
    pub fn reward(&mut self, used: bool) {
        let target = if used { self.min } else { self.max };
        self.threshold += self.lr * (target - self.threshold);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecallAction {
    /// Nothing comes back: low demand, or knowledge gap (caller sees the flag).
    Skip,
    /// Proactive/associative: gentle, top-1 style injection.
    Surface,
    /// Full dual-path recall.
    Retrieve,
}

/// The read gate — mirrors the write gate. `availability = 1 − novelty`
/// (same scalar, read backwards); `demand` is supplied by the CALLER (it has
/// the text/task context; engram sees vectors).
#[derive(Clone, Debug)]
pub struct ReadGate {
    pub threshold: f32,
    lr: f32,
    min: f32,
    max: f32,
    /// Below this demand nothing surfaces deliberately, no matter how
    /// available — the Suppress cell of the 2×2.
    pub demand_floor: f32,
    /// Demand at or above this (with availability) → full Retrieve; the band
    /// between `demand_floor` and this → gentle Surface (top-1 injection).
    pub retrieve_demand: f32,
}

impl Default for ReadGate {
    fn default() -> Self {
        Self {
            threshold: 0.35,
            lr: 0.05,
            min: 0.1,
            max: 0.8,
            demand_floor: 0.25,
            retrieve_demand: 0.6,
        }
    }
}

impl ReadGate {
    pub fn decide(&self, availability: f32, demand: f32) -> RecallAction {
        if demand < self.demand_floor {
            return RecallAction::Skip; // Skip + Suppress cells
        }
        if availability >= self.threshold {
            if demand >= self.retrieve_demand {
                RecallAction::Retrieve
            } else {
                RecallAction::Surface
            }
        } else {
            RecallAction::Skip // knowledge-gap cell; orchestrator flags it
        }
    }

    /// Retrieval helped → retrieve more easily; wasted/distracted → raise the bar.
    pub fn reward(&mut self, helped: bool) {
        let target = if helped { self.min } else { self.max };
        self.threshold += self.lr * (target - self.threshold);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Code = SparseCode<32>;

    #[test]
    fn bitfreq_cold_start_is_maximally_novel() {
        let freq = BitFreq::new(2048);
        let code = Code::from_bits(&(0..40).collect::<Vec<_>>());
        assert!((freq.novelty(&code) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bitfreq_repetition_lowers_novelty() {
        let mut freq = BitFreq::new(2048);
        let code = Code::from_bits(&(0..40).collect::<Vec<_>>());
        let n0 = freq.novelty(&code);
        for _ in 0..5 {
            freq.record(&code);
        }
        let n5 = freq.novelty(&code);
        assert!(n5 < n0 * 0.4, "novelty should drop with repetition: {n5}");
    }

    #[test]
    fn bitfreq_does_not_saturate() {
        // The decision-9 regression at unit level: after many DIVERSE codes,
        // a fresh random code must still score mid-to-high novelty.
        let mut freq = BitFreq::new(2048);
        let mut s = 42u64;
        let mut rand_code = || {
            let mut bits = Vec::with_capacity(40);
            for _ in 0..40 {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                bits.push((s >> 33) as usize % 2048);
            }
            Code::from_bits(&bits)
        };
        for _ in 0..500 {
            let c = rand_code();
            freq.record(&c);
        }
        let fresh = rand_code();
        let n = freq.novelty(&fresh);
        assert!(n > 0.35, "count-based novelty saturated: {n}");
    }

    #[test]
    fn write_gate_bands() {
        let gate = NoveltyGate::default();
        let id = EpisodeId(7);
        assert_eq!(gate.decide(0.1, 0.9, Some(id)), Decision::Strengthen(id));
        assert_eq!(gate.decide(0.5, 0.5, Some(id)), Decision::Link(id));
        assert_eq!(gate.decide(0.9, 0.1, Some(id)), Decision::WriteNew);
        assert_eq!(gate.decide(0.9, 0.0, None), Decision::WriteNew);
        assert_eq!(gate.decide(0.05, 0.2, Some(id)), Decision::Skip);
    }

    #[test]
    fn link_band_survives_a_maxed_threshold() {
        // "Raise the bar" means write less trivia, NOT stop forming
        // associations: with the threshold pushed to its clamp, a Link-band
        // overlap must still Link (and a dup must still Strengthen).
        let mut gate = NoveltyGate::default();
        for _ in 0..1000 {
            gate.reward(false);
        }
        let id = EpisodeId(3);
        assert_eq!(gate.decide(0.2, 0.5, Some(id)), Decision::Link(id));
        assert_eq!(gate.decide(0.2, 0.9, Some(id)), Decision::Strengthen(id));
        // ...while over-familiar unrelated content is what gets skipped.
        assert_eq!(gate.decide(0.2, 0.1, Some(id)), Decision::Skip);
    }

    #[test]
    fn rewards_move_thresholds_and_respect_clamps() {
        let mut gate = NoveltyGate::default();
        let t0 = gate.threshold;
        gate.reward(false);
        assert!(gate.threshold > t0);
        for _ in 0..1000 {
            gate.reward(false);
        }
        assert!(gate.threshold <= 0.6 + 1e-4);
        for _ in 0..1000 {
            gate.reward(true);
        }
        assert!(gate.threshold >= 0.05 - 1e-4);

        let mut rg = ReadGate::default();
        for _ in 0..1000 {
            rg.reward(false);
        }
        assert!(rg.threshold <= 0.8 + 1e-4);
    }

    #[test]
    fn read_gate_2x2() {
        let rg = ReadGate::default();
        // low demand suppresses even high availability
        assert_eq!(rg.decide(0.9, 0.1), RecallAction::Skip);
        // high demand + high availability retrieves
        assert_eq!(rg.decide(0.9, 0.9), RecallAction::Retrieve);
        // high demand + low availability = knowledge gap → Skip (flagged upstream)
        assert_eq!(rg.decide(0.1, 0.9), RecallAction::Skip);
        // moderate demand + availability surfaces gently
        assert_eq!(rg.decide(0.9, 0.4), RecallAction::Surface);
    }
}
