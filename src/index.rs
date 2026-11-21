//! Exact inverted index over the SDR bit-space: for every bit, the slots whose
//! code has that bit set (postings), plus each slot's popcount.
//!
//! It makes the fused scan touch only episodes that share at least one bit
//! with the cue, and compute their jaccard from the overlap COUNT instead of
//! re-reading the 256-byte code: `inter / (|cue| + |ep| − inter)` is the same
//! pair of integers `SparseCode::jaccard` divides, so the f32 is bit-identical.
//! Episodes sharing no bit have jaccard exactly `+0.0` (also when either code
//! is empty), which `MemoryGraph` uses to keep the full scan's `BestMatch`
//! semantics without visiting them.
//!
//! Derived state: never serialized, rebuilt from the slots on restore. Posting
//! order is irrelevant — every consumer is order-independent (best = max
//! jaccard, ties to the lowest slot; seeds are sorted by slot before use).
//!
//! Memory: 4 bytes per set bit per live episode (≈160 B/episode at k = 40)
//! plus a 4-byte popcount per slot, plus `W*64` empty `Vec` headers.

use alloc::vec;
use alloc::vec::Vec;

use crate::sdr::SparseCode;

/// Removing more episodes than this in one pass switches from per-bit
/// `swap_remove` (O(k · list length) each) to one `retain` over every list.
pub(crate) const BULK_REMOVE: usize = 32;

#[derive(Clone, Debug, Default)]
pub(crate) struct BitIndex {
    postings: Vec<Vec<u32>>,
    pops: Vec<u32>,
}

impl BitIndex {
    pub(crate) fn new(bits: usize) -> Self {
        Self {
            postings: vec![Vec::new(); bits],
            pops: Vec::new(),
        }
    }

    #[inline]
    pub(crate) fn postings(&self, bit: usize) -> &[u32] {
        &self.postings[bit]
    }

    #[inline]
    pub(crate) fn pop(&self, slot: u32) -> u32 {
        self.pops[slot as usize]
    }

    pub(crate) fn insert<const W: usize>(&mut self, slot: u32, code: &SparseCode<W>) {
        let s = slot as usize;
        if self.pops.len() <= s {
            self.pops.resize(s + 1, 0);
        }
        let mut pop = 0;
        for b in code.iter_ones() {
            self.postings[b].push(slot);
            pop += 1;
        }
        self.pops[s] = pop;
    }

    pub(crate) fn remove<const W: usize>(&mut self, slot: u32, code: &SparseCode<W>) {
        for b in code.iter_ones() {
            let list = &mut self.postings[b];
            if let Some(p) = list.iter().position(|&s| s == slot) {
                list.swap_remove(p);
            }
        }
        if let Some(p) = self.pops.get_mut(slot as usize) {
            *p = 0;
        }
    }

    /// Bulk removal: drop every posting whose slot is marked `gone`.
    /// Idempotent with slots already removed one by one.
    pub(crate) fn retain_live(&mut self, gone: &[bool]) {
        for list in &mut self.postings {
            list.retain(|&s| !gone[s as usize]);
        }
        for (p, &g) in self.pops.iter_mut().zip(gone) {
            if g {
                *p = 0;
            }
        }
    }

    /// Drop growth slack (after a bulk rebuild, e.g. restore).
    pub(crate) fn shrink_to_fit(&mut self) {
        for list in &mut self.postings {
            list.shrink_to_fit();
        }
        self.pops.shrink_to_fit();
    }

    /// Heap bytes held by the index (postings capacity + popcounts + list
    /// headers) — for the iOS memory budget.
    pub(crate) fn heap_bytes(&self) -> usize {
        self.postings.capacity() * core::mem::size_of::<Vec<u32>>()
            + self
                .postings
                .iter()
                .map(|l| l.capacity() * 4)
                .sum::<usize>()
            + self.pops.capacity() * 4
    }
}

/// Per-call working memory for the indexed scan, kept between calls so the
/// hot path doesn't allocate/zero O(slots) every observation. Invariant
/// between calls: `counts` and `act` are all zero, the lists are empty.
#[derive(Clone, Debug, Default)]
pub(crate) struct ScanScratch {
    /// Overlap count per slot during phase 1.
    pub(crate) counts: Vec<u32>,
    pub(crate) act: Vec<f32>,
    pub(crate) touched: Vec<u32>,
    /// Bitmap of slots with activation (seeds + spread targets): iterated
    /// word by word it yields ascending slot order without sorting.
    pub(crate) active: Vec<u64>,
    pub(crate) sources: Vec<(u32, f32)>,
}

impl ScanScratch {
    pub(crate) fn ensure(&mut self, n: usize) {
        if self.counts.len() < n {
            self.counts.resize(n, 0);
            self.act.resize(n, 0.0);
            self.active.resize(n.div_ceil(64), 0);
        }
    }
}
