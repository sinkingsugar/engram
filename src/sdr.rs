//! Sparse Distributed Representations: fixed-width bit-packed codes and the
//! fly-hash sparsifier (random expansion + k-winners-take-all — Dasgupta,
//! Stevens & Navlakha 2017). Settled decision 10: dense→sparse is projection
//! THEN k-WTA, never k-WTA over raw embedding dims, never sign-only SimHash.

use alloc::vec::Vec;

/// Fixed-width sparse distributed representation. `W` is the number of `u64`
/// lanes, so the code is `W * 64` bits wide.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SparseCode<const W: usize> {
    bits: [u64; W],
}

impl<const W: usize> SparseCode<W> {
    pub const BITS: usize = W * 64;

    pub const fn empty() -> Self {
        Self { bits: [0; W] }
    }

    pub fn from_bits(indices: &[usize]) -> Self {
        let mut code = Self::empty();
        for &i in indices {
            code.set(i);
        }
        code
    }

    #[inline]
    pub fn set(&mut self, i: usize) {
        debug_assert!(i < Self::BITS);
        self.bits[i / 64] |= 1u64 << (i % 64);
    }

    #[inline]
    pub fn get(&self, i: usize) -> bool {
        debug_assert!(i < Self::BITS);
        self.bits[i / 64] & (1u64 << (i % 64)) != 0
    }

    #[inline]
    pub fn count_ones(&self) -> u32 {
        self.bits.iter().map(|w| w.count_ones()).sum()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bits.iter().all(|&w| w == 0)
    }

    /// popcount(self & other) — autovectorizes to POPCNT.
    #[inline]
    pub fn overlap(&self, other: &Self) -> u32 {
        let mut acc = 0u32;
        for i in 0..W {
            acc += (self.bits[i] & other.bits[i]).count_ones();
        }
        acc
    }

    #[inline]
    pub fn jaccard(&self, other: &Self) -> f32 {
        let mut inter = 0u32;
        let mut union = 0u32;
        for i in 0..W {
            inter += (self.bits[i] & other.bits[i]).count_ones();
            union += (self.bits[i] | other.bits[i]).count_ones();
        }
        if union == 0 {
            0.0
        } else {
            inter as f32 / union as f32
        }
    }

    /// Iterate the indices of set bits, ascending.
    pub fn iter_ones(&self) -> Ones<'_, W> {
        Ones {
            bits: &self.bits,
            word: 0,
            cur: if W > 0 { self.bits[0] } else { 0 },
        }
    }

    pub fn as_words(&self) -> &[u64; W] {
        &self.bits
    }

    pub fn from_words(words: [u64; W]) -> Self {
        Self { bits: words }
    }

    /// One-shot fly-hash for callers/tests without a reusable [`FlyHasher`].
    /// Builds the projection taps every call — prefer `FlyHasher` on hot paths.
    pub fn from_dense_fly(dense: &[f32], k: usize) -> Self {
        let mut hasher = FlyHasher::new(dense.len(), k);
        hasher.sparsify(dense)
    }
}

pub struct Ones<'a, const W: usize> {
    bits: &'a [u64; W],
    word: usize,
    cur: u64,
}

impl<'a, const W: usize> Iterator for Ones<'a, W> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        while self.cur == 0 {
            self.word += 1;
            if self.word >= W {
                return None;
            }
            self.cur = self.bits[self.word];
        }
        let t = self.cur.trailing_zeros() as usize;
        self.cur &= self.cur - 1;
        Some(self.word * 64 + t)
    }
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Fly-hash sparsifier with precomputed projection taps.
///
/// Each of the `W*64` output units samples a fixed pseudo-random subset of
/// input dims with signs (deterministic seed — codes are stable across runs
/// and processes). `sparsify` accumulates the projections and keeps the top-k
/// as set bits: exact-k density, cosine-preserving.
#[derive(Clone, Debug)]
pub struct FlyHasher<const W: usize> {
    dim: usize,
    k: usize,
    samples: usize,
    /// (input index, sign) pairs, `samples` per output unit, laid out flat.
    taps: Vec<(u32, f32)>,
    /// Reused scratch (scores + selection order) — sparsify allocates nothing
    /// after the first call.
    scores: Vec<f32>,
    order: Vec<u32>,
}

impl<const W: usize> FlyHasher<W> {
    pub fn new(dim: usize, k: usize) -> Self {
        let bits = W * 64;
        let k = k.min(bits);
        let samples = if dim == 0 {
            0
        } else {
            dim.min(32.max(dim / 12))
        };
        let mut taps = Vec::with_capacity(bits * samples);
        for j in 0..bits {
            let mut state = (j as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
            for _ in 0..samples {
                let h = splitmix64(&mut state);
                // Lemire multiply-shift range reduction: no modulo on the hot path.
                let idx = ((h as u128 * dim as u128) >> 64) as u32;
                let sign = if h & 1 == 0 { 1.0 } else { -1.0 };
                taps.push((idx, sign));
            }
        }
        Self {
            dim,
            k,
            samples,
            taps,
            scores: Vec::new(),
            order: Vec::new(),
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn sparsify(&mut self, dense: &[f32]) -> SparseCode<W> {
        let bits = W * 64;
        if dense.len() != self.dim || self.dim == 0 || self.k == 0 {
            return SparseCode::empty();
        }
        let mut norm = 0.0f32;
        for &v in dense {
            norm += v * v;
        }
        if !norm.is_finite() || norm == 0.0 {
            return SparseCode::empty();
        }

        self.scores.clear();
        for j in 0..bits {
            let base = j * self.samples;
            let mut acc = 0.0f32;
            for &(idx, sign) in &self.taps[base..base + self.samples] {
                acc += sign * dense[idx as usize];
            }
            self.scores.push(acc);
        }

        self.order.clear();
        self.order.extend(0..bits as u32);
        let scores = &self.scores;
        self.order.select_nth_unstable_by(self.k - 1, |&a, &b| {
            scores[b as usize].total_cmp(&scores[a as usize])
        });
        let mut code = SparseCode::empty();
        for &i in &self.order[..self.k] {
            code.set(i as usize);
        }
        code
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Code = SparseCode<32>; // 2048 bits

    fn lcg_vec(seed: u64, dim: usize) -> Vec<f32> {
        let mut s = seed;
        (0..dim)
            .map(|_| {
                let h = splitmix64(&mut s);
                (h as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn density_is_exactly_k() {
        let v = lcg_vec(1, 384);
        let code = Code::from_dense_fly(&v, 40);
        assert_eq!(code.count_ones(), 40);
    }

    #[test]
    fn overlap_symmetry_and_jaccard_bounds() {
        let a = Code::from_dense_fly(&lcg_vec(2, 384), 40);
        let b = Code::from_dense_fly(&lcg_vec(3, 384), 40);
        assert_eq!(a.overlap(&b), b.overlap(&a));
        let j = a.jaccard(&b);
        assert!((0.0..=1.0).contains(&j));
        assert!((a.jaccard(&a) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn similar_vectors_share_bits_orthogonal_dont() {
        let base = lcg_vec(4, 384);
        let noise = lcg_vec(5, 384);
        let similar: Vec<f32> = base.iter().zip(&noise).map(|(b, n)| b + 0.15 * n).collect();
        let unrelated = lcg_vec(6, 384);

        let ca = Code::from_dense_fly(&base, 40);
        let cs = Code::from_dense_fly(&similar, 40);
        let cu = Code::from_dense_fly(&unrelated, 40);

        let j_sim = ca.jaccard(&cs);
        let j_unrel = ca.jaccard(&cu);
        assert!(j_sim > 0.5, "similar vectors should share bits: {j_sim}");
        assert!(j_unrel < 0.15, "unrelated should not: {j_unrel}");
        assert!(j_sim > j_unrel * 3.0);
    }

    #[test]
    fn degenerate_inputs_yield_empty() {
        assert!(Code::from_dense_fly(&[], 40).is_empty());
        assert!(Code::from_dense_fly(&[0.0; 384], 40).is_empty());
        assert!(Code::from_dense_fly(&[f32::NAN; 8], 4).is_empty());
    }

    #[test]
    fn iter_ones_roundtrip() {
        let idxs = [0usize, 63, 64, 100, 2047];
        let code = Code::from_bits(&idxs);
        let out: Vec<usize> = code.iter_ones().collect();
        assert_eq!(out, idxs);
    }

    #[test]
    fn deterministic_across_hashers() {
        let v = lcg_vec(7, 384);
        let a = FlyHasher::<32>::new(384, 40).sparsify(&v);
        let b = FlyHasher::<32>::new(384, 40).sparsify(&v);
        assert_eq!(a, b);
    }

    #[test]
    fn scratch_reuse_is_consistent() {
        // Same hasher, many calls: scratch reuse must not leak state between
        // sparsifications.
        let mut h = FlyHasher::<32>::new(384, 40);
        let v1 = lcg_vec(8, 384);
        let v2 = lcg_vec(9, 384);
        let first = h.sparsify(&v1);
        let _ = h.sparsify(&v2);
        assert_eq!(h.sparsify(&v1), first);
    }
}
