//! §3 performance targets:
//!   novelty check (~40 count reads)      < 100 ns
//!   SDR overlap (2048-bit)               ~ns
//!   best-match scan of 10k episodes      < 150 µs
//!   fly-hash sparsify (384 → 2048)       < 25 µs

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use engram::{BitFreq, FlyHasher, MemoryGraph, SparseCode};

type Code = SparseCode<32>;

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

fn bench_novelty(c: &mut Criterion) {
    let mut rng = Lcg(1);
    let mut freq = BitFreq::new(2048);
    for _ in 0..1000 {
        let code = rng.code();
        freq.record(&code);
    }
    let probe = rng.code();
    c.bench_function("novelty_check_2048", |b| {
        b.iter(|| black_box(freq.novelty(black_box(&probe))))
    });
}

fn bench_overlap(c: &mut Criterion) {
    let mut rng = Lcg(2);
    let a = rng.code();
    let bcode = rng.code();
    c.bench_function("overlap_2048", |b| {
        b.iter(|| black_box(black_box(&a).overlap(black_box(&bcode))))
    });
    c.bench_function("jaccard_2048", |b| {
        b.iter(|| black_box(black_box(&a).jaccard(black_box(&bcode))))
    });
}

fn bench_scan(c: &mut Criterion) {
    let mut rng = Lcg(3);
    let mut graph: MemoryGraph<32> = MemoryGraph::new();
    for i in 0..10_000 {
        graph.advance_tick();
        graph.write(rng.code(), i as u64);
    }
    let probe = rng.code();
    c.bench_function("best_match_scan_10k", |b| {
        b.iter(|| black_box(graph.best_match(black_box(&probe))))
    });
}

/// The per-observation scan at scale: `recall_with_best` (the fused scan
/// `observe` and `recall` pay) and `best_match`, at 10k / 50k / 200k episodes.
/// Two workloads: uniform random codes (40 random bits of 2048 — the
/// worst-case-flat bit distribution) and clustered codes (fly-hashed dense
/// vectors around 64 centroids — skewed bit popularity, closer to real
/// embeddings, where a cue genuinely resembles part of the store).
fn bench_scan_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("scan_scale");
    group.sample_size(20);
    for &n in &[10_000usize, 50_000, 200_000] {
        // Uniform.
        let mut rng = Lcg(5);
        let mut graph: MemoryGraph<32> = MemoryGraph::new();
        for i in 0..n {
            graph.advance_tick();
            graph.write(rng.code(), i as u64);
        }
        let probe = rng.code();
        group.bench_function(format!("recall_with_best_uniform_{}k", n / 1000), |b| {
            b.iter(|| black_box(graph.recall_with_best(black_box(&probe), 2, 8)))
        });
        group.bench_function(format!("best_match_uniform_{}k", n / 1000), |b| {
            b.iter(|| black_box(graph.best_match(black_box(&probe))))
        });
        // The pre-index full scan (kept as the reference implementation),
        // measured in the same run for a like-for-like before/after.
        group.bench_function(
            format!("recall_with_best_uniform_{}k_fullscan", n / 1000),
            |b| b.iter(|| black_box(graph.recall_with_best_scan(black_box(&probe), 2, 8))),
        );
        eprintln!(
            "uniform {}k: index heap {} B ({} B/episode)",
            n / 1000,
            graph.index_heap_bytes(),
            graph.index_heap_bytes() / n
        );
        drop(graph);

        // Clustered.
        let dim = 64;
        let mut hasher: FlyHasher<32> = FlyHasher::new(dim, 40);
        let centroids: Vec<Vec<f32>> = (0..64).map(|_| rng.dense(dim)).collect();
        let mut clustered = |rng: &mut Lcg| {
            let c = &centroids[(rng.next() % 64) as usize];
            let noise = rng.dense(dim);
            let v: Vec<f32> = c.iter().zip(&noise).map(|(a, b)| a + 0.6 * b).collect();
            hasher.sparsify(&v)
        };
        let mut graph: MemoryGraph<32> = MemoryGraph::new();
        for i in 0..n {
            graph.advance_tick();
            let code = clustered(&mut rng);
            graph.write(code, i as u64);
        }
        let probe = clustered(&mut rng);
        group.bench_function(format!("recall_with_best_clustered_{}k", n / 1000), |b| {
            b.iter(|| black_box(graph.recall_with_best(black_box(&probe), 2, 8)))
        });
        group.bench_function(format!("best_match_clustered_{}k", n / 1000), |b| {
            b.iter(|| black_box(graph.best_match(black_box(&probe))))
        });
        // The pre-index full scan (kept as the reference implementation),
        // measured in the same run for a like-for-like before/after.
        group.bench_function(
            format!("recall_with_best_clustered_{}k_fullscan", n / 1000),
            |b| b.iter(|| black_box(graph.recall_with_best_scan(black_box(&probe), 2, 8))),
        );
        eprintln!(
            "clustered {}k: index heap {} B ({} B/episode)",
            n / 1000,
            graph.index_heap_bytes(),
            graph.index_heap_bytes() / n
        );
    }
    group.finish();
}

fn bench_sparsify(c: &mut Criterion) {
    let mut rng = Lcg(4);
    let mut hasher: FlyHasher<32> = FlyHasher::new(384, 40);
    let dense = rng.dense(384);
    c.bench_function("fly_sparsify_384_to_2048", |b| {
        b.iter(|| black_box(hasher.sparsify(black_box(&dense))))
    });
}

criterion_group!(
    benches,
    bench_novelty,
    bench_overlap,
    bench_scan,
    bench_scan_scale,
    bench_sparsify
);
criterion_main!(benches);
