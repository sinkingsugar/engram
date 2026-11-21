# CLAUDE.md — `engram`: a low-power, self-consolidating memory gate

> **What this is:** a project brief *and* a settled set of design decisions for a
> Rust crate that gives an agent (or a sensor rig) a **hippocampus** — a cheap,
> always-on memory that decides *what's worth remembering*, retrieves by
> association, and *forgets* on purpose. Read the whole thing before writing code;
> the "Settled decisions" section exists so we don't re-litigate things that were
> already reasoned through.
>
> *Rev. 2026-07-06 — amended after design review: vectors-in API boundary (no
> in-crate text encoder), count-based novelty (binary Bloom union saturates),
> fly-hash sparsification (not raw-dim k-WTA), ESP32 demoted to bonus,
> persistence added. Then: generic fixed-size payload handle + DB-friendly store
> (owned `get`, `remove`-on-forget). See decisions 3, 4, 9, 10, 11, 12.*
>
> *Rev. 2026-07-07 — three-way implementation review applied:
> fused single scan per observe + availability pre-gate on the passive channel;
> real demand through recall (the 2×2 is live, Surface reachable); knowledge gap
> got a TTL + cue-relatedness; decide() = structure-from-jaccard,
> write-worthiness-from-bit-novelty (Link precedes the threshold Skip); one
> reward step per feedback call; embedding-model fingerprint in the snapshot
> header; recency capped to a ≤2× modulator in recall scoring.*
>
> *Rev. 2026-07-07 (3) — encoder-wiring recipe settled (asymmetric models +
> anisotropy). E5-style `query:`/`passage:` prefixes map onto engram's split:
> passage-embeddings → observe/import, query-embeddings → recall_with_demand;
> the passive channel is passage↔passage, which is the semantically right
> relation for "reminds me of". Anisotropy (the shared cone in all contrastive
> embedders — it biases k-WTA toward the same winners and collapses jaccard) is
> fixed by subtracting a **MODEL-LEVEL** mean: computed once per model+engine
> from generic text, frozen, versioned into the model_tag. NEVER corpus-level —
> cold-start hole, drift at t+100 (silent space change), and it subtracts the
> user's topical signature, which is signal. Tooling: `tools/engram-fit`
> (center / health / calibrate) against any OpenAI-compatible /v1/embeddings
> endpoint so the artifact comes from the production engine binary; runtime
> application via `engram::center_in_place`, doc center for stored content,
> query center for cues. Note: mistral.rs embedding archs are EmbeddingGemma +
> Qwen3-Embedding (no E5/BERT) — both asymmetric, same recipe. Jaccard bands
> should be calibrated per model (`engram-fit calibrate`); making DUP/LINK
> `Params` is on the wiring milestone.*
>
> *Rev. 2026-07-07 (2) — hardening pass, pre-Edge-Talk wiring:
> spreading activation clamped to [0,1] per hop (hubs can't go supernova);
> deliberate retrieval gained a two-stage injection confidence gate — direct
> ANCHOR (`recall_anchor`: best scan jaccard must relate to the cue, or it's a
> knowledge gap, not a retrieval) + score FLOOR (`recall_floor`) — so unknown
> high-demand queries at 10k episodes inject nothing; passive surfaced items
> are feedback-trainable (both channels share the reward loop, feedback targets
> the immediately previous interaction); sleep() invalidates pending feedback
> and feedback validates payload-slot identity (no stale id can mark a reused
> slot); a resolving write no longer immediately re-arms its own knowledge gap;
> `import()` added for gate-bypassing bulk backfill.*

---

## 0. The one-paragraph story

Classic RAG is a filing cabinet: embed everything, cosine-search over all of it,
never forget, and offload the "should I even store this?" decision to an
expensive, unreliable LLM tool call. Brains don't work that way. The insight
driving this project: **the "when to write" problem in RAG *is* a novelty-detection
problem, and novelty detection is a frequency/membership check** — which is
nanoseconds, not a tool call. So we build the missing organ: a **hippocampus**
(fast, sparse, one-shot, plastic) that sits in front of the **neocortex** (a big
slow vector store / the LLM). It takes an embedding the caller already computed,
sparsifies it, gates writes by novelty, retrieves by spreading activation, and
consolidates ("sleeps") in the background — strengthening what recurs, letting
the trivial fade.

---

## 1. Conceptual foundation (why the architecture looks like this)

**Complementary Learning Systems (McClelland & O'Reilly).** The brain uses two
memory systems because one can't do both jobs:

- **Neocortex** — slow, huge, overlapping codes, learns statistics over many
  exposures. → our **durable store** (vector DB / the LLM's frozen weights). Great
  at scale & generalization; catastrophic if you write to it fast.
- **Hippocampus** — fast, sparse, *pattern-separated*, one-shot episodic binding.
  → **this crate.** Great at "saw it once, remember it now"; bad at scale.
- **Sleep replay** — offline consolidation from hippocampus → cortex. → our
  background `sleep()` job.

RAG is all-neocortex with no hippocampus and no sleep. We're adding the two
missing organs.

**The write-gate = novelty detection.** A **Sparse Distributed Representation
(SDR)** derived from a *semantic embedding* sets bits by concept, so similar
inputs share bits (Kanerva's Sparse Distributed Memory, 1988). Novelty = how
*surprising* a code's active bits are given everything stored so far — a per-bit
frequency lookup, nanoseconds. High novelty → write. Low → strengthen the
existing trace instead of duplicating it.

**The neuroscience checks out:** the hippocampus runs a match/mismatch novelty
detector (CA1), and novelty drives a hippocampus→VTA dopamine loop that *gates
encoding*. We're implementing the dopaminergic encoding gate as arithmetic. The
dense→sparse step itself is the fly's mushroom-body circuit (Dasgupta, Stevens
& Navlakha 2017): random expansion + k-winners-take-all, a provably good LSH.

---

## 2. SETTLED DECISIONS — do not re-open without discussing with Gio

1. **NO literal spiking neural network.** On von-Neumann hardware a simulated LIF
   neuron costs you all the temporal-integration work and gives you *none* of the
   neuromorphic energy win (that only exists on event-driven silicon like Loihi).
   We keep the *properties* spikes were a proxy for — **sparsity, cheap local
   writes, event-driven-ness** — and implement them as boring fast arithmetic. If
   we ever target Loihi-class hardware, revisit. Not now.

2. **The gate is the cheap bouncer; the model is the club.** The novelty gate runs
   on *every* observation and must cost nanoseconds-to-sub-µs, so we can afford to
   gate everything and only escalate to expensive work (episode scan, full recall,
   the LLM) when the gate says "interesting."

3. **Encoding lives OUTSIDE the crate. The API boundary is a dense vector.**
   `observe()` takes an embedding the caller already computed (edge talk already
   has an embedding pipeline; MiniLM-class INT8 is single-digit ms on Apple
   silicon, which is nothing at human message rates). Engram owns NO text encoder.
   The previously-planned tiered encoder (SimHash tier-1 → semantic tier-2) is
   **rejected for language input**, in both escalation directions:
   - *Escalate-on-ambiguous* fails because lexically-novel text (the case that
     skips escalation) is exactly where paraphrases hide.
   - *Escalate-on-novel* fails because natural language is lexically diverse —
     nearly every chat/LLM message looks novel to n-gram hashing, so you embed
     everything anyway and the tier is pure overhead. Worse, trusting the
     "recognized" verdict *loses information*: "meeting moved to 3pm" vs "moved to
     4pm" share ~90% of n-grams and differ in the only part that matters; same for
     negations.
   A lexical pre-gate is sound only where surface form is honest about meaning
   (telemetry, logs, structured events). If such a stream ever shows up, the
   pre-gate is a *caller-side* filter in front of `observe()` — engram never
   knows about it.

4. **Keep the core small, portable, allocation-light.** Hot-path structures are
   fixed-size bit arrays and flat count arrays; the scan is branch-light and
   `POPCNT`-friendly. `no_std` compatibility and the ESP32 sensor node are a
   **bonus, not a driver** — don't contort the design for them, but don't
   gratuitously require an OS either.

5. **Forgetting is a feature.** Consolidation decays unused traces and prunes them.
   A memory that only grows is a bug.

6. **The only "learning" is a scalar (two, actually).** The adaptive novelty
   threshold nudges toward usefulness (was a written memory later
   retrieved-and-used = reward, or did it sit dead = waste). Reward-modulated
   online update, no backprop, no ML framework, no autodiff. Both thresholds are
   **hard-clamped** to sane bounds — the reward signal is sparse, delayed, and
   noisy, and an unbounded scalar on a noisy signal wanders. "Sat dead" has a
   definition: pruned at consolidation having never been surfaced or used.

7. **Don't retrieve on every query.** Retrieval is *triggered*, not automatic —
   gated by **availability × demand** (§6b). Over-retrieval poisons context: a read
   gate that injects *nothing* is a feature, not a failure. Reading free-rides on
   the write gate's frequency computation — the two gates share one novelty scalar
   and one reward channel (novelty and recognition are the same number, read in
   opposite directions).

8. **Recall is CONTINUOUS by default; the control problem is inhibition, not
   triggering.** The passive channel surfaces memory cue-driven every tick; an
   explicit query is just a synthetic high-demand observation (so `recall` is sugar
   over `observe` — one loop). The engineered hard part is *suppression* —
   refractory/habituation, a per-tick surfacing budget, graded injection — NOT
   "when to fire." Activation is cheap and emergent; brakes are the work. (See §6b.)

9. **Novelty comes from per-bit COUNTS, not a binary Bloom union.** A binary
   union only fills: with 2048-bit codes at ~40 active bits, the expected unset
   fraction after N diverse episodes is ≈ `e^(−N/51)` — the gate is blind after
   ~150 episodes ("everything is recognized"). Instead keep a flat `u32` count per
   bit (the bit-space is only 2048 wide — direct indexing, no sketch needed):
   novelty = mean of `bg / (bg + count[bit])` over the code's active bits, where
   `bg` is the *expected background count per bit* given total records. The
   background normalization is what prevents soft saturation at scale — a fresh
   code scores ~0.5 even after hundreds of diverse episodes (regression-tested).
   When the episode scan runs anyway (the ambiguous band), `1 − best_jaccard` is
   ground truth and overrides the estimate. Doubles as the read gate's
   availability signal. The count array **rebuilds from surviving
   episodes at consolidation** so forgetting actually forgets.

10. **Dense→sparse is fly-hash: random projection THEN k-WTA.** Not k-WTA over
    raw embedding dims (transformer dims aren't features individually), and not
    pure sign-of-projection SimHash (that yields ~50% bit density, which destroys
    the sparse-overlap statistics). Project to `BITS` pseudo-random directions
    (deterministic seed, sparse projection), keep the top-k as set bits. Exact k
    density, cosine-preserving, zero learned parameters.

11. **Persistence is part of the design, not an afterthought.** An episodic memory
    that dies on reboot is a cache. The graph (episodes, edges, counts, thresholds,
    tick) snapshots to a compact flat format and loads back; payload durability is
    the `PayloadStore`'s job.

12. **The payload boundary is a generic, fixed-size HANDLE — not text.** The
    embedding is the *key* (engram gates/searches on it, then discards it); the
    payload is the *value* (engram stores a handle to it and hands it back on
    recall). They're supplied separately because an embedding is neither invertible
    to text nor derivable by engram (it owns no encoder). The episode stores an
    `S::Ref` — a `PayloadRef` (`Copy`, `Eq`, fixed-width serialize): a store index,
    a DB rowid, or a 128-bit UUID. `recall` returns handles; the caller resolves
    content. `PayloadStore::get` returns *owned* values (a DB backend can't lend a
    borrow that outlives the row — this is why the old `-> Option<&str>` was
    in-memory-only), and `remove` lets forgetting reclaim durable storage (default
    **no-op**, so a store fronting caller-owned data is never made to delete it).
    Three impls ship: `InMemoryStore<P>` (engram assigns a `u64` index and holds
    content), `IdentityStore<R>` (the handle IS your id — zero content duplication,
    the edge-talk shape), and a documented **SQLite adapter sketch** (no SQL
    dependency in-crate — `put`=INSERT, `get`=SELECT, `remove`=DELETE against a
    table engram owns).

---

## 3. Performance targets (measured, single-thread, Apple-silicon class)

| Operation | Target | Measured (M-series Mac, 2026-07-06) |
|---|---|---|
| Novelty/availability check (count lookup over ~40 active bits) | **< 100 ns** | 30.7 ns |
| SDR overlap (2048-bit, AND+POPCNT) | ~ns, memory-bound | 5.0 ns (jaccard 10.3 ns) |
| Full overlap scan of 10k episodes | < 150 µs | 104 µs |
| Indexed best-match, 10k / 200k random episodes (inverted index, exact) | — | 11 µs / 0.47 ms (x86 desktop, loaded; full scan there: 300 µs / 20 ms) |
| Fly-hash sparsify (384-dim dense → 2048-bit SDR) | < 25 µs | 21.2 µs |
| Embedding (caller-side, MiniLM INT8, per sentence) | low ms — **outside the crate** | — |

If any hot-path op allocates or exceeds these by an order of magnitude, something
is wrong.

---

## 4. Crate layout

Workspace crate name: **`engram`** (the physical trace of a memory).

```
engram/
├─ Cargo.toml            # features: std (default). Core kept alloc-only-friendly.
├─ src/
│  ├─ lib.rs             # re-exports + the `Memory` orchestrator
│  ├─ sdr.rs             # SparseCode: bit-packed [u64; W], fly-hash, overlap, jaccard
│  ├─ gate.rs            # BitFreq (per-bit counts) + NoveltyGate + ReadGate, reward updates
│  ├─ graph.rs           # MemoryGraph: episodes, Hebbian edges, spreading activation, consolidate, snapshot/load
│  ├─ index.rs           # exact inverted index (bit → slots) + scan scratch; derived, rebuilt on restore, never serialized
│  └─ store.rs           # PayloadRef + PayloadStore (neocortex); InMemoryStore, IdentityStore; SQLite sketch (interface only)
├─ benches/              # criterion: gate, overlap scan, sparsify
└─ tests/                # gate behavior, saturation regression, capacity, consolidation, inhibition brakes
```

No `encoder.rs` — see decision 3. `bloom.rs` is gone; the count array lives in
`gate.rs` (decision 9).

---

## 5. Core types & hot-path signatures (implement to these shapes)

```rust
// ---------- sdr.rs ----------
/// Fixed-width sparse distributed representation. W = BITS/64.
pub struct SparseCode<const W: usize> { bits: [u64; W] }

impl<const W: usize> SparseCode<W> {
    pub const BITS: usize = W * 64;

    /// Fly-hash: project dense onto BITS pseudo-random sparse directions
    /// (deterministic seed), keep top-k as set bits. Exact-k density.
    pub fn from_dense_fly(dense: &[f32], k: usize) -> Self;

    /// popcount(self & other) — autovectorizes to POPCNT.
    #[inline] pub fn overlap(&self, other: &Self) -> u32;
    #[inline] pub fn jaccard(&self, other: &Self) -> f32;
}

// ---------- gate.rs ----------
/// Per-bit observation counts over the SDR bit-space. THE novelty/availability
/// source (decision 9). Flat array, direct-indexed; rebuilt from surviving
/// episodes at sleep (that rebuild IS the decay — no separate decay method).
pub struct BitFreq { counts: Vec<u32> /* len = BITS */ }
impl BitFreq {
    /// novelty ∈ [0,1]: mean of bg/(bg+count) over the code's active bits,
    /// bg = expected background count per bit (decision 9 — never saturates).
    pub fn novelty<const W: usize>(&self, code: &SparseCode<W>) -> f32;
    pub fn record<const W: usize>(&mut self, code: &SparseCode<W>);
}

pub enum Decision { WriteNew, Link(EpisodeId), Strengthen(EpisodeId), Skip }

pub struct NoveltyGate { threshold: f32, lr: f32 }  // threshold ADAPTIVE + clamped
impl NoveltyGate {
    /// Hot path. Two signals, two jobs: `best_jaccard` (episode scan) picks the
    /// STRUCTURE — Strengthen a near-dup, Link a relative (Link precedes the
    /// threshold Skip: a raised bar means "write less trivia," not "stop forming
    /// associations") — while `novelty` (BitFreq: global familiarity) gates
    /// write-worthiness for the unrelated remainder.
    pub fn decide(&self, novelty: f32, best_jaccard: f32, best: Option<EpisodeId>) -> Decision;
    /// Reward-modulated update: was a written memory later useful?
    pub fn reward(&mut self, used: bool);
}

// ---------- gate.rs (READ side — symmetric) ----------
pub enum RecallAction { Skip, Surface, Retrieve } // Surface = proactive/associative; Retrieve = full dual-path

pub struct ReadGate { threshold: f32, lr: f32 }    // adaptive + clamped, mirrors NoveltyGate
impl ReadGate {
    /// availability = 1 − novelty (same scalar, read backwards); demand supplied by the CALLER
    /// (it has the text/task context; engram sees vectors). A heuristic helper ships for convenience.
    pub fn decide(&self, availability: f32, demand: f32) -> RecallAction;
    pub fn reward(&mut self, helped: bool);
}

// ---------- store.rs ----------
pub trait PayloadRef: Copy + Eq { const SIZE: usize; /* fixed-width LE serialize */ }
pub trait PayloadStore {
    type Ref: PayloadRef;   // handle on the episode: store index / rowid / UUID
    type Payload;           // what the caller gets back
    fn put(&mut self, p: Self::Payload) -> Self::Ref;
    fn get(&self, r: Self::Ref) -> Option<Self::Payload>;  // OWNED — DB-friendly
    fn remove(&mut self, _r: Self::Ref) {}                 // reclaim on forget; default no-op
}
pub struct InMemoryStore<P = String>;   // engram assigns a u64 index, holds content
pub struct IdentityStore<R = u64>;      // the handle IS your id; get is identity, remove no-op
// A SqliteStore etc. implements PayloadStore in the CALLER's crate (see store.rs docs).

// ---------- graph.rs ----------
pub struct Episode<const W: usize, R> {
    code: SparseCode<W>,
    payload: R,           // fixed-size handle into the durable store (neocortex)
    strength: f32,
    count: u32,
    last_seen: u64,       // logical tick
    last_surfaced: u64,   // refractory: don't re-surface into context every tick
    edges: Vec<(EpisodeId, f32)>, // Hebbian links: co-occurrence + temporal order
}
pub struct MemoryGraph<const W: usize, R: PayloadRef = u64> { /* slot-stable episodes, tick */ }
impl<const W: usize, R: PayloadRef> MemoryGraph<W, R> {
    pub fn write(&mut self, code: SparseCode<W>, payload: R) -> EpisodeId;
    pub fn strengthen(&mut self, id: EpisodeId);
    pub fn link(&mut self, a: EpisodeId, b: EpisodeId, w: f32);
    /// Retrieval = seed from cue + spread activation 2–3 hops along edges.
    pub fn recall(&self, cue: &SparseCode<W>, hops: u8, top: usize) -> Vec<(EpisodeId, f32)>;
    /// THE fused scan: one jaccard pass yields both the best match (write gate)
    /// and the seed activations (passive channel). observe() calls this — never
    /// two scans per observation (decision 2).
    pub fn recall_with_best(&self, cue: &SparseCode<W>, hops: u8, top: usize)
        -> (RecallHits, BestMatch);
    /// Sleep: decay strengths + edges, prune below floor, report prunes for gate reward.
    pub fn consolidate(&mut self) -> ConsolidationReport;
}
```

### The orchestrator (`lib.rs`)

```rust
pub struct Memory<const W: usize, S: PayloadStore> {
    freq: BitFreq, gate: NoveltyGate, read_gate: ReadGate,
    graph: MemoryGraph<W, S::Ref>, store: S,
    inhibition: f32,   // §6b — ONE dial scaling all three brakes
}

impl<const W: usize, S: PayloadStore> Memory<W, S> {
    /// ONE observation: dense embedding in (computed by the caller), an opaque
    /// payload handle to store if the gate writes, demand for the read side.
    /// Returns the decision AND anything the passive channel surfaced.
    /// Cost discipline: 30ns bit-novelty first; then at most ONE fused scan
    /// (recall_with_best), and only when the gate is ambiguous or the passive
    /// pre-gate (availability ≥ floor) clears. Never two scans.
    pub fn observe(&mut self, dense: &[f32], payload: S::Payload, demand: f32) -> Observation<S::Ref>;
    /// GENERAL read path: caller demand flows through the full 2×2 —
    /// Skip/Suppress/Surface(top-1)/Retrieve, gap armed on high-demand misses.
    pub fn recall_with_demand(&mut self, dense: &[f32], demand: f32, top: usize) -> Vec<S::Ref>;
    /// Sugar: recall_with_demand with demand FORCED to 1.0 ("asked directly").
    pub fn recall(&mut self, dense: &[f32], top: usize) -> Vec<S::Ref>;
    /// Bulk backfill: write directly, bypassing the gate (no scan, no dedup;
    /// temporal binding still applies). Migration/import only — normal
    /// traffic goes through observe().
    pub fn import(&mut self, code: SparseCode<W>, payload: S::Payload) -> Option<EpisodeId>;
    /// Expose the decision so callers know *why* nothing came back.
    /// (&mut for the lazily-built hasher cache — divergence from the original
    /// &self sketch, accepted.)
    pub fn should_recall(&mut self, dense: &[f32], demand: f32) -> RecallAction;
    /// Background. Off the hot path. Closes the "sat dead" reward loop AND evicts
    /// pruned payloads via PayloadStore::remove.
    pub fn sleep(&mut self) -> ConsolidationReport<S::Ref>;
    /// Close the reward loop: caller reports which recalled handles actually
    /// helped. ONE reward step per call per gate, however many hits were used.
    pub fn feedback(&mut self, used: &[S::Ref]);
    /// §6b inhibition dial, 1.0 = focused … 0.0 = association flood.
    pub fn set_inhibition(&mut self, i: f32);
    /// Persistence (decision 11). `model_tag` fingerprints the embedding model
    /// (see fn model_tag): restore under a different encoder is REFUSED —
    /// the codes would be semantic garbage with intact structure. Params and
    /// gate lr/clamps reset to defaults on restore; only the two learned
    /// thresholds travel. Counts rebuild from survivors (= post-sleep state).
    pub fn snapshot(&self, model_tag: u64) -> Vec<u8>;
    pub fn restore(bytes: &[u8], model_tag: u64, store: S) -> Option<Self>;
}
```

---

## 6. The three verbs (behavioral spec)

**WRITE (`observe`)**
1. Fly-hash the caller's dense vector → `SparseCode`.
2. Novelty from `BitFreq` (O(active bits)). Very novel → decide **WriteNew**
   without even scanning episodes (cheap-reject, same job the Bloom had).
3. Ambiguous middle → scan for `best_jaccard`. Then:
   - `best_jaccard ≥ ~0.82` → **Strengthen** (near-duplicate; the write you DON'T do).
   - `~0.34 ≤ best_jaccard < 0.82` → **Link** (write + Hebbian edge to the match).
   - else → **WriteNew**.
   - Thresholds start fixed but the **novelty threshold is adaptive** (§7 reward).
4. On write: bind Hebbian edges to episodes active in the last few ticks
   (temporal + co-occurrence). Update `BitFreq`.

**READ (`recall`)** — dual path, fused:
- **Path A (cortex):** ANN/top-k over the durable store → precision.
  **Status: PENDING, not implemented** — needs a vector-capable `PayloadStore`;
  today's recall is Path B only. Don't claim fusion until it exists.
- **Path B (hippocampus):** fly-hash the query, seed graph, spread activation 2–3
  hops → associative, temporally-adjacent hits cosine search can't surface.
- Fuse, weighting Path B by edge strength × recency (recency is a modulator,
  ≤2× — an old exact match must beat a fresh coincidental overlap).

**CONSOLIDATE (`sleep`)** — background:
- Strengthen links that keep co-firing; **decay** the rest.
- Prune episodes below a strength floor (graceful forgetting). Prunes that were
  never surfaced/used feed `gate.reward(false)`.
- Decay + rebuild `BitFreq` from survivors (decision 9).
- Migrate stable, recurrent associations into the durable store's structure.

---

## 6b. The read gate — when to *pull* (symmetric to the write gate)

RAG retrieves on every query — the read-side equivalent of "write everything." It's
wrong and actively harmful: irrelevant retrieved memory *poisons the context* and
distracts the model. Retrieval must be **triggered**.

**The symmetry.** The read gate and the write gate are the SAME per-bit frequency
scalar, read in opposite directions:
- novelty ≈ 1 → nothing relevant stored → **novel** → write, *don't* retrieve.
- novelty ≈ 0 → relevant memory exists → **recognized** → candidate to retrieve.

Novelty and recognition are two ends of one number — and that number is *already
computed on every observation* for the write gate. Reading free-rides on writing's
arithmetic; the passive/associative "this reminds me of…" channel costs nothing
extra.

**The second axis reading has (and writing doesn't): availability vs. demand.**
- **Availability** = "do I have anything relevant?" — `1 − novelty`. A
  metacognitive *feeling-of-knowing* estimate, computed BEFORE the expensive
  spreading-activation recall (cheap-before-expensive, decision 2).
- **Demand** = "does the current task actually want memory?" — from the query/task,
  not the store. **Supplied by the caller** (it has the text; engram sees vectors).
  A cheap heuristic helper (anaphora / recall-verb / question detection) ships as
  a convenience function for callers that don't have anything better.

Gate on both → a 2×2. The off-diagonal cells are the point:

| | low demand | high demand |
|---|---|---|
| **low availability** | **Skip** | **Knowledge gap** — need it, don't have it → don't fabricate; flag for *eager write* when the answer arrives. The read gate tells the write gate to pay attention. The flag has a **TTL** (ticks) and remembers its **cue code**: only a later observation jaccard-related to the cue gets the eager write — the answer to the question we asked, not the next thing through the door. |
| **high availability** | **Suppress** — recognize it but don't need it; injecting would only poison context (RAG never does this) | **Retrieve** — the sweet spot |

**The synthesis, one line: activation is continuous; injection is gated.**
Background activation is cheap and always on (it rides the write path's single
scan); what reaches context is controlled by inhibition + refractory + budget +
demand.

**Two channels:**
1. **Passive / associative (proactive reminding).** Every observation already
   computes novelty for the write gate. If a cue's spreading activation crosses
   threshold, the memory *surfaces on its own* — being reminded, no query needed.
   Free.
2. **Active / deliberate (query-time).** On an explicit query, run
   availability × demand; fire the expensive dual-path recall ONLY in the Retrieve
   cell. Even then, **gate injection depth by match confidence** — one strong hit
   beats five mediocre ones. Concretely, two floors guard injection: a direct
   **anchor** (the scan's best jaccard must actually relate to the cue — the
   bit-frequency availability estimate can say "familiar bits" at capacity while
   no stored episode anchors them; that's a knowledge gap, not a retrieval) and a
   final **score floor** (activation × strength × recency, all ≤ 1 since
   activation clamps per hop — hubs can't ride over the floor).

**Automatic recall is the DEFAULT; explicit query is a special case.** The passive
channel makes recall *continuous* — every observation's novelty already decides
what surfaces, so memory bubbles up cue-driven with no query (involuntary,
Proustian recall). An explicit query is a synthetic observation with forced high
demand. So `recall()` is sugar over `observe()`; there is one loop, not two.

**This moves the hard problem from triggering to INHIBITING** — exactly where brains
put it (most of cortex is inhibitory; focus is suppression, not activation). Three
cheap mechanisms, without which "automatic recall" degrades into intrusive
thoughts / context flood (the mirror failure of the beauty):
- **Refractory / habituation.** Once an episode surfaces it's IN context, so its
  overlap stays high → it re-fires every tick → rumination loop. Suppress
  re-surfacing for a cooldown (`last_surfaced`), decaying back. Repetition
  suppression, straight from biology. One field, big effect.
- **Surfacing budget.** Working context is finite. Cap surfaced items per tick to
  top-K by activation; they *compete*. Attention as a scarce resource.
- **Graded intensity.** Passive surfacing is gentle (flag "available" / light
  inject); only the active Retrieve cell does full injection. The ambient channel
  must not drown the deliberate one.

**The inhibition dial (creativity, dreams — and yes, "mescaline mode" 🌀).** The
three brakes above are really one global **inhibition / entropy** parameter. This
isn't a gimmick — detuning it sweeps the whole phenomenological spectrum from one
knob, which is a *validation* the model captures something real (cf. the REBUS /
entropic-brain account of psychedelics: they relax precision, they don't add
signal):
- **default (high inhibition)** → focused, on-task recall.
- **loosened ~20%** → *divergent / ideation mode*: remote low-overlap associations
  cross threshold → analogy, metaphor, brainstorming. Genuinely useful — expose it.
- **brakes off during `consolidate()`** → *dream-like recombination*: Hebbian replay
  + low pattern separation = novel synthesis. This is where insight is made.
- **fully off** → association flood / ego-dissolution regime — mostly a curiosity,
  but a great stress test that the inhibition machinery actually holds.

Implementation: a single `inhibition: f32` that scales (surface threshold ↑,
budget ↓, refractory ↑, pattern-separation ↑) together. One dial, whole spectrum.

**Adaptive + reward (symmetric):** the read gate's threshold learns from the same
`feedback` signal — did retrieval help (used) or waste/distract (ignored)? Two
scalars now (write-novelty + read-availability), both drifting toward usefulness,
both fed by the one reward channel.

## 7. Reward loop (the only learning)

The caller reports which recalled payloads actually helped downstream
(`feedback`). For memories the gate chose to write:
- used later → `gate.reward(true)` nudges the threshold to keep writing similar.
- **sat dead** — pruned at consolidation, never surfaced, never used —
  → `gate.reward(false)` raises the bar (fired from `sleep()`, capped per cycle).

No backprop. Two scalars — the write-novelty threshold and the read-availability
threshold — both drifting toward usefulness, both **hard-clamped** (decision 6),
both fed by this one `feedback` channel. Each gate takes at most **one reward
step per feedback call** (and a capped few per sleep), however many hits were
used — the signal is sparse and gentle by design; a good retrieval session must
not slam a threshold to its clamp. The tell of a dumb gate (either gate) is a
constant threshold. Grading "was it used?" is the *caller's* job, from implicit
signals (did the injected payload survive into the answer / get quoted) — no
second LLM pass. `sleep()` is host-called (e.g. on idle); no threads in-crate.

---

## 8. Testing / correctness

- **SDR:** overlap symmetry; jaccard in [0,1]; fly-hash density is exactly k;
  nearby dense vectors → high jaccard, orthogonal → near-zero.
- **Saturation regression (the decision-9 test):** store hundreds of diverse
  codes; a brand-new random code must STILL score high novelty. A binary union
  fails this; the count array must not.
- **Capacity/interference:** store N patterns, verify recall accuracy degrades
  gracefully (not catastrophically) as N grows for given W. Chart the curve.
- **Gate behavior:** exact-dup → Strengthen; related (partial bit overlap) → Link;
  unrelated → WriteNew. Reward moves each threshold the right way and respects
  its clamps.
- **Consolidation:** unused traces fall below floor and get pruned; strong ones
  survive; `BitFreq` rebuilds consistently from survivors.
- **Inhibition brakes:** same cue on consecutive ticks surfaces once (refractory);
  surfacing per tick never exceeds budget; lowering inhibition widens what
  surfaces.
- **Persistence:** snapshot → restore → identical behavior.
- **Bench (criterion):** assert the §3 targets; fail CI if the gate regresses past
  a ceiling.

---

## 9. Build order (suggested)

1. `sdr.rs` + benches — fly-hash, overlap/jaccard: fast and correct first.
2. `gate.rs` — `BitFreq` counts + novelty + fixed thresholds + `Decision`.
3. `graph.rs` — write/strengthen/link + consolidate. Then spreading-activation recall.
4. `Memory` orchestrator + `InMemoryStore`. End-to-end `observe`/`recall`/`sleep`.
5. **Read gate:** availability × demand; proactive surfacing off the write path's
   novelty; the three brakes + inhibition dial. Prove over-retrieval drops —
   irrelevant injections on unrelated queries should go to ~0 (Suppress/Skip fire).
6. Reward loop (trains BOTH gates, incl. sat-dead from sleep).
7. Persistence: graph snapshot/restore.
8. Wire to a real embedding source (edge talk's pipeline) and run on real traffic.

---

## 10. Non-goals (for now)

- Not a vector DB replacement — the durable store stays; we're the layer in front.
- Not a spiking simulator (see decision 1).
- **Not a text encoder** — embeddings come in from the caller (decision 3).
- Not distributed/multi-threaded — single core saturates the workload.
- No GPU. If it needs a GPU it's in the wrong crate.
- ESP32/`no_std` is a bonus target, not a driver (decision 4).

---

*Origin: designed in conversation between Gio (Fragcolor) and Claude — CLS theory
+ "novelty-as-write-gate" + "drop the spikes, keep the shape." Revised 2026-07-06
after the design review that killed the binary union (saturation math), the
in-crate encoder (language lies in both lexical directions), and raw-dim k-WTA
(fly-hash instead). The whole point is a memory that keeps learning at inference
time, for nanoseconds, on hardware small enough to sit next to the app.*
