# engram

**The hippocampus your vector store is missing.** A small, fast Rust crate that
decides *what's worth remembering*, retrieves *by association*, and *forgets on
purpose* — the memory-gating organ that Retrieval-Augmented Generation skips.

> An *engram* is the physical trace a memory leaves behind.

You give engram embeddings you already computed (from any model) plus a small
handle to your content; it answers three questions cheaply and on its own:
*is this worth storing?* (or is it a repeat to strengthen), *should anything be
recalled right now?*, and *what can be forgotten?* No LLM calls, no GPU, no
background threads: tens of microseconds of plain arithmetic per observation
(a ~30 ns novelty check behind a ~20 µs sparsify), `no_std`-friendly, with
snapshot/restore. A companion tool,
[`engram-fit`](tools/engram-fit/README.md), fits it to a real embedding model.

Status: early, but measured end to end against real models (see
[Status](#status)). The API may still change.

---

## Why

Classic RAG is a filing cabinet: embed everything, cosine-search over all of it,
never forget, and offload the "should I even store this?" decision to an
expensive, unreliable LLM tool call. Brains don't work that way. They use two
memory systems (Complementary Learning Systems): a slow **neocortex** that learns
statistics over many exposures, and a fast, sparse **hippocampus** that binds
one-shot episodes and gates what gets consolidated — with novelty driving the
encoding.

engram is the hippocampus. It sits *in front of* your durable store (a vector DB,
the LLM's weights — the neocortex) and adds the two missing organs: a **write
gate** driven by novelty, and a **sleep** cycle that strengthens what recurs and
prunes what doesn't.

The core insight: **"when should I write this?" is a novelty-detection problem,
and novelty detection is a frequency check — nanoseconds, not a tool call.**

## What it does — three verbs

- **`observe(embedding, payload, demand)`** — gate a new observation. The gate
  decides `WriteNew`, `Link` (write + Hebbian edge to a related memory),
  `Strengthen` (near-duplicate — the write you *don't* do), or `Skip`. Every
  observation also *surfaces* associations for free ("this reminds me of…").
- **`recall_with_demand(embedding, demand, top)`** — retrieval is **triggered,
  not automatic**. It fires only when there's both *availability* (we actually
  have something relevant) and *demand* (the task actually wants memory), and
  even then every hit must clear an injection confidence gate (a direct-anchor
  check + a score floor) — an unknown query at 10k episodes injects *nothing*,
  not coincidental overlap. Over-retrieval poisons context; a read gate that
  injects nothing is a feature. (`recall(embedding, top)` is sugar for forced
  high demand. Recall today is the hippocampal associative path; cortex /
  vector-store fusion is pending.)
- **`sleep()`** — background consolidation: decay unused traces, prune below a
  floor, rebuild. A memory that only grows is a bug.

## Design in one breath

- **Encoding lives outside the crate.** The API boundary is a *dense vector* you
  already computed. engram owns no text encoder — it takes your embedding,
  sparsifies it with a **fly-hash** (random projection + k-winners-take-all,
  the fruit-fly olfactory circuit; Dasgupta, Stevens & Navlakha 2017), and works
  in sparse codes from there.
- **Novelty comes from per-bit counts, not a Bloom union.** A binary union
  saturates ("everything looks familiar") after a few hundred memories; a
  count-based estimate normalized against the background rate does not.
- **The read gate and write gate are one number read two ways.** novelty ≈ 1 →
  write, don't retrieve. novelty ≈ 0 → recognized → candidate to retrieve. That
  number is already computed on every write, so reading rides for free.
- **The hard problem is inhibition, not triggering.** Recall is continuous by
  default; the engineering is in the brakes — refractory periods, a per-tick
  surfacing budget, graded injection — all hanging off a single `inhibition`
  dial. Turn it down and you get divergent/associative "ideation mode"; off, a
  dream-like recombination during consolidation.
- **The only learning is two clamped scalars** — the write-novelty and
  read-availability thresholds — nudged by a reward signal (did a written memory
  later get used? did a retrieval help?). No backprop, no ML framework.

The full rationale and the settled design decisions live in
[`CLAUDE.md`](./CLAUDE.md) (the design brief, also read by coding agents).

## Quickstart

```rust
use engram::{Engram, InMemoryStore, PayloadStore};

// You bring the embeddings — engram never touches text→vector.
// (your existing pipeline / any MiniLM-class model)
fn embed(text: &str) -> Vec<f32> { /* your encoder */ todo!() }

let mut mem = Engram::new(InMemoryStore::<String>::new());

// WRITE — the gate decides whether this is worth remembering.
let obs = mem.observe(
    &embed("aeroponic misting runs every 15 minutes"),
    "aeroponic misting runs every 15 minutes".to_string(),
    0.0, // demand: this is a statement, not a query
);
// obs.decision  -> WriteNew | Link | Strengthen | Skip
// obs.surfaced  -> memories this observation was "reminded of" (free)

// A paraphrase later does NOT create a second copy:
mem.observe(
    &embed("the fogger cycles on a quarter-hour interval"),
    "the fogger cycles on a quarter-hour interval".to_string(),
    0.0,
); // -> Strengthen or Link, not a fresh episode

// READ — nothing comes back unless availability AND demand are both present.
// recall returns handles; resolve them through the store.
let hits = mem.recall(&embed("when does the misting run?"), 4);
for h in &hits {
    println!("{}", mem.store.get(*h).unwrap());
}
mem.feedback(&hits); // close the reward loop: which hits actually helped

// SLEEP — decay the trivial, keep what recurs (and evict what it forgets).
let report = mem.sleep();
println!("forgot {} traces", report.pruned);
```

Persist and reload the whole graph (structure + learned thresholds). Snapshots
are **fingerprinted with the embedding model** that produced the codes — restore
under a different encoder is refused rather than silently loading semantic
garbage:

```rust
let tag = engram::model_tag("all-MiniLM-L6-v2-int8"); // your encoder's identity
let bytes = mem.snapshot(tag);

// Snapshots carry engram structure and payload HANDLES, not payload contents —
// durability is your store's job. Reopen the same store state alongside the
// snapshot (trivial for a DB-backed store; for InMemoryStore, keep the store).
let restored = Engram::restore(&bytes, tag, /* your reopened store */ store).unwrap();

// A different model tag (or a different handle width) refuses to load:
assert!(Engram::<InMemoryStore<String>>::restore(
    &bytes, engram::model_tag("some-other-encoder"), InMemoryStore::new()
).is_none());
```

### Payloads: bring your own id

`observe` takes an **embedding** (the key engram searches on — sparsified, then
discarded) and a **payload handle** (the value engram stores and returns on
recall). They're separate because an embedding isn't invertible to text and
engram owns no encoder. What the handle *is* is up to the store you pick:

| Store | Handle (`Ref`) | Use it when |
|---|---|---|
| `InMemoryStore<P>` | `u64` index engram assigns | you want engram to hold the content (demos, ephemeral) |
| `IdentityStore<R>` | **your own id** (`u128`/UUID/rowid) | you already have a database; engram just carries the id |
| your `impl PayloadStore` | whatever you choose | SQLite/Postgres/… — see `store.rs` for the interface |

The `IdentityStore` path is the zero-duplication case — engram carries your
UUID and hands it back; you resolve id → content yourself:

```rust
use engram::{IdentityStore, Memory};

let mut mem: Memory<32, IdentityStore<u128>> = Memory::new(IdentityStore::new());
mem.observe(&embed(text), message.uuid.as_u128(), 0.0);
let ids = mem.recall(&embed(query), 4);   // -> Vec<u128>, straight back
// resolve `ids` against your own store
```

A durable backend implements the same three-method trait; `store.rs` carries a
complete **SQLite adapter sketch** (`put`=INSERT, `get`=SELECT, `remove`=DELETE)
with no SQL dependency pulled into engram itself.

## Fitting engram to your embedding model (`engram-fit`)

Contrastively-trained embedders (E5, EmbeddingGemma, Qwen3-Embedding, …) are
**anisotropic**: every vector shares a dominant common component. Pushed through
the fly-hash, that biases the k-WTA toward the same winning bits for every
input and collapses jaccard discrimination. The fix is subtracting a
**model-level** mean — computed once per model from generic text, frozen, and
versioned into the `model_tag`. Never a corpus-level mean: that has a
cold-start hole, drifts as the corpus grows (a silent space change), and
subtracts the user's topical signature, which is signal.

The workspace ships `tools/engram-fit` for exactly this. It talks to any
OpenAI-compatible `/v1/embeddings` endpoint (e.g. `mistralrs serve`), so the
artifact is computed by the **same engine binary** production uses:

```console
$ engram-fit center --endpoint http://localhost:8080 --model embeddinggemma-300m \
    --engine-id mistralrs-metal --out center.json
# -> doc + query centers (asymmetric models have per-role means), plus the
#    exact tag string/u64 for Memory::snapshot/restore

$ engram-fit health --endpoint http://localhost:8080 --model embeddinggemma-300m \
    --center center.json
# -> the one-number check: mean pairwise jaccard of unrelated codes,
#    raw vs centered, with a HEALTHY / MARGINAL / COLLAPSED verdict

$ engram-fit calibrate --endpoint http://localhost:8080 --model embeddinggemma-300m \
    --center center.json --pairs pairs.tsv
# -> jaccard percentiles for dup/related/unrelated pairs and suggested
#    DUP/LINK bands + recall_anchor for this model
```

At runtime, apply the artifact with `engram::center_in_place` (subtract +
renormalize) using the **doc center** for stored content and the **query
center** for recall cues — asymmetric `query:`/`passage:` prefixes map onto
engram's split: passage-embeddings feed `observe`/`import`, query-embeddings
feed `recall_with_demand`.

## Performance

Single-thread, measured on an Apple-silicon Mac (`cargo bench`):

| Operation | Target | Measured |
|---|---|---|
| Novelty check (2048-bit code) | < 100 ns | **30.7 ns** |
| SDR overlap (2048-bit, AND+POPCNT) | ~ns | **5.0 ns** |
| Jaccard (2048-bit) | ~ns | **10.3 ns** |
| Best-match scan of 10k episodes | < 150 µs | **104 µs** |
| Fly-hash sparsify (384-dim → 2048-bit) | < 25 µs | **21.2 µs** |

The nanosecond gate is the point: it runs on *every* observation, so you can
afford to gate everything and only escalate to expensive work (the episode scan,
the LLM) when the gate says "interesting."

## Status

Early but real. The crate is feature-complete through the read gate, the
inhibition dial, the reward loop (both the deliberate and the passive channel
are trainable), and model-tagged persistence. Behavior tests cover saturation,
refractory suppression, the full availability×demand 2×2, hub-activation
bounds, snapshot/model-tag refusal, and 10k-episode capacity regressions
(unknown queries inject nothing; old exact matches still win). Recall is the
hippocampal graph path only — cortex/vector-store fusion is pending.
`engram-fit` runs a six-week lifecycle evaluation on real embeddings from two
models (multilingual-E5-small and EmbeddingGemma-300m) against a cosine-RAG
baseline: same answer rate, and silence on never-discussed topics where RAG
always injects something. Not yet published to crates.io.

Add it as a git dependency for now:

```toml
[dependencies]
engram = { git = "https://github.com/sinkingsugar/engram" }
```

`no_std` builds (`--no-default-features`) are supported as a bonus target — the
hot path is fixed-size bit arrays and flat count arrays with no allocation on the
critical path.

## Non-goals

- Not a vector DB replacement — the durable store stays; we're the layer in front.
- Not a spiking-neural-network simulator — we keep the *properties* spikes were a
  proxy for (sparsity, cheap local writes, event-driven-ness) as plain arithmetic.
- Not a text encoder — embeddings come in from the caller.
- No GPU, no threads. A single core saturates the workload.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this work, as defined in the Apache-2.0 license, shall be dual
licensed as above, without any additional terms or conditions.
