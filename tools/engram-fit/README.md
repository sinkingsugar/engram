# engram-fit — runbook

Fitting engram to an embedding model, end to end. Written to be followed by a
human or a Claude instance verbatim; every step was executed for real against
`multilingual-e5-small-q8_0.gguf` on 2026-07-07 and against
`embeddinggemma-300M-qat-Q4_0.gguf` on 2026-07-08 (see "Second fitted model"
below), and the outputs live in this repo as test fixtures.

## What "fitting" means

Engram consumes dense embeddings and gates on fly-hash jaccard. Two things
about any real encoder must be measured, not assumed:

1. **Anisotropy.** Contrastive embedders put all vectors in a shared cone;
   uncorrected, the k-WTA picks the same winning bits for every input and
   unrelated content collides into the Link band. Fix: subtract a frozen
   **model-level** mean ("center"). Never a corpus-level mean (cold-start hole,
   drifts at t+100, subtracts the user's topical signature).
   Measured on mE5-small-q8: raw unrelated mean jaccard **0.251** (collapsed,
   p95 above the Link band) → centered **0.012** (healthy).
2. **Band geometry.** The default DUP/LINK bands come from synthetic code
   geometry; real paraphrases land much lower (centered dup mean ≈ 0.15–0.20,
   cross-lingual lower). Bands and `k` must be tuned from labeled pairs, with
   **asymmetric loss**: a false Strengthen silently drops a payload (costly);
   a missed Strengthen becomes a Link (harmless).

## Prerequisites

- An engine serving OpenAI-compatible `/v1/embeddings` for your model.
  For GGUF encoder models that's llama.cpp (`brew install llama.cpp`);
  mistral.rs works for its supported embedding archs (EmbeddingGemma,
  Qwen3-Embedding — it does NOT support E5/BERT).
- The prefixes your model expects (E5: `passage: ` / `query: ` — the tool's
  defaults; EmbeddingGemma and Qwen3 use different instruction prefixes, pass
  `--doc-prefix`/`--query-prefix`).

## Step 0 — old BERT-family GGUFs may need a metadata patch

Symptom: `llama_model_load: error loading model: bert model needs to define
token type count`. Old conversions lack `tokenizer.ggml.token_type_count`.
Find the right value in the source model's `config.json` (`type_vocab_size` —
2 for mE5-small), then:

```console
$ python3 -m venv /tmp/ggufenv && /tmp/ggufenv/bin/pip install gguf
$ /tmp/ggufenv/bin/python scripts/patch_gguf.py \
    model.gguf model.fixed.gguf 2     # writes a patched COPY
```

## Step 1 — serve the model

```console
$ llama-server -m model.fixed.gguf --embeddings --pooling mean --port 8931
```

`--pooling mean` matters: E5-family models are mean-pooled; wrong pooling is a
silently different space. Record engine+version+pooling+quant in `--engine-id`
below — they are all part of the encoder identity.

## Step 2 — compute the frozen center artifact

```console
$ engram-fit center --endpoint http://localhost:8931 --model me5-small-q8 \
    --engine-id llama.cpp-b9890-metal-mean --out me5-small-q8.center.json
```

Output: doc + query centers (asymmetric models have per-role means), the tag
string, and `tag_u64` for `Memory::snapshot`/`restore`. The artifact is
FROZEN — changing it (new pack, new engine, new quant) is a new tag and a
wipe-and-reimport event, by design.

## Step 3 — health check

```console
$ engram-fit health --endpoint ... --model ... --center me5-small-q8.center.json
```

One number: mean pairwise jaccard of unrelated codes. HEALTHY < 0.10,
MARGINAL < 0.20, else COLLAPSED (don't ship). Expect raw ≈ 0.2–0.3 and
centered ≈ 0.01–0.04 for E5-class models.

## Step 4 — capture fixtures (so all of this becomes `cargo test`)

```console
$ engram-fit dump --endpoint ... --model ... \
    --out tests/fixtures/<model>.embs                       # pack, doc+query
$ engram-fit dump --endpoint ... --model ... \
    --pairs assets/pairs.tsv \
    --out tests/fixtures/<model>.pairs.embs                 # labeled pairs
$ engram-fit dump-stream --endpoint ... --model ... \
    --out tests/fixtures/<model>.stream.embs                # lifecycle stream
```

The bundled fixtures come from `multilingual-e5-small-q8_0.gguf` in
`cstr/multilingual-e5-small-GGUF` (patched with step 0, value 2; served with
`--pooling mean`) and `embeddinggemma-300M-qat-Q4_0.gguf` in
`ggml-org/embeddinggemma-300M-qat-q4_0-GGUF` (model-default pooling, the
prefixes below), on llama.cpp `b9890`. The pack and pairs fixtures were
captured on Metal and the stream fixtures (2026-11-21) on CPU (x86_64); the
two backends agree to cosine ≥ 0.9998 on every row.

Fixtures store RAW embeddings (centering is applied at test/tune time, so
centering variants stay testable). `tests/real_model.rs` runs the anisotropy
regression, label-ordering, and the full-circle gate-decision test against
them — offline, no engine.

## Step 5 — tune k and the bands (offline, from fixtures)

```console
$ engram-fit tune --pack-embs tests/fixtures/<model>.embs \
    --pairs-embs tests/fixtures/<model>.pairs.embs --pairs assets/pairs.tsv
```

Measured result for mE5-small-q8 (2026-07-07):

```text
k=128 is best by combined margin →
  Params.k             = 128
  gate.dup_jaccard     = 0.17   (asymmetric: above the related tail)
  gate.link_jaccard    = 0.05
  params.recall_anchor = 0.09   (blocks coincidence, NOT cousins — see below)
```

(2026-07-08: the anchor formula changed from unrelated-q95 × 1.5 to
unrelated-MAX × 1.5 — with ~20 labeled pairs the quantile index discards the
tail, and measured stream coincidence tops out ~0.07–0.08 at k=128 for both
fitted models while the q95 anchor let real coincidence through.
`tests/regimes_probe.rs` holds the measurements. mE5's anchor moved
0.08 → 0.09 and its lifecycle silence went 2/3 → 3/3 with no lost answers.)

**The three jaccard regimes (measured, k=128, centered).** Query-time best
matches land in three bands:

1. **Same-content, cross-role** (the stored sentence queried as itself):
   0.28–0.75. Easy.
2. **Real question→answer AND same-topic cousins**: ~0.10–0.21. These two
   populations OVERLAP — no jaccard threshold separates "the answer to this
   question" from "a same-topic non-answer" at this model scale. Don't try:
   a cousin-proof anchor refuses real answers too (measured — it zeroed the
   lifecycle eval). Cousins surfacing on a high-demand query are *context*,
   not junk; the downstream LLM judges them.
3. **Pure coincidence** (genuinely unrelated query): < ~0.08 even at
   ~100 stored episodes.

The anchor therefore sits between regimes 3 and 2 (`tune` derives it from the
unrelated-pairs q95 × 1.5) — it exists to block coincidence and fabrication,
not topical relevance. Boundary leaks are weak cousins (e.g. "what did the
mechanic say?" surfacing a repair-kit message), never inventions.

Also intended, not a bug: cross-lingual paraphrases score in the Link range →
they Link instead of Strengthen (asymmetric loss: a false Strengthen drops a
payload).

## Step 5b — the verdict: the lifecycle eval

`tests/lifecycle.rs` runs six simulated weeks of messages (`assets/stream.tsv`,
a synthetic diary of a fictional person: the people, places and events are
invented; fixture: `me5-small-q8.stream.embs` from `dump-stream`) through the
FULL loop — observe → passive channel → recall → feedback → weekly sleep —
against a brute-force cosine-RAG baseline. Measured 2026-11-21 (the stream
was rewritten then; the 2026-07 numbers on its first revision were the same
verdicts):

```text
answerable queries:  engram 8/8    RAG 8/8   (incl. a fact stated once,
                                              5 weeks earlier, queried in
                                              Italian and English)
silence queries:     engram 3/3    RAG 0/3   (RAG always injects)
memory:              51 episodes for 91 messages (44% compression),
                     35 pruned after quiet weeks, all queried facts survived
thresholds learned:  write 0.15→0.28 (stream is ephemera-heavy: bar rose)
                     read  0.35→0.27 (retrievals helped: trust grew)
```

For production-grade bands, replace/extend `assets/pairs.tsv` with real app
traffic (actual edge talk messages, labeled) and re-run steps 4–5.

## Second fitted model — EmbeddingGemma-300m QAT q4_0 (2026-07-08)

The 2025-era on-device model (official first-party GGUF from
`ggml-org/embeddinggemma-300M-qat-q4_0-GGUF`, 278MB on disk, 768-dim, needs
no metadata patching — step 0 does not apply). Prefixes differ from E5:

```console
$ engram-fit <cmd> ... --doc-prefix "title: none | text: " \
    --query-prefix "task: search result | query: "
```

Artifact tag (llama.cpp b9890, metal, model-default pooling):
`tag_u64 0x22b2ae2df2335eb8`; artifact saved next to the model as
`embeddinggemma-300m-qat-q4.center.json`.

Measured vs mE5-small-q8, same pack/pairs/stream, k=128:

| | mE5-small-q8 | gemma-300m-qat-q4 |
|---|---|---|
| raw unrelated jaccard (anisotropy) | 0.251 COLLAPSED | **0.044 HEALTHY** |
| centered unrelated jaccard | 0.012 | 0.011 |
| dup mean / related mean | 0.20 / 0.08 (~2.5×) | **0.34 / 0.09 (~3.9×)** |
| tuned bands | dup .17 link .05 anchor .09 | dup .17 link .05 anchor .07 |
| cross-role top-1 (150 stored) | 150/150 | 150/150 |
| lifecycle answerable | 8/8 | 8/8 |
| lifecycle silence | 3/3 | 2/3 |
| answer jaccard head-room (2026-11 stream) | ans 0.10–0.20 | **ans 0.17–0.29** |

Reading: gemma's space is dramatically healthier (centering keeps helping but
is no longer load-bearing) and answers score ~2× stronger — more head-room
over the anchor. The flip side: it's MORE cousin-sensitive (its one lifecycle
leak surfaces a booked-flights message for "did we ever book the ferry
crossing to the Azores?" — a travel cousin the downstream LLM can use; arguably
correct memory behavior). The answer/cousin overlap does NOT close at 300M:
gemma's weakest answer (0.169, cross-lingual; 0.089 on the first stream
revision) still scores below its hottest cousin (0.196), so the anchor stays a
coincidence blocker on every model
scale measured. Both models are covered by the same per-model test suite
(`real_model.rs`, `lifecycle.rs` — parameterized; `regimes_probe.rs` prints
the raw regime data).

## Storage diet (2026-07-08) — int8 wins, MRL-512 rejected

Durable embedding rows (the caller's neocortex table) don't have to be
f32[768] = 3KB. Three diets measured through the FULL battery (transform →
center → gate; fixtures reused offline, no re-encode):

| variant | bytes/msg | lifecycle | notes |
|---|---|---|---|
| f32[768] | 3072 | 8/8 · 2/3 | baseline |
| f16[768] | 1536 | 8/8 · 2/3 (identical) | needs custom pack; NOT sqlite-vec readable |
| **int8 "unit" [768]** | **768** | **8/8 · 2/3 (identical)** | **sqlite-vec native (`vec_int8` + cosine) — the pick** |
| MRL-512 + f16 | 1024 | 7/8 · 0/3 — **REJECTED** | measured 2026-07 on the first stream revision: truncation sank the cross-lingual answer (0.089→0.067) below the coincidence tail and heated cousins |

int8 "unit" is sqlite-vec's quantization for L2-normalized vectors
(`x → round(clamp(x,−1,1)×127)`): 4× smaller than f32, still
`vec_distance_cosine(vec_int8(…))`-queryable in SQL. Rules:

- **Determinism**: round-trip EVERY embedding through the int8 pack before
  gating (fresh-encoded ones too), so all devices gate on identical values.
- **Fit consistency**: compute the center artifact from int8-round-tripped
  vectors — `engram-fit center --pack-embs <fixture> --int8` (offline), and
  suffix the model name (`-int8`): the storage format is part of the encoder
  identity, so it's a new tag.
- f16 helpers stay in the crate (`engram::f32_to_f16_bits`/`f16_bits_to_f32`,
  exhaustively tested) for stores without sqlite-vec.
- Do NOT MRL-truncate at this model scale: the tail dims carry the
  cross-lingual signal (measured — see `lifecycle_six_weeks` variants and
  `regimes_probe.rs`).

## Step 6 — wire it

```rust
let artifact: CenterArtifact = serde_json::from_str(&fs::read_to_string("center.json")?)?;
let mut m: Engram<_> = Engram::new(store);
m.params.k = 128;                        // from tune
m.gate.dup_jaccard = 0.17;               // from tune
m.gate.link_jaccard = 0.05;              // from tune
m.params.recall_anchor = 0.08;           // from tune (coincidence blocker)

// store path: "passage: <text>" → embed → center → observe
let mut v = embed(format!("passage: {text}"));
engram::center_in_place(&mut v, &artifact.doc_center);
m.observe(&v, handle, demand);

// query path: "query: <text>" → embed → center (QUERY center) → recall
let mut q = embed(format!("query: {question}"));
engram::center_in_place(&mut q, &artifact.query_center);
let hits = m.recall_with_demand(&q, demand, 4);

// persistence: the artifact's tag guards the snapshot
let bytes = m.snapshot(artifact.tag_u64);
```

## Troubleshooting

- **Endpoint refuses / wrong dims**: check the model actually serves
  embeddings (`curl :8931/v1/embeddings -d '{"model":"x","input":["hi"]}'`).
- **HEALTHY raw space**: some models are less anisotropic; centering is still
  recommended (it never hurts) but not load-bearing.
- **dup/related distributions overlap badly in `calibrate`**: the model can't
  separate paraphrase from topic-mate at this size/quant — try the fp16
  weights, a larger model, or accept Link-instead-of-Strengthen for hard
  paraphrases.
- **Changing anything about the encoder** (model, quant, pooling, prefixes,
  center, pack): new tag → old snapshots refuse to load → re-`import` the
  payload corpus. This is intended.
