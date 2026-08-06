# CUDA kernels

Kernels for moving inference onto the GPU. Everything under `kernels/`, `common/` and
`bench/` was brought over from https://github.com/Madfyre/CUDA and keeps its original
names, so the `NOTES.md` in each directory still applies.

## The workload

Llama-3.2-1B, straight from the checkpoint's `config.json`:

```
hidden 2048   heads_q 32   heads_k 8 (GQA 4:1)   head_dim 64
intermediate 8192   n_layers 16   vocab 128256   tie_word_embeddings: true
```

Weights are BF16, activations are F32. Per-layer shapes, stored `[out, in]`:

| tensor | shape | params |
|---|---|---|
| `q_proj`, `o_proj` | [2048, 2048] | 4.2 M each |
| `k_proj`, `v_proj` | [512, 2048] | 1.0 M each |
| `gate_proj`, `up_proj` | [8192, 2048] | 16.8 M each |
| `down_proj` | [2048, 8192] | 16.8 M |
| | **per layer** | **60.8 M** |

16 layers = 973 M, plus `embed_tokens` [128256, 2048] = 263 M. Total 1.236 B params,
2.47 GB in BF16 — exactly the size of `model.safetensors`.

**Decode reads all of it for every single token**: layer weights (1.95 GB) plus the tied
LM head (525 MB). Arithmetic intensity is about 1 flop per byte, so decode is bound by
memory bandwidth and nothing else. The KV cache is small by comparison: 32 KB per token
across all layers, 16 MB at a 512-token context.

Prefill is the opposite — the same weights are read once for all S tokens while FLOPs
grow with S, so it is compute-bound and wants tensor cores.

**Prefill and decode need two different families of kernels.** A single kernel
parameterised by sequence length will be wrong for one of them.

## What was brought over and why

| Directory | Role here |
|---|---|
| `03-dot-product` | `warpReduce` / `blockReduce` and the partial→reduce split. The base primitive for RMSNorm, softmax, GEMV split-K and argmax. |
| `08-rms-norm-gated` | RMSNorm skeleton. Ours is simpler (no gate, no per-head split), but the reduction and the fp16 handling carry over. |
| `02-gemm-v0` | Baseline tiled GEMM — the readable starting point for the prefill path. |
| `04-gemm-v1` | Prefill GEMM: FP16, shared-memory tiling, register blocking, bank-conflict padding. |
| `09-cutlass-gemm` | CUTLASS with a fused epilogue. The epilogue pattern is what SwiGLU, RoPE and the KV-cache write should hang off. |
| `05-quantization` | INT8 weight quantization. Not a side quest — decode is bandwidth-bound, so this is the direct route to halving time per token. |
| `06-moe-topk` | Top-K reduction, reusable for sampling and as the argmax skeleton. |
| `07-moe-topk-hist` | Histogram-based Top-K. Likely the better shape for a 128256-wide vocabulary. |
| `08-prefix-sum` | Cumulative probabilities, needed for top-p sampling. |
| `common/`, `bench/` | Error checking, L2 query, and the self-contained test/bench driver pattern. |

Left behind: `01-grayscale`, `01-reverse-string`, `05-quaternions` (unrelated), and
`03-transpose-v0` / `04-transpose-v1` — weights are already stored `[out, in]`, which
gives each output element a contiguous 4 KB row to read, so nothing needs transposing.

## Still to write

| Kernel | Shape at decode (S=1) | Notes |
|---|---|---|
| Embedding gather + bf16→f32 | 1×2048 out of [128256, 2048] | Conversion must be inline in every kernel that touches weights, never a separate pass. |
| RMSNorm | 2048, row reduction | Fuse with the residual add: one kernel updates the residual and emits the normalized copy. |
| **GEMV (decode)** | 1×2048 @ 2048×N | The whole ballgame. Tiling and register blocking buy nothing — every weight is read exactly once. Judge it purely by % of peak bandwidth; below 80% is not done. |
| RoPE | Q [1, 2048], K [1, 512] | Half-split convention (`r1 = x[i]`, `r2 = x[i + head_dim/2]`), matching HF `rotate_half`. |
| KV-cache append | 2 × 512 elements | Layout: `[layer][2][heads_k][max_seq][head_dim]`. Append happens once per token, reads happen S times, so favour the reads. |
| QK^T with GQA + scale | 32 × (1 × S_k) | 4 query heads share one KV head. Scale by `1/sqrt(64)`. |
| Causal mask | — | Folds into softmax. With a KV cache the condition is `col > pos_offset + row`, not `col > row`. |
| Online softmax | over S_k | Streaming max/sum, no scores materialization. |
| AV | 32 × (1×S_k @ S_k×64) | |
| **Flash-decode attention** | fuses the four above | With S_q=1 a block-per-head grid is only 32 blocks. Split over K, produce partial `(max, sum, acc)`, combine in a second pass. |
| SwiGLU | 8192 | `silu(gate) * up`, an epilogue on the fused gate+up GEMV. |
| LM head + argmax | 1×2048 @ 2048×128256 | 525 MB per token, 21% of all decode traffic. For greedy decoding fuse argmax into the epilogue and never write 128256 floats. |

Fusions worth doing once the naive versions are correct: residual+RMSNorm, Q/K/V into one
[3072, 2048] GEMV, gate+up into one [16384, 2048] GEMV, LM head + argmax.

## Testing

`ref_instruct/h0..h16.bin` hold the per-layer hidden states from the reference
implementation, and `model::CompareRef` already checks against them — but that is
layer granularity, and a layer contains seven operations.

Before writing kernels, extend the dump to the intermediates of a single layer (Q, K, V
before and after RoPE, scores, post-softmax, attention output, gate/up/down). The
`set_layer_hook` on `model::Llama` is the place to hang it. Without that, kernel debugging
is guesswork.

Suggested order — it reaches a working decode path as early as possible:

1. embedding gather, RMSNorm, argmax (warm-up, gets the harness working)
2. **GEMV** — do not rush this one, it is ~80% of inference time
3. RoPE, KV append
4. naive attention as four kernels
5. SwiGLU → **full working GPU decode at this point**
6. fused flash-decode attention
7. the fusions listed above
8. prefill GEMM / CUTLASS
9. runtime: CUDA graphs first, streams later

## Runtime, later

With sensible fusion a decode step is about 116 launches (7 per layer × 16, plus gather,
final norm, LM head). At 3–5 µs of launch overhead that is 0.35–0.6 ms against a ~2.45 ms
compute floor on a 4090 — 15–25% wasted. The graph shape never changes between tokens
(fixed `max_seq_len`, varying length handled by the mask), so capture once and replay.

Streams are a different story. Single-request decode is a strict dependency chain with
nothing to overlap. Streams start paying off with request batching, overlapping weight
upload with compute, or moving sampling and detokenization off the critical path — all of
which belong to a request queue, not to making one token faster.

## Build

See `bench/README.md`. Each driver is a single self-contained executable; adjust `-arch`
to the target GPU.
