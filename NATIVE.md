# Native expert weights

erisinky loads only the Q4_K_M GGUF. That file re-rounds the released 4-bit experts onto Q4_K's grid, and its next-token predictions drift far from the released weights. This document describes how inkserve's erisinky branch instead runs the routed experts in the formats Thinking Machines released.

## Why

Measured teacher-forced against Q8_0 (a near-lossless copy of the release) on ~3K-token agent prompts (`notes/RESULTS.md` → Weight drift):

| Weights | Top token agrees with Q8_0 | Mean KL (nats) |
|---|---|---|
| Q4_K_M (erisinky today) | 85–89% | 0.21–0.25 |
| omlx oQ4e (native NVFP4 experts in layers 0–19) | 91–93.5% | 0.11–0.13 |
| any engine against itself (the noise floor) | 93–94% | 0.10–0.13 |

On 12 Go agent tasks, inkserve with Q4_K_M wrote 2.85× omlx's output tokens. Drifting weights plausibly cause part of that.

## What the release contains

From `thinkingmachines/Inkling-2-Mini` at commit `b85c3cfa` (`hf_quant_config.json` and the safetensors header):

| Tensors | Format |
|---|---|
| Routed experts `w13` (gate and up) and `w2` (down), layers 0–19 | **NVFP4**: FP4 E2M1 codes, one FP8 E4M3 scale per 16 weights, one FP32 scale per expert |
| Routed experts, layers 20–23 | **MXFP8**: FP8 E4M3 values, one E8M0 (power of two) scale per 32 weights |
| Attention, embeddings, output head, router | BF16 |
| Norms, loop controllers, router bias | BF16 or F32 |

- An NVFP4 weight is `fp4(code) × e4m3(scale) × amax[expert] / outer_scale_denominator`.
- The official rows are stored `[expert][output row][input]`, two codes per byte.
- The release also asks for activations quantized to NVFP4 or MXFP8 before the expert matmuls (`required_activation_quantization`). No engine we use does this; all of them, including the Q8_0 reference, keep activations in higher precision.

## The file: NV4 and MX8

`tools/inkconvert` builds a GGUF from the Q4_K_M file and the official safetensors. It replaces every `ffn_gate_up_exps` and `ffn_down_exps` tensor and copies everything else. Two new block types hold the experts. They are private to this branch, and llama.cpp can't load them.

| Type | Id | Block (32 weights) | Bits per weight | Holds |
|---|---|---|---|---|
| NV4 | 200 | 2 E4M3 scale bytes (one per 16 weights), then 16 bytes: low nibble *j* is weight *j*, high nibble is weight *j*+16 | 4.5 | NVFP4, bit for bit |
| MX8 | 201 | fp16 scale (the exact power of two), then 32 E4M3 bytes | 8.5 | MXFP8, bit for bit |

- Each NV4 tensor has a companion F32 tensor `<name minus .weight>.gscale` with the 128 per-expert scales. Kernels multiply a row's dot product by it.
- NV4 is 18 bytes per block, like Q4_0, so it costs the same bandwidth as Q4_K (4.5 bits).
- MX8 costs 8.5 bits per weight, like Q8_0, for the 4 layers that were released at 8 bits.
- `inkconvert probe` checks the reading of the official layout (nibble order, gate/up row order, global scale) against the Q8_0 GGUF. `inkconvert verify` compares a finished file's experts with Q8_0 row by row.

## Kernels

Decoding a block needs no tables:
- **FP4 E2M1 to half:** place the 3 magnitude bits at bit 9 of a half (`(n & 7) << 9`) and the sign at bit 15. That half equals the FP4 value × 2^-14, for normal and subnormal codes alike.
- **E4M3 to half:** place the 7 magnitude bits at bit 7 (`(b & 0x7F) << 7`). That half equals the E4M3 value × 2^-8.
- The constant factors fold into the block scale, computed in fp32.

Where the new types go:

| Path | Kernels | Change |
|---|---|---|
| Decode, gate/up | `k_moe_gu` | an NV4 / MX8 row dot next to `dot_q4k`; the result × gscale |
| Decode, down | `k_moe_down` (`down_rows`) | NV4 / MX8 branches next to Q8_0 / Q5_0; the router weight × gscale |
| Prefill | `deq16` (used by `k_moe_gemm`, `k_moe_gemm_t`, `k_moe_sg`) | NV4 / MX8 branches; the epilogues multiply by gscale before the activation |
| Loader | `gguf.rs`, `engine.rs` | the type table, the expert asserts, and the gscale offsets |

The gscale tensors live in the mapped GGUF, so kernels read them through the weight buffer by file offset. No new buffer bindings are needed.

### Decode layout

Decode is bandwidth-bound, and the lane layout decides how well the reads stream. The decode kernels for NV4, MX8 and Q8_0 give **4 lanes to each block**: each lane reads 4 code bytes (NV4) or 8 values (MX8, Q8_0), and a simdgroup covers 8 blocks per pass. A lane decodes 8 FP4 weights with two `half4` bit-tricks (`fp4x8`) and both E4M3 scales at once (`e4m3f2`). `NV4_K` is applied once per row, with the expert's global scale. The down projection keeps its flattened (expert, block) items, 8 per pass.

Measured on an M5 Pro (decode step at position 1200, erisinky's `prof`; raw: `notes/raw/kernel-tune-20261123.txt`):

| Kernel class | One block per lane | 4 lanes per block |
|---|---|---|
| gate/up (NV4/MX8) | 3.00 ms (420 GB/s) | 2.53 ms (487 GB/s) |
| down (NV4/MX8) | 1.68 ms | 1.66 ms |
| q/k/v/gate/rel (Q8_0) | 2.07 ms | 2.00 ms |
| full step | 11.64 ms | 11.15 ms |

What didn't help: more rows per simdgroup (4 instead of 2, in both the matvec and down), and the cheaper decode alone, without the new layout. Q8_0 in the matvec was already near the bandwidth limit; most of its remaining cost is a fixed ~4 µs per dispatch [estimate from two dispatch sizes].

## Beyond the experts: Q8_0 attention, embedding and output head

Native experts alone barely reduced the drift. The rest of the gap came from the other matrices, which the release ships in BF16 and Q4_K_M cuts to Q4_K (attention, embedding) or Q6_K (output head). `inkconvert convert -q8 FILE -q8-tensors REGEX` copies matching tensors from the ggml-org Q8_0 GGUF. The patch adds Q8_0 to the decode matvec (`dot_q8` in `k_mv`) and to `k_embed`, which tells a Q8_0 row from a Q4_K row by its length. Prefill's `deq16` already handled Q8_0.

## Results

Teacher-forced against llama.cpp Q8_0 on three ~3K-token prompts, decode path (details: `notes/RESULTS.md` → erisinky with native experts):

| File | Top-1 agree | Mean KL | Decode tok/s, 1.2K / 11K / 38K |
|---|---|---|---|
| Q4_K_M | 84.6–90.0% | 0.19–0.26 | 96.8 / 86.1 / 66.5 |
| native experts | 85.6–90.1% | 0.19–0.22 | 94.1 / 84.2 / 65.1 |
| native experts + Q8_0 attention | 91.3–93.8% | 0.12–0.14 | 85.5 / 77.1 / 60.7 |
| **native-q8rest** (+ Q8_0 embedding and output) | **93.6–95.6%** | **0.08–0.10** | **84.1 / 76.0 / 60.2** |
| omlx oQ4e, for reference | 91.3–93.5% | 0.11–0.13 | 64 / 53 / 46 (served) |

`native-q8rest` was the recommended file until the bit-ablation study below (now `native-q8rest-o6`): about at the noise floor, and still ~1.3× omlx's decode. In a 12-task Go agent eval it passed 12/12 at `high` effort, with 31% fewer output tokens than Q4_K_M. At `medium` it passed 12/12 again in 116 agent minutes (omlx at `medium`: 10/12 in 89 min; one slow task makes up the gap).

## Where do bits matter?

A bit-ablation study (`notes/bench/run-bits.sh`, 2026-11-23) measured drift and speed for variants of `native-q8rest` with one piece changed. The patch also accepts Q8_0 gate/up experts, so the plain Q8_0 GGUF loads as a control. Mean KL against Q8_0 (baseline 0.086; Q4_K_M 0.223):

| Change | ΔKL | Decode Δ |
|---|---|---|
| output head Q6_K | +0.001 | +1.6% |
| all attention Q6_K, output head Q6_K | +0.043 | +6.4% |
| embedding Q4_K | +0.042 | 0% |
| Q4_K_M's experts | +0.059 | +2.7% |
| one attention class Q4_K (q, gate, rel/sink; k/v; output projection) | +0.037 to +0.081 | 0–2% |
| attention of layers 18–23 / 0–5 / 6–17 (looped core) Q4_K | +0.015 / +0.066 / +0.104 | +1% / +1% / +7% |
| the plain Q8_0 file (same weights as the reference) | −0.029 | −13% |

- The output head can be Q6_K at no cost. Everything else in `native-q8rest` earns its bits.
- So the recommended file is now **`native-q8rest-o6`**: `native-q8rest` with a Q6_K output head (`-q6-tensors '^output\.weight$'`). Measured with the inkserve engine binary: 93.98% top-1, KL 0.087, decode 85.6 / 77.1 / 61.0 tok/s (+1.7–2.4%). With the tuned decode kernels (Decode layout, above): 93.95%, KL 0.089, decode **88.8 / 79.5 / 62.4** tok/s.
- Q6_K attention is the faster option at omlx oQ4e's drift.
- Bits buy little speed. Kernel work (`dot_q8`, the sampling loop) is the bigger lever.

Full table and method: `notes/RESULTS.md` → Bit-ablation study.
