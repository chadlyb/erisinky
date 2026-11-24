# My changes

## Summary

This fork makes erisinky run Inkling 2 Mini closer to how its makers released it, and serve it well to coding agents.

- **Closer to the release.** The model's experts were released in NVFP4 and MXFP8, two low-bit float formats. Stock erisinky reads a GGUF that re-quantized them to Q4_K. This fork reads the released bits as they are, and keeps attention and the embedding at 8 bits. Its top predictions agree with a near-lossless reference 93.6–95.6% of the time, against 85–89% for Q4_K_M.
- **Fast to serve.** Tuned kernels win back most of the cost of the bigger file. Sampling on the GPU lets sampled decoding run as fast as erisinky's greedy loop. Served decode is 89 / 79 / 63 tok/s at ~1K / 11K / 35K context.
- **Ready for agents.** `inkling serve` keeps the model loaded and reuses the cached prefix across requests. inkserve (in [inkserve/](inkserve/)) puts an OpenAI-style API in front: tool calls, the model's chat template, reasoning effort and a thinking budget.
- **Speculative decoding, measured honestly.** DSpark, with the release's own drafter, is built and exact in distribution. On agent text it gains only ~3%, because the drafter's guesses are accepted too rarely to repay a verify pass. It's opt-in.

## Results and experiments

All numbers are from an M5 Pro, measured 2026-11-22 to 2026-11-23.

### Accuracy

Teacher-forced against llama.cpp's Q8_0 (a near-lossless copy of the release) on three ~3K-token agent prompts:

| File | Top-1 agreement with Q8_0 | Mean KL |
|---|---|---|
| Q4_K_M (stock erisinky's file) | 85–89% | 0.20–0.25 |
| `native-q8rest-o6` (this fork) | 93.6–95.6% | 0.08–0.10 |
| omlx's oQ4e (for comparison) | — | 0.11–0.13 |

**Bit-ablation study.** One piece of `native-q8rest` at a time went to fewer bits, measuring drift and speed:
- The output head can drop to Q6_K for free: +1.6% decode, same drift. That's `native-q8rest-o6`.
- Attention (above all the looped core, layers 6–17) and the embedding need 8 bits.
- Q6_K attention plus a Q6_K output head gives +6.4% decode, at oQ4e's level of drift (KL 0.129, against 0.086 for `native-q8rest`).

Method and full tables: [NATIVE.md](NATIVE.md).

### Speed

Served through inkserve with `native-q8rest-o6` at temperature 1.0: **89 / 79 / 63 tok/s** decode at ~1K / 11K / 35K context. That matches erisinky's own GPU-chained loop on the same file.

Served against omlx 0.8.2 (oQ4e weights, DSpark on), on the same streamed code-review prompts at temperature 1.0. These rows used Q4_K_M and an earlier CPU sampler:

| Context | Prefill | Decode | Next turn, time to first token |
|---|---|---|---|
| ~1K | 3,100 tok/s (2.1× omlx) | 86 tok/s (1.36×) | 0.17 s |
| ~11K | 4,200 tok/s (2.4×) | 77 tok/s (1.45×) | 0.14 s |
| ~37K | 3,800 tok/s (2.35×) | 62 tok/s (1.35×) | 0.2 s |
| ~67K | 3,200 tok/s (2.35×) | 50 tok/s (~1.5×) | 0.3 s |

- **Kernel tuning.** The 8-bit file first decoded 8–12% slower than Q4_K_M. A per-kernel profile showed where: Q8_0 attention, and NV4 gate/up reading at 420 GB/s instead of Q4_K's 481. With 4 lanes per block in the NV4, MX8 and Q8_0 matvecs, the gap closed to 6–8%. See [NATIVE.md](NATIVE.md) → Decode layout.
- **Prefill** runs 3–5% below Q4_K_M's.
- **On real agent turns** (6 Go tasks at `medium` effort): 6/6 passed, at ~1.2× omlx per token.

### DSpark speculative decoding

The release ships a 3-layer drafter that proposes the next few tokens. One batched verify pass of the main model then checks them.
- **A new small-batch verify path:** 4 tokens in 21.7 ms at ~2.4K context, against 11.3 ms for one decode step. erisinky's prefill path would have cost as much as 4 decode steps.
- **Exact in distribution:** standard speculative sampling (accept with probability min(1, p/q), else draw the residual). Rollback restores the state of rejected positions.
- **Results:** on agent text the drafter keeps 2.0–2.5 tokens per cycle, near the ~2.4 needed to beat plain decoding. Through the serve loop that gives 0.95–1.15× by prompt, ~1.03× on average (measured with an earlier version that timed the cycles to pick the depth).
- **A fixed-cost model** picks the draft depth (1–3) per cycle, or plain steps when no depth would pay off. Costs are fixed, so a seed always gives the same reply.

The verify pass is bandwidth-bound on the union of its tokens' experts. A larger gain needs a drafter that fits agent text better. Full log: [DSPARK.md](DSPARK.md).

## Details

This part breaks the changes into pieces that stand mostly on their own, for anyone who wants to review them, rebase them, or offer pieces upstream. In this repository they are one commit on top of erisinky's `gpl-license` branch and upstream pull request #2. That commit adds about 2,460 lines and removes 106, in seven files, `src/serve.rs` (476 lines) among them. Line counts below are approximate.

### Building

1. **Engine:** `cargo build --release --locked` at the repository's root gives `target/release/inkling`.
2. **Weights** (`native-q8rest-o6`), with `tools/inkconvert` (Go 1.27):
   ```
   cd tools/inkconvert && GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go build -trimpath -o build/inkconvert-darwin-arm64 .
   ```
   It needs the official `model-00001-of-00001.safetensors` (`thinkingmachines/Inkling-2-Mini` @ `b85c3cfa`), the Q4_K_M GGUF (for metadata and small tensors) and the ggml-org Q8_0 GGUF:
   ```
   inkconvert probe -official model-00001-of-00001.safetensors -ref Inkling-2-Mini-Q8_0.gguf
   inkconvert convert -official model-00001-of-00001.safetensors -base Inkling-2-Mini-Q4_K_M.gguf -w13 interleaved \
     -q8 Inkling-2-Mini-Q8_0.gguf -q8-tensors '^token_embd\.weight$|^blk\.[0-9]+\.attn_.*\.weight$' \
     -q6-tensors '^output\.weight$' -out Inkling-2-Mini-native-q8rest-o6.gguf
   inkconvert verify -in Inkling-2-Mini-native-q8rest-o6.gguf -ref Inkling-2-Mini-Q8_0.gguf
   ```
   `probe` should show ~0.6% error for `il=true hi=false`, and ~140% for the other readings. Other `convert` options:
   - `-q6-tensors REGEX`: re-quantize matching Q8_0 tensors to Q6_K (llama.cpp's reference encoder, ported; `inkconvert q6check -base Q4_K_M.gguf -ref Q8_0.gguf` compares its error with llama.cpp's own Q6_K)
   - `-experts base`: keep the base file's experts instead of the release's
   - Tensors matching neither `-q6-tensors` nor `-q8-tensors` stay as in `-base`.

   The DSpark drafter has its own file, converted from the release's `mtp.safetensors`:
   ```
   inkconvert mtp -in mtp.safetensors -official model-00001-of-00001.safetensors -out Inkling-2-Mini-dspark.gguf
   ```
   `go test` in `tools/inkconvert` checks the NV4 / MX8 packing and the FP4 / FP8 decoders.
3. **Server:** [inkserve/README.md](inkserve/README.md).

### The parts at a glance

| # | Part | Kind | Size | Depends on |
|---|---|---|---|---|
| 1 | Native expert formats (NV4 / MX8) | accuracy | ~230 | — |
| 2 | Q8_0 for the non-expert tensors | accuracy | ~60 | — |
| 3 | Decode kernel tuning | speed | inside 1 and 2 | 1, 2 |
| 4 | GPU sampling | feature + speed | ~70 | — |
| 5 | The serve protocol (`serve.rs`) | feature | 476 (own file) | 4 |
| 6 | Small-batch verify path | speed (infrastructure) | ~750 | 1 (MoE kernels) |
| 7 | DSpark speculative decoding | feature | ~820 | 4, 5, 6 |
| 8 | Dev tools | tooling | ~330 | 6, 7 |
| 9 | Bug fixes | fixes | small | see below |

The diagram shows the dependencies. Parts 1, 2 and 4 stand alone against upstream. Part 6 is the base for DSpark, and for a future multi-conversation mode.

```
 1 native experts ──┬── 3 tuning
 2 Q8_0 tensors ────┘
 1 ─────────────────── 6 verify path ──┐
 4 GPU sampling ── 5 serve protocol ───┴── 7 DSpark
```

### 1. Native expert formats (accuracy)

The release ships its experts as NVFP4 (gate/up) and MXFP8 (down). Stock erisinky reads them requantized to Q4_K / Q8_0, which costs accuracy. This part reads the release's own bits. See [NATIVE.md](NATIVE.md).
- **Block types** (`gguf.rs`): `T_NV4` = 200 (18-byte blocks of 32 weights) and `T_MX8` = 201 (34 bytes).
- **Expert scale** (`engine.rs` `qw`): NV4 tensors carry one FP32 scale per expert in a companion `.gscale` tensor (`QW::gs`).
- **Type checks** (`Engine::load`): experts may be Q4_K, Q8_0, NV4 or MX8.
- **Decode kernels** (`kernels.metal`):
  - table-free decoding (`fp4f`, `e4m3f`, `fp4x8`, `e4m3x8`): the magnitude bits placed into a half give the value times a fixed power of two
  - `dot_nv4` and `dot_mx8`
  - NV4 and MX8 branches in `k_moe_gu` and `down_rows` / `k_moe_down`
  - the expert scale through `gscale_at`
- **Prefill kernels** (`prefill.metal`):
  - `deq16_q4k` dequantizes NV4 / MX8 tiles to half
  - `k_moe_gemm`, `k_moe_gemm_t` and `k_moe_sg` apply the expert scale
- **Argument structs:** new `gs_lo` / `gs_hi` (and `gu_type`) fields in `MoeArgs`, `MoeGemmArgs` and `MoeGemmHArgs`.
- Outside the patch, `tools/inkconvert` writes these files.

### 2. Q8_0 for the non-expert tensors (accuracy)

The `native-q8rest-o6` file keeps attention and the token embedding at Q8_0, and the output head at Q6_K. Stock erisinky expects Q4_K for those tensors.
- `Engine::load` accepts Q8_0 (and Q6_K) for attention, `token_embd` and `output`.
- `k_embed` reads Q8_0 rows.
- `k_mv` gets a `dot_q8` path. Its `sumf` now starts as `{}`, which zeroes any row count.

### 3. Decode kernel tuning (speed)

This tuning applies to the kernels of parts 1 and 2, so it has no meaning without them. It lives in the same functions.
- **`dot_q8`:** llama.cpp's layout, with 4 lanes per block and 8 blocks per simdgroup pass. All lanes stay busy at K = 2304 and 3072.
- **`dot_nv4` (`NV4_BAL`):** 4 lanes per block, 4 code bytes each.
- **NV4 / MX8 expert down:** 8 (expert, block) items per pass, so TOPK × 36 blocks = 288 split evenly.
- **Decoding without tables**, as in part 1.

Measured effect: `native-q8rest-o6` went from 8–12% to 6–8% slower than Q4_K_M (NATIVE.md → Decode layout).

### 4. GPU sampling (feature + speed)

Stock erisinky's chained decoding is greedy. This part draws each token on the GPU, so sampled decoding keeps the chained speed. See inkserve/ARCHITECTURE.md → Sampling.
- **`SampArgs`** and **`Engine::samp`:** when set, a step's argmax becomes a draw.
- **`k_gumbel`:** adds Gumbel noise to softcap(logit) / T. The noise is a hash of (seed, position, token), and the argmax of logit plus noise is an exact softmax draw.
- **`k_sampstate`:** the thinking budget's state machine, in three spare token slots past the context.
- **`sample_tail`** (the decode step's tail) and **`encode_sample`** (a draw from the last prefill's logits).

### 5. The serve protocol (feature)

`src/serve.rs` is a new file. It adds `inkling serve`, a long-lived engine that inkserve drives over stdin and stdout. See inkserve/ARCHITECTURE.md → Engine protocol.
- the line protocol (`G` / `X` in; `R` / `P` / `T` / `E` / `F` out)
- **Prefix cache:** reuse the live context, or a snapshot taken two tokens before each prompt's end
- the thinking budget
- **Two sampling paths:** `gen_chained` (part 4) and a CPU sampler for top-k / top-p (`Pick`)
- cancel, and the token and context limits
- the DSpark path, `gen_dspark` (part 7)
- In `main.rs`, the `serve` command and its `--ref` (drafter) argument.

### 6. Small-batch verify path (speed infrastructure)

This part is one pass of the main model over 2–8 tokens that is much cheaper than erisinky's prefill path at that size. DSpark verifies its drafts with it. Batched decoding of several conversations could use it too. It is gated by `Engine::verify`, so prefill stays bit-identical. See DSPARK.md → The verify path.
- **`encode_verify(cb, p0, t)`:** the batched path with the small-T kernels, and logits for every position (`ar.vlog`, `verify_logits`). It needs a 128-row batch arena (`VERIFY_BATCH`).
- **`k_mvT`:** a decode-layout matvec for T ≤ 8 (Q8_0, Q6_K, BF16). It replaces the 64-row GEMM tiles, which leave the GPU idle at small T.
- **`k_attn_hs` + `k_attn_hs_comb`:** `k_attn_h` split over the keys, about 10 ranges.
- **`k_moe_routeT` / `k_moe_guT` / `k_moe_downT`:**
  - the MoE over the union of the tokens' experts, each expert read once
  - the down kernel splits each row group across 4 simdgroups (`DNT_KS`)
  - NV4 / MX8 only, so it builds on part 1
- **`k_mix_bw`:** the residual mixes with one 256-thread threadgroup per token.
- **Refactor:** the batched MoE moves into `moe_ffn`, and the old tiled path moves there unchanged. This is most of the patch's 97 removed lines.
- Measured: 21.7 ms for 4 tokens at ~2.4K, against 11.3 ms for one decode step.

### 7. DSpark speculative decoding (feature)

This is the release's bundled drafter, used for speculative decoding. It's opt-in (`inkserve -dspark`), and its output is exact in distribution. See [DSPARK.md](DSPARK.md).
- **Taps:**
  - `k_tapcopy` writes four feature vectors per position into a 1,024-position ring (`ar.taps`)
  - `tap_feature` / `visit_of` choose the layer visits
  - `taps_on` / `tap_from` control capture, and a tapped prefill turns tail trimming off
- **The drafter:** `engine/dspark.rs` (`Drafter::load`, `DScratch`, `draft_reset`, `encode_draft_commit`, `encode_draft`) and the `k_d*` kernels in `prefill.metal`.
- **Rollback:**
  - `k_state_upd` records the conv history in verify passes (`StateArgs::hist_off`, `ar.vhist`)
  - `swa_save` and `verify_rollback` restore the state on the CPU
- **Speculative sampling:** `encode_verify_spec` with `k_slse`, `k_sresid`, `k_sdecide` and `k_dcopyf`. It accepts a draft with probability min(1, p/q), or else samples the residual. `encode_verify_sample` gives the coupled and greedy variants.
- **Serve loop:** `gen_dspark` in `serve.rs`:
  - the per-cycle depth choice, plain-step stretches, and the `tap_hi` window
  - CPU handling of the thinking budget, eos and limits over accepted tokens

### 8. Dev tools (tooling)

These are commands in `main.rs` and one engine method, all marked "dev". None runs in serving.
- `prof_verify`: verify timing, agreement with decode steps, and a breakdown by kernel class
- `kcheck` (`Engine::kcheck`): each small-T kernel against the original on random inputs
- `tapcheck`: taps from prefill against taps from decode steps
- `dcheck`: drafter acceptance on a transcript
- `spectest`: DSpark against plain decoding, with speeds
- **Environment switches:** `INK_NO_MOET`, `INK_NO_MVT` and `INK_NO_ATTNT` turn off one small-T kernel family each.

### 9. Bug fixes

Few of our changes fix upstream code. Most of the fixes below are to our own additions, and they ship inside those parts.
- **Upstream: one-token prefill.** At `3890bd5` a one-token prefill on a fresh batch arena gives wrong logits. Upstream pull request [Eriskii/erisinky#2](https://github.com/Eriskii/erisinky/pull/2) fixes it, and this repository merges it. `serve.rs` `feed()` still feeds one token through a decode step, which is cheaper.
- **Upstream: `INK_OLD_MIXB` kept.** This upstream switch sends batched passes through the decode `k_mix` kernel, which doesn't write the half copies the fp16 GEMMs read. Setting it breaks verify passes, and probably fp16 prefill too (not checked).
- **Ours, part 4:** the Gumbel noise's uniform could round to 1.0 and make a token's noise infinite (~1% of steps). Fixed with 23 random bits plus a half step.
- **Ours, part 6:** with a batch arena under 128 rows, verify passes silently fell back to the prefill MoE (6 ms slower). `encode_verify` now asserts the size.
- **Ours, part 7:** after a snapshot restore, stale taps from an earlier reply fed the drafter. That inflated acceptance, but it never affected correctness. Fixed with `tap_hi`.

### Outside the engine

The Go server is a separate program, in `inkserve/`. The weight converter is in `tools/inkconvert/`.
- **inkserve** (`*.go`, ~1,500 lines):
  - the OpenAI-style HTTP API
  - the chat template, checked byte for byte against `testdata/golden.json`
  - tool-call parsing
  - effort, thinking budget and sampling defaults
  - `-dspark`
- **inkconvert** (`tools/inkconvert`, ~1,700 lines): builds the GGUFs of parts 1 and 2 (`convert`, `probe`, `verify`, `q6check`) and the drafter file (`mtp`).

### Offering parts upstream

Parts 1, 2 and 4 are self-contained and useful to anyone running erisinky on the release's weights. Part 3 rides along with 1 and 2. Parts 6 and 7 are larger and tied to speculative decoding. They make sense upstream only if erisinky wants DSpark or batched decoding. inky's "lanes" work overlaps part 6. Nothing can be offered until the license question is settled (TODO.md → Blocking).
