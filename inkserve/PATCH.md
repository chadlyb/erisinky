# What our engine changes contain

This note breaks inkserve's changes to erisinky into parts that stand mostly on their own. It is for anyone who wants to review the patch, rebase it, or offer pieces upstream.

In this repository the engine changes are one commit on top of erisinky's `gpl-license` branch and upstream pull request #2. It adds about 2,460 lines and removes 106, in seven files. One of them is the new `src/serve.rs` (476 lines), the `inkling serve` command.

Line counts below are approximate `+` lines in the patch.

## The parts at a glance

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

## 1. Native expert formats (accuracy)

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
- Outside the patch, `cmd/inkconvert` writes these files.

## 2. Q8_0 for the non-expert tensors (accuracy)

The `native-q8rest-o6` file keeps attention and the token embedding at Q8_0, and the output head at Q6_K. Stock erisinky expects Q4_K for those tensors.
- `Engine::load` accepts Q8_0 (and Q6_K) for attention, `token_embd` and `output`.
- `k_embed` reads Q8_0 rows.
- `k_mv` gets a `dot_q8` path. Its `sumf` now starts as `{}`, which zeroes any row count.

## 3. Decode kernel tuning (speed)

This tuning applies to the kernels of parts 1 and 2, so it has no meaning without them. It lives in the same functions.
- **`dot_q8`:** llama.cpp's layout, with 4 lanes per block and 8 blocks per simdgroup pass. All lanes stay busy at K = 2304 and 3072.
- **`dot_nv4` (`NV4_BAL`):** 4 lanes per block, 4 code bytes each.
- **NV4 / MX8 expert down:** 8 (expert, block) items per pass, so TOPK × 36 blocks = 288 split evenly.
- **Decoding without tables**, as in part 1.

Measured effect: `native-q8rest-o6` went from 8–12% to 6–8% slower than Q4_K_M (NATIVE.md → Decode layout).

## 4. GPU sampling (feature + speed)

Stock erisinky's chained decoding is greedy. This part draws each token on the GPU, so sampled decoding keeps the chained speed. See ARCHITECTURE.md → Sampling.
- **`SampArgs`** and **`Engine::samp`:** when set, a step's argmax becomes a draw.
- **`k_gumbel`:** adds Gumbel noise to softcap(logit) / T. The noise is a hash of (seed, position, token), and the argmax of logit plus noise is an exact softmax draw.
- **`k_sampstate`:** the thinking budget's state machine, in three spare token slots past the context.
- **`sample_tail`** (the decode step's tail) and **`encode_sample`** (a draw from the last prefill's logits).

## 5. The serve protocol (feature)

`src/serve.rs` is a new file. It adds `inkling serve`, a long-lived engine that inkserve drives over stdin and stdout. See ARCHITECTURE.md → Engine protocol.
- the line protocol (`G` / `X` in; `R` / `P` / `T` / `E` / `F` out)
- **Prefix cache:** reuse the live context, or a snapshot taken two tokens before each prompt's end
- the thinking budget
- **Two sampling paths:** `gen_chained` (part 4) and a CPU sampler for top-k / top-p (`Pick`)
- cancel, and the token and context limits
- the DSpark path, `gen_dspark` (part 7)
- In `main.rs`, the `serve` command and its `--ref` (drafter) argument.

## 6. Small-batch verify path (speed infrastructure)

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

## 7. DSpark speculative decoding (feature)

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

## 8. Dev tools (tooling)

These are commands in `main.rs` and one engine method, all marked "dev". None runs in serving.
- `prof_verify`: verify timing, agreement with decode steps, and a breakdown by kernel class
- `kcheck` (`Engine::kcheck`): each small-T kernel against the original on random inputs
- `tapcheck`: taps from prefill against taps from decode steps
- `dcheck`: drafter acceptance on a transcript
- `spectest`: DSpark against plain decoding, with speeds
- **Environment switches:** `INK_NO_MOET`, `INK_NO_MVT` and `INK_NO_ATTNT` turn off one small-T kernel family each.

## 9. Bug fixes

Few of our changes fix upstream code. Most of the fixes below are to our own additions, and they ship inside those parts.
- **Upstream: one-token prefill.** At `3890bd5` a one-token prefill on a fresh batch arena gives wrong logits. Upstream pull request [Eriskii/erisinky#2](https://github.com/Eriskii/erisinky/pull/2) fixes it, and this repository merges it. `serve.rs` `feed()` still feeds one token through a decode step, which is cheaper.
- **Upstream: `INK_OLD_MIXB` kept.** This upstream switch sends batched passes through the decode `k_mix` kernel, which doesn't write the half copies the fp16 GEMMs read. Setting it breaks verify passes, and probably fp16 prefill too (not checked).
- **Ours, part 4:** the Gumbel noise's uniform could round to 1.0 and make a token's noise infinite (~1% of steps). Fixed with 23 random bits plus a half step.
- **Ours, part 6:** with a batch arena under 128 rows, verify passes silently fell back to the prefill MoE (6 ms slower). `encode_verify` now asserts the size.
- **Ours, part 7:** after a snapshot restore, stale taps from an earlier reply fed the drafter. That inflated acceptance, but it never affected correctness. Fixed with `tap_hi`.

## Outside the engine

The Go code is a separate program, in `inkserve/`.
- **inkserve** (`*.go`, ~1,500 lines):
  - the OpenAI-style HTTP API
  - the chat template, checked byte for byte against `testdata/golden.json`
  - tool-call parsing
  - effort, thinking budget and sampling defaults
  - `-dspark`
- **inkconvert** (`cmd/inkconvert`, ~1,700 lines): builds the GGUFs of parts 1 and 2 (`convert`, `probe`, `verify`, `q6check`) and the drafter file (`mtp`).

## Offering parts upstream

Parts 1, 2 and 4 are self-contained and useful to anyone running erisinky on the release's weights. Part 3 rides along with 1 and 2. Parts 6 and 7 are larger and tied to speculative decoding. They make sense upstream only if erisinky wants DSpark or batched decoding. inky's "lanes" work overlaps part 6. Nothing can be offered until the license question is settled (TODO.md → Blocking).
