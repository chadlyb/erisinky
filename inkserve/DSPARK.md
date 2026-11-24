# DSpark speculative decoding: design notes

Status: **built and deployed, opt-in** (2026-11-23). `inkserve -dspark Inkling-2-Mini-dspark.gguf` turns it on; it's off by default. On agent replies it gains about 3% (0.95–1.15× by prompt; Serving modes → DSpark in the engine's serve loop). This note covers the design, the build and the measurements of speculative decoding with Inkling's bundled DSpark drafter in erisinky.

## Why it matters

On real agent turns at `medium` effort, omlx with DSpark decoded 61.8 tok/s, and inkserve's previous build 63.5, both counting prefill (`notes/RESULTS.md` → Where agent time goes). On `serverbench.py` prompts, inkserve looked 1.35–1.45× faster. The gap is DSpark: its drafts are accepted far more often on agent code than on the bench (omlx: 2.70 tokens per cycle at depth 3 on an agent task). inkserve's GPU-sampling build is faster again, but DSpark is the one large lever omlx has and inkserve doesn't.

## The drafter

It ships as `mtp.safetensors` next to the model, described by `mtp_config` in `config.json`. omlx's implementation is `omlx/patches/inkling2/dspark.py` (Apache-2.0) plus its generic DFlash loop.

| Part | Shape |
|---|---|
| Layers | 3 dense transformer layers, hidden 2304, 24 query / 8 K/V heads × 128, MLP 9216 (SiLU-gated), RoPE θ = 1e6 |
| Attention | over the target's context (its own `k_proj_ctx` / `v_proj_ctx` and `k_norm_ctx`) and over the block, both directions. Every layer sees only the newest 1,023 context positions |
| Context taps | the target's hidden state at layer visits 5, 23, 41 and 48 (counting each pass through the looped core, so 48 = the final norm's input). Each draft layer mixes the 4 taps with its own softmax weights (`fuse_weight`), RMS-normalizes, and projects them to context K/V |
| Block | the anchor (last accepted) token plus `block_size − 1` = 7 copies of `mask_embed`. Only the masks propose |
| Proposals | the target's LM head on the draft's output, plus a per-step bias from an RNN "Markov head" (rank 256). The bias reads a `[V, 256]` matrix, the LM head folded into `hidden_out` once at load. The RNN makes proposals sequential |
| Size | 318M parameters (636 MB BF16; omlx runs the layers as 8-bit affine) |

omlx proposes 3 tokens per cycle (`mtp_fixed_depth: 3`), so its verify pass is 4 tokens long.

## Drafter spec (from omlx 0.8.2, read 2026-11-23)

Sources: `omlx/patches/inkling2/dspark.py` (the model), `omlx/speculative/dflash/drafter.py` (`_draft_batched`: the forward pass), `variants.py`, `selection.py` (`select_markov`), and `omlx/patches/inkling2/model.py` (the taps). omlx is Apache-2.0. Config, from `mtp_config` in the model's `config.json`: hidden 2304, 3 layers, 24 query heads and 8 K/V heads × 128, MLP 9216, RMSNorm eps 1e-6, RoPE θ 1e6 (not traditional), block 8, Markov rank 256, window 1024, taps `[5, 23, 41, 48]`.

**Taps** (the target's features, per position): for ids 5, 23 and 41, the partial block sum `h` *entering* layer visit i. Visits count every pass through the looped core:
- 0–5: prelude
- 6–17, 18–29, 30–41: the core's three passes
- 42–47: coda

In erisinky that's `cur` just before the i-th `layer()` / `layer_b()` call. Id 48 is the final-norm output: the output head's input (`fout`, or the verify pass's half rows).

**Context K/V, per draft layer l, kept in a ring of the newest 1,023 committed positions.**
1. α_l = softmax(`fuse_weight[l]`) over the 4 taps.
2. mixed = RMSNorm(Σ α_l[i] · tap_i, `layers.l.dflare_norm`), computed in fp32.
3. k = `k_norm_ctx`(`k_proj_ctx`(mixed)) per head, then RoPE at the position's absolute index.
4. v = `v_proj_ctx`(mixed).

Every layer, the full-attention one included, attends only the newest 1,023 context positions.

**One draft pass.** The block is the anchor (the newest committed token, not yet run through the target) plus 7 masks, at positions n … n+7, where n = the number of committed context positions.
1. Embed the block:
   - The anchor gets the target's embedding plus its embedding RMSNorm (erisinky's `k_embed`).
   - The masks get the raw `mask_embed` vector.
2. Per layer:
   - x = RMSNorm(h, `input_layernorm`).
   - q = `q_norm`(`q_proj` x); block k = `k_norm`(`k_proj` x); block v = `v_proj` x. RoPE q and block k at positions n + i.
   - Attention, scaled by 1/√128, with no sink and no gate. Every block row sees the context ring and all 8 block rows, in both directions.
   - h += `o_proj`(attention).
   - h += `down`(SiLU(`gate` x′) · `up` x′), with x′ = RMSNorm(h, `post_attention_layernorm`).
3. The draft hidden states are RMSNorm(h, `norm`) at the 7 mask positions. Their base logits come from the **target's** output head, applied directly, without the target's final norm.

**Markov head, sequential, one step per proposal.** It's seeded with:
- rows_f = RMSNorm(Σ α_3[i] · tap_i of the newest committed position, `markov_head.dflare_norm`)
- state = 0
- prev = anchor

Step s:
1. in = [state, `markov_w1`[prev], hidden_s, rows_f], 2 × 256 + 2 × 2304 = 5,120 wide.
2. p = `rnn_proj`(in): 768 outputs, with bias.
3. gate = σ(p[0:256]); state = gate · state + (1 − gate) · tanh(p[256:512]); latent = tanh(p[512:768]).
4. logits_s = softcap(base_s + W_f · latent + b_f).
5. Choose proposal s from logits_s; it becomes prev.

W_f = output head · `hidden_out.weight` ([V, 256]) and b_f = output head · `hidden_out.bias`, folded once at load.

**Proposals with coupled noise** (our choice; omlx samples drafts and accepts by probability ratio instead):
- Proposal s is argmax(logits_s / T + G(seed, n + 1 + s, token)), using the same Gumbel noise `k_gumbel` gives the target at that position.
- The verify pass draws the target's tokens with the same noise. A draft is accepted while it equals the target's draw, and the first mismatch takes the target's token.
- So the output is exactly plain decoding's for the seed, and shared noise makes the two draws agree as often as their distributions allow.

**Per-cycle cost** [E]:
- 3 layers × 87M parameters: ~1 ms in BF16, ~0.5 ms at 8 bits.
- The output head over the proposals used: one `k_mvT` pass, ~0.7 ms.
- 3 Markov steps, each reading W_f (100 MB in BF16): ~0.55 ms.
- Projecting the newly committed positions into the context rings: ~0.05 ms.

## Build log

- **Taps (2026-11-23, tuning copy `~/erisinky-eval/tune`):**
  - A tap ring in the decode arena (`ar.taps`): [1024 positions, by position mod 1024][4 features][E], in f32, because hidden states can exceed half's range.
  - `k_tapcopy` fills it before layer visits 5, 23 and 41, and after the final norm, in decode steps, prefill and verify passes. It's on while `Engine::taps_on` is set, for positions ≥ `tap_from`.
  - A tapped prefill runs the final-norm mix over every row and turns the last-layer tail trim off.
  - **Checks** (`inkling tapcheck`):
    - Two prefills split at 512 and at 96 tokens give identical taps, bit for bit. Prefill doesn't depend on the split, so the row-to-position mapping is right.
    - Prefill against decode steps: feature 0 matches to 2–3 × 10⁻⁴. The deeper features differ by 0.6–15%, growing with depth. That's the model amplifying rounding through its looped core (the same effect behind the 6–8% top-1 gap between the two paths), and it affects acceptance, not correctness.

- **Drafter weights (2026-11-23):** `inkconvert mtp -in mtp.safetensors -official model-00001-of-00001.safetensors -out Inkling-2-Mini-dspark.gguf` (`cmd/inkconvert/mtp.go`). It took 4.8 s on the M5 Pro and wrote 445 MB (`/Volumes/TB5/llm/inkling-speed/Inkling-2-Mini-dspark.gguf`). Tensors, named `mtp.<layer>.<name>` and `mtp.markov.*`:
  - Q8_0: the 9 projections per layer, and the folded head `fold_w` [201,024 × 256]
  - BF16: the Markov embedding `w1` and `rnn`
  - F32: norms, `mask_embed`, biases, and `alpha` = softmax(`fuse_weight`), 4 × 4
  - The fold W_f = unembed · `hidden_out` uses the release's BF16 output head. It took 4 s on 18 cores.

- **The drafter in erisinky (2026-11-23, tuning copy):** `src/engine/dspark.rs`, a child module of `engine`, with its kernels appended to `prefill.metal`.
  - `Drafter::load(gpu, path)` maps the drafter GGUF.
  - `Engine::draft_reset`, `encode_draft_commit(d, cb, p0, n)` (taps into the 3 context rings) and `encode_draft(d, cb, n, depth, samp)`, which writes proposals into toks[n + 1 ..= n + depth], where `encode_verify` reads them.
  - The block pass uses `k_mvT` for every projection. The output head runs over the proposal rows only, and the Markov steps run sequentially on the GPU, with no CPU round trip.
  - **Acceptance** (`inkling dcheck`, greedy, inside the model's own messages of two agent transcripts): **2.74 and 2.83 tokens per cycle** at depth 3 (74–87% per depth), matching omlx's 2.84 over its whole eval. Over whole transcripts, tool output included: 1.67. Raw: `notes/raw/dspark-20261123/verify-path-progress.txt`.

- **Rollback and the first end-to-end runs (2026-11-23, tuning copy):**
  - Rollback pieces:
    - In verify passes, `k_state_upd` records [old state, the T inputs] per conv channel (`ar.vhist`).
    - `Engine::swa_save` / `verify_rollback(saved, p, t, keep)` rebuild the conv state for the kept prefix and put back the window rows of rejected positions, on the CPU between cycles (0.1–0.2 ms).
    - `encode_verify_sample` draws the target's token at each verified position.
  - **`inkling spectest`** generates with DSpark and with plain chained decoding from the same prompt and seed, then compares tokens and speed.
  - Results (coupled-Gumbel acceptance, depth 3, raw in `…/raw/dspark-20261123/verify-path-progress.txt`):
    - **~11K, greedy: 1.31×**, identical to plain decoding over all 300 tokens. So rollback keeps the state exact.
    - ~38K: 1.16–1.17× at temperature 0 and 1.
    - ~1K, greedy: 1.05×.
    - **~11K at temperature 1.0: 0.99×** (2.26 tokens per cycle).
  - Per cycle at ~1K: draft 2.49 ms, verify 23.55 ms, CPU 0.2 ms, wall 27.1 ms.
  - **Next:** replace coupled-Gumbel acceptance with standard speculative sampling: accept with probability min(1, p/q), and on rejection sample the residual (p − q)⁺. It accepts the most possible (omlx gets 2.84 tokens per cycle at temperature 1.0 with it). It's exact in distribution, though no longer token-identical to plain decoding for a seed.

- **Standard speculative sampling (2026-11-23):** `Engine::encode_verify_spec`, which uses the kernels `k_slse` / `k_sresid` / `k_sdecide`. The drafter keeps its pre-noise logits (`qlog`) for the test. `inkling spectest` uses it when sampling; `INK_SPEC_COUPLED=1` switches back to the coupled test.
  - **Temperature 1.0, five seeds on the 11K prompt:** median 2.34 tokens per cycle, against 2.19 coupled. It's about even and more consistent, so the rule wasn't the limit there. The content was: open-ended reasoning over a code dump.
  - **Agent transcripts continued at temperature 1.0 (~3K context, 6 runs):** 2.17–3.21 tokens per cycle, mean 2.53, and **0.90–1.29×, ~1.03× on average**.
  - At ~3K, a cycle (~27–30 ms) needs ~2.3 tokens to beat plain decoding (12.1 ms per token). Long context does better (1.15–1.3× at 38K).
  - **So:**
    - an adaptive policy is needed, falling back to plain decoding when running acceptance or context says DSpark won't pay
    - and the verify pass has to get cheaper for a broad win

- **Cheaper verify pass (2026-11-23):** at 4 tokens and ~2.4K context, 23.65 → 21.72 ms. Agreement with decode steps holds: top-1 107/108 and 92/92, mean KL ≤ 0.0073.
  - `k_mix_bw`: the residual mixes run one 256-thread threadgroup per token in verify passes. `k_mix_b` gave the whole pass one threadgroup. Saves 1.4 ms.
  - `k_moe_downT` splits each row group's expert blocks across 4 simdgroups (`DNT_KS`). Saves 0.5 ms.
  - Expert gate/up and down now take ~7.2 and ~6.3 ms. Both look bandwidth-bound on the union of the tokens' experts, so little is left there.
  - End to end (spectest, depth 3, 300 tokens):

    | Prompt | Temp | Before | After | Tokens/cycle |
    |---|---|---|---|---|
    | ~11K | 0 | 1.31× | **1.39×** | 3.06 |
    | ~11K | 1.0 | 0.99× | **1.10×** | 2.44 |
    | ~38K | 0 | 1.17× | **1.24×** | 2.40 |
    | ~38K | 1.0 | 1.16× | **1.21×** | 2.36 |

## Serving modes

Latency mode exists as the `-dspark` flag (below). Throughput mode is still planned.

inkserve gets one server flag:
- **`-mode latency`** (default): one conversation, DSpark on. Each task gets faster.
- **`-mode throughput`**: several conversation slots, each with its own K/V cache and recurrent state, so one agent decodes while another waits on tools. Requests are served one at a time at first; batched decoding of several slots comes later.

The modes share their foundation, so neither is wasted:
- the small-T pass (DSpark verifies 4 tokens of one stream; batching decodes 1 token from each of several)
- state save and restore (draft rollback, slot switching)
- a serve loop that plans each step

The flag is per server, since slots reserve memory at startup (~1.3 GB per 40K-token slot). DSpark can also be switched off per request for debugging.

### DSpark in the engine's serve loop (deployed 2026-11-23)

`inkling serve --ref Inkling-2-Mini-dspark.gguf` turns DSpark on. The default sampling path (no top-k or top-p) then runs `gen_dspark` in `serve.rs` instead of `gen_chained`. `INKSERVE_DSPARK_DEPTH` sets the draft depth (default 3).
- **Taps:** each prompt's prefill captures taps for its last 1,024 positions (`tap_from`). Prefill results stay bit-identical: proto_test's logit hashes match a run without the drafter.
- **Drafter context:** each request resets the drafter and projects the last 1,023 positions' taps into its rings.
  - Only positions whose tap-ring slot still holds their own tap count. A reply that ran past the current prompt's end (the snapshot path) overwrote some slots. The serve loop tracks `tap_hi`, the highest position whose tap was written, and starts the window at `tap_hi − 1022`.
  - Without that check, a repeated request saw the previous reply's taps as "context" and accepted unrealistically many drafts (3.48 tokens per cycle against 2.1).
- **Batch arena:** verify passes need `ensure_batch(VERIFY_BATCH)` (128 rows). The small-T MoE keeps its expert list in the arena, and with fewer rows it silently fell back to the prefill MoE, costing 6 ms more per cycle (32.7 against 26.4 ms). `encode_verify` now asserts the size.
- **Fall-back to plain decoding:** a cycle pays off only if it keeps more tokens than plain steps would make in its time (~2.4 at 3K).
  - Generation starts with 16 plain steps, to time a step.
  - After that, DSpark cycles run. When the last 4 cycles' tokens times the step time fall below those cycles' time, it goes back to plain steps: 64 tokens, doubling up to 256 while it keeps losing.
  - The first cycle after a stretch also projects the stretch's taps, so it is left out of the window.
  - `INKSERVE_DSPARK_ALWAYS=1` turns the fall-back off.
- **Speed through the serve loop** (`serve_bench.py`: a cold request after a warm-up, 300 tokens, two seeds; raw output in `…/raw/dspark-20261123/verify-path-progress.txt`):

  | Prompt | Context | Temp | Fall-back on | DSpark always |
  |---|---|---|---|---|
  | dagrun transcript | 3.0K | 0 | 1.12× | 1.24× |
  | dagrun transcript | 3.0K | 1.0 | 1.04–1.08× | 1.05–1.08× |
  | orderedpool transcript | 2.6K | 0 | 0.95× | 0.89× |
  | orderedpool transcript | 2.6K | 1.0 | 0.96–0.97× | 0.95–0.96× |
  | p10k (user message) | 11K | 0 | 0.97–0.98× | 1.05× |
  | p10k (user message) | 11K | 1.0 | 1.02–1.03× | 0.97–1.03× |
  | p32k (user message) | 38K | 0 | 1.08× | 0.87× |
  | p32k (user message) | 38K | 1.0 | 0.99–1.03× | 1.22–1.35× |

  - **So far DSpark is roughly a wash on this content:** about 1.03× with the fall-back (worst 0.95×), and about 1.05× without it (0.87–1.35×).
  - Acceptance here is 2.0–2.5 tokens per cycle. A 4-token cycle costs ~2.3 plain steps at 3K and ~1.9 at 38K, so most cycles sit near break-even.
  - The higher numbers from `spectest` (1.2–1.4×) came from prompts in the first-turn template, with a system message. The drafter accepts more on that kind of text.
- **Depth per cycle (replaces the window fall-back):**
  - The loop keeps the running acceptance of draft j, given that the drafts before it were accepted. A cycle's cost is fixed in plain steps: 1.64 / 1.88 / 2.22 for depth 1 / 2 / 3 (prof_verify at ~2.4K).
  - Each cycle takes the depth with the most expected tokens per plain step. When no depth beats a plain step after 8 cycles, it runs a stretch of plain steps.
  - The costs are fixed rather than timed, so a seed always gives the same reply. A first version timed the cycles, and `sample_test.py` caught timing noise changing the reply. The results below come from that timed version.
  - `INKSERVE_DSPARK_DEPTH` is now the maximum depth.
  - Results: 0.95–1.15× per row, **mean ~1.03×**, about the same as the window fall-back.
    - Middling acceptance mostly picks depth 2. At 38K every cycle chose depth 2.
    - Best rows: dagrun 1.11–1.14×, p32k 1.09–1.15×.
  - `sample_test.py` under DSpark: the temperature-0.01 checks now ask for 200 matching tokens. Different depths mean different verify batch sizes, so the known near-tie at token 286 can flip.
- **Where this leaves DSpark:** on agent-style content it gains a few percent at best. Two limits:
  - The drafter keeps 1.7–2.8 tokens per cycle here, against 2.84 in omlx's eval.
  - A verify pass reads the union of the tokens' experts, which is bandwidth-bound: at 4 tokens it costs ~2× a decode step.
  - A larger win needs better acceptance (a drafter trained on agent transcripts) or a cheaper verify. The multi-slot throughput mode doesn't depend on either.
- **The anchor:** the last streamed token, not yet fed through the model. Each cycle verifies anchor + drafts, and the target's next token becomes the new anchor.
- **The CPU walks each cycle's kept tokens in order.** It applies what the GPU sampler state does in plain decoding:
  - the thinking budget (force `<|end_message|>`, then ban `<|content_thinking|>`)
  - eos
  - the token limit and the context limit
  - A token it changes or stops at cuts the cycle there, and `verify_rollback` drops the rest. Afterwards it writes the sampler state slots, so a plain step could take over.
- **At the end,** one decode step feeds the last anchor (the stop token included), as `gen_chained` does. So the live prefix-cache path sees the same state.
- **Checks:**
  - `proto_test.py` passes all 15 with `INKSERVE_TEST_REF` set (prefix cache paths, thinking budget, prefill hashes).
  - `sample_test.py` passes, except greedy against the CPU sampler: the texts part at token 286 of 300. The verify pass's batched logits differ slightly from a decode step's, so a near-tie can flip. With DSpark the test now asks for 100 matching tokens.

## What erisinky needs

1. **The drafter as Metal kernels.** Dense BF16 or Q8_0 layers: erisinky's matvec and attention pieces mostly fit. One new piece is block attention over [context K/V | block] in both directions. Context K/V must be kept per draft layer, for 1,023 positions, and appended as tokens are accepted. That needs the four tap vectors per token from the target step, so the step has to write them out.
2. **A small-batch verify path. This is the hard part.** Verifying T = 4 tokens needs one pass that reads each dense weight once for all T tokens, and each routed expert once per group of tokens that chose it.
   - The one-token decode step costs ~11 ms.
   - erisinky's batched prefill path is tuned for large chunks. It has a large fixed cost: today's bench prefilled 30–170 new tokens in 0.15–0.2 s, and the inky fork measured "58 ms + 0.21 ms per token" on its hardware.
   - So verifying through the prefill path would cost more than T decode steps.
   - inky's roadmap lists the same kernel ("a kernel for decode-size batches") for its lanes. The two efforts overlap.
3. **Rollback of rejected drafts.**
   - Full-attention K/V: nothing to do. Positions past the accepted ones get overwritten later.
   - Sliding-window rings (1,024 keys each): a write at position p + j overwrites key p + j − 1024, which later queries still need when j ≥ accepted + 2. So save those ring rows before the verify: ≤ 3 rows × K/V × 8 heads × 128 × 40 layer slots ≈ 0.6 MB, one blit.
   - Convolution state (3 taps per channel, on K, V and both residual branches): the state after the last accepted token is the last 3 per-token inputs. The verify pass must keep them, as omlx does (`_omlx_verify_xp` in its `model.py`).
4. **Sampling.** Accept a draft token if it matches the target's own draw at that position. With the GPU sampler's Gumbel noise keyed by (seed, position, token), the target's draw at each position is fixed. So greedy-style matching gives exact sampling: the output is what non-speculative decoding with the same seed would give. The thinking budget's state machine must then step through accepted tokens only.

## Measurements (2026-11-23)

Raw data and method: `notes/raw/dspark-20261123/`.

**The existing batched path can't verify.** Over T real transcript tokens at position 2,400 (`inkling prof_verify`, a dev command in the tuning copy):

| T | batched pass | T decode steps | ratio |
|---|---|---|---|
| 2 | 37–39 ms | 22.8 ms | 1.7 |
| 4 | 43–45 ms | 45.5 ms | 0.94–0.99 |
| 8 | 53–54 ms | 91 ms | 0.58 |

It has a fixed cost of about 35 ms, so a 4-token verify costs what 4 plain steps do. A verify path built for 2–8 tokens is required.

**Consecutive tokens share fewer experts than hoped.** On 400 generated tokens each from two agent transcripts (`inkling routes`, union per layer execution, averaged over all 48):

| T | distinct experts per layer | expert reads vs one token |
|---|---|---|
| 2 | 12.7–13.0 | 1.6× |
| 3 | 16.4–17.0 | 2.1× |
| 4 | 19.6–20.4 | 2.5× |
| 8 | 29–30 | 3.7× |

## Acceptance and the measured estimate (2026-11-23)

omlx logs DSpark statistics per request. Over its whole tuned `medium` eval (158 agent requests, 190K tokens, depth 3, temperature 1.0; `notes/raw/dspark-20261123/omlx-acceptance.txt`):

| | `medium` | `high` checks |
|---|---|---|
| tokens per cycle | **2.84** | 2.69 |
| draft acceptance | 77.6% (77 / 77 / 79% at depths 1 / 2 / 3, each given the one before) | 73.8% |
| drafter time per cycle | 2.70 ms | 2.70 ms |

omlx's target is oQ4e, ours the native file. Both sit near the release, which the drafter was trained against, so the acceptance should carry over closely [E].

With the measured verify cost, per token:

| draft depth (verify T) | ~2.4K | ~30K |
|---|---|---|
| 1 (2) | 1.04× | 1.12× |
| 2 (3) | 1.21× | 1.33× |
| **3 (4)** | **1.23×** | **1.40×** |
| 5 (6) | 1.08× | 1.32× |
| 7 (8) | 0.96× | 1.19× |

- **Depth 3 is best:** ~1.3× decode on typical agent turns.
- Decode is about half of agent time on the Go tasks, so that's **~12% off agent time** [E].
- Recovering the verify path's remaining ~2–3 ms at T = 4 would add ~0.1×.

The earlier, pre-measurement estimate follows for the record.

## Expected gain [estimate, before the measurements]

Per cycle at depth 3 (verify 4 tokens), from the decode profile at ~1.2K context. One decode step takes 11.2 ms: experts 4.2 ms, attention projections 2.9 ms, output head 0.6 ms, attention 1.2 ms, small latency-bound kernels ~2.3 ms.

| Part | Cost |
|---|---|
| experts: 4.2 ms × 2.5 (measured union) | ~10.5 ms |
| attention projections and output head, read once for 4 tokens | ~3.6 ms |
| attention: 4 queries over one cache, read once | ~1.5 ms |
| small kernels | ~2.5 ms |
| **verify, 4 tokens** | **~18 ms** |
| drafter: 3 layers (~1 ms at 8 bits), output head over 7 positions (~0.7 ms), 3 Markov steps (~0.5 ms) | ~2–2.5 ms |
| **cycle, for ~2.7 tokens** (omlx's acceptance on agent code, bonus token included) | **~20.5 ms → 7.6 ms per token, ~1.45×** |

- At ~30K context, attention (~5 ms per step) is shared by the 4 queries, so the ratio rises to ~1.6× [E].
- Depth 2 (verify 3 tokens, expert union 2.1×) comes out a little lower, ~1.4×, if acceptance falls to ~2.3 tokens per cycle.
- **Realistic: 1.25–1.4×,** allowing for kernel inefficiency at small T, and lower acceptance on prose than on code.
- Decode is about half of agent time on the Go tasks, so agent time would fall by roughly **10–15%**.

## The verify path (2026-11-23)

**API** (in `src/engine.rs`):
- `Engine::encode_verify(cb, p0, t)`, for 2 ≤ t ≤ 8, encodes one pass over `toks[p0..p0+t]`.
- `Engine::verify_logits(i)` then gives position i's logits.

The small-T kernels run only while `encode_verify` is encoding, so prefill is unchanged. That's checked: prefill logit hashes are identical to the previous build, and `proto_test.py` passes.

Every position gets logits: one batched output mix into half rows, then `k_mvT` over the output head (Q6_K included). Dev commands (`inkling prof_verify`, `inkling kcheck`) measure and check it.

It keeps the batched path's structure and, for T ≤ 8, swaps in small-T kernels class by class:

- **`k_mvT`:** the decode matvec's layout (8 rows per threadgroup) with T tokens per weight read. It's used for the attention projections, the router and the loop controllers (Q8_0 and BF16). The batched path's 64-row GEMM tiles left too few threadgroups at small T.
- **`k_moe_routeT` / `k_moe_guT` / `k_moe_downT`:** the union of the T tokens' experts. Each expert is read once in the decode kernels' layout (NV4 and MX8) and applied to every token that chose it. This replaces the gather, the tiled GEMMs and the combine.

- **`k_attn_hs` / `k_attn_hs_comb`:** the batched fp16 attention (`k_attn_h`, tensor units) split over ~10 key ranges, as the decode path splits. The original ran one threadgroup per head over every key. The kernel is derived from `k_attn_h`'s own source, so scores, bias and masks are unchanged.

With logits for every position:

| T | original batched path (last logits only) | verify, ~2.4K | T decode steps, ~2.4K | verify, 30K | T decode steps, 30K |
|---|---|---|---|---|---|
| 2 | 37.5 ms | 17.1 | 22.7 | 21.3 | 29.8 |
| 4 | 42.7 | **23.5** | 45.5 | **27.7** | 59.6 |
| 8 | 53.2 | 42.6 | 91.1 | 45.3 | 119.3 |

**Agreement:** at every position of each span, the verify logits pick the same top token as the decode steps:
- 108/108 and 91/92 at T = 4 on the two ~3K transcripts
- 32/32 at T = 4 on the 38K bench prompt
- mean KL ≤ 0.006

**Per-layer check** (`inkling kcheck`, in the tuning copy): each new kernel against the original on identical random inputs, one layer at a time. The relative error is 2–4 × 10⁻⁴ (half-precision rounding; the original keeps half intermediates), and 5 × 10⁻⁷ for the BF16 matvec. So the kernels are right, and the rare single-position logit outliers come from the model amplifying rounding differences.

**DSpark estimate with the verify path as it is** [E]: a cycle is verify(4) + drafter (~2.5 ms), for ~2.7 tokens. That's ~1.16× decode at ~2.4K context and ~1.34× at 30K, where the 4 queries share one attention read; ~1.25× on typical agent turns.

- **Agreement with T decode steps:** at every 50th position of two transcripts, top-1 agreement matches the original path's.
- **Single-position outliers** (KL 0.5–2.7) show up on every variant, the original included. That fits this model's sensitivity to rounding, but a per-layer check on identical inputs should confirm it.
- Raw: `notes/raw/dspark-20261123/verify-path-progress.txt`.

**Left on the verify path** (worth ~2–3 ms at T = 4 together):

- **Expert down:** ~6 ms net, against ~4.2 at decode efficiency. Sharing an expert across tokens costs ~1.6 activation loads and dots per weight block. 4 rows per simdgroup is the best of 2 / 4 / 8; more needs a different design.
- **`k_moe_routeT`:** ~0.5 ms of fixed dispatch cost per pass.
- **Remaining overhead:** the mixes, kvprep and the post kernels (~2 ms above their decode cost).

## Order of work

1. ~~Measure the expert union for consecutive tokens and the batched path's cost at T = 2–8.~~ Done (above).
2. ~~Build the small-batch verify path and check it against T decode steps.~~ Done (above).
3. The drafter, with per-layer context K/V; check its acceptance against omlx's on the same prompts.
4. Rollback and the loop in `serve.rs`. Test against non-speculative output with the same seed.

Each step is a few days of focused engine work; steps 1–2 alone also pay off for running several streams at once.
