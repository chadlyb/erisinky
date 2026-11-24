# TODO

## Blocking

- **License.** This repository builds on erisinky's commit `d1d5263` ("License under GPL-3.0-or-later"). It sits on the upstream branch `gpl-license` and in upstream pull request #1, not yet merged into `main` (checked 2026-11-23). The inky fork carries the same license on its `main`. Before pushing, confirm the license is on upstream `main`, or ask the author whether the branch is final. Our changes and `inkserve/` are GPL-3.0-or-later too (the root README's "This fork" section).

## Next

- **Current build eval'd on 6 of the 12 Go tasks** (2026-11-23, `medium`): 6/6, ~1.2× omlx per token on real agent turns (notes/RESULTS.md → `custom-inkling-o6gpu-medium-r1`). The other 6 tasks are still to run, if a full comparison is wanted.
- **Optional: eval `attn-q6-out-q6`** (+6.4% decode, drift at omlx oQ4e's level) if that speed is worth the drift.
- **Attention at long context.** At 30K, attention takes ~5 ms of a 15 ms step. It reads the K/V cache well below the weights' rate (inky's notes, measured on other hardware: ~210 vs ~565 GB/s). A faster split-K kernel is the largest long-context lever; the 8-bit cache (`--kv8`, not exact) is the cheap one, and needs a drift check.
- **More decode speed for the 8-bit file.** Decode kernels are tuned (NATIVE.md → Decode layout): `native-q8rest-o6` now decodes 6–8% below Q4_K_M, down from 8–12%. What's left:
  - the small, latency-bound kernels (loop controllers, router, mixes, combine): ~2.3 ms per step for ~0.3 ms of data. Fuse them.
  - the fixed ~4 µs per dispatch (about 0.8 ms per token across the four matvec classes): fuse or overlap dispatches
  - the NV4 down projection reaches only ~370 GB/s
  - try a 16-entry LUT for FP4 against the bit trick; a 32-lanes-per-superblock `dot_q4k` (it has the same ragged tail at K = 2304)
  - check once on the GPU that subnormal halves (FP4 ±0.5, small E4M3 codes) aren't flushed to zero; drift at the noise floor suggests they aren't
- **Take inky's two engine fixes** (robotgamedevelopers/inky, GPL-3.0-or-later like erisinky):
  - `715c0b6`: `mix_t` always uses the batch mix kernel, which fixes the one-token prefill. Then `feed()` can drop its workaround.
  - `79c5157`: zero the sliding-window K/V scratch at the start of each prefill (`k_zero`), since a zero attention weight doesn't cancel a NaN left there.
- **Drift past 3K tokens**, for `native-q8rest-o6` (needs a reference tool with a larger context; erisinky's README tests only to ~16K).
- **Running the eval again:** relay alias `inkling-erisinky`; restart the relay with `-upstream http://127.0.0.1:8045` only when no eval is running, then put it back on 8044 (omlx).
- **m5relay alias for `inkling-erisinky-dspark`**, if the eval should reach the DSpark entry (m5ctl has both entries; README → Under m5ctl).

## Later

- **DSpark is in, opt-in** (`-dspark`, off by default; [DSPARK.md](DSPARK.md)): about 1.03× on agent replies (0.95–1.15×). Levers left:
  - acceptance: the drafter keeps 1.7–2.8 tokens per cycle on agent text. A drafter fine-tuned on agent transcripts would help most.
  - verify cost: a 4-token pass reads the union of its tokens' experts, ~2× a decode step. That's bandwidth, so kernel tuning has little left.
  - an end-to-end agent eval with `-dspark`, to see the gain on real turns (the serve benchmark used 300-token continuations)
- **Throughput mode** (DSPARK.md → Serving modes): several conversation slots, so agents waiting on tools don't leave the GPU idle. With DSpark's gain small, this is the bigger lever for parallel agent work.
- **Several conversations at once.** inky ("lanes": one engine, a K/V cache and state per stream) is building this. Worth reusing if parallel agent tasks matter; today one snapshot serves one conversation.
- Prefix match on tokens that differ only by retokenization: the live path needs the client's re-render to tokenize exactly as generated. It falls back to the snapshot otherwise.
- Top-k / top-p on the GPU, so those requests chain too (today they take the CPU path).

## Known issues

- Requests with top-k or top-p sample on the CPU, one step at a time: ~1 ms per token slower.
- The model can write control tokens the parser doesn't know (other `<|content_…|>` types). Their text is dropped.
- User text containing literal control tokens (`<|end_message|>`) becomes those tokens, as in llama.cpp's default.
