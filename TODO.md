# TODO

Engine work. The server's list is [inkserve/TODO.md](inkserve/TODO.md).

## Blocking

- **License on upstream `main`.** This repository builds on erisinky's commit `d1d5263` ("License under GPL-3.0-or-later"). That commit is on the upstream branch `gpl-license` and in upstream pull request #1, not yet on `main` (checked 2026-11-23). Watch for it to reach `main`, or ask the author whether the branch is final.

## Next

- **Optional: `attn-q6-out-q6`:** +6.4% decode, with drift at omlx oQ4e's level. Worth it if that speed matters more than the drift (NATIVE.md).
- **Attention at long context.** At 30K, attention takes ~5 ms of a 15 ms step. It reads the K/V cache well below the weights' rate (inky's notes, on other hardware: ~210 vs ~565 GB/s). Options:
  - a faster split-K kernel, the largest long-context lever
  - the 8-bit cache (`--kv8`, not exact), the cheap one. It needs a drift check.
- **More decode speed for the 8-bit file.** The decode kernels are tuned (NATIVE.md → Decode layout): `native-q8rest-o6` decodes 6–8% below Q4_K_M. What's left:
  - the small, latency-bound kernels (loop controllers, router, mixes, combine): ~2.3 ms per step for ~0.3 ms of data. Fuse them.
  - the fixed ~4 µs per dispatch (about 0.8 ms per token across the four matvec classes): fuse or overlap dispatches
  - the NV4 down projection reaches only ~370 GB/s
  - try a 16-entry LUT for FP4 against the bit trick, and a 32-lanes-per-superblock `dot_q4k` (it has the same ragged tail at K = 2304)
  - check once on the GPU that subnormal halves (FP4 ±0.5, small E4M3 codes) aren't flushed to zero. Drift at the noise floor suggests they aren't.
- **inky's other engine fix** (robotgamedevelopers/inky `79c5157`): zero the sliding-window K/V scratch at the start of each prefill (`k_zero`), since a zero attention weight doesn't cancel a NaN left there. Its one-token prefill fix (`715c0b6`) matches upstream pull request #2, merged here.
- **Drift past 3K tokens** for `native-q8rest-o6`. It needs a reference tool with a larger context; erisinky's README tests only to ~16K.

## Later

- **DSpark** is in, opt-in (DSPARK.md): about 1.03× on agent replies (0.95–1.15×). Levers left:
  - acceptance: the drafter keeps 1.7–2.8 tokens per cycle on agent text. A drafter fine-tuned on agent transcripts would help most.
  - verify cost: a 4-token pass reads the union of its tokens' experts, ~2× a decode step. That's bandwidth, so kernel tuning has little left.
  - an end-to-end agent eval with DSpark on, to see the gain on real turns (the serve benchmark used 300-token continuations)
- **Several conversations at once.** inky ("lanes": one engine, a K/V cache and state per stream) is building this. It would serve inkserve's throughput mode. The verify pass's small-batch kernels could serve it too.
