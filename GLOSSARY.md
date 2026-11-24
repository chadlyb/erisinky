# Glossary

Terms for the engine and its changes. The server's terms are in [inkserve/GLOSSARY.md](inkserve/GLOSSARY.md).

- **erisinky:** a third-party Rust + Metal inference engine for Inkling 2 Mini (github.com/Eriskii/erisinky). GPL-3.0-or-later (upstream commit `d1d5263`, on branch `gpl-license`).
- **inky:** a GPL-3.0-or-later fork of erisinky by Glenn Fiedler and others (github.com/robotgamedevelopers/inky). It adds several streams in one engine ("lanes") and some engine fixes.
- **GPU sampling / chained generation:** drawing each token on the GPU (Gumbel-max), so the next decode step can be committed before the CPU sees the token. See inkserve/ARCHITECTURE.md → Sampling.
- **Routed experts:** the 128 small MLPs per layer, 8 of which run for each token. They hold almost all of the model's weights. Each has a gate/up projection (`w13`, GGUF `ffn_gate_up_exps`) and a down projection (`w2`, `ffn_down_exps`).
- **NVFP4 / MXFP8:** the formats the experts were released in. NVFP4 is 4-bit floats with an FP8 scale per 16 weights and an FP32 scale per expert. MXFP8 is 8-bit floats with a power-of-two scale per 32 weights. See [NATIVE.md](NATIVE.md).
- **NV4 / MX8:** this project's GGUF block types (ids 200 and 201) that hold NVFP4 and MXFP8 bit for bit. Only the patched engine reads them.
- **gscale:** the per-expert FP32 scale of an NVFP4 tensor, stored in a companion tensor named `….gscale`.
- **Native-experts file:** `Inkling-2-Mini-native-experts.gguf`, built by `tools/inkconvert`: the Q4_K_M file with its experts replaced by NV4 and MX8.
- **DSpark:** speculative decoding with the release's small drafter model (`mtp.safetensors`, converted to `Inkling-2-Mini-dspark.gguf`). Off unless inkserve gets `-dspark`. See [DSPARK.md](DSPARK.md).
- **Drafter:** DSpark's 3-layer model. It reads taps from the main model and proposes the next few tokens.
- **Taps:** four hidden-state vectors per position (three layer visits plus the final-norm output) that the main model writes to a 1,024-position ring for the drafter.
- **Cycle:** one DSpark round: the drafter proposes 1–3 tokens (the **depth**), one verify pass checks them, and the accepted ones are kept.
- **Verify pass:** one batched pass of the main model over 2–8 tokens that returns the logits of every position. `Engine::encode_verify`.
- **Anchor:** the last token streamed, not yet fed through the model. Each cycle verifies the anchor plus its drafts, and the target's next token becomes the new anchor.
- **Acceptance:** how many drafts a cycle keeps. Standard speculative sampling accepts a draft with probability min(1, p/q), so the output distribution stays exactly plain decoding's.
- **Stretch:** a run of plain decode steps inside DSpark generation, used while no draft depth beats a plain step.
