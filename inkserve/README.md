# inkserve

An OpenAI-style HTTP server for Thinking Machines' Inkling 2 Mini on Apple silicon. It wraps [erisinky](https://github.com/Eriskii/erisinky), a hand-fused Rust + Metal engine for the model, and adds what an agent harness needs:

- `/v1/chat/completions`, streamed or not, with `reasoning_content` and `usage`
- tool calls, rendered and parsed in Inkling's own format
- a **prefix cache** across requests, so an agent turn reads only its new tokens
- a **thinking budget**, which ends runaway reasoning and lets the model answer
- the model's chat template, matched byte for byte (checked against `transformers`)

**Status:** an experiment, shared as a fork of erisinky. The repository's root is erisinky (its GPL-3.0-or-later `gpl-license` branch, plus upstream pull request #2) with our engine changes; this folder holds the Go server and converter. All of it is GPL-3.0-or-later (the repository's [LICENSE](../LICENSE)). Measurements cite raw data in the author's private notes (`notes/…`), which are not in this repository.

## How it fits together

```
client ──HTTP──▶ inkserve (Go) ──stdin/stdout──▶ inkling serve (Rust + Metal: this repository's engine, src/serve.rs)
```

- **inkserve** renders the chat into a prompt, sends it to the engine, and turns the tokens that come back into reasoning, text and tool calls.
- **`inkling serve`** keeps the model loaded, finds the cached prefix, prefills the rest, and samples. It runs one request at a time; inkserve queues the others.

Details: [ARCHITECTURE.md](ARCHITECTURE.md). Native weights: [NATIVE.md](NATIVE.md). Speculative decoding: [DSPARK.md](DSPARK.md). What the engine patch contains: [PATCH.md](PATCH.md). Terms: [GLOSSARY.md](GLOSSARY.md).

## Requirements

- An Apple silicon Mac with Metal 4 (built and tested on an M5 Pro).
- The engine at the repository's root, built (below).
- A GGUF the patched engine loads. Two choices:
  - **Recommended: `native-q8rest-o6`**, built by `cmd/inkconvert` (below). Its experts are the official release's NVFP4 and MXFP8 bits; attention and embedding are Q8_0, and the output head is Q6_K. It's the closest to the release of any engine we've measured. See [NATIVE.md](NATIVE.md).
  - The **Q4_K_M** GGUF from `lmstudio-community/Inkling-2-Mini-GGUF`, as stock erisinky uses. It's 9–13% faster to decode, but its predictions drift much further from the release.
- ~17.6 GB of memory for the weights, plus the KV cache: 32 KiB per token of context, plus 160 MiB for the sliding-window layers.

## Building

1. **Engine** (on the Mac, needs Rust):
   - at the repository's root, run `cargo build --release --locked`
   - The binary is `target/release/inkling`. Its changes from erisinky are in [PATCH.md](PATCH.md).
2. **Server and converter** (anywhere with Go 1.27), in this folder:
   ```
   GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go build -trimpath -o build/inkserve-darwin-arm64 .
   GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go build -trimpath -o build/inkconvert-darwin-arm64 ./cmd/inkconvert
   ```
3. **Weights** (`native-q8rest-o6`): needs the official `model-00001-of-00001.safetensors` (`thinkingmachines/Inkling-2-Mini` @ `b85c3cfa`), the Q4_K_M GGUF (for metadata and small tensors) and the ggml-org Q8_0 GGUF:
   ```
   inkconvert probe -official model-00001-of-00001.safetensors -ref Inkling-2-Mini-Q8_0.gguf
   inkconvert convert -official model-00001-of-00001.safetensors -base Inkling-2-Mini-Q4_K_M.gguf -w13 interleaved \
     -q8 Inkling-2-Mini-Q8_0.gguf -q8-tensors '^token_embd\.weight$|^blk\.[0-9]+\.attn_.*\.weight$' \
     -q6-tensors '^output\.weight$' -out Inkling-2-Mini-native-q8rest-o6.gguf
   inkconvert verify -in Inkling-2-Mini-native-q8rest-o6.gguf -ref Inkling-2-Mini-Q8_0.gguf
   ```
   `probe` should show ~0.6% error for `il=true hi=false` and ~140% for the other readings.

   Other `convert` options, for experiments:
   - `-q6-tensors REGEX`: re-quantize matching Q8_0 tensors to Q6_K (llama.cpp's reference encoder, ported; `inkconvert q6check -base Q4_K_M.gguf -ref Q8_0.gguf` compares its error with llama.cpp's own Q6_K)
   - `-experts base`: keep the base file's experts instead of the release's
   - tensors matching neither `-q6-tensors` nor `-q8-tensors` stay as in `-base`

   The DSpark drafter (optional, see `-dspark` below and [DSPARK.md](DSPARK.md)) has its own file, converted from the release's `mtp.safetensors`:
   ```
   inkconvert mtp -in mtp.safetensors -official model-00001-of-00001.safetensors -out Inkling-2-Mini-dspark.gguf
   ```

## Running

```
inkserve -listen 127.0.0.1:8045 -engine ~/erisinky-eval/src/target/release/inkling \
  -model /Volumes/TB5/llm/inkling-speed/Inkling-2-Mini-native-q8rest-o6.gguf -ctx 131072
```

| Flag | Default | Meaning |
|---|---|---|
| `-listen` | `127.0.0.1:8045` | API address |
| `-name` | `inkling-erisinky` | model id the API reports and accepts |
| `-ctx` | 131072 | context length in tokens |
| `-effort` | `high` | default reasoning effort |
| `-budget` | 8192 | default thinking budget in tokens (0: none) |
| `-temperature`, `-top-p`, `-top-k` | 1.0, 1.0, 0 | default sampling (the model's own defaults). With top-p or top-k set, sampling moves to the CPU, ~1 ms per token slower (ARCHITECTURE.md → Sampling) |
| `-max-tokens` | 0 | default reply limit (0: up to the end of the context) |
| `-dspark` | off | path to `Inkling-2-Mini-dspark.gguf`: speculative decoding with the drafter. It's exact in distribution, and about 3% faster on agent replies (0.95–1.15× by prompt; DSPARK.md → DSpark in the engine's serve loop). It applies to the default sampling path (no top-k / top-p) |

Per request, it reads:
- `reasoning_effort`, or `chat_template_kwargs.reasoning_effort`
- `thinking_budget` (omlx's name) or `thinking_budget_tokens` (LM Studio's name)
- `max_tokens` or `max_completion_tokens`
- `temperature`, `top_p`, `top_k`, `seed` and `stop`
- `tools`, plus `tool_choice` of `auto` or `none`

It has no authentication, so keep it on loopback behind m5relay. Images and audio are rejected.

### Under m5ctl

On the author's Mac, m5ctl starts and stops inkserve, so its memory budget counts it. There are two catalog entries, both on port 8045 and served under their own ids:
- `inkling-erisinky` (22 GiB): plain decoding.
- `inkling-erisinky-dspark` (23 GiB): with `-dspark`.

The pieces:
- `deploy/serve-m5ctl.sh`, installed as `~/erisinky-eval/serve-m5ctl.sh`. It starts inkserve; give it the argument `dspark` for the DSpark entry.
- `deploy/m5ctl-entries.json`: the two entries as they sit in `~/.config/m5ctl/catalog.json`. Each `match` pattern ends at its own last argument, so only the running entry matches.
- The GGUFs, copied to `~/erisinky-eval/models/` on the internal disk. macOS blocks m5ctl's launchd daemon from reading TB5.

After a catalog change, restart m5ctl with `pkill -f "m5ctl serve"`. launchd starts it again within about 10 s, and loaded models keep running. Unloading stops inkserve, and the engine exits when inkserve's pipe closes.

## Testing

- `go test ./...`: the template against [testdata/golden.json](testdata/golden.json) (rendered by `transformers` from the model's own `chat_template.jinja`; regenerate with [testdata/golden.py](testdata/golden.py)), plus the output parser.
- [engine/proto_test.py](engine/proto_test.py): the engine protocol on the Mac. It checks the prefix cache's paths against a whole-prompt prefill (logit hashes must match) and the thinking budget. Run it once more with `INKSERVE_CPU_SAMPLE=1` for the CPU sampler.
- [engine/sample_test.py](engine/sample_test.py): the GPU sampler against the CPU one. It checks seeds, greedy output, the first token's distribution at temperatures 1 and 2, and the budget.
- `go test ./cmd/inkconvert`: the NV4 / MX8 packing and the FP4 / FP8 decoders.
- [testdata/e2e.py](testdata/e2e.py): HTTP end to end against a running server. It covers plain and streamed replies, tool calls, the cache across an agent turn, the budget, stop strings, disconnects and errors.

The Python tests use only the standard library, plus `transformers` for the golden file.

## Speed

Measured on an M5 Pro (2026-11-22), against omlx 0.8.2 with oQ4e and DSpark, on the same streamed code-review prompts at temperature 1.0:

| Context | Prefill | Decode | Next turn, time to first token |
|---|---|---|---|
| ~1K | 3,100 tok/s (2.1× omlx) | 86 tok/s (1.36×) | 0.17 s |
| ~11K | 4,200 tok/s (2.4×) | 77 tok/s (1.45×) | 0.14 s |
| ~37K | 3,800 tok/s (2.35×) | 62 tok/s (1.35×) | 0.2 s |
| ~67K | 3,200 tok/s (2.35×) | 50 tok/s (~1.5×) | 0.3 s |

These rows used Q4_K_M and the earlier CPU sampler. The current build, with `native-q8rest-o6`, tuned kernels and sampling on the GPU, decodes **89 / 79 / 63 tok/s** served at ~1K / 11K / 35K and temperature 1.0 (2026-11-23). That's as fast as erisinky's own GPU-chained loop on the same file. Prefill is 3–5% below Q4_K_M's.

Details and raw data: `notes/RESULTS.md`.
