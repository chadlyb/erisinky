# inkserve

An OpenAI-style HTTP server for Thinking Machines' Inkling 2 Mini on Apple silicon. It wraps [erisinky](https://github.com/Eriskii/erisinky), a hand-fused Rust + Metal engine for the model, and adds what an agent harness needs:

- `/v1/chat/completions`, streamed or not, with `reasoning_content` and `usage`
- tool calls, rendered and parsed in Inkling's own format
- a **prefix cache** across requests, so an agent turn reads only its new tokens
- a **thinking budget**, which ends runaway reasoning and lets the model answer
- the model's chat template, matched byte for byte (checked against `transformers`)

**Status:** an experiment, shared as a fork of erisinky. The repository's root is erisinky (its GPL-3.0-or-later `gpl-license` branch, plus upstream pull request #2) with our engine changes; this folder holds the Go server. All of it is GPL-3.0-or-later (the repository's [LICENSE](../LICENSE)). Measurements cite raw data in the author's private notes (`notes/…`), which are not in this repository.

## How it fits together

```
client ──HTTP──▶ inkserve (Go) ──stdin/stdout──▶ inkling serve (Rust + Metal: this repository's engine, src/serve.rs)
```

- **inkserve** renders the chat into a prompt, sends it to the engine, and turns the tokens that come back into reasoning, text and tool calls.
- **`inkling serve`** keeps the model loaded, finds the cached prefix, prefills the rest, and samples. It runs one request at a time; inkserve queues the others.

Details: [ARCHITECTURE.md](ARCHITECTURE.md). Terms: [GLOSSARY.md](GLOSSARY.md). The engine's changes (native weights, GPU sampling, DSpark), the weights and their results are in the repository's [README](../README.md#my-changes).

## Requirements

- An Apple silicon Mac with Metal 4 (built and tested on an M5 Pro).
- The engine at the repository's root, built: `cargo build --release --locked` there gives `target/release/inkling`.
- A GGUF the engine loads. `native-q8rest-o6` (recommended) is built with `tools/inkconvert`, and so is the DSpark drafter. The repository's README has the steps. The Q4_K_M from `lmstudio-community/Inkling-2-Mini-GGUF` also works, with more drift.
- ~17.6 GB of memory for the weights, plus the KV cache: 32 KiB per token of context, plus 160 MiB for the sliding-window layers.

## Building

Anywhere with Go 1.27, in this folder:
```
GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go build -trimpath -o build/inkserve-darwin-arm64 .
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
- With `INKSERVE_TEST_REF=Inkling-2-Mini-dspark.gguf`, both engine tests run the engine with DSpark.
- [testdata/e2e.py](testdata/e2e.py): HTTP end to end against a running server. It covers plain and streamed replies, tool calls, the cache across an agent turn, the budget, stop strings, disconnects and errors.

The Python tests use only the standard library, plus `transformers` for the golden file.

## Speed

Served speed, and how it compares with omlx: [MYCHANGES.md](../MYCHANGES.md#results).
