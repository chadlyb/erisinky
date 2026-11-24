# Architecture

## Processes

inkserve (Go) starts `inkling serve` (Rust) as a child process and talks to it over its stdin and stdout. If the engine exits, inkserve exits with status 1, so a supervisor can restart both. Closing the engine's stdin stops it.

The engine side is [src/serve.rs](../src/serve.rs). The engine changes it relies on (native weights, GPU sampling, the verify pass, DSpark) are listed in [MYCHANGES.md](../MYCHANGES.md). Two invariants keep upstream speed-ups easy to pick up:
- Q4_K_M's logits are bit-identical to stock erisinky.
- Prefill is bit-identical with and without the verify kernels, which run only inside `encode_verify`.

## Engine protocol

It's line-based, with one request at a time:

| Direction | Line | Meaning |
|---|---|---|
| → engine | `G max= temp= top_p= top_k= seed= budget= len=N` + N bytes | generate from this prompt text (control tokens written literally) |
| → engine | `X` | stop the running generation |
| ← engine | `R ctx= vocab= eos=` | loaded (once) |
| ← engine | `P prompt= cached= us= h=` | prefill done; `h` hashes the last logits, for tests |
| ← engine | `T ID s\|n HEX` | one token: control (`s`) or text (`n`), its bytes in hex |
| ← engine | `E REASON gen= think= us=` | end: `stop`, `length`, `cancel` or `context` |
| ← engine | `F MESSAGE` | malformed request |

The engine tokenizes, so inkserve never needs the vocabulary. Hex keeps token bytes (which may split a UTF-8 character) safe in a line protocol.

## Prefix cache

Inkling's state per token is the K/V cache plus small recurrent state: convolutions on K, V and both residual branches.
- The full-attention K/V only grows, so any prefix of it stays valid.
- The 40 sliding-window layers keep a 1,024-key ring each, and the convolution state is overwritten as the context advances. Going back needs a copy of those: a **snapshot** (≤160 MiB, from erisinky's `Engine::snapshot`).

The engine keeps the live context's token list and one snapshot, taken two tokens before the end of each prompt. A new prompt reuses:
1. **the live context,** if the prompt extends it. This happens when the client sends back exactly what the model generated, reasoning included.
2. **the snapshot,** if the prompt extends that. This happens when the client re-renders the reply differently, for example without the reasoning; the agent harnesses here do that.
3. **nothing** otherwise: a full prefill.

Why two tokens: after the snapshot, the last two prompt tokens are prefilled together. Prefill results don't depend on where a prompt is split, so a cached request gives bit-identical logits to a whole-prompt prefill (`proto_test.py` checks the hashes).

**Upstream bug avoided:** at erisinky `3890bd5`, a *one-token* prefill on a freshly allocated batch arena gives wrong logits: the reply starts with junk tokens. Two tokens or more are fine. So `feed()` sends single tokens through a decode step instead. The cause: `mix_t` sent a one-token batch to the decode mix kernel. Upstream pull request #2, merged in this repository, fixes it (the inky fork found it too, `715c0b6`). `feed()` keeps the decode step, which is the cheaper way to feed one token.

## Thinking budget

The engine counts tokens inside `<|content_thinking|>`. At the budget it emits `<|end_message|>` itself, which closes the thinking message, and bans `<|content_thinking|>` for the rest of the reply. The model then writes text or calls a tool. Unlike an output cap, the reply still has an answer.

## Sampling

There are two paths. Both draw from softmax(softcap(logits) / temperature) and apply the budget above.

- **GPU, chained** (`gen_chained`): used without top-k or top-p, which covers the model's defaults. Each decode step ends with three kernels:
  - `k_gumbel` adds Gumbel noise to softcap(logit) / T. The noise comes from a hash of (seed, position, token).
  - erisinky's argmax picks the winner and writes it to the token buffer, where the next step reads it. Argmax of logit plus Gumbel noise is an exact draw from the softmax.
  - `k_sampstate` runs the thinking budget's state machine. Its state sits in three spare token slots past the context.

  Because the next token never has to visit the CPU, the engine commits step *q*+1 while step *q* runs, and streams token *q* meanwhile. Decoding then runs at erisinky's GPU-chained speed.
- **GPU, speculative** (`gen_dspark`): replaces the chained path when inkserve runs with `-dspark`.
  - Each cycle, the DSpark drafter proposes 1–3 tokens, and one verify pass of the main model checks them.
  - Standard speculative sampling keeps the output distribution exact.
  - The CPU applies the thinking budget, eos and the token limit to the accepted tokens. Rejected positions are rolled back.
  - It falls back to stretches of chained-style plain steps while no draft depth beats a plain step.
  - See [DSPARK.md](../DSPARK.md) → DSpark in the engine's serve loop.
- **CPU** (`Pick`): used with top-k or top-p, or with `INKSERVE_CPU_SAMPLE=1`. It reads the logits back after each step, ignoring tokens more than 30 nats below the best. A raw-logit pre-filter keeps `tanh` off the rest. This costs ~1 ms per token: the sampler itself, plus the GPU idling while the CPU samples.

The paths use different random streams, so a given seed gives different replies on each. `engine/sample_test.py` checks they draw the same distribution. The noise's uniform `u` uses 23 random bits plus a half step, which float holds exactly. With 24 bits, the top value rounds to 1.0, the noise becomes infinite, and a random token wins on ~1% of steps. The test's temperature-0.01 check catches that.

**The stop token's step.** On the GPU path, the step that consumes the stop token has already run when the CPU sees the token. Its state is still valid: the template writes `<|content_model_end_sampling|>` after every assistant message, so the next prompt holds that same token there. The live context includes it.

**Cache growth.** Growing the K/V cache copies it, so it needs an idle GPU. The chained loop waits for the step in flight before it grows the cache, which happens about every 1/16 of the context.

## Prompt format

[template.go](template.go) follows the model's `chat_template.jinja` line by line:
- tool declarations as sorted, compact JSON
- the effort line before the first non-system message, as a number (`high` = 0.9)
- reasoning, text and tool-call messages for the assistant
- `<|content_model_end_sampling|>` after each assistant turn

Numbers in tool arguments are printed as Python would print them (`2.50` → `2.5`), since the template's `tojson` re-encodes parsed values.

[parse.go](parse.go) reads the reply the same way in reverse. Text is emitted only in whole UTF-8 characters. Stop strings hold back text that could be the start of one. A tool-call message whose JSON doesn't parse is passed on as text and logged.
