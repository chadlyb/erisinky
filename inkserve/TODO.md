# TODO

Server work. Engine work (kernels, weights, DSpark) is in the repository's [TODO.md](../TODO.md).

## Next

- **Current build eval'd on 6 of the 12 Go tasks** (2026-11-23, `medium`): 6/6, ~1.2× omlx per token on real agent turns (notes/RESULTS.md → `custom-inkling-o6gpu-medium-r1`). The other 6 tasks are still to run, if a full comparison is wanted.
- **Running the eval again:** relay alias `inkling-erisinky`; restart the relay with `-upstream http://127.0.0.1:8045` only when no eval is running, then put it back on 8044 (omlx).
- **m5relay alias for `inkling-erisinky-dspark`**, if the eval should reach the DSpark entry (m5ctl has both entries; README → Under m5ctl).

## Later

- **Throughput mode** (DSPARK.md → Serving modes): several conversation slots, so agents waiting on tools don't leave the GPU idle. With DSpark's gain small, this is the bigger lever for parallel agent work.
- Prefix match on tokens that differ only by retokenization: the live path needs the client's re-render to tokenize exactly as generated. It falls back to the snapshot otherwise.
- Top-k / top-p on the GPU, so those requests chain too (today they take the CPU path).

## Known issues

- Requests with top-k or top-p sample on the CPU, one step at a time: ~1 ms per token slower.
- The model can write control tokens the parser doesn't know (other `<|content_…|>` types). Their text is dropped.
- User text containing literal control tokens (`<|end_message|>`) becomes those tokens, as in llama.cpp's default.
