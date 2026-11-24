# Glossary

Terms for the server. The engine's terms (NV4, DSpark and others) are in the repository's [GLOSSARY.md](../GLOSSARY.md).

- **Engine:** erisinky's binary (`inkling`) running its `serve` command, with the model loaded. inkserve starts it as a child process.
- **Control token:** a token like `<|message_model|>` or `<|end_message|>` that structures the conversation. The protocol marks them `s`.
- **Live context:** the tokens whose state the engine holds right now: the last prompt plus what it generated (on the GPU path, including the stop token).
- **Snapshot:** a copy of the sliding-window rings and the convolution state at one position, so the engine can go back to it. Taken two tokens before the end of each prompt.
- **Prefix cache:** reusing the live context or the snapshot when a new prompt starts with it.
- **Thinking budget:** the most reasoning tokens a reply may use. At the limit the engine closes the thinking message itself.
- **Effort:** the "Thinking effort level" line in the prompt, a number from 0 to 0.99 (`high` = 0.9). It steers how long the model thinks; the budget is the hard limit.
- **Golden prompts:** `testdata/golden.json`, the model's official template rendered by `transformers` for the test cases. The Go renderer must match them byte for byte.
