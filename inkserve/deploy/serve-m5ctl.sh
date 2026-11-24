#!/bin/sh
# Serves Inkling 2 Mini through inkserve + erisinky (model id inkling-erisinky) on 127.0.0.1:8045, for m5ctl's
# entries inkling-erisinky and inkling-erisinky-dspark (each served under its m5ctl id, as m5ctl and m5relay expect). With the argument "dspark" it turns on DSpark speculative
# decoding (D:\src\chadlyb\inkserve\DSPARK.md). The model files sit on the internal disk: macOS blocks the m5ctl
# launchd daemon from reading TB5. exec, so m5ctl's SIGTERM reaches inkserve; the engine exits when inkserve's pipe closes.
set -eu
D="$HOME/erisinky-eval"
set -- "$@" ""
if [ "$1" = dspark ]; then
  exec "$D/inkserve" -listen 127.0.0.1:8045 -engine "$D/src/target/release/inkling" \
    -model "$D/models/Inkling-2-Mini-native-q8rest-o6.gguf" -ctx 131072 -name inkling-erisinky-dspark -dspark "$D/models/Inkling-2-Mini-dspark.gguf"
fi
exec "$D/inkserve" -listen 127.0.0.1:8045 -engine "$D/src/target/release/inkling" \
  -model "$D/models/Inkling-2-Mini-native-q8rest-o6.gguf" -ctx 131072
