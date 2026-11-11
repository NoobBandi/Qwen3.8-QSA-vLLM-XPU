#!/usr/bin/env bash
# QSA against dense at matched context depths: the data for docs/qsa-vs-dense.png.
# Both runs use the same vLLM build and scripts/serve.sh settings. Dense loads
# the release as plain Qwen3.5 (an overlay whose config.json has architectures
# ["Qwen3_5ForConditionalGeneration"] and no "qsa" section), so the only
# difference is the attention in the 16 full-attention layers.
#
#   bench/sweep.sh QSA_OVERLAY DENSE_OVERLAY [lengths]
#   SERVE="my-wrapper" bench/sweep.sh ...    # to start the server another way
set -euo pipefail
cd "$(dirname "$0")/.."
QSA=$1 DENSE=$2
LENGTHS=${3:-500,4000,16000,64000,128000,256000}
PORT=${PORT:-5899}
SERVE=${SERVE:-scripts/serve.sh}
OUT=${OUT:-docs/results}
URL=http://127.0.0.1:$PORT
mkdir -p "$OUT"
for mode in qsa dense; do
  model=$QSA; [[ $mode == dense ]] && model=$DENSE
  log=$OUT/$mode-server.log
  setsid $SERVE "$model" --host 127.0.0.1 --port "$PORT" > "$log" 2>&1 &
  server=$!
  until grep -q "Application startup complete" "$log"; do
    kill -0 $server 2>/dev/null || { echo "$mode: server exited"; tail -5 "$log"; exit 1; }
    sleep 10
  done
  # One short request first, so no measurement includes the first request's
  # kernel compiles.
  python3 bench/prefill.py --url "$URL" --lengths 1000 > /dev/null
  python3 -u bench/prefill.py --url "$URL" --lengths "$LENGTHS" | tee "$OUT/$mode-prefill.jsonl"
  python3 -u bench/decode.py --url "$URL" --lengths "$LENGTHS" | tee "$OUT/$mode-decode.jsonl"
  echo "$mode: device lost $(grep -c DEVICE_LOST "$log" || true)"
  kill -TERM -- -$server
  wait $server || true
  sleep 5
done
