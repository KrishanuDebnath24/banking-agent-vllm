#!/usr/bin/env bash
# E1 and E3: how many requests at once one GPU serves within the latency targets.
# Runs `vllm bench serve` once per concurrency level and saves each result as JSON.
#
# On the phase 0 GPU machine (runs inside the vllm container, results in bench/results/):
#   bash bench/sweep.sh                       # the full sweep, 1 to 256 requests at once
#   CONCURRENCY="1" LABEL=e1 bash bench/sweep.sh   # E1: one request at a time
# Then:  python -m bench.summarise
#
# Settings (environment variables):
#   MODEL          served model name (default: VLLM_MODEL from .env, else openai/gpt-oss-20b)
#   PREFIX_LEN     shared start of every prompt, in tokens (default 2500: system prompt + tools)
#   INPUT_LEN      the varying part (default 500)       Take both from real calls in LangSmith.
#   OUTPUT_LEN     tokens per answer (default 200)
#   CONCURRENCY    levels to test (default "1 2 4 8 16 32 64 128 256")
#   PER_LEVEL      requests per concurrency level = level x this, at least 20 (default 10)
#   GOODPUT        latency targets for goodput (default "ttft:1000 tpot:50", milliseconds)
#   LABEL          prefix for the result files (default e3)
#   BASE_URL       where vLLM listens, seen from where the benchmark runs (default http://127.0.0.1:8000)
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .env ]; then
  # the value only: no trailing "# comment", spaces or quotes
  VLLM_MODEL_FROM_ENV=$(grep -E '^VLLM_MODEL=' .env | tail -1 | cut -d= -f2- \
    | sed -e 's/[[:space:]]#.*$//' -e 's/[[:space:]]*$//' -e 's/^["'"'"']//' -e 's/["'"'"']$//' || true)
fi
MODEL=${MODEL:-${VLLM_MODEL_FROM_ENV:-openai/gpt-oss-20b}}
PREFIX_LEN=${PREFIX_LEN:-2500}
INPUT_LEN=${INPUT_LEN:-500}
OUTPUT_LEN=${OUTPUT_LEN:-200}
CONCURRENCY=${CONCURRENCY:-"1 2 4 8 16 32 64 128 256"}
PER_LEVEL=${PER_LEVEL:-10}
GOODPUT=${GOODPUT:-"ttft:1000 tpot:50"}
LABEL=${LABEL:-e3}
BASE_URL=${BASE_URL:-http://127.0.0.1:8000}

if command -v vllm >/dev/null; then
  run() { vllm bench serve "$@"; }                     # already inside the vLLM image (Kubernetes Job)
  RESULTS=${RESULTS:-/results}
else
  COMPOSE=(docker compose -f docker-compose.yml -f deploy/phase0/compose.vllm.yml)
  run() { "${COMPOSE[@]}" exec -T vllm vllm bench serve "$@"; }
  RESULTS=/results                                     # = bench/results/ on the machine
fi

for c in $CONCURRENCY; do
  n=$(( c * PER_LEVEL )); [ "$n" -lt 20 ] && n=20
  echo "== ${LABEL}: ${c} at once, ${n} requests"
  # shellcheck disable=SC2086
  run --backend openai-chat --endpoint /v1/chat/completions --base-url "$BASE_URL" \
      --model "$MODEL" --dataset-name random \
      --random-prefix-len "$PREFIX_LEN" --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
      --ignore-eos --num-prompts "$n" --max-concurrency "$c" \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
      --goodput $GOODPUT \
      --save-result --result-dir "$RESULTS" --result-filename "${LABEL}-c${c}.json" \
    || echo "   (level ${c} failed; check the flags against 'vllm bench serve --help' for your vLLM version)"
done
echo "Done. Summary:  python -m bench.summarise"
