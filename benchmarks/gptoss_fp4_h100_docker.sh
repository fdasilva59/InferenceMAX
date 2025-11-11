#!/usr/bin/env bash

# === Required Env Vars ===
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# ISL
# OSL
# MAX_MODEL_LEN
# RANDOM_RANGE_RATIO
# TP
# CONC
# RESULT_FILENAME

set -euo pipefail

# Create a basic vLLM config
cat > config.yaml << EOF
compilation-config: '{"cudagraph_mode":"PIECEWISE"}'
async-scheduling: true
no-enable-prefix-caching: true
cuda-graph-sizes: 2048
max-num-batched-tokens: 8192
max-model-len: 10240
EOF

SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)
export TORCH_CUDA_ARCH_LIST="9.0"
PORT=${PORT:-8888}

# Start server in the background
set -x
PYTHONNOUSERSITE=1 vllm serve "$MODEL" --host=0.0.0.0 --port="$PORT" \
  --config config.yaml \
  --gpu-memory-utilization=0.9 \
  --tensor-parallel-size="$TP" \
  --max-num-seqs="$CONC" \
  --disable-log-requests > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
set +x

cleanup() {
  set +e
  if ps -p "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Wait for server readiness via log message
echo "Waiting for vLLM server to become ready on port $PORT ..."
READY=0
for i in $(seq 1 600); do
  if grep -q "Application startup complete" "$SERVER_LOG"; then
    READY=1
    break
  fi
  if ! ps -p "$SERVER_PID" >/dev/null 2>&1; then
    echo "vLLM server exited prematurely. Last 200 lines of log:" >&2
    tail -n 200 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  sleep 1
done
if [[ "$READY" -ne 1 ]]; then
  echo "Timed out waiting for vLLM server readiness." >&2
  tail -n 200 "$SERVER_LOG" >&2 || true
  exit 1
fi

RUN_MODE=${RUN_MODE:-benchmark}

if [[ "$RUN_MODE" == "eval" ]]; then
  # Run evaluation in-container using lm-evaluation-harness
  EVAL_RESULT_DIR=${EVAL_RESULT_DIR:-eval_out}
  OPENAI_SERVER_BASE="http://0.0.0.0:${PORT}"
  OPENAI_CHAT_BASE="$OPENAI_SERVER_BASE/v1/chat/completions"
  export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}

  python3 -m pip install -q --upgrade pip || true
  python3 -m pip install -q --no-cache-dir "lm-eval[api]" || true
  # Temporary: workaround known harness issue
  python3 -m pip install -q --no-cache-dir --no-deps "git+https://github.com/EleutherAI/lm-evaluation-harness.git@main" || true

  # Basic health check
  if command -v curl >/dev/null 2>&1; then
    curl -fsS "$OPENAI_SERVER_BASE/health" >/dev/null || { echo "Health check failed"; exit 1; }
  else
    python3 - "$OPENAI_SERVER_BASE/health" <<'PY' || exit 1
import sys, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=5) as r:
        pass
except Exception as e:
    print("Health check failed:", e)
    sys.exit(1)
PY
  fi

  # Infer served model name if not provided
  if [[ -z "${OPENAI_MODEL_NAME:-}" ]]; then
    if command -v curl >/dev/null 2>&1; then
      OPENAI_MODEL_NAME_COMPUTED=$(curl -fsS "$OPENAI_SERVER_BASE/v1/models" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print((d.get('data') or [{}])[0].get('id',''))")
    else
      OPENAI_MODEL_NAME_COMPUTED=$(python3 - "$OPENAI_SERVER_BASE" <<'PY'
import json, sys, urllib.request
base = sys.argv[1]
with urllib.request.urlopen(f"{base}/v1/models") as r:
    d=json.load(r)
print((d.get('data') or [{}])[0].get('id',''))
PY
)
    fi
  else
    OPENAI_MODEL_NAME_COMPUTED="$OPENAI_MODEL_NAME"
  fi
  echo "Using model: $OPENAI_MODEL_NAME_COMPUTED"

  # Optional: quick POST to chat endpoint
  if command -v curl >/dev/null 2>&1; then
    curl -fsS -X POST "$OPENAI_CHAT_BASE" -H "Content-Type: application/json" \
      -d "{\"model\":\"$OPENAI_MODEL_NAME_COMPUTED\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" >/dev/null || { echo "Chat POST failed"; exit 1; }
  fi

  set -x
  python3 -m lm_eval --model local-chat-completions \
    --tasks ${EVAL_TASK:-gsm8k} \
    --apply_chat_template \
    --num_fewshot ${NUM_FEWSHOT:-5} \
    --limit ${LIMIT:-200} \
    --batch_size 1 \
    --output_path "/workspace/${EVAL_RESULT_DIR}" \
    --model_args "model=$OPENAI_MODEL_NAME_COMPUTED,base_url=$OPENAI_CHAT_BASE,api_key=$OPENAI_API_KEY,eos_string=</s>,max_retries=3" \
    --gen_kwargs "max_tokens=16384,temperature=0,top_p=1"
  set +x

  echo "Evaluation completed. Results in /workspace/${EVAL_RESULT_DIR}"
else
  # Default values for optional vars used by the benchmark
  ISL=${ISL:-256}
  OSL=${OSL:-256}
  RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-1.0}
  RESULT_FILENAME=${RESULT_FILENAME:-results}

  # Install deps and run benchmark in the same container
  python3 -m pip install -q --upgrade pip || true
  python3 -m pip install -q --no-cache-dir datasets pandas || true
  if [[ ! -d bench_serving ]]; then
    git clone https://github.com/kimbochen/bench_serving.git
  fi

  set -x
  python3 bench_serving/benchmark_serving.py \
    --model="$MODEL" \
    --backend=vllm \
    --base-url="http://0.0.0.0:$PORT" \
    --dataset-name=random \
    --random-input-len="$ISL" --random-output-len="$OSL" --random-range-ratio="$RANDOM_RANGE_RATIO" \
    --num-prompts=$(( CONC * 10 )) --max-concurrency="$CONC" \
    --request-rate=inf --ignore-eos \
    --save-result --percentile-metrics='ttft,tpot,itl,e2el' \
    --result-dir=/workspace/ \
    --result-filename="$RESULT_FILENAME.json"
  set +x
fi
