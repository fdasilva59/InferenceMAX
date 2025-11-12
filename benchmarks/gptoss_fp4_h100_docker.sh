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

# Start server in the background, shld be openai/gpt-oss-120b
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
echo "vLLM server up"
RUN_MODE=${RUN_MODE:-benchmark}

if [[ "$RUN_MODE" == "eval" ]]; then
  EVAL_RESULT_DIR=${EVAL_RESULT_DIR:-eval_out}
  mkdir -p "/workspace/${EVAL_RESULT_DIR}"
  OPENAI_SERVER_BASE="http://0.0.0.0:${PORT}"
  export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}

  # --- Install Lighteval with LiteLLM backend ---
  python3 -m pip install -q --upgrade pip || true
  python3 -m pip install -q --no-cache-dir "lighteval[litellm]" || true

  echo "Using model: $MODEL"

  # --- Lighteval (LiteLLM) model config pointing at your vLLM server ---
  # Docs: provider=openai, base_url, model_name, api_key, generation parameters
  # https://huggingface.co/docs/lighteval/en/use-litellm-as-backend
  cat > /workspace/lighteval_litellm.yaml <<YAML
model_parameters:
  provider: "openai"
  model_name: "${MODEL}"
  base_url: "${OPENAI_SERVER_BASE}/v1"
  api_key: "${OPENAI_API_KEY}"
  parallel_calls_count: 16
  # Optional: system_prompt: "You are a helpful math assistant."
  generation_parameters:
    temperature: 0.0
    top_p: 1.0
    max_new_tokens: 8192
YAML

  # --- Clean any previous results ---
  rm -rf "/workspace/${EVAL_RESULT_DIR:?}/"* 2>/dev/null || true

  # --- Run Lighteval on the requested task ---
  # Task spec format: {suite}|{task}|{num_few_shot}|{truncate_ok}
  # e.g., lighteval|gsm8k|5|1  (allow automatic reduction if too-long prompt)
  set -x
  lighteval endpoint litellm \
    "/workspace/lighteval_litellm.yaml" \
    "lighteval|${EVAL_TASK:-gsm8k}|${NUM_FEWSHOT:-5}|1" \
    --output-dir "/workspace/${EVAL_RESULT_DIR}" \
    --save-details
  set +x

  # --- Append a Markdown table to the GitHub Actions job summary ---
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    python3 - <<'PY' >> "$GITHUB_STEP_SUMMARY" || true
import glob, json, os, time
out_dir = os.environ.get("EVAL_RESULT_DIR", "eval_out")
model = os.environ.get("MODEL", "unknown_model")
task  = os.environ.get("EVAL_TASK", "gsm8k")
nshot = int(os.environ.get("NUM_FEWSHOT", "5"))

# Find the newest results_*.json under results/<model>/
pattern = os.path.join("/workspace", out_dir, "results", model, "results_*.json")
paths = sorted(glob.glob(pattern), key=os.path.getmtime)
if not paths:
    print("\n**Lighteval**: no results found.\n")
    raise SystemExit(0)

with open(paths[-1], "r") as f:
    data = json.load(f)

key = f"{task}|{nshot}"
res = data.get("results", {}).get(key, {})
em = res.get("em")
em_stderr = res.get("em_stderr")
maj8 = res.get("maj@8")
maj8_stderr = res.get("maj@8_stderr")

print(f"\n### Lighteval — {task} (n-shot = {nshot})\n")
print("| Metric | Value | Stderr |")
print("|---|---:|---:|")
def fmt(x): return "—" if x is None else f"{x:.4f}" if isinstance(x, (int,float)) else str(x)
print(f"| EM | {fmt(em)} | {fmt(em_stderr)} |")
if maj8 is not None:  # present for GSM8K by default
    print(f"| maj@8 | {fmt(maj8)} | {fmt(maj8_stderr)} |")
print(f"\n<sub>File: `{os.path.basename(paths[-1])}`</sub>\n")
PY
  fi

  echo "Evaluation completed. Results in /workspace/${EVAL_RESULT_DIR}"
  exit 0

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
