#!/usr/bin/bash

HF_HUB_CACHE_MOUNT="/home/ubuntu/hf_hub_cache/"
PORT=8888

server_name="bmk-server"
client_name="bmk-client"
RUN_MODE="${RUN_MODE:-benchmark}"

set -x
## Pre-clean any leftover containers to avoid name conflicts
if docker ps -a --format '{{.Names}}' | grep -Fxq "$server_name"; then
  echo "Found existing container $server_name; removing..."
  docker rm -f "$server_name" || true
fi
if docker ps -a --format '{{.Names}}' | grep -Fxq "$client_name"; then
  echo "Found existing container $client_name; removing..."
  docker rm -f "$client_name" || true
fi

docker run --rm -d --network=host --name=$server_name \
--runtime=nvidia --gpus=all --ipc=host --privileged --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
-v $HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
-v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
-e HF_TOKEN -e HF_HUB_CACHE -e MODEL -e TP -e CONC -e MAX_MODEL_LEN -e ISL -e OSL -e PORT=$PORT \
-e TORCH_CUDA_ARCH_LIST="9.0" -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
--entrypoint=/bin/bash \
$IMAGE \
benchmarks/"${EXP_NAME%%_*}_${PRECISION}_h100_docker.sh"

set +x
while IFS= read -r line; do
    printf '%s\n' "$line"
    if [[ "$line" =~ Application\ startup\ complete ]]; then
        break
    fi
done < <(docker logs -f --tail=0 $server_name 2>&1)

if ! docker ps --format "{{.Names}}" | grep -q "$server_name"; then
    echo "Server container launch failed."
    exit 1
fi

if [[ "$RUN_MODE" == "eval" ]]; then
  mkdir -p "${EVAL_RESULT_DIR:-eval_out}"

  OPENAI_SERVER_BASE="http://localhost:${PORT:-8888}"
  OPENAI_CHAT_BASE="$OPENAI_SERVER_BASE/v1/chat/completions"

  LM_EVAL_IMAGE="${LM_EVAL_IMAGE:-$IMAGE}"

  set -x
  docker run --rm --network=host --name="$client_name" \
    -v "$GITHUB_WORKSPACE:/workspace/" -w /workspace/ \
    -e OPENAI_API_KEY=EMPTY \
    -e OPENAI_SERVER_BASE="$OPENAI_SERVER_BASE" \
    -e OPENAI_CHAT_BASE="$OPENAI_CHAT_BASE" \
    -e OPENAI_MODEL_NAME_COMPUTED="${OPENAI_MODEL_NAME:-}" \
    --entrypoint=/bin/bash \
    "$LM_EVAL_IMAGE" \
    -lc 'python3 -m pip install -q --upgrade pip || true; \
         python3 -m pip install -q --no-cache-dir "lm-eval[api]"; \
         # 0) Solves issue lm-evals #3355
         python3 -m pip install -q --no-cache-dir --no-deps "git+https://github.com/EleutherAI/lm-evaluation-harness.git@main"; \
         # 1) Sanity: /health is GET-able (avoid 405 on POST-only endpoints)
         curl -fsS "$OPENAI_SERVER_BASE/health" >/dev/null || { echo "Health check failed"; exit 1; }; \
         # 2) Get served model id if not provided
         if [ -z "$OPENAI_MODEL_NAME_COMPUTED" ]; then \
           OPENAI_MODEL_NAME_COMPUTED=$(curl -fsS "$OPENAI_SERVER_BASE/v1/models" \
             | python3 -c "import sys,json; d=json.load(sys.stdin); print((d.get(\"data\") or [{}])[0].get(\"id\",\"\"))"); \
         fi; \
         echo "Using model: $OPENAI_MODEL_NAME_COMPUTED"; \
         # 3) Optional: quick POST sanity to chat endpoint
         curl -fsS -X POST "$OPENAI_CHAT_BASE" -H "Content-Type: application/json" \
           -d "{\"model\":\"$OPENAI_MODEL_NAME_COMPUTED\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" >/dev/null || { echo "Chat POST failed"; exit 1; }; \
         # 4) Run lm-eval
         python3 -m lm_eval --model local-chat-completions \
           --tasks ${EVAL_TASK:-gsm8k} \
           --apply_chat_template \
           --num_fewshot ${NUM_FEWSHOT:-5} \
           --limit ${LIMIT:-200} \
           --batch_size 1 \
           --output_path /workspace/${EVAL_RESULT_DIR:-eval_out} \
           --model_args "model=$OPENAI_MODEL_NAME_COMPUTED,base_url=$OPENAI_CHAT_BASE,api_key=$OPENAI_API_KEY,eos_string=</s>,max_retries=3" \
           --gen_kwargs "max_tokens=16384,temperature=0,top_p=1"'

  set +x

  ##############################################################################
  # Append a Markdown table to the GitHub Actions job summary
  ##############################################################################
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    RES_DIR="${EVAL_RESULT_DIR:-eval_out}"
    RES_FILE="$(ls -1t "$RES_DIR"/*.json 2>/dev/null | head -n1)"

    {
      echo "### GSM8K Evaluation"
      echo ""
      if [ -z "$RES_FILE" ]; then
        echo "> No result JSON found in \`$RES_DIR\`."
      else
        # Prefer Python (usually available on self-hosted); fall back to jq if desired
        if command -v python3 >/dev/null 2>&1; then
python3 - "$FRAMEWORK" "$PRECISION" "$TP" "$EP_SIZE" "$DP_ATTENTION" "$RES_FILE" <<'PY'
import sys, json, re, os
framework, precision, tp, ep, dp, path = sys.argv[1:7]
with open(path, 'r') as f:
    data = json.load(f)

pe = data.get("pretty_env_info","")
gpu_lines = [l for l in pe.splitlines() if l.startswith("GPU ")]
names = [re.sub(r"GPU \d+:\s*", "", l).strip() for l in gpu_lines]
from collections import Counter
c = Counter(names)
gpu_summary = " + ".join([f"{n}\u00D7 {name}" for name, n in c.items()]) if c else "Unknown GPU"
cpu_line = next((l.split(":",1)[1].strip() for l in pe.splitlines() if l.startswith("Model name:")), None)
hardware = gpu_summary + (f" ({cpu_line})" if cpu_line else "")

res = data.get("results",{}).get("gsm8k", {})
strict = res.get("exact_match,strict-match")
flex   = res.get("exact_match,flexible-extract")
strict_se = res.get("exact_match_stderr,strict-match")
flex_se   = res.get("exact_match_stderr,flexible-extract")
n_eff = data.get("n-samples",{}).get("gsm8k",{}).get("effective")

def pct(x): return f"{x*100:.2f}%" if isinstance(x,(int,float)) else "N/A"
def se(x):  return f" \u00B1{(x*100):.2f}%" if isinstance(x,(int,float)) else ""

print("| Hardware | Framework | Precision | TP | EP | DP Attention | EM Strict | EM Flexible | N (eff) |")
print("|---|---|---:|--:|--:|:--:|--:|--:|--:|")
print(f"| {hardware} | {framework} | {precision} | {tp} | {ep} | {str(dp).lower()} | {pct(strict)}{se(strict_se)} | {pct(flex)}{se(flex_se)} | {n_eff or ''} |")

model = data.get("model_name") or data.get("configs",{}).get("gsm8k",{}).get("metadata",{}).get("model")
limit = data.get("config",{}).get("limit")
fewshot = data.get("n-shot",{}).get("gsm8k")
lim_str = str(int(limit)) if isinstance(limit,(int,float)) else str(limit)
print(f"\n_Model_: `{model}` &nbsp;&nbsp; _k-shot_: **{fewshot}** &nbsp;&nbsp; _limit_: **{lim_str}**  \n_Source_: `{os.path.basename(path)}`")
PY
        else
          # Minimal jq fallback (prints only metrics without hardware/CPU inference)
          jq -r --arg fw "$FRAMEWORK" --arg prec "$PRECISION" --arg tp "$TP" --arg ep "$EP_SIZE" --arg dp "$DP_ATTENTION" '
            def pct: (. * 100 | tostring) + "%";
            . as $root
            | "| Hardware | Framework | Precision | TP | EP | DP Attention | EM Strict | EM Flexible | N (eff) |",
              "|---|---|---:|--:|--:|:--:|--:|--:|--:|",
              ("| Unknown GPU | \($fw) | \($prec) | \($tp) | \($ep) | \($dp) | "
               + (.results.gsm8k["exact_match,strict-match"]    | pct) + " | "
               + (.results.gsm8k["exact_match,flexible-extract"]| pct) + " | "
               + (.["n-samples"].gsm8k.effective|tostring) + " |")
          ' "$RES_FILE"
        fi
      fi
      echo ""
    } >> "$GITHUB_STEP_SUMMARY" || true
  fi
  ##############################################################################

else
    # Benchmark mode: original throughput client
    git clone https://github.com/kimbochen/bench_serving.git

    set -x
    docker run --rm --network=host --name=$client_name \
    -v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
    -e HF_TOKEN -e PYTHONPYCACHEPREFIX=/tmp/pycache/ \
    --entrypoint=/bin/bash \
    $IMAGE \
    -lc "pip install -q datasets pandas && \
    python3 bench_serving/benchmark_serving.py \
    --model=$MODEL \
    --backend=vllm \
    --base-url=\"http://localhost:$PORT\" \
    --dataset-name=random \
    --random-input-len=$ISL --random-output-len=$OSL --random-range-ratio=$RANDOM_RANGE_RATIO \
    --num-prompts=$(( $CONC * 10 )) --max-concurrency=$CONC \
    --request-rate=inf --ignore-eos \
    --save-result --percentile-metrics='ttft,tpot,itl,e2el' \
    --result-dir=/workspace/ \
    --result-filename=$RESULT_FILENAME.json"
fi

docker stop $server_name