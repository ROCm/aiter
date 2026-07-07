#!/usr/bin/env bash
set -euo pipefail

NODE_RANK="${NODE_RANK:-0}"
NODE0_ADDR="${NODE0_ADDR:-127.0.0.1}"
IPADDRS="${IPADDRS:-127.0.0.1}"
RUN_DIR="${RUN_DIR:-/run_logs/slurm_job-${SLURM_JOB_ID:-local}}"

MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
BACKEND="${BACKEND:-atom}"
TOPOLOGY="${TOPOLOGY:-unknown}"
DISPLAY_TOPOLOGY="${DISPLAY_TOPOLOGY:-${TOPOLOGY}}"

xP="${xP:-1}"
yD="${yD:-1}"
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-8}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-8}"
PREFILL_ENABLE_DP="${PREFILL_ENABLE_DP:-false}"
DECODE_ENABLE_DP="${DECODE_ENABLE_DP:-false}"

PREFILL_PORT="${PREFILL_PORT:-8010}"
DECODE_PORT="${DECODE_PORT:-8020}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
ROUTER_POLICY="${ROUTER_POLICY:-random}"
PROMETHEUS_PORT="${PROMETHEUS_PORT:-29100}"
HANDSHAKE_PORT="${HANDSHAKE_PORT:-6301}"

KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"
PREFILL_EXTRA_SERVER_ARGS="${PREFILL_EXTRA_SERVER_ARGS:-}"
DECODE_EXTRA_SERVER_ARGS="${DECODE_EXTRA_SERVER_ARGS:-}"
PREFILL_SERVER_ARGS="${EXTRA_SERVER_ARGS}"
DECODE_SERVER_ARGS="${EXTRA_SERVER_ARGS}"
if [[ -n "${PREFILL_EXTRA_SERVER_ARGS}" ]]; then
  PREFILL_SERVER_ARGS="${PREFILL_SERVER_ARGS:+${PREFILL_SERVER_ARGS} }${PREFILL_EXTRA_SERVER_ARGS}"
fi
if [[ -n "${DECODE_EXTRA_SERVER_ARGS}" ]]; then
  DECODE_SERVER_ARGS="${DECODE_SERVER_ARGS:+${DECODE_SERVER_ARGS} }${DECODE_EXTRA_SERVER_ARGS}"
fi

ISL_LIST="${ISL_LIST:-8192}"
OSL="${OSL:-1024}"
CONC_LIST="${CONC_LIST:-4,8}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-${CONC_LIST//,/x}}"
BENCH_NUM_PROMPTS_MULTIPLIER="${BENCH_NUM_PROMPTS_MULTIPLIER:-10}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"
REQUEST_RATE="${REQUEST_RATE:-inf}"

RUN_EVAL="${RUN_EVAL:-false}"
RUN_BENCHMARK="${RUN_BENCHMARK:-true}"
EVAL_TASK="${EVAL_TASK:-gsm8k}"
EVAL_FEWSHOT="${EVAL_FEWSHOT:-3}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-16}"

WAIT_SERVER_TIMEOUT="${WAIT_SERVER_TIMEOUT:-2500}"
WAIT_ROUTER_TIMEOUT="${WAIT_ROUTER_TIMEOUT:-300}"

mkdir -p "${RUN_DIR}"/{logs,benchmark_results,eval_results}

role_tp="${PREFILL_TP_SIZE}"
if [[ "${NODE_RANK}" -ge "${xP}" ]]; then
  role_tp="${DECODE_TP_SIZE}"
fi
if [[ -z "${HIP_VISIBLE_DEVICES:-}" ]]; then
  export HIP_VISIBLE_DEVICES="$(seq -s, 0 "$((role_tp - 1))")"
fi
rm -rf /root/.cache/atom/* 2>/dev/null || true
echo "[runtime] HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"

ensure_nvcc_probe_safe() {
  if command -v nvcc >/dev/null 2>&1 && nvcc --version >/dev/null 2>&1; then
    return 0
  fi
  local fake_bin="/tmp/aiter-ci-bin"
  mkdir -p "${fake_bin}"
  cat > "${fake_bin}/nvcc" <<'EOF'
#!/usr/bin/env bash
echo "nvcc not available in ROCm container"
exit 0
EOF
  chmod +x "${fake_bin}/nvcc"
  export PATH="${fake_bin}:${PATH}"
  echo "[runtime] installed nvcc probe shim at ${fake_bin}/nvcc"
}

ensure_nvcc_probe_safe

apply_prefixed_env() {
  local prefix="$1"
  local role_ip="$2"
  local name raw value
  while IFS='=' read -r name raw; do
    [[ "${name}" == "${prefix}"* ]] || continue
    value="${raw//\$\{ROLE_IP\}/${role_ip}}"
    export "${name#${prefix}}=${value}"
  done < <(env)
}

host_ip="$(echo "${IPADDRS}" | tr ',' '\n' | sed -n "$((NODE_RANK + 1))p")"
if [[ -z "${host_ip}" ]]; then
  host_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi
host_name="$(hostname)"

apply_prefixed_env "AITER_CI_ENV_" "${host_ip}"

IFS=',' read -r -a IP_ARRAY <<< "${IPADDRS}"

prefill_args=()
prefill_ips=()
for idx in $(seq 0 $((xP - 1))); do
  prefill_ips+=("${IP_ARRAY[$idx]}")
  prefill_args+=(--prefill "http://${IP_ARRAY[$idx]}:${PREFILL_PORT}")
done

decode_args=()
decode_ips=()
for idx in $(seq 0 $((yD - 1))); do
  node_idx=$((xP + idx))
  decode_ips+=("${IP_ARRAY[$node_idx]}")
  decode_args+=(--decode "http://${IP_ARRAY[$node_idx]}:${DECODE_PORT}")
done

prefill_parallel=(-tp "${PREFILL_TP_SIZE}")
if [[ "${PREFILL_ENABLE_DP}" == "true" ]]; then
  prefill_parallel+=("--enable-dp-attention")
fi

decode_parallel=(-tp "${DECODE_TP_SIZE}")
if [[ "${DECODE_ENABLE_DP}" == "true" ]]; then
  decode_parallel+=("--enable-dp-attention")
fi

cudagraph_args=()
case "${DECODE_CUDAGRAPH:-}" in
  ""|none|None|NONE|false|False|FALSE|off|Off|OFF|disabled|Disabled|DISABLED)
    cudagraph_args=()
    ;;
  *)
    cudagraph_args=(--cudagraph-capture-sizes "${DECODE_CUDAGRAPH}")
    ;;
esac

server_common=(
  --model "${MODEL_PATH}"
  --host 0.0.0.0
  --trust-remote-code
  --kv_cache_dtype "${KV_CACHE_DTYPE}"
  --block-size "${BLOCK_SIZE}"
  --gpu-memory-utilization "${MEM_FRACTION}"
  --no-enable_prefix_caching
)

wait_http() {
  local url="$1"
  local name="$2"
  local timeout="$3"
  local pid="${4:-}"
  local deadline=$(( $(date +%s) + timeout ))
  echo "[wait] ${name} ${url} timeout=${timeout}s"
  until curl -sf --max-time 10 "${url}" >/dev/null 2>&1; do
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      set +e
      wait "${pid}"
      local rc=$?
      set -e
      [[ "${rc}" -eq 0 ]] && rc=1
      echo "[wait][FAIL] ${name} process exited before becoming ready rc=${rc}" >&2
      exit "${rc}"
    fi
    if [[ "$(date +%s)" -ge "${deadline}" ]]; then
      echo "[wait][FAIL] ${name} not ready after ${timeout}s" >&2
      exit 1
    fi
    sleep 10
  done
  echo "[wait][OK] ${name}"
}

wait_router_closed() {
  local miss_count=0
  local max_misses=3
  echo "[wait] router shutdown http://${NODE0_ADDR}:${ROUTER_PORT}/health"
  while true; do
    if curl -sf --max-time 10 "http://${NODE0_ADDR}:${ROUTER_PORT}/health" >/dev/null 2>&1; then
      miss_count=0
      if [[ -n "${server_pid:-}" ]] && ! kill -0 "${server_pid}" 2>/dev/null; then
        set +e
        wait "${server_pid}"
        local rc=$?
        set -e
        [[ "${rc}" -eq 0 ]] && rc=1
        echo "[wait][FAIL] worker process exited while router was still alive rc=${rc}" >&2
        exit "${rc}"
      fi
    else
      miss_count=$((miss_count + 1))
      if [[ "${miss_count}" -ge "${max_misses}" ]]; then
        break
      fi
      echo "[wait] router health miss ${miss_count}/${max_misses}; continuing"
    fi
    sleep 10
  done
  echo "[wait][OK] router closed"
}

write_metadata() {
  cat > "${RUN_DIR}/metadata-rank-${NODE_RANK}.json" <<EOF
{
  "rank": ${NODE_RANK},
  "host": "${host_name}",
  "ip": "${host_ip}",
  "model": "${MODEL_NAME}",
  "model_path": "${MODEL_PATH}",
  "backend": "${BACKEND}",
  "topology": "${TOPOLOGY}",
  "display_topology": "${DISPLAY_TOPOLOGY}",
  "prefill_ips": "$(IFS=,; echo "${prefill_ips[*]}")",
  "decode_ips": "$(IFS=,; echo "${decode_ips[*]}")"
}
EOF
}

start_prefill() {
  local log_name="$1"
  apply_prefixed_env "AITER_CI_PREFILL_ENV_" "${host_ip}"
  echo "[prefill] rank=${NODE_RANK} host=${host_name} ip=${host_ip}"
  python3 -m atom.entrypoints.openai_server \
    "${server_common[@]}" \
    --server-port "${PREFILL_PORT}" \
    "${prefill_parallel[@]}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --kv-transfer-config "{\"kv_role\":\"kv_producer\",\"kv_connector\":\"mooncake\",\"proxy_ip\":\"${host_ip}\",\"handshake_port\":${HANDSHAKE_PORT}}" \
    ${PREFILL_SERVER_ARGS} \
    2>&1 | tee "${RUN_DIR}/logs/${log_name}.log" &
  server_pid=$!
}

start_decode() {
  apply_prefixed_env "AITER_CI_DECODE_ENV_" "${host_ip}"
  local max_conc
  max_conc="$(echo "${BENCH_MAX_CONCURRENCY}" | tr 'x,' '\n' | sort -n | tail -1)"
  local decode_max_num_seqs="${MAX_NUM_SEQS}"
  if [[ "${ISL_LIST}" == "1024" && "${OSL}" == "1024" ]]; then
    decode_max_num_seqs="${max_conc}"
  fi
  echo "[decode] rank=${NODE_RANK} host=${host_name} ip=${host_ip} cudagraph=${DECODE_CUDAGRAPH:-none}"
  python3 -m atom.entrypoints.openai_server \
    "${server_common[@]}" \
    --server-port "${DECODE_PORT}" \
    "${decode_parallel[@]}" \
    --max-num-seqs "${decode_max_num_seqs}" \
    --kv-transfer-config "{\"kv_role\":\"kv_consumer\",\"kv_connector\":\"mooncake\",\"proxy_ip\":\"${host_ip}\",\"handshake_port\":${HANDSHAKE_PORT}}" \
    "${cudagraph_args[@]}" \
    ${DECODE_SERVER_ARGS} \
    2>&1 | tee "${RUN_DIR}/logs/decode-rank-${NODE_RANK}.log" &
  server_pid=$!
}

start_router() {
  echo "[router] prefill=${prefill_args[*]} decode=${decode_args[*]}"
  /usr/local/bin/atomesh launch \
    --host 0.0.0.0 \
    --port "${ROUTER_PORT}" \
    --pd-disaggregation \
    "${prefill_args[@]}" \
    "${decode_args[@]}" \
    --policy "${ROUTER_POLICY}" \
    --backend atom \
    --log-level info \
    --disable-health-check \
    --disable-circuit-breaker \
    --prometheus-port "${PROMETHEUS_PORT}" \
    2>&1 | tee "${RUN_DIR}/logs/router.log" &
  router_pid=$!
}

run_benchmark() {
  [[ "${RUN_BENCHMARK}" == "true" ]] || [[ "${RUN_BENCHMARK}" == "1" ]] || {
    echo "[bench] RUN_BENCHMARK=${RUN_BENCHMARK}; skipping serving benchmark"
    return 0
  }
  local bench_dir="/tmp/aiter-ci-bench-serving"
  if [[ ! -d "${bench_dir}/bench_serving" ]]; then
    rm -rf "${bench_dir}"
    mkdir -p "${bench_dir}"
    git clone --depth 1 https://github.com/kimbochen/bench_serving.git "${bench_dir}/bench_serving"
  fi
  IFS=',' read -r -a isls <<< "${ISL_LIST}"
  IFS=',' read -r -a concs <<< "${CONC_LIST}"
  local safe_model="${MODEL_NAME//\//-}"
  for isl in "${isls[@]}"; do
    for conc in "${concs[@]}"; do
      local result_file="pd-${BACKEND}-${safe_model}-${TOPOLOGY}-isl${isl}-osl${OSL}-conc${conc}-${RANDOM_RANGE_RATIO}.json"
      echo "[bench] ${result_file}"
      PYTHONDONTWRITEBYTECODE=1 python "${bench_dir}/bench_serving/benchmark_serving.py" \
        --model="${MODEL_PATH}" \
        --backend=vllm \
        --base-url="http://127.0.0.1:${ROUTER_PORT}" \
        --dataset-name=random \
        --random-input-len="${isl}" \
        --random-output-len="${OSL}" \
        --random-range-ratio "${RANDOM_RANGE_RATIO}" \
        --num-prompts="$(( conc * BENCH_NUM_PROMPTS_MULTIPLIER ))" \
        --max-concurrency="${conc}" \
        --trust-remote-code \
        --num-warmups="$(( 2 * conc ))" \
        --request-rate="${REQUEST_RATE}" \
        --ignore-eos \
        --save-result \
        --percentile-metrics='ttft,tpot,itl,e2el' \
        --result-dir="${RUN_DIR}/benchmark_results" \
        --result-filename="${result_file}"
    done
  done
}

run_eval() {
  [[ "${RUN_EVAL}" == "true" ]] || [[ "${RUN_EVAL}" == "1" ]] || return 0
  if [[ "${EVAL_TASK}" != "gsm8k" ]]; then
    echo "[eval] unsupported task ${EVAL_TASK}; skipping"
    return 0
  fi
  if ! command -v lm_eval >/dev/null 2>&1; then
    python3 -m pip install 'lm-eval[api]'
  fi
  local limit_arg=()
  if [[ -n "${EVAL_LIMIT}" ]]; then
    limit_arg=(--limit "${EVAL_LIMIT}")
  fi

  IFS=',' read -r -a eval_concs <<< "${EVAL_CONCURRENCY}"
  local eval_conc tag result_dir
  for eval_conc in "${eval_concs[@]}"; do
    eval_conc="${eval_conc//[[:space:]]/}"
    [[ -n "${eval_conc}" ]] || continue
    tag="$(date +%Y%m%d%H%M%S)_gsm8k_${TOPOLOGY}_c${eval_conc}"
    result_dir="${RUN_DIR}/eval_results/${tag}"

    echo ""
    echo "========================================="
    echo "[eval] gsm8k concurrent=${eval_conc}"
    echo "========================================="

    lm_eval --model local-completions \
      --model_args "model=${MODEL_PATH},base_url=http://127.0.0.1:${ROUTER_PORT}/v1/completions,num_concurrent=${eval_conc},max_retries=3,tokenized_requests=False,trust_remote_code=True" \
      --tasks gsm8k \
      --num_fewshot "${EVAL_FEWSHOT}" \
      "${limit_arg[@]}" \
      --output_path "${result_dir}"

    python3 - "${result_dir}" "${eval_conc}" <<'PY'
import json
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
eval_conc = sys.argv[2]
json_files = list(result_dir.rglob("*.json")) if result_dir.is_dir() else []
if not json_files:
    print("[eval] ERROR: no result JSON found")
    raise SystemExit(1)

result_file = max(json_files, key=lambda path: path.stat().st_mtime)
data = json.loads(result_file.read_text(encoding="utf-8"))
score = (
    data.get("results", {})
    .get("gsm8k", {})
    .get("exact_match,flexible-extract", "N/A")
)
print("=========================================")
print(f"[eval] concurrent={eval_conc} exact_match,flexible-extract = {score}")
print("=========================================")
print(json.dumps(data.get("results", {}), indent=2))
PY
  done

  echo "[eval] gsm8k runs done, results saved to ${RUN_DIR}/eval_results"
}

write_metadata

if [[ "${NODE_RANK}" -eq 0 ]]; then
  start_prefill "prefill-rank-0"
  trap 'kill ${router_pid:-0} ${server_pid:-0} 2>/dev/null || true' EXIT
  for ip in "${prefill_ips[@]}"; do
    wait_http "http://${ip}:${PREFILL_PORT}/health" "prefill-${ip}" "${WAIT_SERVER_TIMEOUT}" "${server_pid}"
  done
  for ip in "${decode_ips[@]}"; do
    wait_http "http://${ip}:${DECODE_PORT}/health" "decode-${ip}" "${WAIT_SERVER_TIMEOUT}"
  done
  start_router
  wait_http "http://127.0.0.1:${ROUTER_PORT}/v1/models" "router" "${WAIT_ROUTER_TIMEOUT}"
  run_eval
  run_benchmark
  kill "${router_pid}" "${server_pid}" 2>/dev/null || true
elif [[ "${NODE_RANK}" -lt "${xP}" ]]; then
  start_prefill "prefill-rank-${NODE_RANK}"
  trap 'kill ${server_pid:-0} 2>/dev/null || true' EXIT
  wait_http "http://${NODE0_ADDR}:${ROUTER_PORT}/health" "router" "${WAIT_SERVER_TIMEOUT}" "${server_pid}"
  wait_router_closed
  kill "${server_pid}" 2>/dev/null || true
else
  start_decode
  trap 'kill ${server_pid:-0} 2>/dev/null || true' EXIT
  wait_http "http://${NODE0_ADDR}:${ROUTER_PORT}/health" "router" "${WAIT_SERVER_TIMEOUT}" "${server_pid}"
  wait_router_closed
  kill "${server_pid}" 2>/dev/null || true
fi
