#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  pd_spur_submit.sh --cell-json <json-or-file> [options]

Options:
  --result-dir <dir>       Local result copy directory. Default: aiter-ci-results
  --scripts-root <dir>     Shared script directory visible on compute nodes.
                           Default: this script's directory
  --atom-repo <dir>        Optional ATOM checkout mounted into the container.
  --dry-run                Expand env and write a dry-run JSON, but do not submit.
  --no-wait                Submit and return immediately.
  -h, --help               Show this help.
USAGE
}

CELL_JSON_ARG=""
RESULT_DIR="${RESULT_DIR:-aiter-ci-results}"
DRY_RUN=0
WAIT_JOB=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AITER_CI_SCRIPTS_ROOT="${AITER_CI_SCRIPTS_ROOT:-${SCRIPT_DIR}}"
ATOM_REPO_ROOT="${ATOM_REPO_ROOT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cell-json)
      CELL_JSON_ARG="$2"
      shift 2
      ;;
    --result-dir)
      RESULT_DIR="$2"
      shift 2
      ;;
    --scripts-root)
      AITER_CI_SCRIPTS_ROOT="$2"
      shift 2
      ;;
    --atom-repo)
      ATOM_REPO_ROOT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-wait)
      WAIT_JOB=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${CELL_JSON_ARG}" ]]; then
  echo "ERROR: --cell-json is required" >&2
  usage >&2
  exit 2
fi

if [[ -f "${CELL_JSON_ARG}" ]]; then
  CELL_JSON="$(<"${CELL_JSON_ARG}")"
else
  CELL_JSON="${CELL_JSON_ARG}"
fi
export CELL_JSON

mkdir -p "${RESULT_DIR}"
JOB_SCRIPT="${AITER_CI_SCRIPTS_ROOT%/}/pd_spur_job.sh"

eval "$(
python3 - <<'PY'
import json
import os
import shlex

cell = json.loads(os.environ["CELL_JSON"])
runner = cell.get("runner", {})
service = cell.get("service", {})
prefill = service.get("prefill", {})
decode = service.get("decode", {})
router = service.get("router", {})
server_args = cell.get("server_args", {})
accuracy = cell.get("accuracy", {})

def shell_value(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return value

def csv_value(value):
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)

def q(value):
    return shlex.quote(str(shell_value(value)))

exports = {
    "AITER_CI_CELL_ID": cell["id"],
    "MODEL_NAME": cell["model"],
    "BACKEND": cell["backend"],
    "DOCKER_IMAGE": cell["image"],
    "MODEL_PATH": cell["model_path"],
    "PRECISION": cell.get("precision", ""),
    "TOPOLOGY": cell["topology"],
    "DISPLAY_TOPOLOGY": cell.get("display_topology", cell["topology"]),
    "NODE_LIST": ",".join(cell["nodes"]),
    "NUM_NODES": cell["num_nodes"],
    "ISL_LIST": ",".join(str(v) for v in cell["isl"]),
    "OSL": cell["osl"],
    "CONC_LIST": ",".join(str(v) for v in cell["concurrency"]),
    "BENCH_MAX_CONCURRENCY": cell["concurrency_x"],
    "RANDOM_RANGE_RATIO": cell["random_range_ratio"],
    "REQUEST_RATE": cell["request_rate"],
    "BENCH_NUM_PROMPTS_MULTIPLIER": cell["num_prompts_multiplier"],
    "WAIT_SERVER_TIMEOUT": cell["wait_server_timeout"],
    "WAIT_ROUTER_TIMEOUT": cell["wait_router_timeout"],
    "PREFILL_WORKERS": prefill.get("workers", 1),
    "DECODE_WORKERS": decode.get("workers", 1),
    "PREFILL_TP": prefill.get("tp", 8),
    "DECODE_TP": decode.get("tp", 8),
    "PREFILL_ENABLE_DP": str(prefill.get("enable_dp_attention", False)).lower(),
    "DECODE_ENABLE_DP": str(decode.get("enable_dp_attention", False)).lower(),
    "DECODE_CUDAGRAPH": decode.get("cudagraph", ""),
    "PREFILL_PORT": prefill.get("port", 8010),
    "DECODE_PORT": decode.get("port", 8020),
    "ROUTER_PORT": router.get("port", 8000),
    "ROUTER_POLICY": router.get("policy", "random"),
    "PROMETHEUS_PORT": router.get("prometheus_port", 29100),
    "KV_CACHE_DTYPE": server_args.get("kv_cache_dtype", "fp8"),
    "BLOCK_SIZE": server_args.get("block_size", 16),
    "MEM_FRACTION": server_args.get("gpu_memory_utilization", 0.85),
    "MAX_NUM_SEQS": server_args.get("max_num_seqs", 256),
    "EXTRA_SERVER_ARGS": server_args.get("extra_args", ""),
    "PREFILL_EXTRA_SERVER_ARGS": prefill.get("extra_args", ""),
    "DECODE_EXTRA_SERVER_ARGS": decode.get("extra_args", ""),
    "RUN_EVAL": str(cell.get("run_eval", False)).lower(),
    "RUN_BENCHMARK": str(cell.get("run_benchmark", True)).lower(),
    "EVAL_TASK": accuracy.get("task", "gsm8k"),
    "EVAL_FEWSHOT": accuracy.get("fewshot", 3),
    "EVAL_LIMIT": "" if accuracy.get("limit") is None else accuracy.get("limit"),
    "EVAL_CONCURRENCY": csv_value(accuracy.get("concurrency") or cell.get("concurrency", [])),
    "SPUR_ACCOUNT": runner.get("slurm_account", "amd-frameworks"),
    "SPUR_PARTITION": runner.get("slurm_partition", "amd-frameworks"),
    "SPUR_CPUS_PER_TASK": runner.get("cpus_per_task", 114),
    "SPUR_GPUS_PER_NODE": runner.get("gpus_per_node", 8),
    "SPUR_TIME_LIMIT": runner.get("time_limit", "08:00:00"),
    "SPUR_LOG_ROOT": runner.get("log_root", "/data/jiaolyu/aiter_ci/logs/"),
}

for key, value in exports.items():
    print(f"export {key}={q(value)}")

for key, value in cell.get("env", {}).get("common", {}).items():
    print(f"export AITER_CI_ENV_{key}={q(value)}")
for key, value in cell.get("env", {}).get("prefill", {}).items():
    print(f"export AITER_CI_PREFILL_ENV_{key}={q(value)}")
for key, value in cell.get("env", {}).get("decode", {}).items():
    print(f"export AITER_CI_DECODE_ENV_{key}={q(value)}")
PY
)"

export RESULT_DIR
export AITER_CI_SCRIPTS_ROOT
export ATOM_REPO_ROOT
export LOG_ROOT="${SPUR_LOG_ROOT%/}/${AITER_CI_CELL_ID}-${GITHUB_RUN_ID:-local}-$(date +%Y%m%d%H%M%S)"
export SLURM_OUTPUT="/tmp/spur-%j.out"
export SLURM_ERROR="/tmp/spur-%j.err"

echo "=== AITER CI Spur benchmark cell ==="
echo "cell=${AITER_CI_CELL_ID}"
echo "model=${MODEL_NAME}"
echo "topology=${DISPLAY_TOPOLOGY}"
echo "nodes=${NODE_LIST}"
echo "isl=${ISL_LIST} osl=${OSL} concurrency=${CONC_LIST}"
echo "scripts_root=${AITER_CI_SCRIPTS_ROOT}"
echo "atom_repo=${ATOM_REPO_ROOT:-<image-installed>}"
echo "log_root=${LOG_ROOT}"
echo "job_script=${JOB_SCRIPT}"

mkdir -p "${RESULT_DIR}"

copy_log_root_to_result() {
  [[ -d "${LOG_ROOT}" ]] || return 0
  mkdir -p "${RESULT_DIR}/${AITER_CI_CELL_ID}"
  cp -R "${LOG_ROOT}/." "${RESULT_DIR}/${AITER_CI_CELL_ID}/" || true
}

write_dashboard_summary() {
  [[ -n "${JOB_ID:-}" ]] || return 0
  local run_dir="${LOG_ROOT%/}/spur_job-${JOB_ID}"
  local out_dir="${RESULT_DIR}/${AITER_CI_CELL_ID}"
  local processor="${AITER_CI_SCRIPTS_ROOT%/}/process_result.py"
  [[ -f "${processor}" && -d "${run_dir}" ]] || return 0
  mkdir -p "${out_dir}"
  python3 "${processor}" "${run_dir}" \
    --output "${out_dir}/${AITER_CI_CELL_ID}-benchmark-action.json" \
    --summary "${out_dir}/summary.md" || true
}

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "=== dry-run only; sbatch is not invoked ==="
  python3 - <<'PY'
import json
import os
from pathlib import Path
cell = json.loads(os.environ["CELL_JSON"])
Path(os.environ["RESULT_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["RESULT_DIR"], f"{cell['id']}-spur-dry-run.json").write_text(
    json.dumps(
        {
            "cell": cell,
            "log_root": os.environ["LOG_ROOT"],
            "scripts_root": os.environ["AITER_CI_SCRIPTS_ROOT"],
            "atom_repo": os.environ.get("ATOM_REPO_ROOT", ""),
        },
        indent=2,
    ),
    encoding="utf-8",
)
PY
  exit 0
fi

if [[ ! -f "${JOB_SCRIPT}" ]]; then
  echo "ERROR: job script not found: ${JOB_SCRIPT}" >&2
  echo "Use --scripts-root with a directory visible to Spur compute nodes." >&2
  exit 2
fi

mkdir -p "${LOG_ROOT}"

SUBMIT_MODE="${AITER_CI_SUBMIT_MODE:-sbatch}"
if [[ "${SUBMIT_MODE}" == "srun" ]]; then
  if ! command -v srun >/dev/null 2>&1; then
    echo "ERROR: srun not found. Ensure Spur symlinks are on PATH or run through a Spur submit host." >&2
    exit 127
  fi
  SRUN_CMD=(
    srun
    --partition "${SPUR_PARTITION}"
    --nodes "${NUM_NODES}"
    --ntasks "${NUM_NODES}"
    --time "${SPUR_TIME_LIMIT}"
    "${JOB_SCRIPT}"
  )
  echo "=== running Spur job with srun ==="
  printf ' %q' "${SRUN_CMD[@]}"
  echo
  set +e
  "${SRUN_CMD[@]}" 2>&1 | tee "${LOG_ROOT}/srun.out"
  SUBMIT_RC=${PIPESTATUS[0]}
  set -e
  copy_log_root_to_result
  exit "${SUBMIT_RC}"
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Ensure Spur symlinks are on PATH or run through a Spur submit host." >&2
  exit 127
fi

SUBMIT_MODE="${AITER_CI_SUBMIT_MODE:-spur-script}"
if [[ "${SUBMIT_MODE}" == "spur-script" ]]; then
  SUBMIT_SCRIPT="${LOG_ROOT}/submit-${AITER_CI_CELL_ID}.sbatch.sh"
  cat > "${SUBMIT_SCRIPT}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${AITER_CI_CELL_ID}
#SBATCH --nodes=${NUM_NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --time=${SPUR_TIME_LIMIT}
#SBATCH --chdir=/tmp
EOF
  if [[ -n "${SPUR_PARTITION}" ]]; then
    printf '#SBATCH --partition=%s\n' "${SPUR_PARTITION}" >> "${SUBMIT_SCRIPT}"
  fi
  if [[ "${AITER_CI_USE_NODELIST:-0}" == "1" && -n "${NODE_LIST}" ]]; then
    printf '#SBATCH --nodelist=%s\n' "${NODE_LIST}" >> "${SUBMIT_SCRIPT}"
  fi
  cat >> "${SUBMIT_SCRIPT}" <<EOF
#SBATCH --output=${SLURM_OUTPUT}
#SBATCH --error=${SLURM_ERROR}
EOF
  cat >> "${SUBMIT_SCRIPT}" <<'EOF'
mkdir -p "${LOG_ROOT}"
exec > "${LOG_ROOT}/spur-job-${SPUR_JOB_ID:-${SLURM_JOB_ID:-unknown}}.log" 2>&1
EOF
  {
    printf 'export %q=%q\n' AITER_CI_SCRIPTS_ROOT "${AITER_CI_SCRIPTS_ROOT}"
    printf 'export %q=%q\n' ATOM_REPO_ROOT "${ATOM_REPO_ROOT:-}"
    printf 'export %q=%q\n' LOG_ROOT "${LOG_ROOT}"
    printf 'exec bash %q\n' "${JOB_SCRIPT}"
  } >> "${SUBMIT_SCRIPT}"
  chmod +x "${SUBMIT_SCRIPT}"
  if [[ -n "${SPUR_CONTROLLER_ADDR:-}" ]]; then
    SBATCH_CMD=(sbatch --controller "${SPUR_CONTROLLER_ADDR}" "${SUBMIT_SCRIPT}")
  else
    SBATCH_CMD=(sbatch "${SUBMIT_SCRIPT}")
  fi
else
  SBATCH_CMD=(
    sbatch
    --partition "${SPUR_PARTITION}"
    --nodes "${NUM_NODES}"
    --ntasks "${NUM_NODES}"
    --ntasks-per-node 1
    --cpus-per-task "${SPUR_CPUS_PER_TASK}"
    --time "${SPUR_TIME_LIMIT}"
  )
  if [[ -n "${SPUR_ACCOUNT}" ]]; then
    SBATCH_CMD+=(--account "${SPUR_ACCOUNT}")
  fi
  if [[ -n "${SPUR_GPUS_PER_NODE}" && "${SPUR_GPUS_PER_NODE}" != "0" ]]; then
    SBATCH_CMD+=(--gres "gpu:amdgpu-0x75b0:${SPUR_GPUS_PER_NODE}")
  fi
  if [[ "${AITER_CI_USE_NODELIST:-0}" == "1" && -n "${NODE_LIST}" ]]; then
    SBATCH_CMD+=(--nodelist "${NODE_LIST}")
  fi
  SBATCH_CMD+=(
    --output "${SLURM_OUTPUT}"
    --error "${SLURM_ERROR}"
    "${JOB_SCRIPT}"
  )
fi

echo "=== submitting Spur job ==="
printf ' %q' "${SBATCH_CMD[@]}"
echo

set +e
SUBMIT_OUTPUT="$("${SBATCH_CMD[@]}")"
SUBMIT_RC=$?
set -e
echo "${SUBMIT_OUTPUT}"
if [[ "${SUBMIT_RC}" -ne 0 ]]; then
  exit "${SUBMIT_RC}"
fi

JOB_ID="$(printf '%s\n' "${SUBMIT_OUTPUT}" | sed -nE 's/.*batch job ([0-9]+).*/\1/p; s/^([0-9]+).*/\1/p' | head -n 1)"
if [[ -z "${JOB_ID}" ]]; then
  echo "WARNING: failed to parse job id from submit output" >&2
  exit 0
fi
echo "${JOB_ID}" | tee "${RESULT_DIR}/${AITER_CI_CELL_ID}.spur-job-id"

if [[ "${WAIT_JOB}" -eq 0 ]]; then
  echo "submitted job ${JOB_ID}; not waiting"
  exit 0
fi

echo "=== monitoring Spur job ${JOB_ID} ==="
while true; do
  set +e
  QUEUE_OUTPUT="$(squeue -h -j "${JOB_ID}" 2>/dev/null)"
  QUEUE_RC=$?
  set -e
  if [[ "${QUEUE_RC}" -ne 0 || -z "${QUEUE_OUTPUT}" ]]; then
    break
  fi
  printf '%s\n' "${QUEUE_OUTPUT}"
  sleep "${SPUR_LOG_POLL_INTERVAL:-30}"
done

if command -v scontrol >/dev/null 2>&1; then
  scontrol show job "${JOB_ID}" | tee "${RESULT_DIR}/${AITER_CI_CELL_ID}.spur-job-info" || true
fi

copy_log_root_to_result
write_dashboard_summary

echo "=== job ${JOB_ID} finished or left active queue ==="
