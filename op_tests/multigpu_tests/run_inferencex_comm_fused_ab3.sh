#!/usr/bin/env bash
# SPDX-License-Identifier: MIT

set -Eeuo pipefail
umask 022
ulimit -c 0

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AITER_REPO=${AITER_REPO:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}
CONTAINER=${CONTAINER:-}
ATOM_REPO=${ATOM_REPO:-}
MODEL=${MODEL:-}
DATA_ROOT=${DATA_ROOT:-${TMPDIR:-/tmp}/aiter_comm_fused_moe}
HF_HOME_DIR=${HF_HOME_DIR:-}
TEST_ROOT=${TEST_ROOT:-$DATA_ROOT/comm_fused_moe_tests}
JIT_DIR=${JIT_DIR:-$DATA_ROOT/comm_fused_moe_inferencex_ab3_jit}
FMOE_TABLE=${FMOE_TABLE:-$AITER_REPO/aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv}
COMM_FUSED_TABLE=${COMM_FUSED_TABLE:-$AITER_REPO/aiter/configs/comm_fused_moe.csv}
ROCPROF=${ROCPROF:-/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/bin/rocprofv3}
ROCM_CORE=${ROCM_CORE:-/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core}
ROCM_DEVEL=${ROCM_DEVEL:-/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel}
ROCPROF_COMPAT_ROOT=${ROCPROF_COMPAT_ROOT:-$TEST_ROOT/rocprof_core_compat}
PORT=${PORT:-8000}
PHASE=all
SCENARIO_FILTER=
RESULT_ROOT=
RESUME=0
DRY_RUN=0
TRACE_ATTACH_SECONDS=${TRACE_ATTACH_SECONDS:-180}
SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT:-1800}
ARM_TIMEOUT=${ARM_TIMEOUT:-14400}
COOLDOWN_SECONDS=${COOLDOWN_SECONDS:-30}
SAVE_DETAILED=${SAVE_DETAILED:-0}
SEED_BASE=${SEED_BASE:-1000}
PAIR_FILTER=${PAIR_FILTER:-}
LATEST_POINTER=$TEST_ROOT/.comm_fused_ab3_latest
MAIN_BASHPID=$BASHPID

usage() {
    cat <<'EOF'
Usage: run_inferencex_comm_fused_ab3.sh [options]

  --phase preflight|trace|synthetic|persistent|agentic|summarize|all
  --scenario ID       Run one scenario from the selected phase.
  --result-root PATH  Explicit result directory.
  --resume            Resume from completed pair.done markers.
  --dry-run           Validate inputs and print the selected matrix only.
  -h, --help

Fresh runs create DATA_ROOT/comm_fused_moe_inferencex_ab3_<UTC>.
Resume without --result-root uses the last result directory recorded outside
the repositories in .comm_fused_ab3_latest.
EOF
}

while (($#)); do
    case "$1" in
        --phase)
            PHASE=${2:?missing value for --phase}
            shift 2
            ;;
        --scenario)
            SCENARIO_FILTER=${2:?missing value for --scenario}
            shift 2
            ;;
        --result-root)
            RESULT_ROOT=${2:?missing value for --result-root}
            shift 2
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$PHASE" in
    preflight|trace|synthetic|persistent|agentic|summarize|all) ;;
    *)
        echo "invalid --phase: $PHASE" >&2
        exit 2
        ;;
esac

if [[ -z "$RESULT_ROOT" ]]; then
    if ((RESUME)); then
        [[ -s "$LATEST_POINTER" ]] || {
            echo "--resume requires --result-root or $LATEST_POINTER" >&2
            exit 2
        }
        RESULT_ROOT=$(<"$LATEST_POINTER")
    else
        RESULT_ROOT=$DATA_ROOT/comm_fused_moe_inferencex_ab3_$(date -u +%Y%m%dT%H%M%SZ)
    fi
fi

RUN_LOG=$RESULT_ROOT/run.log
MANIFEST=$RESULT_ROOT/scenario_manifest.tsv
OWNED_PID_FILE=$RESULT_ROOT/server.pid
CURRENT_ARM=none
ENABLE_ROCP_ATTACH=0
CREATED_CORE_ATTACH_LINK=0

timestamp() {
    date -u '+%F %T UTC'
}

log() {
    local line
    line="[$(timestamp)] $*"
    echo "$line"
    if [[ -d "$RESULT_ROOT" ]]; then
        echo "$line" >> "$RUN_LOG"
    fi
}

die() {
    log "FATAL $*"
    exit 1
}

emit_scenarios() {
    local variant isl conc
    printf 'id\tphase\tvariant\tisl\tosl\tconcurrency\tmax_num_batched_tokens\tmax_model_len\tmax_num_seqs\n'
    for variant in base mtp3; do
        for isl in 1024 8192; do
            for conc in 32 64 128 256; do
                printf '%s\t%s\t%s\t%s\t1024\t%s\t16384\t9472\t512\n' \
                    "${variant}_${isl}i_1024o_c${conc}" synthetic "$variant" "$isl" "$conc"
            done
        done
    done
    for isl in 1024 8192; do
        for conc in 64 128 256; do
            printf '%s\t%s\t%s\t%s\t1024\t%s\t16384\t9472\t512\n' \
                "tbo_${isl}i_1024o_c${conc}" synthetic tbo "$isl" "$conc"
        done
    done
    printf 'base_8192i_1024o_c64_mnbt32768\tpersistent\tbase\t8192\t1024\t64\t32768\t9472\t512\n'
    for conc in 16 32 48; do
        printf '%s\t%s\t%s\t0\t0\t%s\t16384\t131072\t96\n' \
            "agentic_c${conc}" agentic agentic "$conc"
    done
}

emit_trace_scenarios() {
    cat <<'EOF'
id	variant	isl	osl	concurrency	max_num_batched_tokens	max_model_len	max_num_seqs	expected
T1_atomic	base	1024	16	256	16384	9472	512	atomic
T2_full_window	base	8192	16	32	16384	9472	512	full_or_window
T3_persistent	base	8192	16	64	32768	9472	512	persistent
T4_mtp3	mtp3	1024	16	64	16384	9472	512	any_fused
T5_tbo	tbo	1024	16	64	16384	9472	512	any_fused
T6_dpa_negative	dpa	1024	16	64	16384	9472	512	none
EOF
}

variant_server_args() {
    case "$1" in
        base) ;;
        mtp3) printf '%s\n' '--method' 'mtp' '--num-speculative-tokens' '3' ;;
        tbo) printf '%s\n' '--enable-tbo' ;;
        dpa) printf '%s\n' '--enable-dp-attention' ;;
        agentic) ;;
        *) die "unknown variant $1" ;;
    esac
}

variant_benchmark_args() {
    case "$1" in
        mtp3) printf '%s\n' '--use-chat-template' ;;
        base|tbo|dpa|agentic) ;;
        *) die "unknown variant $1" ;;
    esac
}

selected_scenarios() {
    emit_scenarios | awk -F '\t' -v phase="$PHASE" -v wanted="$SCENARIO_FILTER" '
        NR == 1 { next }
        (phase == "all" || $2 == phase) && (wanted == "" || $1 == wanted) { print }
    '
}

check_static_inputs() {
    [[ -n "$CONTAINER" ]] || die "set CONTAINER to the ATOM Docker container"
    [[ -n "$ATOM_REPO" ]] || die "set ATOM_REPO to the ATOM repository"
    [[ -n "$MODEL" ]] || die "set MODEL to the model directory"
    [[ -d "$AITER_REPO/.git" ]] || die "missing AITer repo: $AITER_REPO"
    [[ -d "$ATOM_REPO/.git" ]] || die "missing ATOM repo: $ATOM_REPO"
    [[ -d "$MODEL" ]] || die "missing model: $MODEL"
    [[ -r "$FMOE_TABLE" ]] || die "missing FMoE table: $FMOE_TABLE"
    [[ -r "$COMM_FUSED_TABLE" ]] || die "missing comm-fused table: $COMM_FUSED_TABLE"
    command -v docker >/dev/null || die "docker is unavailable"
    docker inspect "$CONTAINER" >/dev/null 2>&1 || die "missing container: $CONTAINER"
    docker exec "$CONTAINER" test -x "$ROCPROF" || die "rocprofv3 is unavailable: $ROCPROF"

    local count
    count=$(selected_scenarios | sed '/^$/d' | wc -l)
    if [[ "$PHASE" != preflight && "$PHASE" != trace && "$PHASE" != summarize && "$count" -eq 0 ]]; then
        die "no scenario selected for phase=$PHASE filter=$SCENARIO_FILTER"
    fi
}

container_processes() {
    docker exec "$CONTAINER" bash -lc \
        "ps -eo pid=,pgid=,stat=,args= | grep '[a]tom.entrypoints.openai_server'" 2>/dev/null || true
}

stop_owned_server() {
    [[ -s "$OWNED_PID_FILE" ]] || return 0
    local pid
    pid=$(<"$OWNED_PID_FILE")
    if [[ "$pid" =~ ^[0-9]+$ ]]; then
        docker exec "$CONTAINER" bash -lc \
            "kill -TERM -- -$pid 2>/dev/null || true" || true
        for _ in $(seq 1 30); do
            if ! docker exec "$CONTAINER" kill -0 "$pid" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        docker exec "$CONTAINER" bash -lc \
            "kill -KILL -- -$pid 2>/dev/null || true" || true
    fi
    rm -f -- "$OWNED_PID_FILE"
}

prepare_rocprof_attach() {
    local core_attach=$ROCM_CORE/lib/librocprofiler-sdk-attach.so
    mkdir -p "$ROCPROF_COMPAT_ROOT/bin" "$ROCPROF_COMPAT_ROOT/lib/rocprofiler-sdk"
    docker exec "$CONTAINER" install -m 0755 \
        "$ROCM_DEVEL/bin/rocprof-attach" \
        "$ROCPROF_COMPAT_ROOT/bin/rocprof-attach"
    docker exec "$CONTAINER" ln -sfn \
        "$ROCM_CORE/lib/librocprofiler-sdk-rocattach.so.1" \
        "$ROCPROF_COMPAT_ROOT/lib/librocprofiler-sdk-rocattach.so"
    docker exec "$CONTAINER" ln -sfn \
        "$ROCM_CORE/lib/librocprofiler-sdk.so.1" \
        "$ROCPROF_COMPAT_ROOT/lib/librocprofiler-sdk.so"
    docker exec "$CONTAINER" ln -sfn \
        "$ROCM_CORE/lib/librocprofiler-sdk-roctx.so.1" \
        "$ROCPROF_COMPAT_ROOT/lib/librocprofiler-sdk-roctx.so"
    docker exec "$CONTAINER" ln -sfn \
        "$ROCM_CORE/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so.1" \
        "$ROCPROF_COMPAT_ROOT/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so"
    docker exec "$CONTAINER" ln -sfn \
        "$ROCM_CORE/lib/rocprofiler-sdk/librocprofiler-sdk-tool-kokkosp.so.1" \
        "$ROCPROF_COMPAT_ROOT/lib/rocprofiler-sdk/librocprofiler-sdk-tool-kokkosp.so"
    docker exec "$CONTAINER" ln -sfn \
        "$ROCM_CORE/lib/rocprofiler-sdk/librocprofv3-list-avail.so.1" \
        "$ROCPROF_COMPAT_ROOT/lib/rocprofiler-sdk/librocprofv3-list-avail.so"
    if ! docker exec "$CONTAINER" test -e "$core_attach"; then
        docker exec "$CONTAINER" ln -s \
            librocprofiler-sdk-attach.so.1 "$core_attach"
        CREATED_CORE_ATTACH_LINK=1
    fi
}

cleanup_rocprof_attach() {
    if [[ "$CREATED_CORE_ATTACH_LINK" -eq 1 ]]; then
        docker exec "$CONTAINER" rm -f -- \
            "$ROCM_CORE/lib/librocprofiler-sdk-attach.so" || true
        CREATED_CORE_ATTACH_LINK=0
    fi
}

find_trace_target() {
    docker exec "$CONTAINER" bash -lc '
        for proc in /proc/[0-9]*; do
            pid=${proc#/proc/}
            cmd=$(tr "\0" " " < "$proc/cmdline" 2>/dev/null) || continue
            case "$cmd" in
                *"::TP0"*|*"::DP0TP0"*)
                    if grep -qsx rocp-bg-attach "$proc"/task/*/comm; then
                        printf "%s\n" "$pid"
                    fi
                    ;;
            esac
        done
    '
}

on_exit() {
    local rc=$?
    [[ "$BASHPID" -eq "$MAIN_BASHPID" ]] || return 0
    stop_owned_server || true
    cleanup_rocprof_attach || true
    if [[ "$rc" -ne 0 ]]; then
        log "EXIT rc=$rc current_arm=$CURRENT_ARM"
    fi
}
trap on_exit EXIT INT TERM

capture_provenance() {
    local out=$RESULT_ROOT/provenance.txt
    {
        echo "start_utc=$(timestamp)"
        echo "host=$(hostname)"
        echo "slurm_job_id=${SLURM_JOB_ID:-unset}"
        echo "container=$CONTAINER"
        echo "container_image=$(docker inspect -f '{{.Image}}' "$CONTAINER")"
        echo "aiter_head=$(git -C "$AITER_REPO" rev-parse HEAD)"
        echo "aiter_branch=$(git -C "$AITER_REPO" branch --show-current)"
        echo "aiter_status_begin"
        git -C "$AITER_REPO" status --short
        echo "aiter_status_end"
        echo "atom_head=$(git -C "$ATOM_REPO" rev-parse HEAD)"
        echo "atom_branch=$(git -C "$ATOM_REPO" branch --show-current)"
        echo "atom_status_begin"
        git -C "$ATOM_REPO" status --short
        echo "atom_status_end"
        sha256sum "$FMOE_TABLE" "$COMM_FUSED_TABLE"
        echo "model=$MODEL"
        echo "jit_dir=$JIT_DIR"
        echo "phase=$PHASE"
        echo "scenario_filter=${SCENARIO_FILTER:-all}"
        echo "save_detailed=$SAVE_DETAILED"
        echo "seed_base=$SEED_BASE"
        docker exec "$CONTAINER" bash -lc \
            "python3 --version; $ROCPROF --version; rocm-smi --showproductname --showdriverversion" 2>&1
    } > "$out"
}

run_preflight() {
    log "STEP1 preflight begin"
    check_static_inputs
    [[ -n "${SLURM_JOB_ID:-}" ]] || die "must run through the GPU mailbox allocation"
    local stale
    stale=$(container_processes)
    [[ -z "$stale" ]] || die "unowned ATOM server is already running: $stale"

    mkdir -p "$RESULT_ROOT" "$JIT_DIR"
    emit_scenarios > "$MANIFEST"
    emit_trace_scenarios > "$RESULT_ROOT/trace_manifest.tsv"
    capture_provenance

    docker exec "$CONTAINER" bash -lc "
        set -e
        test -d '$MODEL'
        test -x '$ROCPROF'
        python3 -m atom.benchmarks.benchmark_serving --help >/dev/null
        python3 -m atom.entrypoints.openai_server --help >/dev/null
        if test -x /workspace/venvs/aiperf-sa/bin/aiperf; then
            echo /workspace/venvs/aiperf-sa/bin/aiperf
        elif command -v aiperf >/dev/null 2>&1; then
            command -v aiperf
        else
            echo unavailable
        fi
    " > "$RESULT_ROOT/preflight_container.txt" 2>&1

    if rg -q '^unavailable$' "$RESULT_ROOT/preflight_container.txt"; then
        log "STEP1 agentic harness unavailable; synthetic/trace remain runnable"
    else
        log "STEP1 agentic harness detected"
    fi
    log "STEP1 preflight complete result_root=$RESULT_ROOT"
}

wait_server() {
    local server_log=$1
    local elapsed=0
    while ((elapsed < SERVER_READY_TIMEOUT)); do
        local code
        code=$(curl -s --connect-timeout 2 --max-time 5 \
            -o /dev/null -w '%{http_code}' \
            "http://localhost:$PORT/v1/models" 2>/dev/null || true)
        if [[ "$code" == 200 ]]; then
            log "SERVER_READY arm=$CURRENT_ARM elapsed_s=$elapsed"
            return 0
        fi
        local pid=
        [[ -s "$OWNED_PID_FILE" ]] && pid=$(<"$OWNED_PID_FILE")
        if [[ -n "$pid" ]] && ! docker exec "$CONTAINER" kill -0 "$pid" 2>/dev/null; then
            tail -n 200 "$server_log" || true
            die "server died before readiness arm=$CURRENT_ARM pid=$pid"
        fi
        if ((elapsed > 0 && elapsed % 60 == 0)); then
            log "SERVER_WAIT arm=$CURRENT_ARM elapsed_s=$elapsed"
            rg -n 'Loading|Captur|Traceback|ERROR|Error' "$server_log" | tail -30 || true
        fi
        sleep 10
        elapsed=$((elapsed + 10))
    done
    tail -n 300 "$server_log" || true
    die "server readiness timeout arm=$CURRENT_ARM"
}

launch_server() {
    local arm_dir=$1 variant=$2 disable=$3 mnbt=$4 max_model_len=$5 max_seqs=$6 prefix_cache=$7
    local server_log=$arm_dir/server.log
    local -a args=(
        python3 -m atom.entrypoints.openai_server
        --model "$MODEL"
        -tp 8
        --moe-backend standard
        --kv_cache_dtype fp8
        --trust-remote-code
        --gpu-memory-utilization 0.9
        --max-num-batched-tokens "$mnbt"
        --max-model-len "$max_model_len"
        --max-num-seqs "$max_seqs"
        --level 0
        --cudagraph-mode FULL
        --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}'
    )
    if [[ "$prefix_cache" == 0 ]]; then
        args+=(--no-enable_prefix_caching)
    fi
    while IFS= read -r arg; do
        [[ -n "$arg" ]] && args+=("$arg")
    done < <(variant_server_args "$variant")

    printf '%q ' "${args[@]}" > "$arm_dir/server_command.txt"
    printf '\n' >> "$arm_dir/server_command.txt"

    local command_q log_q pid_q
    printf -v command_q '%q ' "${args[@]}"
    printf -v log_q '%q' "$server_log"
    printf -v pid_q '%q' "$OWNED_PID_FILE"

    local -a envs=(
        -e "PYTHONPATH=$AITER_REPO:$ATOM_REPO"
        -e "AITER_META_DIR=$AITER_REPO"
        -e "CK_DIR=$AITER_REPO/3rdparty/composable_kernel"
        -e "AITER_JIT_DIR=$JIT_DIR"
        -e "AITER_CONFIG_FMOE=$FMOE_TABLE"
        -e "AITER_DISABLE_COMM_FUSED_MOE=$disable"
        -e AITER_LOG_LEVEL=INFO
        -e AITER_USE_SYSTEM_TRITON=1
        -e AITER_BF16_FP8_MOE_BOUND=0
        -e ATOM_MOE_GU_ITLV=1
        -e ATOM_NUMA_BIND=1
        -e CU_NUM=256
        -e HF_HUB_DISABLE_XET=1
        -e HSA_COREDUMP_PATTERN=/dev/null
        -e PYTHONUNBUFFERED=1
        -e MORI_SHMEM_HEAP_SIZE=2G
    )
    if [[ -n "$HF_HOME_DIR" ]]; then
        envs+=(-e "HF_HOME=$HF_HOME_DIR")
    fi
    if [[ "$variant" == tbo ]]; then
        envs+=(-e GPU_MAX_HW_QUEUES=5)
    fi
    if ((ENABLE_ROCP_ATTACH)); then
        envs+=(-e ROCP_TOOL_ATTACH=1)
        envs+=(-e "LD_PRELOAD=$ROCM_CORE/lib/librocprofiler-register.so.0")
    fi

    local existing
    existing=$(container_processes)
    [[ -z "$existing" ]] || die "refusing to kill unowned ATOM process: $existing"

    log "SERVER_LAUNCH arm=$CURRENT_ARM variant=$variant disable=$disable mnbt=$mnbt"
    docker exec -d "${envs[@]}" -w "$ATOM_REPO" "$CONTAINER" bash -lc \
        "setsid $command_q > $log_q 2>&1 & echo \$! > $pid_q"
    wait_server "$server_log"
}

run_random_benchmark() {
    local arm_dir=$1 variant=$2 isl=$3 osl=$4 conc=$5 prompts=$6 warmups=$7 seed=$8
    local -a args=(
        python3 -m atom.benchmarks.benchmark_serving
        --model "$MODEL"
        --backend vllm
        --base-url "http://localhost:$PORT"
        --dataset-name random
        --random-input-len "$isl"
        --random-output-len "$osl"
        --random-range-ratio 0.8
        --max-concurrency "$conc"
        --num-prompts "$prompts"
        --num-warmups "$warmups"
        --request-rate inf
        --ignore-eos
        --seed "$seed"
        --disable-tqdm
        --percentile-metrics ttft,tpot,itl,e2el
        --metric-percentiles 90,99,99.9
        --save-result
        --result-dir "$arm_dir"
        --result-filename result.json
    )
    if ((SAVE_DETAILED)); then
        args+=(--save-detailed)
    fi
    while IFS= read -r arg; do
        [[ -n "$arg" ]] && args+=("$arg")
    done < <(variant_benchmark_args "$variant")
    printf '%q ' "${args[@]}" > "$arm_dir/benchmark_command.txt"
    printf '\n' >> "$arm_dir/benchmark_command.txt"

    timeout -s KILL "$ARM_TIMEOUT" docker exec \
        -e "PYTHONPATH=$AITER_REPO:$ATOM_REPO" \
        -e PYTHONUNBUFFERED=1 \
        -w "$ATOM_REPO" \
        "$CONTAINER" "${args[@]}" > "$arm_dir/benchmark.log" 2>&1
}

verify_random_arm() {
    local arm_dir=$1 prompts=$2
    python3 - "$arm_dir" "$prompts" <<'PY'
import json
import re
import sys
from collections import Counter
from pathlib import Path

arm = Path(sys.argv[1])
expected = int(sys.argv[2])
result = json.loads((arm / "result.json").read_text())
if result.get("completed") != expected:
    raise SystemExit(
        f"completed={result.get('completed')} expected={expected} in {arm}"
    )
server = (arm / "server.log").read_text(errors="replace")
fatal = re.compile(
    r"Traceback \(most recent call last\)|GPU memory fault|Memory access fault|"
    r"HTTP/[0-9.]+\" 5[0-9][0-9]|RuntimeError:|KeyError:"
)
matches = fatal.findall(server)
if matches:
    raise SystemExit(f"fatal server markers in {arm}: {Counter(matches)}")

raw = Counter(
    int(m.group(1))
    for m in re.finditer(r"Scheduled prefill batch: \d+ reqs, (\d+) new tokens", server)
)
def padded(n: int) -> int:
    if n < 32768:
        return 1 if n <= 1 else 1 << (n - 1).bit_length()
    return 131072 if n >= 131072 else 32768

padded_histogram = Counter()
for raw_m, count in raw.items():
    padded_histogram[padded(raw_m)] += count

summary = {
    "completed": result["completed"],
    "num_prompts": result["num_prompts"],
    "total_input_tokens": result["total_input_tokens"],
    "total_output_tokens": result["total_output_tokens"],
    "raw_prefill_histogram": dict(sorted(raw.items())),
    "padded_prefill_histogram": dict(sorted(padded_histogram.items())),
}
(arm / "arm_validation.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY
}

wait_gpu_idle() {
    stop_owned_server
    local elapsed=0
    while ((elapsed < 120)); do
        local busy
        busy=$(rocm-smi --showuse 2>/dev/null | rg -c 'GPU use \(%\): ([1-9][0-9]*)' || true)
        if [[ "$busy" -eq 0 ]]; then
            break
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    sleep "$COOLDOWN_SECONDS"
}

write_arm_meta() {
    local arm_dir=$1 scenario=$2 pair=$3 arm=$4 seed=$5 disable=$6
    python3 - "$arm_dir/meta.json" "$scenario" "$pair" "$arm" "$seed" "$disable" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path

path, scenario, pair, arm, seed, disable = sys.argv[1:]
Path(path).write_text(json.dumps({
    "scenario": scenario,
    "pair": int(pair),
    "arm": arm,
    "seed": int(seed),
    "disable_comm_fused": int(disable),
    "start_utc": datetime.now(timezone.utc).isoformat(),
}, indent=2, sort_keys=True) + "\n")
PY
}

run_random_arm() {
    local scenario=$1 pair=$2 arm=$3 disable=$4 variant=$5 isl=$6 osl=$7 conc=$8 mnbt=$9
    shift 9
    local max_model_len=$1 max_seqs=$2 seed=$3 arm_dir=$4
    CURRENT_ARM=$scenario/pair$pair/$arm
    mkdir -p "$arm_dir"
    write_arm_meta "$arm_dir" "$scenario" "$pair" "$arm" "$seed" "$disable"
    launch_server "$arm_dir" "$variant" "$disable" "$mnbt" "$max_model_len" "$max_seqs" 0
    run_random_benchmark "$arm_dir" "$variant" "$isl" "$osl" "$conc" \
        "$((conc * 10))" "$((conc * 2))" "$seed"
    verify_random_arm "$arm_dir" "$((conc * 10))"
    wait_gpu_idle
    log "ARM_DONE arm=$CURRENT_ARM"
    CURRENT_ARM=none
}

prepare_pair_dir() {
    local scenario_dir=$1 pair=$2
    local pair_dir=$scenario_dir/pair$pair
    if [[ -f "$pair_dir/pair.done" ]]; then
        return 0
    fi
    if [[ -d "$pair_dir" ]]; then
        local archive=$scenario_dir/attempts/pair${pair}_$(date -u +%Y%m%dT%H%M%SZ)
        mkdir -p "$scenario_dir/attempts"
        mv "$pair_dir" "$archive"
        log "ARCHIVE incomplete_pair=$pair_dir to=$archive"
    fi
    mkdir -p "$pair_dir"
}

validate_pair() {
    local pair_dir=$1
    local variant=$2
    python3 - "$pair_dir" "$variant" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
variant = sys.argv[2]
a = json.loads((root / "A" / "result.json").read_text())
b = json.loads((root / "B" / "result.json").read_text())
for key in ("num_prompts", "completed", "total_input_tokens"):
    if a[key] != b[key]:
        raise SystemExit(f"pair mismatch {key}: A={a[key]} B={b[key]}")

output_delta = b["total_output_tokens"] - a["total_output_tokens"]
if variant == "mtp3":
    # A speculative step may commit up to num_speculative_tokens extra tokens
    # at a request's max-token boundary. Acceptance decisions can therefore
    # make repeated MTP3 runs differ slightly even with the same workload.
    max_delta = 3 * a["num_prompts"]
    if abs(output_delta) > max_delta:
        raise SystemExit(
            "MTP3 output-token drift exceeds protocol bound: "
            f"A={a['total_output_tokens']} B={b['total_output_tokens']} "
            f"bound={max_delta}"
        )
else:
    if output_delta:
        raise SystemExit(
            "pair mismatch total_output_tokens: "
            f"A={a['total_output_tokens']} B={b['total_output_tokens']}"
        )

(root / "pair_validation.json").write_text(json.dumps({
    "variant": variant,
    "num_prompts": a["num_prompts"],
    "input_tokens": a["total_input_tokens"],
    "output_tokens_A": a["total_output_tokens"],
    "output_tokens_B": b["total_output_tokens"],
    "output_token_delta_B_minus_A": output_delta,
}, indent=2, sort_keys=True) + "\n")
PY
}

summarize_scenario() {
    local scenario_dir=$1
    python3 - "$scenario_dir" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
throughput = ("request_throughput", "output_throughput", "total_token_throughput")
latency = (
    "median_ttft_ms", "p90_ttft_ms", "p99_ttft_ms", "p99.9_ttft_ms",
    "median_tpot_ms", "p90_tpot_ms", "p99_tpot_ms", "p99.9_tpot_ms",
    "median_itl_ms", "p90_itl_ms", "p99_itl_ms", "p99.9_itl_ms",
    "median_e2el_ms", "p90_e2el_ms", "p99_e2el_ms", "p99.9_e2el_ms",
)

def count_log(path, needle):
    return path.read_text(errors="replace").count(needle) if path.exists() else 0

def fmt(value, digits=5, suffix=""):
    return "-" if value is None else f"{value:.{digits}f}{suffix}"

pairs = []
for pair in range(1, 4):
    pair_dir = root / f"pair{pair}"
    if not (pair_dir / "pair.done").exists():
        continue
    a = json.loads((pair_dir / "A" / "result.json").read_text())
    b = json.loads((pair_dir / "B" / "result.json").read_text())
    deltas = {}
    for key in throughput:
        deltas[key] = (b[key] / a[key] - 1.0) * 100.0
    for key in latency:
        deltas[key] = (1.0 - b[key] / a[key]) * 100.0
    validation_path = pair_dir / "pair_validation.json"
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else {}
    diagnostics = {
        "output_token_delta_B_minus_A": validation.get("output_token_delta_B_minus_A"),
        "acceptance_rate_A": a.get("acceptance_rate"),
        "acceptance_rate_B": b.get("acceptance_rate"),
        "hipblas_recovery_A": count_log(pair_dir / "A" / "server.log", "HIPBLAS_STATUS_INTERNAL_ERROR"),
        "hipblas_recovery_B": count_log(pair_dir / "B" / "server.log", "HIPBLAS_STATUS_INTERNAL_ERROR"),
    }
    if diagnostics["acceptance_rate_A"] is not None and diagnostics["acceptance_rate_B"] is not None:
        diagnostics["acceptance_rate_delta_pp"] = 100.0 * (
            diagnostics["acceptance_rate_B"] - diagnostics["acceptance_rate_A"]
        )
    pairs.append({
        "pair": pair,
        "A": a,
        "B": b,
        "delta_pct": deltas,
        "diagnostics": diagnostics,
    })

summary = {"scenario": root.name, "completed_pairs": len(pairs), "pairs": pairs}
if pairs:
    summary["median_paired_delta_pct"] = {
        key: statistics.median(item["delta_pct"][key] for item in pairs)
        for key in (*throughput, *latency)
    }
if len(pairs) == 3:
    med = summary["median_paired_delta_pct"]
    core = (
        "total_token_throughput", "output_throughput",
        "median_tpot_ms", "median_itl_ms", "median_e2el_ms",
    )
    tail = (
        "p90_tpot_ms", "p99_tpot_ms",
        "p90_itl_ms", "p99_itl_ms",
        "p90_e2el_ms", "p99_e2el_ms",
    )
    ttft = ("median_ttft_ms", "p90_ttft_ms", "p99_ttft_ms")
    fail_reasons = []
    inconclusive_reasons = []
    for key in (*core, *tail):
        if med[key] < -1.0:
            fail_reasons.append(f"{key} median regression {med[key]:.3f}% exceeds 1%")
    for key in ttft:
        if med[key] < -2.0:
            fail_reasons.append(f"{key} median regression {med[key]:.3f}% exceeds 2%")
    for key in (*core, *tail, *ttft):
        regressions = sum(item["delta_pct"][key] < 0.0 for item in pairs)
        if regressions >= 2 and not any(reason.startswith(key + " ") for reason in fail_reasons):
            inconclusive_reasons.append(f"{key} regressed in {regressions}/3 pairs")
    if fail_reasons:
        status = "FAIL"
    elif inconclusive_reasons:
        status = "INCONCLUSIVE"
    elif max(med[key] for key in core) >= 1.0:
        status = "BENEFIT"
    else:
        status = "PASS"
    summary["gate"] = {
        "status": status,
        "fail_reasons": fail_reasons,
        "inconclusive_reasons": inconclusive_reasons,
    }
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

lines = [f"# {root.name}", "", f"Completed pairs: {len(pairs)}/3", ""]
if "gate" in summary:
    gate = summary["gate"]
    lines += [f"Gate: **{gate['status']}**", ""]
    for reason in (*gate["fail_reasons"], *gate["inconclusive_reasons"]):
        lines.append(f"- {reason}")
    if gate["fail_reasons"] or gate["inconclusive_reasons"]:
        lines.append("")
if pairs:
    lines += [
        "| Metric | " + " | ".join(f"Pair {p['pair']}" for p in pairs) + " | Median |",
        "| --- | " + " | ".join("---:" for _ in pairs) + " | ---: |",
    ]
    med = summary["median_paired_delta_pct"]
    for key in (*throughput, *latency):
        values = [p["delta_pct"][key] for p in pairs]
        lines.append(
            f"| {key} | " + " | ".join(f"{v:+.3f}%" for v in values) + f" | {med[key]:+.3f}% |"
        )
    if any(p["diagnostics"]["output_token_delta_B_minus_A"] is not None for p in pairs):
        lines += [
            "",
            "| Pair | Output token B-A | Acceptance A | Acceptance B | Acceptance B-A | hipBLAS A/B |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for p in pairs:
            d = p["diagnostics"]
            output_delta = d["output_token_delta_B_minus_A"]
            acceptance_a = d["acceptance_rate_A"]
            acceptance_b = d["acceptance_rate_B"]
            acceptance_delta = d.get("acceptance_rate_delta_pp")
            lines.append(
                f"| {p['pair']} | {output_delta if output_delta is not None else '-'} | "
                f"{fmt(acceptance_a)} | {fmt(acceptance_b)} | "
                f"{('-' if acceptance_delta is None else f'{acceptance_delta:+.3f} pp')} | "
                f"{d['hipblas_recovery_A']}/{d['hipblas_recovery_B']} |"
            )
(root / "summary.md").write_text("\n".join(lines) + "\n")
PY
}

run_scenario_ab3() {
    local scenario=$1 variant=$2 isl=$3 osl=$4 conc=$5 mnbt=$6 max_model_len=$7 max_seqs=$8
    local scenario_dir=$RESULT_ROOT/runs/$scenario
    mkdir -p "$scenario_dir"
    local pair seed pair_dir
    for pair in 1 2 3; do
        if [[ -n "$PAIR_FILTER" && "$pair" != "$PAIR_FILTER" ]]; then
            continue
        fi
        pair_dir=$scenario_dir/pair$pair
        if ((RESUME)) && [[ -f "$pair_dir/pair.done" ]]; then
            log "PAIR_SKIP scenario=$scenario pair=$pair"
            continue
        fi
        prepare_pair_dir "$scenario_dir" "$pair"
        seed=$((SEED_BASE + pair))
        log "PAIR_BEGIN scenario=$scenario pair=$pair seed=$seed"
        run_random_arm "$scenario" "$pair" A 1 "$variant" "$isl" "$osl" "$conc" \
            "$mnbt" "$max_model_len" "$max_seqs" "$seed" "$pair_dir/A"
        run_random_arm "$scenario" "$pair" B 0 "$variant" "$isl" "$osl" "$conc" \
            "$mnbt" "$max_model_len" "$max_seqs" "$seed" "$pair_dir/B"
        validate_pair "$pair_dir" "$variant"
        touch "$pair_dir/pair.done"
        summarize_scenario "$scenario_dir"
        log "PAIR_DONE scenario=$scenario pair=$pair summary=$scenario_dir/summary.md"
    done
}

parse_trace_arm() {
    local arm_dir=$1
    python3 - "$arm_dir" <<'PY'
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
kernels = Counter()
for path in root.rglob("*kernel_trace.csv"):
    with path.open(newline="", errors="replace") as handle:
        for row in csv.DictReader(handle):
            name = row.get("Kernel_Name", "")
            if name:
                kernels[name] += 1

families = Counter()
bucket_counts = Counter()
patterns = {
    "atomic": re.compile(r"comm_fused_moe_atomic_m(\d+)_"),
    "full": re.compile(r"flydsl_fused_moe_full_"),
    "window": re.compile(r"flydsl_fused_moe_win_"),
    "persistent": re.compile(r"flydsl_fused_moe_pwin_"),
}
for name, count in kernels.items():
    for family, pattern in patterns.items():
        match = pattern.search(name)
        if match:
            families[family] += count
            if match.lastindex:
                bucket_counts[f"{family}:M{match.group(1)}"] += count
            break

server = (root / "server.log").read_text(errors="replace")
raw = Counter(
    int(m.group(1))
    for m in re.finditer(r"Scheduled prefill batch: \d+ reqs, (\d+) new tokens", server)
)
def padded(n: int) -> int:
    if n < 32768:
        return 1 if n <= 1 else 1 << (n - 1).bit_length()
    return 131072 if n >= 131072 else 32768

padded_histogram = Counter()
for raw_m, count in raw.items():
    padded_histogram[padded(raw_m)] += count

result = {
    "kernel_trace_files": [str(p) for p in root.rglob("*kernel_trace.csv")],
    "fused_family_dispatches": dict(sorted(families.items())),
    "fused_bucket_dispatches": dict(sorted(bucket_counts.items())),
    "raw_prefill_histogram": dict(sorted(raw.items())),
    "padded_prefill_histogram": dict(sorted(padded_histogram.items())),
}
(root / "trace_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
PY
}

run_trace_arm() {
    local trace_id=$1 arm=$2 disable=$3 variant=$4 isl=$5 osl=$6 conc=$7 mnbt=$8
    shift 8
    local max_model_len=$1 max_seqs=$2
    local arm_dir=$RESULT_ROOT/trace/$trace_id/$arm
    CURRENT_ARM=trace/$trace_id/$arm
    mkdir -p "$arm_dir/rocprof"
    prepare_rocprof_attach
    ENABLE_ROCP_ATTACH=1
    launch_server "$arm_dir" "$variant" "$disable" "$mnbt" "$max_model_len" "$max_seqs" 0
    ENABLE_ROCP_ATTACH=0

    local server_pid trace_target
    local -a trace_targets
    server_pid=$(<"$OWNED_PID_FILE")
    mapfile -t trace_targets < <(find_trace_target)
    if [[ "${#trace_targets[@]}" -ne 1 ]]; then
        docker exec "$CONTAINER" bash -lc \
            'ps -eo pid=,ppid=,comm=,args= | grep -E "(ATOM::|openai_server)" || true' \
            > "$arm_dir/trace_target_processes.txt"
        die "expected one attachable TP0 worker, found ${#trace_targets[@]}; see $arm_dir/trace_target_processes.txt"
    fi
    trace_target=${trace_targets[0]}
    printf 'server_pid=%s\ntrace_target_pid=%s\n' "$server_pid" "$trace_target" \
        > "$arm_dir/trace_target.txt"
    log "ROCPROF_ATTACH arm=$CURRENT_ARM server_pid=$server_pid target_pid=$trace_target duration_s=$TRACE_ATTACH_SECONDS"
    docker exec \
        -e "ROCPROF_ATTACH_LIBRARY=$ROCM_CORE/lib/librocprofiler-sdk-rocattach.so.1" \
        "$CONTAINER" "$ROCPROF" \
        --rocm-root "$ROCPROF_COMPAT_ROOT" \
        --kernel-trace \
        --output-format csv \
        --output-directory "$arm_dir/rocprof" \
        --output-file trace \
        --pid "$trace_target" \
        --attach-children false \
        --attach-duration-msec "$((TRACE_ATTACH_SECONDS * 1000))" \
        --attach-sync-output true \
        > "$arm_dir/rocprof.log" 2>&1 &
    local profiler_pid=$!
    sleep 8
    if ! kill -0 "$profiler_pid" 2>/dev/null; then
        wait "$profiler_pid" || true
        tail -n 120 "$arm_dir/rocprof.log" || true
        die "rocprof attach exited before trace workload arm=$CURRENT_ARM"
    fi

    run_random_benchmark "$arm_dir" "$variant" "$isl" "$osl" "$conc" "$conc" 0 9001
    verify_random_arm "$arm_dir" "$conc"
    wait "$profiler_pid"
    parse_trace_arm "$arm_dir"
    wait_gpu_idle
    cleanup_rocprof_attach
    log "TRACE_ARM_DONE arm=$CURRENT_ARM"
    CURRENT_ARM=none
}

verify_trace_gate() {
    python3 - "$RESULT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {
    "T1_atomic": "atomic",
    "T2_full_window": "full_or_window",
    "T3_persistent": "persistent",
    "T4_mtp3": "any_fused",
    "T5_tbo": "any_fused",
    "T6_dpa_negative": "none",
}
report = {"status": "PASS", "traces": {}}
for trace_id, want in expected.items():
    a = json.loads((root / "trace" / trace_id / "A" / "trace_summary.json").read_text())
    b = json.loads((root / "trace" / trace_id / "B" / "trace_summary.json").read_text())
    a_families = a["fused_family_dispatches"]
    b_families = b["fused_family_dispatches"]
    errors = []
    if a_families:
        errors.append(f"baseline unexpectedly used fused kernels: {a_families}")
    if want == "none":
        if b_families:
            errors.append(f"negative control used fused kernels: {b_families}")
    elif want == "any_fused":
        if not b_families:
            errors.append("candidate did not use any fused kernel")
    elif want == "full_or_window":
        if not (b_families.get("full", 0) or b_families.get("window", 0)):
            errors.append(f"candidate missed full/window: {b_families}")
    elif not b_families.get(want, 0):
        errors.append(f"candidate missed {want}: {b_families}")
    report["traces"][trace_id] = {
        "expected": want,
        "baseline": a,
        "candidate": b,
        "errors": errors,
    }
    if errors:
        report["status"] = "FAIL"

path = root / "trace" / "trace_gate.json"
path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
if report["status"] != "PASS":
    raise SystemExit(f"trace gate failed; see {path}")
print(f"trace gate PASS: {path}")
PY
}

verify_trace_pair() {
    local trace_id=$1 expected=$2
    python3 - "$RESULT_ROOT" "$trace_id" "$expected" <<'PY'
import json
import sys
from pathlib import Path

root, trace_id, expected = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
a = json.loads((root / "trace" / trace_id / "A" / "trace_summary.json").read_text())
b = json.loads((root / "trace" / trace_id / "B" / "trace_summary.json").read_text())
a_families = a["fused_family_dispatches"]
b_families = b["fused_family_dispatches"]
errors = []
if a_families:
    errors.append(f"baseline unexpectedly used fused kernels: {a_families}")
if expected == "none":
    if b_families:
        errors.append(f"negative control used fused kernels: {b_families}")
elif expected == "any_fused":
    if not b_families:
        errors.append("candidate did not use any fused kernel")
elif expected == "full_or_window":
    if not (b_families.get("full", 0) or b_families.get("window", 0)):
        errors.append(f"candidate missed full/window: {b_families}")
elif not b_families.get(expected, 0):
    errors.append(f"candidate missed {expected}: {b_families}")

result = {
    "trace_id": trace_id,
    "expected": expected,
    "status": "FAIL" if errors else "PASS",
    "baseline_families": a_families,
    "candidate_families": b_families,
    "candidate_buckets": b["fused_bucket_dispatches"],
    "errors": errors,
}
path = root / "trace" / trace_id / "pair_gate.json"
path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, sort_keys=True))
if errors:
    raise SystemExit(f"trace pair failed; see {path}")
PY
}

run_trace_phase() {
    log "STEP2 trace gate begin"
    while IFS=$'\t' read -r trace_id variant isl osl conc mnbt max_model_len max_seqs expected; do
        [[ "$trace_id" == id ]] && continue
        if [[ -n "$SCENARIO_FILTER" && "$trace_id" != "$SCENARIO_FILTER" ]]; then
            continue
        fi
        local trace_dir=$RESULT_ROOT/trace/$trace_id
        if ((RESUME)) && [[ -f "$trace_dir/trace.done" ]]; then
            log "TRACE_SKIP id=$trace_id"
            continue
        fi
        if [[ -d "$trace_dir" ]]; then
            local archive=$RESULT_ROOT/trace/attempts/${trace_id}_$(date -u +%Y%m%dT%H%M%SZ)
            mkdir -p "$RESULT_ROOT/trace/attempts"
            mv "$trace_dir" "$archive"
        fi
        run_trace_arm "$trace_id" A 1 "$variant" "$isl" "$osl" "$conc" "$mnbt" "$max_model_len" "$max_seqs"
        run_trace_arm "$trace_id" B 0 "$variant" "$isl" "$osl" "$conc" "$mnbt" "$max_model_len" "$max_seqs"
        verify_trace_pair "$trace_id" "$expected"
        touch "$trace_dir/trace.done"
        log "TRACE_PAIR_DONE id=$trace_id expected=$expected"
    done < <(emit_trace_scenarios)

    if [[ -z "$SCENARIO_FILTER" ]]; then
        verify_trace_gate
        log "STEP2 trace gate PASS"
    else
        log "STEP2 single trace complete; full gate intentionally not evaluated"
    fi
}

run_selected_random_scenarios() {
    local selected=0
    while IFS=$'\t' read -r scenario phase variant isl osl conc mnbt max_model_len max_seqs; do
        selected=$((selected + 1))
        log "SCENARIO_BEGIN id=$scenario phase=$phase variant=$variant isl=$isl osl=$osl c=$conc"
        run_scenario_ab3 "$scenario" "$variant" "$isl" "$osl" "$conc" "$mnbt" "$max_model_len" "$max_seqs"
        log "SCENARIO_DONE id=$scenario"
    done < <(selected_scenarios)
    ((selected > 0)) || die "no scenarios selected"
}

find_aiperf() {
    docker exec "$CONTAINER" bash -lc '
        if test -x /workspace/venvs/aiperf-sa/bin/aiperf; then
            echo /workspace/venvs/aiperf-sa/bin/aiperf
        elif command -v aiperf >/dev/null 2>&1; then
            command -v aiperf
        fi
    '
}

run_agentic_phase() {
    local aiperf
    aiperf=$(find_aiperf)
    [[ -n "$aiperf" ]] || die "AIPerf agentic harness is unavailable in $CONTAINER"
    die "agentic execution is intentionally gated until synthetic and persistent summaries pass; AIPerf=$aiperf"
}

summarize_all() {
    python3 - "$RESULT_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted((root / "runs").glob("*/summary.json")):
    data = json.loads(path.read_text())
    med = data.get("median_paired_delta_pct", {})
    rows.append({
        "scenario": data["scenario"],
        "completed_pairs": data["completed_pairs"],
        "gate": data.get("gate", {}).get("status", "PENDING"),
        **med,
    })
if rows:
    keys = ["scenario", "completed_pairs", "gate"] + sorted(set().union(*(r.keys() for r in rows)) - {"scenario", "completed_pairs", "gate"})
    with (root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Comm-Fused MoE InferenceX A/B 进度", "", "| Scenario | Pairs | Gate | Total token throughput | Median TTFT | Median TPOT | Median E2EL |", "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        def fmt(key):
            value = row.get(key)
            return "-" if value is None else f"{float(value):+.3f}%"
        lines.append(f"| {row['scenario']} | {row['completed_pairs']}/3 | {row['gate']} | {fmt('total_token_throughput')} | {fmt('median_ttft_ms')} | {fmt('median_tpot_ms')} | {fmt('median_e2el_ms')} |")
    (root / "summary.md").write_text("\n".join(lines) + "\n")
PY
}

main() {
    if ((DRY_RUN)); then
        echo "phase=$PHASE"
        echo "result_root=$RESULT_ROOT"
        if [[ "$PHASE" == trace ]]; then
            emit_trace_scenarios
        else
            selected_scenarios
        fi
        exit 0
    fi
    check_static_inputs
    mkdir -p "$TEST_ROOT"

    if [[ ! -d "$RESULT_ROOT" ]]; then
        mkdir -p "$RESULT_ROOT"
        printf '%s\n' "$RESULT_ROOT" > "$LATEST_POINTER"
        emit_scenarios > "$MANIFEST"
        emit_trace_scenarios > "$RESULT_ROOT/trace_manifest.tsv"
        capture_provenance
    fi

    case "$PHASE" in
        preflight)
            run_preflight
            ;;
        trace)
            run_preflight
            run_trace_phase
            ;;
        synthetic|persistent)
            [[ -f "$RESULT_ROOT/trace/trace_gate.json" ]] || die "trace gate has not run"
            python3 - "$RESULT_ROOT/trace/trace_gate.json" <<'PY'
import json, sys
if json.load(open(sys.argv[1]))["status"] != "PASS":
    raise SystemExit("trace gate is not PASS")
PY
            run_selected_random_scenarios
            summarize_all
            ;;
        agentic)
            run_agentic_phase
            ;;
        summarize)
            for scenario_dir in "$RESULT_ROOT"/runs/*; do
                [[ -d "$scenario_dir" ]] || continue
                summarize_scenario "$scenario_dir"
            done
            summarize_all
            ;;
        all)
            run_preflight
            run_trace_phase
            PHASE=synthetic
            run_selected_random_scenarios
            summarize_all
            PHASE=persistent
            run_selected_random_scenarios
            summarize_all
            run_agentic_phase
            ;;
    esac
    log "PHASE_DONE phase=$PHASE"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main
fi
