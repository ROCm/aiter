#!/usr/bin/env bash
# DeepSeek V4 end-to-end A/B on the MI350P box: aiter main vs main + PR 5670.
# Run on the host.
#
#   ./e2e.sh build             # build both images from ./Dockerfile
#   ./e2e.sh build-fork        # build IMAGE_FORK: IMAGE_B + the vLLM fork's prefill dispatch
#   ./e2e.sh check             # preflight: images, vLLM all-reduce path, model
#   ./e2e.sh run               # check, then the A/B benchmark, then the summary
#   ./e2e.sh run3              # the same for the 3 arms in ARMS3 (needs build-fork)
#   ./e2e.sh profile           # check, then one torch-profiled run per server
#   ./e2e.sh smoke             # check, then only start each server, smoke-test it, stop it
#   ./e2e.sh summary <dir>     # summary table of a finished run
#   ./e2e.sh all               # build + run
#
# Settings are environment variables, e.g. TPS="8" ./e2e.sh run. See below.
#
# For every (TP, image) "run" starts one server, checks that it answers,
# records which all-reduce path vLLM took, runs the benchmark scenarios and
# stops the server. "smoke" does the same without the benchmark scenarios. The images alternate per TP so that slow drift does not
# favour one of them.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

MODEL=${MODEL:-deepseek-ai/DeepSeek-V4-Pro-0813}
# A hub cache that already holds the model: a fresh download is ~830 GiB and
# may not finish within READY_TIMEOUT.
HF_HUB=${HF_HUB:-${HF_HOME:-$HOME/.cache/huggingface}/hub}
CACHE_ROOT=${CACHE_ROOT:-$HOME/.cache/vllm-e2e}
IMAGE_A=${IMAGE_A:-vllm-aiter-main}   # baseline: aiter main
IMAGE_B=${IMAGE_B:-vllm-aiter-ar}     # aiter main + PR 5670
IMAGE_FORK=${IMAGE_FORK:-vllm-aiter-ar-prefill}   # IMAGE_B + the vLLM fork
# The arms to compare: "tag,image[,VAR=value...]" separated by spaces. Each arm
# gets its own server; VAR=value overrides the image's environment for that arm.
# The first arm is the reference the summary's deltas are against. Tags are \w+.
# Default: the A/B of aiter main vs main + PR 5670. For the 3-way prefill
# comparison use ./e2e.sh run3 (see ARMS3).
ARMS=${ARMS:-"main,$IMAGE_A ar,$IMAGE_B"}
# Stock vLLM | fork with the stock HIP QuickReduce INT4 kernel | fork with FlyDSL
# QuickReduce INT4. INT4 is set explicitly on the fork arm so that it is visible
# here: it is the same quantization in both, only the kernel differs.
ARMS3=${ARMS3:-"base,$IMAGE_A fork,$IMAGE_FORK,AITER_FLY_AR=0,VLLM_ROCM_QUICK_REDUCE_USE_FLYDSL=0,VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4,AITER_QUICK_REDUCE_QUANTIZATION=INT4 flyar,$IMAGE_FORK"}
# V4-Pro (~830 GiB) fits on 144 GiB GPUs only at TP=8; preflight rejects
# TPs whose per-GPU weight share does not fit.
TPS=${TPS:-8}
GPU_GIB=${GPU_GIB:-144}               # memory per GPU (MI350P)
MODEL_GIB=${MODEL_GIB:-}              # checkpoint size; default: measured from the cache
# The Docker images enable shared-expert fusion, but this checkpoint's shared expert
# is block-FP8 and cannot load into the MXFP4 routed slot (KeyError on
# w13_weight_scale_inv). 1 only for checkpoints with MXFP4 shared experts.
FUSE_SHARED_EXPERTS=${FUSE_SHARED_EXPERTS:-0}
GPUS=${GPUS:-}                        # default 0..TP-1
PORT=${PORT:-8001}
READY_TIMEOUT=${READY_TIMEOUT:-7200}
# "input:output:concurrencies". The first one is the recipe's prefill-heavy
# case. The decode-heavy cases are where the small all-reduce payloads are.
SCENARIOS=${SCENARIOS:-"60000:100:1,4 1024:1024:1,4,16,64 8192:1024:4,16"}
RANGE_RATIO=${RANGE_RATIO:-0.1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
# The model's default 1M-token context needs 14.4 GiB of KV cache per GPU,
# more than 0.85 leaves. The longest scenario is ~67k tokens.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
PROMPTS_PER_CONC=${PROMPTS_PER_CONC:-4}   # num_prompts = max(8, conc * this)
NAME=dsv4-e2e

MODE=bench
HF_OFFLINE=0   # set by cmd_check when the model is already cached
RUN_DIR=

log() {
    local line="[$(date +%H:%M:%S)] $*"
    echo "$line"
    [ -n "$RUN_DIR" ] && echo "$line" >>"$RUN_DIR/run.log"
    return 0
}
die() { log "ERROR: $*"; exit 1; }

stop_server() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }

# ---------------------------------------------------------------- build ----

cmd_build() {
    log "building $IMAGE_B (main + PR 5670)"
    docker build -t "$IMAGE_B" -f "$HERE/Dockerfile" "$HERE"
    log "building $IMAGE_A (main only)"
    docker build -t "$IMAGE_A" --build-arg WITH_ALLREDUCE=0 -f "$HERE/Dockerfile" "$HERE"
}

cmd_build_fork() {
    docker image inspect vllm-aiter-ar >/dev/null 2>&1 || die "vllm-aiter-ar missing: ./e2e.sh build"
    log "building $IMAGE_FORK (vllm-aiter-ar + vLLM fork)"
    docker build -t "$IMAGE_FORK" -f "$HERE/Dockerfile.vllm-fork" "$HERE"
}

# ---------------------------------------------------------------- check ----

in_image() {  # image cmd...
    local image=$1
    shift
    docker run --rm --entrypoint "" "$image" "$@"
}

cmd_check() {
    command -v docker >/dev/null || die "docker not found: run this on the host"
    local ok=1 image
    local arm tag env v fly
    for arm in $ARMS; do
        IFS=, read -r tag image env <<<"$arm"
        docker image inspect "$image" >/dev/null 2>&1 \
            || die "image $image missing: ./e2e.sh build / build-fork"
        log "$tag = $image: $(in_image "$image" sh -c \
            'echo "aiter $(git -C /opt/aiter log -1 --format=%h), AITER_FLY_AR=$AITER_FLY_AR, vllm $(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null), fork $(cat /opt/vllm-fork-ref 2>/dev/null || echo none)"')${env:+, overrides: $env}"
        # vLLM reaches aiter's CustomAllreduce (and through it the FlyDSL
        # one-shot and fused AR+RMSNorm) only if it has this switch.
        if ! in_image "$image" python3 -c \
            "import vllm.envs as e, sys; sys.exit(0 if e.VLLM_ROCM_USE_AITER_CUSTOM_AR else 1)" 2>/dev/null; then
            log "WARNING $tag: vLLM has no VLLM_ROCM_USE_AITER_CUSTOM_AR (or it is off);" \
                "it will not call aiter's all-reduce, so the comparison measures nothing"
            ok=0
        fi
        # An override the image's vLLM does not know is silently ignored.
        for v in ${env//,/ }; do
            case "$v" in VLLM_*=*)
                in_image "$image" python3 -c \
                    "import vllm.envs as e, sys; sys.exit(0 if '${v%%=*}' in e.environment_variables else 1)" 2>/dev/null \
                    || { log "WARNING $tag: vLLM in $image has no ${v%%=*}"; ok=0; } ;;
            esac
        done
        # Effective AITER_FLY_AR: the override, else the image's.
        fly=$(in_image "$image" sh -c 'echo $AITER_FLY_AR')
        for v in ${env//,/ }; do case "$v" in AITER_FLY_AR=*) fly=${v#*=} ;; esac; done
        if [ "$fly" = 1 ]; then
            in_image "$image" sh -c 'AITER_AOT_IMPORT=1 python3 -c "from aiter.ops.flydsl import allreduce_policy"' \
                || die "$tag: FlyDSL all-reduce does not import in $image"
        fi
    done

    mkdir -p "$CACHE_ROOT" 2>/dev/null && [ -w "$CACHE_ROOT" ] \
        || die "$CACHE_ROOT is not writable: set CACHE_ROOT to a writable directory"
    local model_dir="$HF_HUB/models--${MODEL//\//--}"
    local model_gib=$MODEL_GIB
    if [ -d "$model_dir/snapshots" ]; then
        # A cached model only has to be readable (/raid is shared); the server
        # runs offline so it loads exactly this snapshot and never re-downloads.
        [ -r "$model_dir/snapshots" ] || die "$model_dir is not readable"
        HF_OFFLINE=1
        log "model cached: $model_dir ($(du -sh "$model_dir" 2>/dev/null | cut -f1)), serving offline"
        # The index lists every shard's size; a partial download (shards
        # missing) fails at load time, so catch it here.
        local index have want
        index=$(ls "$model_dir"/snapshots/*/model.safetensors.index.json 2>/dev/null | head -1)
        have=$(du -sbL "$model_dir/snapshots" 2>/dev/null | cut -f1)
        want=$(grep -o '"total_size": *[0-9]*' "$index" 2>/dev/null | grep -o '[0-9]*$' || true)
        if [ -n "$want" ]; then
            [ "$have" -ge "$want" ] \
                || die "model download incomplete: $((have >> 30)) of $((want >> 30)) GiB in $model_dir"
            [ -n "$model_gib" ] || model_gib=$((want >> 30))
        else
            [ -n "$model_gib" ] || model_gib=$((have >> 30))
        fi
    else
        mkdir -p "$HF_HUB" 2>/dev/null && [ -w "$HF_HUB" ] \
            || die "model not in $HF_HUB and it is not writable: set HF_HUB to the cache that holds it"
        log "WARNING model not in $HF_HUB ($(df -h --output=avail "$HF_HUB" | tail -1 | tr -d ' ') free)," \
            "the first server start downloads it (set HF_TOKEN)"
        ok=0
    fi
    # Weights alone must leave room for activations and KV cache on each GPU.
    local tp
    if [ -n "$model_gib" ] && [ "$model_gib" -gt 0 ]; then
        for tp in $TPS; do
            [ $((model_gib / tp)) -le $((GPU_GIB * 9 / 10)) ] \
                || die "TP=$tp: $model_gib GiB of weights is $((model_gib / tp)) GiB per GPU," \
                       "more than 90% of $GPU_GIB GiB: drop it from TPS"
        done
    else
        log "WARNING checkpoint size unknown, not checking that TPs $TPS fit (set MODEL_GIB)"
        ok=0
    fi
    if docker ps -a -q -f name="^$NAME$" | grep -q .; then
        log "WARNING a $NAME container exists; it will be replaced"
    fi
    [ $ok = 1 ] && log "preflight ok" || log "preflight finished with warnings"
}

# ------------------------------------------------------------------ run ----

start_server() {  # image tag tp out_dir [VAR=value...]
    local image=$1 tag=$2 tp=$3 out=$4
    shift 4
    local extra_env=() v
    for v in "$@"; do extra_env+=(-e "$v"); done
    local gpus=${GPUS:-$(seq -s, 0 $((tp - 1)))}
    local profile_args=()
    if [ "$MODE" = profile ]; then
        profile_args=(
            --profiler-config.profiler torch
            --profiler-config.torch_profiler_dir /traces
            --profiler-config.torch_profiler_record_shapes true
            --profiler-config.torch_profiler_with_stack false
        )
    fi
    # Per-image cache: torch.compile, triton and FlyDSL caches survive restarts,
    # and the two aiter builds never share compiled artifacts.
    mkdir -p "$CACHE_ROOT/$tag" "$out/traces"

    # The image sets VLLM_ROCM_USE_AITER*, AITER_FLY_AR* and
    # AITER_QUICK_REDUCE_QUANTIZATION; only the run-specific settings go here.
    docker run -d \
        --name "$NAME" \
        --entrypoint "" \
        -e HIP_VISIBLE_DEVICES="$gpus" \
        -e VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
        -e VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS="$FUSE_SHARED_EXPERTS" \
        -e HF_TOKEN="${HF_TOKEN:-}" \
        -e HF_HUB_OFFLINE="$HF_OFFLINE" \
        "${extra_env[@]}" \
        --network host --ipc host \
        -v "$CACHE_ROOT/$tag":/root/.cache \
        -v "$HF_HUB":/root/.cache/huggingface/hub \
        -v "$out/traces":/traces \
        --device /dev/dri:/dev/dri --device /dev/kfd:/dev/kfd \
        --security-opt seccomp=unconfined \
        --shm-size 128G \
        "$image" \
        vllm serve "$MODEL" \
            --async_scheduling \
            --no-enable_log_requests \
            --block_size 256 \
            --max-num-batched-tokens 8192 \
            --compilation_config '{"mode": 3, "cudagraph_mode": "FULL_DECODE_ONLY"}' \
            --distributed-executor-backend mp \
            --gpu_memory_utilization "$GPU_MEM_UTIL" \
            --max-model-len "$MAX_MODEL_LEN" \
            --no_enable_prefix_caching \
            --port "$PORT" \
            --tensor_parallel_size "$tp" \
            --trust_remote_code \
            --kv-cache-dtype fp8_e4m3 \
            --tokenizer-mode deepseek_v4 \
            --tool-call-parser deepseek_v4 \
            --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 \
            "${profile_args[@]}" >/dev/null
    docker logs -f "$NAME" >"$out/server.log" 2>&1 &
}

wait_ready() {
    local t=0
    until curl -sf "localhost:$PORT/health" >/dev/null; do
        if ! docker ps -q -f name="^$NAME$" | grep -q .; then
            log "server exited during startup, see server.log"
            return 1
        fi
        if [ $t -ge "$READY_TIMEOUT" ]; then
            log "server not ready after ${t}s"
            return 1
        fi
        sleep 15
        t=$((t + 15))
    done
    log "ready after ${t}s"
}

smoke() {  # out_dir
    curl -sf "localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
        -d "{\"model\": \"$MODEL\", \"prompt\": \"The capital of France is\", \"max_tokens\": 32, \"temperature\": 0, \"logprobs\": 5}" \
        >"$1/smoke.json" || return 1
    log "smoke: $(grep -o '"text": *"[^"]*"' "$1/smoke.json" | head -1)"
    # Per-token greedy picks and how close each was to the runner-up.
    run_py e2e_smoke.py "$1" >"$1/smoke_tokens.txt" 2>&1 \
        && log "smoke $(tail -1 "$1/smoke_tokens.txt") (all tokens: $1/smoke_tokens.txt)" \
        || log "smoke logprobs unavailable, see $1/smoke_tokens.txt"
}

record_ar_path() {  # out_dir
    # The all-reduce backends vLLM enabled, and what aiter's FlyDSL code said.
    grep -iE "all.?reduce|quick.?reduce|flydsl|fly_ar|AITER_CUSTOM|QUICK_REDUCE" "$1/server.log" \
        | grep -v "^DEBUG" | sort | uniq -c | sort -rn >"$1/ar_path.txt" || true
    log "all-reduce log lines: $(wc -l <"$1/ar_path.txt") ($1/ar_path.txt)"
}

bench() {  # in out conc out_dir [extra args]
    local in=$1 outl=$2 conc=$3 dir=$4
    shift 4
    local n=$((conc * PROMPTS_PER_CONC))
    [ $n -lt 8 ] && n=8
    local file="bench_${in}x${outl}_c${conc}.json"
    docker exec "$NAME" vllm bench serve \
        --backend vllm \
        --model "$MODEL" \
        --port "$PORT" \
        --dataset-name random \
        --random-input-len "$in" \
        --random-output-len "$outl" \
        --random-range-ratio "$RANGE_RATIO" \
        --ignore-eos \
        --request-rate inf \
        --max-concurrency "$conc" \
        --num-prompts "$n" \
        --num-warmups "$conc" \
        --seed 15 \
        --percentile-metrics ttft,tpot,itl,e2el \
        --save-result \
        --result-dir /traces \
        --result-filename "$file" \
        "$@" >>"$dir/bench.log" 2>&1 || return 1
    mv "$dir/traces/$file" "$dir/"
}

run_one() {  # image tag tp [VAR=value...]
    local image=$1 tag=$2 tp=$3
    shift 3
    local out="$RUN_DIR/${tag}_tp${tp}"
    mkdir -p "$out"
    log "=== $tag TP=$tp${*:+ [$*]} ($image $(docker image inspect -f '{{.Id}}' "$image" | cut -c8-19))"
    stop_server
    start_server "$image" "$tag" "$tp" "$out" "$@"
    if ! wait_ready; then
        stop_server
        return 0
    fi
    smoke "$out" || log "smoke request FAILED"
    record_ar_path "$out"
    case "$MODE" in
        profile) bench 60000 100 4 "$out" --profile || log "profile run FAILED" ;;
        bench) run_scenarios "$out" ;;
        smoke) ;;
    esac
    record_ar_path "$out"   # again: the fused engines may log after warmup
    stop_server
}

run_scenarios() {  # out_dir
    local out=$1 s in outl concs c
    for s in $SCENARIOS; do
        IFS=: read -r in outl concs <<<"$s"
        for c in ${concs//,/ }; do
            log "bench ${in}x${outl} c=$c"
            bench "$in" "$outl" "$c" "$out" || log "bench ${in}x${outl} c=$c FAILED, see bench.log"
            if ! docker ps -q -f name="^$NAME$" | grep -q .; then
                log "server died, skipping the remaining scenarios, see server.log"
                return 0
            fi
        done
    done
}

cmd_run() {
    RUN_DIR=${RUN_DIR_OVERRIDE:-$HERE/results/$(date +%Y%m%d-%H%M%S)-$MODE}
    mkdir -p "$RUN_DIR"
    trap stop_server EXIT
    cmd_check
    log "run dir $RUN_DIR, mode $MODE, TPs $TPS"
    local arms=($ARMS) i=0 tp k arm tag image env
    : >"$RUN_DIR/arms.txt"
    for arm in "${arms[@]}"; do echo "${arm%%,*}" >>"$RUN_DIR/arms.txt"; done
    for tp in $TPS; do
        # Rotate the order per TP so that slow drift does not favour one arm.
        for ((k = 0; k < ${#arms[@]}; k++)); do
            arm=${arms[$(((k + i) % ${#arms[@]}))]}
            IFS=, read -r tag image env <<<"$arm"
            run_one "$image" "$tag" "$tp" ${env//,/ }
        done
        i=$((i + 1))
    done
    if [ "$MODE" = profile ]; then
        log "traces under $RUN_DIR/*/traces"
    else
        cmd_summary "$RUN_DIR" | tee "$RUN_DIR/summary.md"
    fi
}

# -------------------------------------------------------------- summary ----

run_py() {  # script.py dir: run a helper next to this script on a result dir
    local script=$1 dir
    dir=$(cd "$2" && pwd)
    if command -v python3 >/dev/null; then
        python3 "$HERE/$script" "$dir"
    else
        # The server image has python3; the host may not.
        docker run --rm --entrypoint "" -v "$HERE":/e2e:ro -v "$dir":/r \
            "$IMAGE_A" python3 "/e2e/$script" /r
    fi
}

cmd_summary() {  # run_dir
    run_py e2e_summary.py "${1:?usage: ./e2e.sh summary <run dir>}"
}

# ----------------------------------------------------------------- main ----

case "${1:-}" in
    build) cmd_build ;;
    build-fork) cmd_build_fork ;;
    check) cmd_check ;;
    run) MODE=bench; cmd_run ;;
    run3) MODE=bench; ARMS=$ARMS3; cmd_run ;;
    profile) MODE=profile; cmd_run ;;
    smoke) MODE=smoke; cmd_run ;;
    summary) cmd_summary "${2:-}" ;;
    all) cmd_build; MODE=bench; cmd_run ;;
    *) sed -n '2,13p' "$0"; exit 1 ;;
esac
