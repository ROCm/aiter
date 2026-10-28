#!/usr/bin/env bash
# DeepSeek V4 end-to-end A/B on the MI350P box: aiter main vs main + FlyDSL comm branch.
# Run on the host.
#
#   ./e2e.sh build             # build both images from ./Dockerfile
#   ./e2e.sh build-fork        # build IMAGE_FORK: IMAGE_B + the vLLM fork's prefill dispatch
#   ./e2e.sh build-eval        # build IMAGE_B and IMAGE_FORK, the images "eval" needs
#   ./e2e.sh check             # preflight: images, vLLM all-reduce path, model
#   ./e2e.sh run               # check, then the A/B benchmark, then the summary
#   ./e2e.sh run3              # the same for the 3 arms in ARMS3 (needs build-fork)
#   ./e2e.sh profile           # check, then one torch-profiled run per server
#   ./e2e.sh smoke             # check, then only start each server, smoke-test it, stop it
#   ./e2e.sh eval              # check, then the accuracy eval (lm-eval) of the arms in ARMS_EVAL
#   ./e2e.sh summary <dir>     # summary table of a finished run
#   ./e2e.sh eval-summary <dir> # accuracy table of a finished eval run
#   ./e2e.sh all               # build + run
#
# Settings are environment variables, e.g. TPS="8" ./e2e.sh run. See below.
# The defaults are for MI350P. On MI350X (8 XCDs, 256 CUs, 252 GiB):
#   AITER_CU_NUM=256 AITER_NUM_XCDS=8 IMAGE_B=vllm-aiter-ar-mi350x \
#     IMAGE_FORK=vllm-aiter-ar-prefill-mi350x ./e2e.sh build-eval
#   IMAGE_FORK=vllm-aiter-ar-prefill-mi350x TPS="4 8" GPU_GIB=252 \
#     GPU_MEM_UTIL=0.92 MAX_MODEL_LEN=16384 ./e2e.sh eval
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
IMAGE_B=${IMAGE_B:-vllm-aiter-ar}     # aiter main + FlyDSL comm branch
IMAGE_FORK=${IMAGE_FORK:-vllm-aiter-ar-prefill}   # IMAGE_B + the vLLM fork
# The arms to compare: "tag,image[,VAR=value...]" separated by spaces. Each arm
# gets its own server; VAR=value overrides the image's environment for that arm.
# The first arm is the reference the summary's deltas are against. Tags are \w+.
# Default: the A/B of aiter main vs main + the FlyDSL comm branch. For the 3-way prefill
# comparison use ./e2e.sh run3 (see ARMS3).
ARMS=${ARMS:-"main,$IMAGE_A ar,$IMAGE_B"}
# base: stock vLLM, no FlyDSL. fork: the vLLM fork with aiter's FlyDSL one-shot and
# fused AR+RMSNorm but not the FlyDSL QuickReduce. flyqr: fork + FlyDSL QuickReduce
# INT4. fork and flyqr differ only by VLLM_ROCM_QUICK_REDUCE_USE_FLYDSL. There is
# no stock-QuickReduce arm: vLLM's HIP QuickReduce is disabled on PCIe-only GPUs
# above TP2 (it needs xGMI), as on this box.
ARMS3=${ARMS3:-"base,$IMAGE_A fork,$IMAGE_FORK,VLLM_ROCM_QUICK_REDUCE_USE_FLYDSL=0 flyqr,$IMAGE_FORK"}
# The accuracy eval: the same image with and without INT4 quick-reduce. exact
# turns quick-reduce off, so every all-reduce is exact (aiter's CustomAllreduce,
# FlyDSL one-shot included). flyqr is the image as built: the FlyDSL INT4 mesh/ring
# serves plain all-reduce, and the quantized fused AR+RMSNorm serves its own
# windows. Add a second exact arm (e.g. exact2) to measure run-to-run noise.
ARMS_EVAL=${ARMS_EVAL:-"exact,$IMAGE_FORK,VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=NONE,AITER_QUICK_REDUCE_QUANTIZATION=NONE flyqr,$IMAGE_FORK"}
# lm-eval settings for "eval". The concurrency sets the decode batch: one
# DeepSeek V4 token is 14 KiB of all-reduce payload, so at 64 concurrent
# requests a decode step moves up to 896 KiB, inside the FlyDSL INT4 mesh
# window (from 384 KiB on gfx950/xGMI, 96 KiB on gfx950/PCIe at TP8).
EVAL_TASKS=${EVAL_TASKS:-gsm8k}
EVAL_FEWSHOT=${EVAL_FEWSHOT:-5}
EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-64}
EVAL_LIMIT=${EVAL_LIMIT:-}            # questions per task; empty = all (quick check: 100)
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
    log "building $IMAGE_B (main + FlyDSL comm branch)"
    docker build -t "$IMAGE_B" $(build_args) -f "$HERE/Dockerfile" "$HERE"
    log "building $IMAGE_A (main only)"
    docker build -t "$IMAGE_A" $(build_args) --build-arg WITH_ALLREDUCE=0 -f "$HERE/Dockerfile" "$HERE"
}

# ./Dockerfile needs BuildKit (RUN --mount); the legacy builder rejects it.
export DOCKER_BUILDKIT=1

# Target build options for ./Dockerfile; empty = its defaults (MI350P).
build_args() {
    [ -n "${AITER_CU_NUM:-}" ] && echo "--build-arg AITER_CU_NUM=$AITER_CU_NUM"
    [ -n "${AITER_NUM_XCDS:-}" ] && echo "--build-arg AITER_NUM_XCDS=$AITER_NUM_XCDS"
    return 0
}

cmd_build_fork() {
    docker image inspect "$IMAGE_B" >/dev/null 2>&1 || die "$IMAGE_B missing: ./e2e.sh build"
    log "building $IMAGE_FORK ($IMAGE_B + vLLM fork)"
    docker build -t "$IMAGE_FORK" --build-arg BASE="$IMAGE_B" -f "$HERE/Dockerfile.vllm-fork" "$HERE"
}

cmd_build_eval() {
    log "building $IMAGE_B (main + FlyDSL comm branch)${AITER_CU_NUM:+, CU_NUM=$AITER_CU_NUM}"
    docker build -t "$IMAGE_B" $(build_args) -f "$HERE/Dockerfile" "$HERE"
    cmd_build_fork
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
        if [ "$MODE" = eval ]; then
            in_image "$image" test -x /opt/lm-eval/bin/lm_eval \
                || die "$tag: $image has no lm-eval: ./e2e.sh build-fork"
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
        eval) run_eval "$out" ;;
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

run_eval() {  # out_dir
    # lm-eval runs in the server's container, against its completions API.
    # Greedy (the tasks' defaults). The server may be offline; lm-eval needs the
    # hub only for the datasets, cached per arm under CACHE_ROOT.
    local out=$1
    log "quick-reduce in the server: $(docker exec "$NAME" sh -c \
        'echo "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=$VLLM_ROCM_QUICK_REDUCE_QUANTIZATION, VLLM_ROCM_QUICK_REDUCE_USE_FLYDSL=$VLLM_ROCM_QUICK_REDUCE_USE_FLYDSL, AITER_FLY_AR=$AITER_FLY_AR"')"
    log "eval $EVAL_TASKS, ${EVAL_FEWSHOT}-shot, concurrency $EVAL_CONCURRENCY${EVAL_LIMIT:+, limit $EVAL_LIMIT}"
    docker exec \
        -e HF_HUB_OFFLINE=0 -e HF_DATASETS_OFFLINE=0 -e HF_ALLOW_CODE_EVAL=1 \
        "$NAME" /opt/lm-eval/bin/lm_eval \
        --model local-completions \
        --model_args "model=$MODEL,base_url=http://localhost:$PORT/v1/completions,num_concurrent=$EVAL_CONCURRENCY,max_retries=3,timeout=3600,tokenized_requests=False,tokenizer_backend=None" \
        --tasks "$EVAL_TASKS" \
        --num_fewshot "$EVAL_FEWSHOT" \
        --batch_size 1 \
        --seed 15 \
        ${EVAL_LIMIT:+--limit "$EVAL_LIMIT"} \
        --output_path /traces/lm_eval \
        --log_samples \
        >>"$out/eval.log" 2>&1 \
        || { log "eval FAILED, see eval.log"; return 0; }
    # Left where the container wrote it: the directory is root's, so the host
    # user may not be able to move it.
    log "eval done (traces/lm_eval): $(grep -E '^\|' "$out/eval.log" | grep -m2 -E 'exact_match|acc' | tr -s ' ' || true)"
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
    elif [ "$MODE" = eval ]; then
        run_py e2e_eval_summary.py "$RUN_DIR" | tee "$RUN_DIR/summary.md"
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
        # The server images have python3; the host may not. Any built one will do.
        local image
        for image in "$IMAGE_A" "$IMAGE_B" "$IMAGE_FORK"; do
            docker image inspect "$image" >/dev/null 2>&1 && break
        done
        docker run --rm --entrypoint "" -v "$HERE":/e2e:ro -v "$dir":/r \
            "$image" python3 "/e2e/$script" /r
    fi
}

cmd_summary() {  # run_dir
    run_py e2e_summary.py "${1:?usage: ./e2e.sh summary <run dir>}"
}

# ----------------------------------------------------------------- main ----

case "${1:-}" in
    build) cmd_build ;;
    build-fork) cmd_build_fork ;;
    build-eval) cmd_build_eval ;;
    check) cmd_check ;;
    run) MODE=bench; cmd_run ;;
    run3) MODE=bench; ARMS=$ARMS3; cmd_run ;;
    profile) MODE=profile; cmd_run ;;
    smoke) MODE=smoke; cmd_run ;;
    eval) MODE=eval; ARMS=$ARMS_EVAL; cmd_run ;;
    summary) cmd_summary "${2:-}" ;;
    eval-summary) run_py e2e_eval_summary.py "${2:?usage: ./e2e.sh eval-summary <run dir>}" ;;
    all) cmd_build; MODE=bench; cmd_run ;;
    *) sed -n '2,22p' "$0"; exit 1 ;;
esac
