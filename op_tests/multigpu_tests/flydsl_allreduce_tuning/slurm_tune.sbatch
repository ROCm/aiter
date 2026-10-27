#!/bin/bash
# Run the FlyDSL all-reduce tuning (tune.sh) as a Slurm job inside Docker.
#
#   mkdir -p ~/flydsl_tuning && cd ~/flydsl_tuning     # the Slurm log lands in the submit dir
#   sbatch /path/to/slurm_tune.sbatch                  # the real run (whole node, 8 GPUs)
#   DRY_RUN=1 bash /path/to/slurm_tune.sbatch          # GPU-free rehearsal, submitted as a CPU-only
#                                                      # Slurm job (log: slurm-<id>.out in the cwd)
#
# Everything lands in $RESULTS/<run name>/ (default ~/flydsl_tuning/job-<id>/):
#   aiter/      the clone, pinned to the commit checked out at the first start
#   out/        tune.sh --out: sweep CSVs, fit.txt, fused/fused_allreduce_ladders.json, ...
#   out-smoke/  the smoke pass that gates the real run
#   home/       container $HOME: the aiter JIT and FlyDSL caches, pip cache
#
# Resume: tune.sh skips finished chunks. A requeued job reuses its own directory; after a
# time-out, resubmit with RUN_NAME=<the old directory name> to continue where it stopped.
#
# DRY_RUN=1 does everything except touch a GPU: pulls the image, clones, starts the
# container WITHOUT /dev/kfd, installs the Python deps, and runs `tune.sh --dry-run all`
# (environment checks, every chunk's command and input files, the fit scripts loading).
# Invoked with DRY_RUN=1 outside Slurm it submits itself via sbatch without GPUs, exclusive
# node or requeue (add a partition with DRY_SBATCH_ARGS='-p <cpu-partition>'). Inside a job
# (or a plain `srun ... bash`) it just runs.
#
# Environment overrides: IMAGE, REPO_URL, REF, RESULTS, RUN_NAME, DRY_RUN, DRY_SBATCH_ARGS,
#   DRY_GPU_ARCHS (arch aiter imports for on the GPU-less dry run; default gfx950),
#   TUNE_ARGS (extra tune.sh options for the real run, e.g. TUNE_ARGS='--tp "8 4"'),
#   TUNE_CMD (the tune.sh command of the smoke, real and dry runs; default all, e.g.
#   TUNE_CMD=fused-sweep for the fused sweep alone).
#SBATCH --job-name=flydsl-ar-tune
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:gfx950-mi355x:8
#SBATCH --time=10:00:00
#SBATCH --requeue

set -euo pipefail

# DRY_RUN=1 from a login shell: submit this script as a GPU-free Slurm job. The #SBATCH
# directives above can't be made conditional, so the GPU/exclusive/requeue ones are stripped
# from a copy that is piped to sbatch. Extra sbatch options: DRY_SBATCH_ARGS='-p <cpu-partition>'.
if [[ ${DRY_RUN:-0} == 1 && -z ${SLURM_JOB_ID:-} ]]; then
  command -v sbatch >/dev/null || { echo "sbatch not found; run on the Slurm head node"; exit 1; }
  # The 8-GPU MI355 node is scarce, so rehearse on a CPU-only node (far better availability).
  # Default to an idle alola 'compile' node (no GPUs, Docker present); override with DRY_SBATCH_ARGS.
  dry_args=${DRY_SBATCH_ARGS:-}
  if [[ -z $dry_args ]]; then
    node=$(sinfo -h -p defq -t idle -N -o '%N' 2>/dev/null | grep -m1 -E 'alola-compile') || true
    dry_args="--partition=defq${node:+ --nodelist=$node}"
    echo "dry run -> ${node:-defq (no idle alola-compile node found; may land on a GPU node)}"
  fi
  # Strip the real-run-only directives so the rehearsal needs no GPUs / exclusive node / requeue.
  sed -E '/^#SBATCH --(exclusive|gres|requeue|time|partition|constraint)/d' "$0" |
    sbatch --job-name=flydsl-ar-tune-dry --nodes=1 --time=02:00:00 --export=ALL,DRY_RUN=1 $dry_args
  exit $?
fi

IMAGE=${IMAGE:-rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0}
REPO_URL=${REPO_URL:-git@github.com:ROCm/aiter.git}
REF=${REF:-users/vpietila/flydsl-fused-comm-kernels-with-tuning}
RESULTS=${RESULTS:-$HOME/flydsl_tuning}
DRY_RUN=${DRY_RUN:-0}
TUNE_ARGS=${TUNE_ARGS:-}
TUNE_CMD=${TUNE_CMD:-all}
JOB=${SLURM_JOB_ID:-local$$}
RUN_NAME=${RUN_NAME:-$([[ $DRY_RUN == 1 ]] && echo "dryrun-$JOB" || echo "job-$JOB")}
RUN_DIR=$RESULTS/$RUN_NAME
TUNE_REL=op_tests/multigpu_tests/flydsl_allreduce_tuning

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

command -v docker >/dev/null || { log "docker not found on $(hostname)"; exit 1; }
log "run dir: $RUN_DIR (dry_run=$DRY_RUN, image=$IMAGE, ref=$REF)"
mkdir -p "$RUN_DIR/home" "$RUN_DIR/out"

# Clone on the host so the job uses your git credentials; the container never needs them.
if [[ -d $RUN_DIR/aiter/.git ]]; then
  log "reusing clone $(git -C "$RUN_DIR/aiter" rev-parse --short HEAD) (not updating, keeps the run pinned)"
else
  git clone "$REPO_URL" "$RUN_DIR/aiter"
  git -C "$RUN_DIR/aiter" checkout "$REF"
fi
log "commit: $(git -C "$RUN_DIR/aiter" log -1 --format='%h %s')"
[[ -x $RUN_DIR/aiter/$TUNE_REL/tune.sh ]] || { log "$TUNE_REL/tune.sh missing at $REF"; exit 1; }

docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"

# The container runs as root (pip must write into the image's Python); hand the files back
# to you however the job ends, using the same image so nothing else has to be pulled.
fix_owner() {
  docker run --rm -v "$RUN_DIR:/work" "$IMAGE" chown -R "$(id -u):$(id -g)" /work || true
}
trap fix_owner EXIT

cat > "$RUN_DIR/inner.sh" <<'INNER'
set -euo pipefail
export HOME=/work/home PIP_CACHE_DIR=/work/home/.cache/pip PIP_BREAK_SYSTEM_PACKAGES=1
git config --global --add safe.directory '*'
# Slurm/the submit shell may carry a GPU pin (e.g. HIP_VISIBLE_DEVICES=7) that would hide GPUs.
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
# The dry run is on a CPU-only node, so aiter's rocminfo arch probe finds no GPU and `import
# aiter` dies. GPU_ARCHS lets aiter import/codegen for a named arch without a live GPU
# (aiter/jit/utils/chip_info.py). Match the real target (gfx950); override with DRY_GPU_ARCHS.
if [[ $DRY_RUN == 1 ]]; then
  # CPU-only rehearsal node. GPU_ARCHS lets aiter import/codegen for a named arch with no GPU,
  # but a few import-time constants (e.g. aiter/utility/dtypes.py) call get_gfx_runtime(), which
  # ALWAYS shells out to rocminfo and ignores GPU_ARCHS -> "No gfx arch found" with no GPU. Shim a
  # fake rocminfo on PATH reporting the target arch so the GPU-less import finishes. The dry run
  # never launches a kernel, so the spoofed arch/CU only feed dtype/CU constants. Override arch
  # with DRY_GPU_ARCHS, CU count with DRY_CU_NUM.
  export GPU_ARCHS=${DRY_GPU_ARCHS:-gfx950}
  export CU_NUM=${DRY_CU_NUM:-256}
  arch=${GPU_ARCHS%%;*}
  mkdir -p /work/home/fakebin
  { printf '#!/bin/sh\n'
    printf 'echo "  Name:                    %s"\n' "$arch"
    printf 'echo "  Marketing Name:          AMD Instinct (dry-run stub)"\n'
    printf 'echo "  Device Type:             GPU"\n'
    printf 'echo "  ASIC Revision:           0"\n'
    printf 'echo "  Compute Unit:            %s"\n' "$CU_NUM"
  } > /work/home/fakebin/rocminfo
  chmod +x /work/home/fakebin/rocminfo
  export PATH=/work/home/fakebin:$PATH
fi
# ROCm pip/venv image shim (applies to the dry AND the real run -- same image). This rocm/pytorch
# build ships ROCm as pip packages with no /opt/rocm. hipcc (ROCM_HOME=<venv>) COMPILES fine --
# it finds its device-library bitcode relative to itself -- but aiter then LINKS with
# `-L<venv>/lib -lamdhip64`, while the hip runtime actually lives in the _rocm_sdk_core package as
# libamdhip64.so.7 with no unversioned soname -> "cannot find -lamdhip64". Fix: drop an unversioned
# libamdhip64.so into <venv>/lib where aiter's linker looks. Deliberately do NOT set ROCM_PATH to
# _rocm_sdk_core -- that moves hipcc off the layout where it locates the device libraries and the
# compile fails instead of the link.
hipcc_bin=$(command -v hipcc || true)
if [[ -n $hipcc_bin ]]; then
  venv=$(dirname "$(dirname "$hipcc_bin")")
  hip_real=$(ls "$venv"/lib/python*/site-packages/_rocm_sdk_core/lib/libamdhip64.so.* 2>/dev/null | head -1)
  [[ -n ${hip_real:-} && ! -e $venv/lib/libamdhip64.so ]] && ln -s "$hip_real" "$venv/lib/libamdhip64.so" 2>/dev/null || true
fi
cd /work/aiter
python -m pip install -r requirements.txt pyzmq tabulate
cd "$TUNE_REL"
./tune.sh setup
if [[ $DRY_RUN == 1 ]]; then
  ./tune.sh --dry-run "$TUNE_CMD"
  ./tune.sh --dry-run --smoke "$TUNE_CMD"
  exit 0
fi
# The smoke pass uses its own --out: its tiny CSVs share chunk names with the real run's
# and would otherwise be skipped over as "already done".
./tune.sh --smoke --out /work/out-smoke "$TUNE_CMD"
eval "./tune.sh --out /work/out $TUNE_ARGS $TUNE_CMD"
INNER

GPU_FLAGS=()
if [[ $DRY_RUN != 1 ]]; then
  GPU_FLAGS=(--device=/dev/kfd --device=/dev/dri --ipc=host --shm-size=16g --ulimit memlock=-1
             --cap-add=SYS_PTRACE --security-opt seccomp=unconfined)
fi

log "starting container (gpu flags: ${GPU_FLAGS[*]:-none})"
docker run --rm ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} -v "$RUN_DIR:/work" \
  -e DRY_RUN="$DRY_RUN" -e TUNE_ARGS="$TUNE_ARGS" -e TUNE_CMD="$TUNE_CMD" -e TUNE_REL="$TUNE_REL" \
  -e DRY_GPU_ARCHS="${DRY_GPU_ARCHS:-}" \
  "$IMAGE" bash /work/inner.sh
log "finished; results in $RUN_DIR/out"
