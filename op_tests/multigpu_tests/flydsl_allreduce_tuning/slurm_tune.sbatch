#!/bin/bash
# Run the FlyDSL all-reduce tuning (tune.sh) as a Slurm job inside Docker.
#
#   mkdir -p ~/flydsl_tuning && cd ~/flydsl_tuning     # the Slurm log lands in the submit dir
#   sbatch /path/to/slurm_tune.sbatch                  # the real run (whole node, 8 GPUs)
#   DRY_RUN=1 bash /path/to/slurm_tune.sbatch          # GPU-free rehearsal, see below
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
# Run it on any host with docker (a login node, or `srun -p <cpu-partition> --pty bash`).
#
# Environment overrides: IMAGE, REPO_URL, REF, RESULTS, RUN_NAME, DRY_RUN,
#   TUNE_ARGS (extra tune.sh options for the real run, e.g. TUNE_ARGS='--tp "8 4"').
#SBATCH --job-name=flydsl-ar-tune
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --requeue

set -euo pipefail

IMAGE=${IMAGE:-rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0}
REPO_URL=${REPO_URL:-git@github.com:ROCm/aiter.git}
REF=${REF:-users/vpietila/flydsl-fused-comm-kernels-with-tuning}
RESULTS=${RESULTS:-$HOME/flydsl_tuning}
DRY_RUN=${DRY_RUN:-0}
TUNE_ARGS=${TUNE_ARGS:-}
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
cd /work/aiter
python -m pip install -r requirements.txt pyzmq tabulate
cd "$TUNE_REL"
./tune.sh setup
if [[ $DRY_RUN == 1 ]]; then
  ./tune.sh --dry-run all
  ./tune.sh --dry-run --smoke all
  exit 0
fi
# The smoke pass uses its own --out: its tiny CSVs share chunk names with the real run's
# and would otherwise be skipped over as "already done".
./tune.sh --smoke --out /work/out-smoke all
eval "./tune.sh --out /work/out $TUNE_ARGS all"
INNER

GPU_FLAGS=()
if [[ $DRY_RUN != 1 ]]; then
  GPU_FLAGS=(--device=/dev/kfd --device=/dev/dri --ipc=host --shm-size=16g --ulimit memlock=-1
             --cap-add=SYS_PTRACE --security-opt seccomp=unconfined)
fi

log "starting container (gpu flags: ${GPU_FLAGS[*]:-none})"
docker run --rm ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} -v "$RUN_DIR:/work" \
  -e DRY_RUN="$DRY_RUN" -e TUNE_ARGS="$TUNE_ARGS" -e TUNE_REL="$TUNE_REL" \
  "$IMAGE" bash /work/inner.sh
log "finished; results in $RUN_DIR/out"
