#!/bin/bash
# Reproduce the gfx1250 f4gemm low-efficiency benchmark + ATT trace end-to-end,
# then copy the whole run (e2e log + 4 ttrace prof dirs) out in a single step.
#
# Everything the run produces lands in ONE timestamped dir inside the container
# (f4gemm_<TS>/), so the copy-back is a single `docker cp` of that dir.
#
# Usage:  ./reproduce_f4gemm_low_eff.sh
set -euo pipefail

CONTAINER=f4gemm_low_eff
IMAGE=rocm/fw-bringup:gfx1250-atom--20260810
BRANCH=gfx1250/f4gemm_low_eff
APP_DIR=/app/aiter

# One timestamp shared by container (dir name) and host (copy target) so both
# sides agree without having to scrape it back out of the container.
TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="f4gemm_${TS}"
DEST="${HOME}/${OUT_DIR}_results"

# 1) Ensure the container is running: reuse if up, start if stopped, else create.
if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "container ${CONTAINER} already running"
elif docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "container ${CONTAINER} exists but stopped; starting it"
  docker start "${CONTAINER}"
else
  echo "creating container ${CONTAINER}"
  docker run -d --name "${CONTAINER}" --network=host \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -v "$HOME:/home/$USER" -v /mnt:/mnt \
    --shm-size=16G --ulimit memlock=-1 --ulimit stack=67108864 \
    "${IMAGE}"
fi

# 2) Sync the branch, build, run. The trailing arg pins the output dir name so it
#    matches OUT_DIR on the host. test_f4gemm.sh already tees its log into it.
docker exec "${CONTAINER}" bash -lc "
  set -euo pipefail
  cd ${APP_DIR}
  git fetch origin
  git checkout -t origin/${BRANCH} 2>/dev/null || git checkout ${BRANCH}
  git pull --ff-only origin ${BRANCH} || true
  git submodule sync && git submodule update --init --recursive
  python setup.py develop
  bash test_f4gemm.sh ${OUT_DIR}
"

# 3) Copy the single run dir back (log + all ttraces together).
mkdir -p "${DEST}"
docker cp "${CONTAINER}:${APP_DIR}/${OUT_DIR}" "${DEST}/"

echo "==================== results copied to: ${DEST}/${OUT_DIR} ===================="
ls -la "${DEST}/${OUT_DIR}"
