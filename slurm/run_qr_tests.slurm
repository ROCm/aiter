#!/bin/bash
#SBATCH --job-name=aiter-qr-tests
#SBATCH --partition=<PARTITION>      # EDIT ME
#SBATCH --account=<ACCOUNT>          # EDIT ME
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Ad-hoc QR test + comm-allreduce benchmark run (not an official CI pipeline).
# Clones the aiter repo fresh, builds a container from slurm/Dockerfile, runs
# the three FlyDSL QR pytest files, then sweeps bench_comm_allreduce.py over
# TP=2,4,8. Neither the tests nor the benchmark use torchrun/mpirun/srun
# --ntasks -- each self-forks its ranks internally, so this runs as a single
# task on one node with all 8 GPUs visible.

set -uo pipefail   # not -e: a failing test/benchmark step must not abort the rest

REPO_URL="https://github.com/ROCm/aiter.git"
REPO_REF="users/vpietila/mi350p-comm-kernels"   # branch/tag to test; EDIT as needed
JOB_DIR="${SLURM_SUBMIT_DIR}/aiter_qr_test_runs/${SLURM_JOB_ID}"
REPO_DIR="${JOB_DIR}/repo"
RESULTS_DIR="${REPO_DIR}/results"
DOCKERFILE="${REPO_DIR}/slurm/Dockerfile"    # comes from the clone itself, fully self-contained
CONTAINER_REPO_DIR="/home/$(id -un)/aiter"   # where REPO_DIR is bind-mounted inside the container
IMAGE_TAG="aiter-qr-tests:${SLURM_JOB_ID}"
CONTAINER_NAME="aiter_qr_tests_${SLURM_JOB_ID}"

mkdir -p "${JOB_DIR}"
git clone --branch "${REPO_REF}" --depth 1 "${REPO_URL}" "${REPO_DIR}"
mkdir -p "${RESULTS_DIR}"

docker build -f "${DOCKERFILE}" -t "${IMAGE_TAG}" \
  --build-arg USERNAME="$(id -un)" --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" \
  "${REPO_DIR}"

cleanup() { docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# Flags mirror the already-proven persistent dev container invocation, minus
# the whole-home mount (job-scoped repo clone only).
docker run -d --init --name "${CONTAINER_NAME}" \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt=seccomp=unconfined --ipc=host \
  --group-add=video --cap-add=SYS_PTRACE \
  -v "${REPO_DIR}:${CONTAINER_REPO_DIR}" \
  "${IMAGE_TAG}" tail -f /dev/null

run_in_container() { docker exec -w "${CONTAINER_REPO_DIR}" "${CONTAINER_NAME}" bash -lc "$1"; }

run_in_container "pip install --no-build-isolation -e ." || { echo "aiter install failed"; exit 1; }

run_in_container "pytest -v \
  op_tests/flydsl_tests/test_flydsl_qr_int4.py \
  op_tests/flydsl_tests/test_flydsl_qr_codec.py \
  op_tests/flydsl_tests/test_flydsl_qr_1stage.py" \
  2>&1 | tee "${RESULTS_DIR}/pytest_qr_results.log"

for tp in 2 4 8; do
  run_in_container "python3 op_tests/multigpu_tests/bench_comm_allreduce.py -tp ${tp} -o results/bench_allreduce_tp${tp}.md" \
    2>&1 | tee "${RESULTS_DIR}/bench_allreduce_tp${tp}.log"
done

echo "Results in: ${RESULTS_DIR}"
