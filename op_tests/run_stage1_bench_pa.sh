#!/usr/bin/env bash
# Run the stage1-only perf comparison (_stage1_bench_pa.py: asm sparse decode
# stage1 vs ATOM gluon pa_decode_sparse stage1) INSIDE the ff_mla container,
# launched from the HOST (bare metal).
#
#   ./run_stage1_bench_pa.sh                  # default sweep from the .py file
#   ./run_stage1_bench_pa.sh 128 2048 4       # one combo: batch ctx split
#   CONTAINER=ff_mla ./run_stage1_bench_pa.sh
#
# Every argument after the script name is forwarded verbatim to the python
# script, so its positional CLI args (batch ctx split) work unchanged.
set -euo pipefail

CONTAINER="${CONTAINER:-ff_mla}"
AITER_DIR="${AITER_DIR:-/home/amd/feifei/aiter}"

# ensure container is running
if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "[host] starting container $CONTAINER ..."
  docker start "$CONTAINER" >/dev/null
  sleep 2
fi

echo "[host] container=$CONTAINER  aiter_dir=$AITER_DIR  args=[$*]"

# Forward all extra args to the python script. "$@" is expanded here (on the
# host) into the remote bash -lc command string.
docker exec "$CONTAINER" bash -lc '
  cd "'"$AITER_DIR"'"
  # clear JIT cache before each run
  sudo rm -rf ../aiter/jit/built
  sudo rm -f ../aiter/jit/*.so
  ENABLE_CK=0 python3 op_tests/_stage1_bench_pa.py '"$*"'
'
