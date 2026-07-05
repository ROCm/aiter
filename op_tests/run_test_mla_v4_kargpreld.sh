#!/usr/bin/env bash
# Run the v4 nm MLA decode accuracy + perf sweep (test_mla_v4_kargpreld.py)
# INSIDE the ff_mla container, launched from the HOST (bare metal).
#
#   ./run_test_mla_v4_kargpreld.sh                          # default sweep
#   ./run_test_mla_v4_kargpreld.sh -b 64 -c 1024 2048       # override shapes
#   ./run_test_mla_v4_kargpreld.sh --split-kv 1 2 4         # sweep splits
#   CONTAINER=ff_mla ./run_test_mla_v4_kargpreld.sh
#
# Every argument after the script name is forwarded verbatim to the python
# script, so its CLI args (-b/--batch, -c/--kv-seq-lens, --variant, --split-kv,
# --attn-sink, --seed, --iters, --warmup) work unchanged.
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
  ENABLE_CK=0 python3 op_tests/test_mla_v4_kargpreld.py '"$*"'
'
