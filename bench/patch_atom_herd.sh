#!/bin/bash
# Build an ATOM copy carrying ATOM#2713 (FlyDSL MXFP4 decode through aiter HERD on gfx1250).
# The source tree is copied, never modified.
#
# Usage: [CHECK_ONLY=1] patch_atom_herd.sh [ATOM_SRC=/app/ATOM] [ATOM_DST=/data/binding/moe_1031/atom_pr2713]
#   CHECK_ONLY=1 only reports whether ATOM_DST is ready or the port applies to ATOM_SRC; nothing is copied.
#
# patches/atom_pr2713.diff is the PR as published (against ATOM main). It does not apply to the
# r1_1012_rebased branch in /app/ATOM, which routes inside Mxfp4MoEMethod.apply() instead of a
# per-method select_experts(); patches/atom_pr2713_port_r1_1012_rebased.diff is the same change
# ported to that branch, plus a one-time "FlyDSL HERD route" info log that run_config.sh checks for.
set -euo pipefail
BENCH=$(cd "$(dirname "$0")" && pwd)
SRC=${1:-/app/ATOM}
DST=${2:-/data/binding/moe_1031/atom_pr2713}
PATCH=$BENCH/patches/atom_pr2713_port_r1_1012_rebased.diff
PORT_BASE=f19778df4

if [ -d "$DST" ]; then
  if git -C "$DST" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "already patched: $DST ($(git -C "$DST" rev-parse --short HEAD), branch $(git -C "$DST" branch --show-current))"
    if [ "$(git -C "$DST" rev-parse HEAD)" != "$(git -C "$SRC" rev-parse HEAD)" ]; then
      echo "warning: $SRC is at $(git -C "$SRC" rev-parse --short HEAD), the copy is at $(git -C "$DST" rev-parse --short HEAD);" \
        "pass a new ATOM_DIR to test the newer ATOM"
    fi
    exit 0
  fi
  echo "$DST exists but does not carry the patch; remove it or pass another destination"
  exit 1
fi

head=$(git -C "$SRC" rev-parse --short=9 HEAD)
if [ "$head" != "$PORT_BASE" ]; then
  echo "warning: $SRC is at $head; the port was made against $PORT_BASE (r1_1012_rebased)"
fi
if ! git -C "$SRC" apply --check "$PATCH" 2>/dev/null; then
  echo "the port does not apply to $SRC/atom/model_ops/moe.py; re-port patches/atom_pr2713.diff by hand"
  exit 1
fi
if [ "${CHECK_ONLY:-0}" = "1" ]; then
  echo "check ok: the port applies to $SRC; $DST would be created"
  exit 0
fi

cp -a "$SRC" "$DST"
git -C "$DST" checkout -q -b "$(git -C "$DST" branch --show-current)_pr2713_port"
git -C "$DST" apply "$PATCH"
python3 -m py_compile "$DST/atom/model_ops/moe.py"
echo "patched ATOM: $DST (branch $(git -C "$DST" branch --show-current))"
git -C "$DST" diff --stat -- atom/model_ops/moe.py
