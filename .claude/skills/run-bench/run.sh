#!/bin/bash
# Run one combo suite, record what produced it, write report.html.
# Usage (from anywhere): run.sh <suite> <out_dir> [gpus]
#   suite: a bench_gfx1250_combo.py suite flag without the dashes (dsr1, dsv41, ...)
#   gpus:  HIP_VISIBLE_DEVICES, default 0; mega_moe needs four (0,1,2,3)
#   env:   OPS="moe mla_v4_decode" to pick ops, AITER_BENCH_TOKENS to override the
#          sweeps, IMAGE=<docker image tag@digest> to record the image.
set -uo pipefail
SUITE=${1:?usage: run.sh <suite> <out_dir> [gpus]}
OUT=${2:?usage: run.sh <suite> <out_dir> [gpus]}
GPU=${3:-0}
SKILL_DIR=$(cd "$(dirname "$0")" && pwd)
AITER=$(cd "$SKILL_DIR/../../.." && pwd)
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
cd "$AITER" || exit 1

ver() { python3 -c "import importlib.metadata as m; print(m.version('$1'))" 2>/dev/null || echo unknown; }
CMD="HIP_VISIBLE_DEVICES=$GPU python3 op_tests/bench_gfx1250_combo.py --$SUITE --json${OPS:+ --ops $OPS}"
{
  echo "date_utc: $(date -u +%FT%TZ)"
  echo "suite: $SUITE"
  echo "host: $(hostname)"
  echo "image: ${IMAGE:-not recorded}"
  echo "aiter: $(git log -1 --format='%H %ci') ($(git status --porcelain --untracked-files=no | wc -l) modified files)"
  echo "rocm: $(ver rocm) $(cat /opt/rocm/.info/version 2>/dev/null)"
  echo "torch: $(python3 -c 'import torch; print(torch.__version__, "hip", torch.version.hip)' 2>/dev/null)"
  echo "triton: $(ver triton)"
  echo "flydsl: $(ver flydsl)"
  echo "combo_sha256: $(sha256sum op_tests/bench_gfx1250_combo.py | cut -c1-64)"
  echo "AITER_BENCH_TOKENS: ${AITER_BENCH_TOKENS:-unset}"
  echo "cmd: $CMD"
} > "$OUT/provenance.txt"

START=$(date -u +%FT%TZ)
HIP_VISIBLE_DEVICES=$GPU python3 op_tests/bench_gfx1250_combo.py --"$SUITE" --json ${OPS:+--ops $OPS} > "$OUT/combo.log" 2>&1
RC=$?
printf 'start_utc: %s\nend_utc: %s\nrc: %s\n' "$START" "$(date -u +%FT%TZ)" "$RC" > "$OUT/status.txt"
python3 "$SKILL_DIR/report.py" "$OUT"
exit $RC
