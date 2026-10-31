#!/bin/bash
# Regenerate the TP1 serve scripts from the original serve_dsv4.sh.
# Only -tp 4 -> -tp 1 and the MoE env vars below change; everything else stays as in the original.
#
# Usage: make_serve_scripts.sh [SRC=/app/scripts/dsv4/serve_dsv4.sh] [OUT_DIR=<this dir>]
#   serve_dsv4_tp1.sh               FlyDSL MoE, HERD off            (ATOM_USE_TRITON_MOE=0)
#   serve_dsv4_tp1_flydsl_herd.sh   FlyDSL MoE + AITER_FLYDSL_USE_HERD=1 (needs the ATOM#2713 patch)
#   serve_dsv4_tp1_triton_herd.sh   Triton/gluon decode MoE + AITER_TRITON_USE_HERD=1 (ATOM_USE_TRITON_MOE=1)
#   serve_dsv4_tp1_triton_noherd.sh Triton/gluon decode MoE, AITER_TRITON_USE_HERD=0
set -euo pipefail
BENCH=$(cd "$(dirname "$0")" && pwd)
S=${1:-/app/scripts/dsv4/serve_dsv4.sh}
D=${2:-$BENCH}
mkdir -p "$D"

sed 's/-tp 4/-tp 1/' "$S" > "$D/serve_dsv4_tp1.sh"
sed 's/-tp 4/-tp 1/; s/^  ATOM_GC_THRESHOLD=20000,50,50 \\$/&\n  AITER_FLYDSL_USE_HERD=1 \\/' "$S" > "$D/serve_dsv4_tp1_flydsl_herd.sh"
sed 's/-tp 4/-tp 1/; s/^  ATOM_USE_TRITON_MOE=0 \\$/  ATOM_USE_TRITON_MOE=1 \\/' "$S" > "$D/serve_dsv4_tp1_triton_herd.sh"
sed 's/-tp 4/-tp 1/; s/^  ATOM_USE_TRITON_MOE=0 \\$/  ATOM_USE_TRITON_MOE=1 \\/; s/^  AITER_TRITON_USE_HERD=1 \\$/  AITER_TRITON_USE_HERD=0 \\/' "$S" > "$D/serve_dsv4_tp1_triton_noherd.sh"

for f in serve_dsv4_tp1.sh serve_dsv4_tp1_flydsl_herd.sh serve_dsv4_tp1_triton_herd.sh serve_dsv4_tp1_triton_noherd.sh; do
  echo "== $f (lines changed vs $(basename "$S"))"
  diff "$S" "$D/$f" | grep '^>' || true
done
