#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# FlyDSL 32x32 vs ASM 1x32, one op_test process each.
#
# FlyDSL: op_tests/test_gemm_a8w8_blockscale.py --flydsl --w-scale-block 32x32
# ASM:    op_tests/test_mxfp8fp4gemm.py --mode perf  (1x32, a_scale not preshuffled)
#
# Both processes measure every shape once, with that kernel first in the
# process, so the two sides see the same kind of cold rotation. Do not use the
# ASM column from test_gemm_a8w8_blockscale.py --w-scale-block 1x32: there ASM
# runs after FlyDSL and reuses its freed memory, so it looks faster.
#
# Usage (gfx1250; wrap in your GPU lock if the GPU is shared. This script
# does not take the lock itself):
#   op_tests/flydsl_tests/mxfp8_gemm_vs_asm/run-optest-32x32.sh [label] [-kernel flydsl|asm] [-data_init uniform|constant]
#
# -kernel flydsl   only the FlyDSL 32x32 op_test, print its table
# -kernel asm      only the ASM 1x32 op_test, print its table
# omit -kernel     run both and print (ASM - FlyDSL) / ASM
#
# Env:
#   FLYDSL_PYTHONPATH  FlyDSL python path put in front of PYTHONPATH (optional)
#   AITER_DIR          aiter checkout                 (default: this repo)
#   M                  M dimension                    (default: 16384)
#   NK                 space-separated "N,K" pairs
#   SCALE_INIT         auto | constant                (default: auto)
#   TIMEOUT            seconds per side               (default: 1800)
#   OUT_DIR            raw logs                       (default: /tmp/mxfp8_optest_32x32)
#
# Writes $OUT_DIR/<label>/{flydsl,asm}-<data_init>.log
# Percent = (ASM - FlyDSL) / ASM; positive means FlyDSL is faster.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LABEL=local
KERNEL=both
DATA_INIT=${DATA_INIT:-uniform}
usage() {
  echo "usage: $0 [label] [-kernel flydsl|asm] [-data_init uniform|constant]" >&2
  exit 2
}
while [ $# -gt 0 ]; do
  case "$1" in
    -kernel)
      [ $# -ge 2 ] || usage
      KERNEL=$2
      shift 2
      ;;
    -kernel=*)
      KERNEL=${1#-kernel=}
      shift
      ;;
    -data_init)
      [ $# -ge 2 ] || usage
      DATA_INIT=$2
      shift 2
      ;;
    -data_init=*)
      DATA_INIT=${1#-data_init=}
      shift
      ;;
    -h|--help)
      usage
      ;;
    -*)
      echo "unknown option: $1" >&2
      usage
      ;;
    *)
      LABEL=$1
      shift
      ;;
  esac
done
case "$KERNEL" in
  flydsl|asm|both) ;;
  *) echo "-kernel must be flydsl, asm, or omitted (both)" >&2; exit 2 ;;
esac
case "$DATA_INIT" in
  uniform|constant) ;;
  *) echo "-data_init must be uniform or constant" >&2; exit 2 ;;
esac

AITER_DIR=${AITER_DIR:-$(cd "$HERE/../../.." && pwd)}
P=${FLYDSL_PYTHONPATH:+$FLYDSL_PYTHONPATH:}$AITER_DIR
M=${M:-16384}
NK=${NK:-"32768,1280 5120,8192 4608,5120 5120,2304 1792,5120 25600,6144"}
SCALE_INIT=${SCALE_INIT:-auto}
TIMEOUT=${TIMEOUT:-1800}
OUT_DIR=${OUT_DIR:-/tmp/mxfp8_optest_32x32}
RES=$OUT_DIR/$LABEL
mkdir -p "$RES"
cd "$AITER_DIR" || exit 1

SHAPES=""
for nk in $NK; do
  SHAPES="$SHAPES ${M},${nk%,*},${nk#*,}"
done

FLY_LOG=$RES/flydsl-$DATA_INIT.log
ASM_LOG=$RES/asm-$DATA_INIT.log
fly_rc=skip
asm_rc=skip

if [ "$KERNEL" = both ] || [ "$KERNEL" = flydsl ]; then
  echo "flydsl 32x32 data_init=$DATA_INIT -> $FLY_LOG" >&2
  env -u PYTHONPATH PYTHONPATH=$P HIP_VISIBLE_DEVICES=0 \
    timeout "$TIMEOUT" python3 -u op_tests/test_gemm_a8w8_blockscale.py \
    --flydsl --w-scale-block 32x32 -m "$M" -nk $NK \
    --data-init "$DATA_INIT" --scale-init "$SCALE_INIT" --table \
    > "$FLY_LOG" 2>&1
  fly_rc=$?
fi

if [ "$KERNEL" = both ] || [ "$KERNEL" = asm ]; then
  echo "asm 1x32 data_init=$DATA_INIT -> $ASM_LOG" >&2
  env -u PYTHONPATH PYTHONPATH=$P HIP_VISIBLE_DEVICES=0 \
    timeout "$TIMEOUT" python3 -u op_tests/test_mxfp8fp4gemm.py \
    --mode perf --intype a8w8 --apre 0 \
    --data-init "$DATA_INIT" --scale-init "$SCALE_INIT" -s $SHAPES \
    > "$ASM_LOG" 2>&1
  asm_rc=$?
fi

python3 - "$KERNEL" "$DATA_INIT" "$FLY_LOG" "$ASM_LOG" "$fly_rc" "$asm_rc" <<'EOF'
import json, re, sys
mode, data_init, fly_log, asm_log, fly_rc, asm_rc = sys.argv[1:]

def load_fly(path):
    rows = {}
    try:
        text = open(path, errors="replace").read()
    except FileNotFoundError:
        return rows
    for line in text.splitlines():
        m = re.match(r"ROW m=(\d+) n=(\d+) k=(\d+) (.*)", line.strip())
        if not m:
            continue
        key = (int(m[1]), int(m[2]), int(m[3]))
        rows[key] = {k.strip(): v for k, v in re.findall(r"([\w /]+)=(\S+)", m[4])}
    if rows:
        return rows
    # Main's op_test prints one JSON summary instead of ROW lines.
    blob = re.search(
        r'\{"name": "gemm_a8w8_blockscale summary", "rows": \[.*\]\}', text
    )
    if not blob:
        return rows
    for row in json.loads(blob.group(0))["rows"]:
        key = (int(row["m"]), int(row["n"]), int(row["k"]))
        rows[key] = {
            "ck us": row.get("ck us"),
            "ck TFLOPS": row.get("ck TFLOPS"),
            "kernelName": row.get("kernelName", ""),
        }
    return rows

def load_asm(path):
    rows = {}
    try:
        lines = open(path, errors="replace")
    except FileNotFoundError:
        return rows
    header = None
    for line in lines:
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            if "asm us" in cells and "M" in cells:
                header = cells
            continue
        if set(cells[0]) <= set("-: "):
            continue
        row = dict(zip(header, cells))
        try:
            key = (int(row["M"]), int(row["N"]), int(row["K"]))
            rows[key] = (float(row["asm us"]), float(row["asm TFLOPS"]))
        except (KeyError, ValueError):
            continue
    return rows

def us_tf(us, tf):
    return f"{us:.1f}µs / {tf:.0f} TF"

fly = load_fly(fly_log) if mode in ("both", "flydsl") else {}
asm = load_asm(asm_log) if mode in ("both", "asm") else {}
keys = list(dict.fromkeys([*fly, *asm]))
print(f"data_init={data_init}")

if mode == "flydsl":
    print("| Shape (MxNxK) | FlyDSL 32x32 | kernel |")
    print("|---|---|---|")
    if not keys:
        print(f"| n/a | rc={fly_rc} | see {fly_log} |")
    for M, N, K in keys:
        f = fly[(M, N, K)]
        kn = re.sub(r".*_t\d+x\d+x\d+_", "", f.get("kernelName", ""))
        print(f"| {M}x{N}x{K} | {us_tf(float(f['ck us']), float(f['ck TFLOPS']))} | {kn} |")
    print(f"exit: flydsl={fly_rc}")
    print(f"log: {fly_log}")
elif mode == "asm":
    print("| Shape (MxNxK) | ASM 1x32 |")
    print("|---|---|")
    if not keys:
        print(f"| n/a | rc={asm_rc} see {asm_log} |")
    for M, N, K in keys:
        us, tf = asm[(M, N, K)]
        print(f"| {M}x{N}x{K} | {us_tf(us, tf)} |")
    print(f"exit: asm={asm_rc}")
    print(f"log: {asm_log}")
else:
    print("| Shape (MxNxK) | FlyDSL 32x32 | ASM 1x32 | (ASM-Fly)/ASM | kernel |")
    print("|---|---|---|---|---|")
    if not keys:
        print(f"| n/a | fly rc={fly_rc} | asm rc={asm_rc} | n/a | see logs |")
    for M, N, K in keys:
        f, a = fly.get((M, N, K)), asm.get((M, N, K))
        fly_cell = us_tf(float(f["ck us"]), float(f["ck TFLOPS"])) if f else "n/a"
        asm_cell = us_tf(*a) if a else "n/a"
        if f and a:
            rel = f"{(a[0] - float(f['ck us'])) / a[0] * 100:+.1f}%"
        else:
            rel = "n/a"
        kn = re.sub(r".*_t\d+x\d+x\d+_", "", f.get("kernelName", "")) if f else ""
        print(f"| {M}x{N}x{K} | {fly_cell} | {asm_cell} | {rel} | {kn} |")
    print(f"exit: flydsl={fly_rc} asm={asm_rc}")
    print(f"logs: {fly_log}")
    print(f"      {asm_log}")
EOF
