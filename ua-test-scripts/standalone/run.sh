#!/usr/bin/env bash
# Fast, torch-free FA4 ATT overlap analysis.
#
#   standalone/run.sh [sq] [hq] [hk] [d] [mask] [iters]
#
# vs att_analysis/run.sh (which traces the PyTorch test harness, ~2.5 min floored
# by libtorch code-object disassembly), this traces the standalone ua_trace
# executable, so rocprofv3 ATT only disassembles ~1 kernel -> seconds.
#
# Steps:
#   1. build ua_trace (only if missing or REBUILD=1) with line tables + slim stubs.
#   2. rocprofv3 --att on the executable (single SIMD, one dispatch).
#   3. render the two-warpgroup overlap timeline + Markdown report.
#
# Env: GPU=2  ARCH=gfx950  DTYPE=fp8  SIMD_MASK=0x1  CU=1  SE_MASK=0x1
#      SIMD=0 (report)  REPORT_ITERS=4  REBUILD=0  TAG=...
set -euo pipefail

SQ="${1:-8192}"; HQ="${2:-16}"; HK="${3:-2}"; D="${4:-128}"; MASK="${5:-0}"; ITERS="${6:-3}"
DTYPE="${DTYPE:-fp8}"; GPU="${GPU:-2}"; ARCH="${ARCH:-gfx950}"
SIMD_MASK="${SIMD_MASK:-0x1}"; CU="${CU:-1}"; SE_MASK="${SE_MASK:-0x1}"
SIMD="${SIMD:-0}"; REPORT_ITERS="${REPORT_ITERS:-4}"

HERE="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS="$(dirname "$HERE")"
EXE="$HERE/build/ua_trace"
MASKTAG=$([[ "$MASK" == "0" ]] && echo "noncausal" || echo "causal")
TAG="${TAG:-att_std_d${D}_${DTYPE}_${MASKTAG}_sq${SQ}}"
RUN_DIR="$SCRIPTS/rocprof_analysis/runs/$TAG"
ATT_DIR="$RUN_DIR/att"

# build.sh self-guards: rebuilds iff REBUILD=1, stamp mismatch (arch/dtype/d/mask),
# or any kernel source/header is newer than the exe. So a stale binary can never
# be silently traced/measured -- no JIT-style "is it stale?" ambiguity.
echo "[std] (1/3) build (self-guarding) ..."
ARCH="$ARCH" DTYPE="$DTYPE" D="$D" MASK="$MASK" bash "$HERE/build.sh"

echo "[std] (2/3) rocprofv3 ATT on executable -> $TAG ..."
rm -rf "$ATT_DIR"; mkdir -p "$ATT_DIR"
ATT_LIB_DIR="/opt/rocm/lib"
HIP_VISIBLE_DEVICES="$GPU" /opt/rocm/bin/rocprofv3 \
    --att --att-library-path "$ATT_LIB_DIR" \
    --att-target-cu "$CU" --att-shader-engine-mask "$SE_MASK" \
    --att-simd-select "$SIMD_MASK" --att-consecutive-kernels 1 --att-activity 8 \
    --kernel-include-regex '.*UnifiedAttentionKernel.*' \
    -o "att" -d "$ATT_DIR" \
    -- "$EXE" "$SQ" "$HQ" "$HK" "$D" "$MASK" "$ITERS" > "$RUN_DIR/att.log" 2>&1 || {
        echo "rocprofv3 failed; tail of $RUN_DIR/att.log:" >&2; tail -40 "$RUN_DIR/att.log" >&2; exit 1; }

UI_DIRS=( "$ATT_DIR"/ui_output_agent_*_dispatch_* )
[[ -d "${UI_DIRS[0]:-}" ]] || { echo "no ui_output produced; tail att.log:" >&2; tail -40 "$RUN_DIR/att.log" >&2; exit 1; }
echo "[ok] traced ${#UI_DIRS[@]} dispatch(es): $(basename "${UI_DIRS[0]}") ($(du -sh "${UI_DIRS[0]}" | awk '{print $1}'))"

echo "[std] (3/3) rendering overlap report (SIMD $SIMD, $REPORT_ITERS iters) ..."
cd "$SCRIPTS"
python3 -m att_analysis.report "$RUN_DIR" --simd "$SIMD" --iters "$REPORT_ITERS"
echo "[std] done -> $RUN_DIR/att_analysis/"
