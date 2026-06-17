#!/usr/bin/env bash
# Collect an Advanced Thread Trace (ATT) for the UA prefill kernel so the
# trace can be opened with ROCprof Compute Viewer (RCV / "attviewer"):
#   https://github.com/ROCm/rocprof-compute-viewer
#
# The viewer reads `ui_output_agent_{N}_dispatch_{N}` directories produced
# by `rocprofv3 --att`. rocprofv3 on this server already converts the raw
# .att/.out trace blobs into RCV's JSON layout if the trace decoder lib is
# present (we checked: /opt/rocm/lib/librocprof-trace-decoder.so exists),
# so the artefacts here can be opened by a *vanilla* RCV build on the
# user's local machine — no decoder-enabled RCV needed.
#
# Workflow:
#   1. Pick a workload that runs the UA kernel for a moderate amount of
#      time (long enough that the trace is meaningful, short enough that
#      one dispatch fits comfortably in the trace buffer).
#   2. `rocprofv3 --att ...` collects a trace for the FIRST matching kernel
#      dispatch on a SINGLE compute unit (CU). Other dispatches in the run
#      still execute but are not traced.
#   3. The output tree contains `ui_output_agent_*_dispatch_*/` per-dispatch
#      subdirs with `filenames.json` — those are what RCV opens.
#   4. tar.gz the run dir; scp to local; open in RCV.
#
# Usage:
#   ua-test-scripts/rocprof_att_prefill.sh fp8  2 2048 2048
#   ua-test-scripts/rocprof_att_prefill.sh bf16 2 2048 2048
#   # Bigger shape, more representative but bigger trace too:
#   ua-test-scripts/rocprof_att_prefill.sh fp8  16 10000 10000
#
# Tunable env vars (all optional):
#   GPU=2                 HIP_VISIBLE_DEVICES (one GPU)
#   HQ=12  HK=2  D=128    head dims; PS=64 page size
#   TAG=...               output subdir name (default: att_prefill_d${D}_${DTYPE}...)
#   CU=1                  --att-target-cu (single CU is traced in detail)
#   SE_MASK=0x1           --att-shader-engine-mask (1 SE is enough for one CU)
#   SIMD_MASK=0xF         --att-simd-select (gfx9: bitmask over 4 SIMDs)
#   N_KERNELS=1           --att-consecutive-kernels (how many UA dispatches to trace)
#   BUF_BYTES=0           --att-buffer-size in BYTES; 0 = rocprofv3 default (256MB)
#   MODE=summary          counter mode (see below)
#
# MODE selects ONE of two counter-overlay strategies — rocprofv3 errors out
# if both `--att-activity` and `--att-perfcounters` are passed:
#   MODE=summary          --att-activity 8: powers RCV's Summary tab
#                         (per-CU MFMA/VALU/LDS hardware utilisation).
#                         Best general-purpose default.
#   MODE=counters         --att-perfcounter-ctrl 3 --att-perfcounters
#                         "SQ_VALU_MFMA_BUSY_CYCLES SQ_INSTS_VALU
#                          SQ_INSTS_MFMA SQ_INST_LEVEL_LDS": populates
#                         RCV's Counters tab with per-time-bucket SQ
#                         counters. Use when you want to overlay
#                         instruction-rate plots on the wave trace.
#   MODE=none             no extra counters; trace + waves only.
#
# After the run completes, the artefacts live under:
#   ua-test-scripts/rocprof_analysis/runs/$TAG/att/
# and can be packaged with:
#   tar -C ua-test-scripts/rocprof_analysis/runs/$TAG -czvf att.tgz att/
# Then on the local machine:
#   tar xzvf att.tgz
#   rocprof-compute-viewer att/ui_output_agent_<N>_dispatch_<N>/

set -euo pipefail

# Canonical prefill shape (see ua-test-scripts/PREFILL_PERF_PLAN.md §1):
#   b=1 sq=sk=75600 hq=16 hk=2 d=128 page_size=64, fp8 (honest 8-bit dtype),
#   contiguous (non-paged) leg, non-causal (mask_type=0) — the Goal-2 focus.
# Positional args are optional and fall back to the canonical shape.
DTYPE="${1:-fp8}"
BATCH="${2:-1}"
SQ="${3:-75600}"
SK="${4:-75600}"

GPU="${GPU:-2}"
HQ="${HQ:-16}"
HK="${HK:-2}"
D="${D:-128}"
PS="${PS:-64}"

# CONTIG=1 → profile the contiguous (is_paged=false) kernel (Goal-2 focus).
# CONTIG=0 → profile the paged kernel (for the eventual Goal-1 work).
CONTIG="${CONTIG:-1}"
# MASK: 0=non-causal (Goal-2 head-to-head), 2=causal.
MASK="${MASK:-0}"

_CTAG=$([[ "$CONTIG" == "1" ]] && echo "ctg" || echo "paged")
_MTAG=$([[ "$MASK" == "0" ]] && echo "noncausal" || echo "causal")
TAG="${TAG:-att_prefill_d${D}_${DTYPE}_${_CTAG}_${_MTAG}_b${BATCH}_sq${SQ}_sk${SK}}"

CU="${CU:-1}"
SE_MASK="${SE_MASK:-0x1}"
SIMD_MASK="${SIMD_MASK:-0xF}"
N_KERNELS="${N_KERNELS:-1}"
# BUF_BYTES: --att-buffer-size takes bytes (rocprofv3 7.x rejects 512 as
# "Invalid buffer size"; the documented "Default 256MB" means 256*1024*1024).
# Leave unset (=0) to use rocprofv3's default; raise to e.g. 1073741824 (1GB)
# if a long-running dispatch overflows.
BUF_BYTES="${BUF_BYTES:-0}"
MODE="${MODE:-summary}"

HERE="$(cd "$(dirname "$0")" && pwd)"
AITER_ROOT="$(dirname "$(dirname "$HERE")")"   # analysis/ -> ua-test-scripts -> repo root
RUN_DIR="$HERE/rocprof_analysis/runs/$TAG"
ATT_DIR="$RUN_DIR/att"
mkdir -p "$ATT_DIR"

# Same demangled-kernel filter as the other rocprof scripts. RCV reads
# whatever dispatches rocprofv3 emits, so narrowing here keeps the trace
# focussed on the UA kernel proper.
KERNEL_FILTER='.*UnifiedAttentionKernel.*'

case "$MODE" in
    summary)  COUNTER_DESC="--att-activity 8 (Summary tab)" ;;
    counters) COUNTER_DESC="--att-perfcounters (Counters tab)" ;;
    none)     COUNTER_DESC="(no counter overlay)" ;;
    *)        echo "MODE must be one of: summary, counters, none (got: $MODE)" >&2; exit 1 ;;
esac

# Minimal CK-only driver, same as the existing rocprof_prefill_d128.sh.
# @perftest still runs 101 timed iters, but --att-consecutive-kernels=$N_KERNELS
# caps how many of those dispatches are actually traced (the rest run untraced).
RUN_CMD=(
    python3 "$AITER_ROOT/op_tests/test_unified_attention_ck.py"
    -b "$BATCH" -sq "$SQ" -sk "$SK"
    --num-heads "${HQ},${HK}"
    --head-size "$D"
    --block-size "$PS"
    --dtype "$DTYPE"
    --num-blocks auto
    --mask-type "$MASK"
    --no-triton
    --no-reference
    --seed 42
)
# Flip the single CK leg to the contiguous (is_paged=false) kernel.
if [[ "$CONTIG" == "1" ]]; then
    RUN_CMD+=( --contiguous )
fi

export HIP_VISIBLE_DEVICES="$GPU"

# rocprofv3 needs to find the trace-decoder lib to convert .att/.out into
# the JSON `ui_output_*` layout that vanilla RCV reads. Confirmed present
# on this box; pass it explicitly so the script is portable.
ATT_LIB_DIR="/opt/rocm/lib"
if [[ ! -f "$ATT_LIB_DIR/librocprof-trace-decoder.so" ]]; then
    echo "ERROR: $ATT_LIB_DIR/librocprof-trace-decoder.so missing —" \
         "rocprofv3 can't decode the ATT blob without it." >&2
    exit 1
fi

echo "============================================================"
echo " rocprofv3 ATT  —  prefill_d${D} ${DTYPE}  [${_CTAG} / ${_MTAG}]"
echo "   b=${BATCH} sq=${SQ} sk=${SK}  hq,hk=${HQ},${HK}  page_size=${PS}"
echo "   GPU=${GPU}  CU=${CU}  SE_mask=${SE_MASK}  SIMD_mask=${SIMD_MASK}"
if [[ "$BUF_BYTES" != "0" ]]; then
    BUF_LABEL="${BUF_BYTES}B"
else
    BUF_LABEL="(default 256MB)"
fi
echo "   trace ${N_KERNELS} consecutive UA kernel(s), buf=${BUF_LABEL}"
echo "   MODE=${MODE} → ${COUNTER_DESC}"
echo "   out:  ${ATT_DIR}"
echo "============================================================"

rm -rf "$ATT_DIR"; mkdir -p "$ATT_DIR"

ATT_FLAGS=(
    --att
    --att-library-path "$ATT_LIB_DIR"
    --att-target-cu "$CU"
    --att-shader-engine-mask "$SE_MASK"
    --att-simd-select "$SIMD_MASK"
    --att-consecutive-kernels "$N_KERNELS"
)
if [[ "$BUF_BYTES" != "0" ]]; then
    ATT_FLAGS+=( --att-buffer-size "$BUF_BYTES" )
fi
case "$MODE" in
    summary)
        ATT_FLAGS+=( --att-activity 8 ) ;;
    counters)
        # --att-perfcounters takes a single argument; the "list" is space-
        # separated counters inside that one string. These are the four
        # most useful for instruction-mix overlay in the RCV Counters tab.
        ATT_FLAGS+=(
            --att-perfcounter-ctrl 3
            --att-perfcounters "SQ_VALU_MFMA_BUSY_CYCLES SQ_INSTS_VALU SQ_INSTS_MFMA SQ_INST_LEVEL_LDS"
        ) ;;
    none) ;;  # no extra flags
esac

/opt/rocm/bin/rocprofv3 \
    "${ATT_FLAGS[@]}" \
    --kernel-include-regex "$KERNEL_FILTER" \
    -o "att" -d "$ATT_DIR" \
    -- "${RUN_CMD[@]}" > "$RUN_DIR/att.log" 2>&1 || {
        echo "rocprofv3 failed; last 40 lines of $RUN_DIR/att.log:" >&2
        tail -40 "$RUN_DIR/att.log" >&2
        exit 1
    }

# Sanity check: did rocprofv3 actually emit the ui_output_* tree that RCV
# reads? If not, dump the log so the failure mode is visible immediately.
UI_DIRS=( "$ATT_DIR"/ui_output_agent_*_dispatch_* )
if [[ ! -d "${UI_DIRS[0]:-}" ]]; then
    echo "WARN: no ui_output_agent_*_dispatch_* directory produced." >&2
    echo "      Contents of $ATT_DIR:" >&2
    ls -la "$ATT_DIR" >&2
    echo "      Last 40 lines of $RUN_DIR/att.log:" >&2
    tail -40 "$RUN_DIR/att.log" >&2
    exit 1
fi

echo
echo "[ok] traced ${#UI_DIRS[@]} dispatch(es). Per-dispatch UI directories:"
for d in "${UI_DIRS[@]}"; do
    sz=$(du -sh "$d" 2>/dev/null | awk '{print $1}')
    printf '   %-6s %s\n' "$sz" "$(basename "$d")"
done

# Pack just the ui_output_* subtree(s) for download. The raw .att/.out and
# .db files in ATT_DIR are ~70-80 MB and not needed by RCV in JSON mode —
# excluding them keeps the tarball small (typically <5 MB per dispatch).
TARBALL="$RUN_DIR/$TAG.att.tgz"
(cd "$ATT_DIR" && tar czf "$TARBALL" ui_output_agent_*_dispatch_*)
TAR_SZ=$(du -h "$TARBALL" | awk '{print $1}')

echo
echo "[ok] tarball ready: $TARBALL  (${TAR_SZ})"
echo
echo "To open locally:"
echo "  # on your laptop:"
echo "  scp <user>@<server>:$TARBALL ."
echo "  tar xzvf $(basename "$TARBALL")"
echo "  rocprof-compute-viewer ui_output_agent_<N>_dispatch_<N>/"
echo
echo "  (Pre-built RCV binaries: https://github.com/ROCm/rocprof-compute-viewer/releases)"
