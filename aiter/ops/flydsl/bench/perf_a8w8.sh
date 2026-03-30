#!/bin/bash
#
# Benchmark FlyDSL A8W8 FP8 GEMM kernel on the FFM simulator.
#
# Usage:
#   ./perf_a8w8.sh [--stats-only] M N K [-- extra launcher args...]
#
# Examples:
#   ./perf_a8w8.sh 32 7168 4096
#   ./perf_a8w8.sh --stats-only 64 5120 2880
#   ./perf_a8w8.sh 64 5120 2880 -- --tile-m 64 --num-buffers 3
#
# With auto-config (default), the script automatically picks the best
# tile config from the JSON configs for the given M/N/K shape.
#
# Environment:
#   TRITON_GFX1250_MODEL_PATH  — path to the FFM/rocdtif installation
#                                (default: /root/rocdtif-7.12-am+ffmlite-mi400.8575807.264-rel-20260305)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AITER_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

TRITON_GFX1250_MODEL_PATH="${TRITON_GFX1250_MODEL_PATH:-/root/rocdtif-7.12-am+ffmlite-mi400.8575807.264-rel-20260305}"
DRAW_LOG="${DRAW_LOG:-./draw.log}"

# --- Parse options ---
SKIP_TRACES=0
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --stats-only) SKIP_TRACES=1; shift ;;
    --) shift; break ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 [--stats-only] M N K [-- extra launcher args...]"
  exit 1
fi
M="$1"; N="$2"; K="$3"; shift 3

# Remaining args are passed to the launcher
EXTRA_ARGS=("$@")

TAG="${M}_${N}_${K}"
LAUNCHER="$SCRIPT_DIR/gemm_a8w8_launch.py"

echo "=== FlyDSL A8W8 GEMM Benchmark ==="
echo "M=$M  N=$N  K=$K"
echo "Extra args: ${EXTRA_ARGS[*]:-<none>}"
echo ""

# --- 1) Capture: run the kernel under roccap ---
echo "=== Step 1: Capture kernel dispatch ==="
source "$TRITON_GFX1250_MODEL_PATH/ffmlite_env.sh"

# roccap capture segfaults on exit (expected FFM behavior) — ignore exit code
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
PYTHONPATH="$AITER_ROOT" \
"$TRITON_GFX1250_MODEL_PATH/tools/roccap/bin/roccap" capture \
  --loglevel error \
  --disp "kernel_gemm_a8w8/0" \
  --file gemm_a8w8.cap \
  python3 "$LAUNCHER" "$M" "$N" "$K" "${EXTRA_ARGS[@]}" || true

# --- 2) Play: replay on the model ---
echo ""
echo "=== Step 2: Replay on FFM model ==="
source "$TRITON_GFX1250_MODEL_PATH/am_env.sh"
export DtifFbBaseLocation=0x200000000

# Pick the largest .cap file
CAP_FILE=""
largest_size=0
for f in gemm_a8w8*.cap; do
  [[ -f "$f" ]] || continue
  fsize=$(stat --format="%s" "$f")
  if (( fsize > largest_size )); then
    largest_size=$fsize
    CAP_FILE="$f"
  fi
done

if [[ -z "$CAP_FILE" ]]; then
  echo "Error: no gemm_a8w8*.cap files found"
  exit 1
fi
echo "Using cap file: $CAP_FILE ($largest_size bytes)"

"$TRITON_GFX1250_MODEL_PATH/tools/roccap/bin/roccap" play \
  -r "0x200000000-0xF00000000" "./$CAP_FILE"

# --- 3) Parse draw.log for start/end timestamps (picoseconds) ---
echo ""
echo "=== Step 3: Parse timing ==="
if [[ ! -f "$DRAW_LOG" ]]; then
  echo "Error: draw.log not found at $DRAW_LOG"
  exit 1
fi

start_ps=""
end_ps=""
while IFS= read -r line; do
  if [[ "$line" =~ Time:([0-9]+)\ DrawId: ]]; then
    start_ps="${BASH_REMATCH[1]}"
  elif [[ "$line" =~ Time:([0-9]+)\ DrawDone: ]]; then
    end_ps="${BASH_REMATCH[1]}"
  fi
done < "$DRAW_LOG"

if [[ -z "$start_ps" || -z "$end_ps" ]]; then
  echo "Error: could not parse start/end time from $DRAW_LOG"
  exit 1
fi

time_taken_ps=$(( end_ps - start_ps ))
time_us=$(echo "scale=2; $time_taken_ps / 1000000" | bc)

# --- 4) Compute bandwidth and TFLOPS ---
total_bytes=$(( M*K + N*K + M*N*2 ))
bw_tb_s=$(echo "scale=4; $total_bytes / $time_taken_ps" | bc)

total_flops=$(( 2 * M * N * K ))
tflops=$(echo "scale=2; $total_flops / $time_taken_ps" | bc)

echo ""
echo "=== Results ==="
echo "Time:      ${time_us} us"
echo "Bandwidth: ${bw_tb_s} TB/s"
echo "TFLOPS:    ${tflops}"
echo ""

# --- Write stats file ---
STATS_FILE="stats_flydsl_a8w8_${TAG}.txt"
cat > "$STATS_FILE" <<EOF
FlyDSL A8W8 GEMM Benchmark
M=$M  N=$N  K=$K
Extra args: ${EXTRA_ARGS[*]:-<none>}
Time (us): $time_us
Time (ps): $time_taken_ps
Total bytes: $total_bytes
Bandwidth: $bw_tb_s TB/s
Total FLOPs: $total_flops
TFLOPS: $tflops
EOF
echo "Stats written to $STATS_FILE"

if [[ "$SKIP_TRACES" -eq 1 ]]; then
  exit 0
fi

# --- 5) Collect WGP00 instruction trace -> Perfetto ---
echo ""
echo "=== Collecting WGP00 instruction trace ==="
if [[ -f "xcc0se0sa0_itrace_emu.mon" ]]; then
  grep -A1 "WGP00" xcc0se0sa0_itrace_emu.mon > wgp0.txt
  if [[ -f "$SCRIPT_DIR/gen_perfetto.py" ]]; then
    python3 "$SCRIPT_DIR/gen_perfetto.py" wgp0.txt "itrace_flydsl_a8w8_${TAG}.json"
    echo "Perfetto trace written to itrace_flydsl_a8w8_${TAG}.json"
  else
    echo "Warning: gen_perfetto.py not found, skipping Perfetto trace generation"
  fi
else
  echo "Warning: xcc0se0sa0_itrace_emu.mon not found, skipping WGP00 trace"
fi

# --- 6) SP3 disassembly + amtool ---
echo ""
echo "=== Collecting SP3 disassembly trace ==="
"$TRITON_GFX1250_MODEL_PATH/tools/roccap/bin/roccap" extract --sp3 0- "./$CAP_FILE"

ISA_BIN=$(ls roc-dump-*-isa-data.bin 2>/dev/null | head -n1)
if [[ -z "$ISA_BIN" ]]; then
  echo "Error: no roc-dump-*-isa-data.bin file found after extract"
  exit 1
fi
echo "Found ISA binary: $ISA_BIN"

"$TRITON_GFX1250_MODEL_PATH/ffm-lite/sp3disasm" "./$ISA_BIN" gemm_a8w8.sp3
echo "SP3 disassembly written to gemm_a8w8.sp3"

"$TRITON_GFX1250_MODEL_PATH/tools/rcv/amtool" "rcv_flydsl_a8w8_${TAG}/" *.mon gemm_a8w8.sp3
echo "amtool output written to rcv_flydsl_a8w8_${TAG}/"

# --- 7) Pack traces ---
echo ""
echo "=== Packing traces ==="
PACK_LIST=("rcv_flydsl_a8w8_${TAG}/" "$STATS_FILE")
if [[ -f "itrace_flydsl_a8w8_${TAG}.json" ]]; then
  PACK_LIST+=("itrace_flydsl_a8w8_${TAG}.json")
fi

tar czf "traces_flydsl_a8w8_${TAG}.tar.gz" "${PACK_LIST[@]}"
echo "Traces packed into traces_flydsl_a8w8_${TAG}.tar.gz"
