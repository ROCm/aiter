#!/bin/bash
#
# Benchmark FlyDSL pa_decode_main + pa_decode_reduce kernels on the FFM
# simulator. Reports per-kernel and combined time + bandwidth.
#
# - Main bandwidth uses (Q + KV + tmp_out + LSE + small) / main_time.
#   "KV useful" BW is the canonical pa_decode metric (compares to vLLM /
#   FlashDecoding numbers).
# - Reduce bandwidth uses (tmp_out read + LSE read + final out write) /
#   reduce_time. Usually tiny — reduce is launch/latency-bound.
# - Combined uses the canonical "effective KV bandwidth" view: KV bytes /
#   (main + reduce time).
#
# Usage:
#   ./perf_pa_decode.sh [--stats-only] [-- launcher args...]
#
# All launcher params have defaults baked into pa_decode_launch.py; pass any
# flags after `--` to override them. Common flags: --head-size --kv-block-size
# --query-group-size --num-kv-heads --num-seqs --seq-len
# --kv-compute-block-size --partition-size --dtype --random-seq-lens --seed
#
# Examples:
#   ./perf_pa_decode.sh
#   ./perf_pa_decode.sh --stats-only
#   ./perf_pa_decode.sh -- --num-kv-heads 16 --seq-len 4096
#
# Both kernels are captured via two --disp flags. draw.log gets one
# DrawId/DrawDone pair per dispatch; we extract per-kernel times and sum.
#
# Environment:
#   TRITON_GFX1250_MODEL_PATH  — path to the FFM/rocdtif installation
#                                (default: /root/rocdtif-7.12-am+ffmlite-mi400.8575807.264-rel-20260305)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AITER_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

TRITON_GFX1250_MODEL_PATH="${TRITON_GFX1250_MODEL_PATH:-/root/rocdtif-7.12-am+ffmlite-mi400.8575807.264-rel-20260305}"
DRAW_LOG="${DRAW_LOG:-./draw.log}"
HBM_PEAK_TBPS="2.8"

# --- Parse options ---
SKIP_TRACES=0
while [[ $# -gt 0 && "${1:-}" == --* ]]; do
  case "$1" in
    --stats-only) SKIP_TRACES=1; shift ;;
    --) shift; break ;;
    *) break ;;  # let unknown flags fall through to launcher
  esac
done

EXTRA_ARGS=("$@")

LAUNCHER="$SCRIPT_DIR/pa_decode_launch.py"
FLYDSL_DEBUG_DIR="$HOME/.flydsl/debug"

echo "=== FlyDSL pa_decode (main+reduce) Benchmark ==="
echo "Launcher args: ${EXTRA_ARGS[*]:-<defaults>}"
echo ""

# --- 0) Clear stale debug dumps ---
if [[ -d "$FLYDSL_DEBUG_DIR" ]]; then
  read -rp "Clear existing debug dumps at $FLYDSL_DEBUG_DIR? [y/N] " answer
  if [[ "$answer" =~ ^[Yy]$ ]]; then
    rm -rf "$FLYDSL_DEBUG_DIR"
    echo "Debug dumps cleared."
  else
    echo "Debug dumps kept."
  fi
fi

# --- 1) Capture: run the kernels under roccap ---
echo ""
echo "=== Step 1: Capture kernel dispatches ==="
source "$TRITON_GFX1250_MODEL_PATH/ffmlite_env.sh"

LAUNCHER_LOG=$(mktemp)
trap 'rm -f "$LAUNCHER_LOG"' EXIT

# Two --disp flags are needed: roccap's "(main|reduce)" regex form silently
# drops one of the alternatives (see commit history). The docs say `--disp
# <capture_exp>...` is repeatable; that's the form we use.
# roccap capture segfaults on exit (expected FFM behavior) — ignore exit code.
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
FLYDSL_DUMP_IR=1 \
FLYDSL_DEBUG_DUMP_ASM=1 \
PYTHONPATH="$AITER_ROOT" \
"$TRITON_GFX1250_MODEL_PATH/tools/roccap/bin/roccap" capture \
  --loglevel error \
  --disp "kernel_pa_decode_main/0" \
  --disp "kernel_pa_decode_reduce/0" \
  --file pa_decode.cap \
  python3 "$LAUNCHER" "${EXTRA_ARGS[@]}" 2>&1 | tee "$LAUNCHER_LOG" || true

echo "MLIR/ASM dumps written to $FLYDSL_DEBUG_DIR"

# --- Extract metrics emitted by the launcher ---
get_metric() {
  grep -E "^METRIC $1=" "$LAUNCHER_LOG" | tail -1 | sed -E "s/^METRIC $1=//"
}

TAG=$(get_metric tag)
head_size=$(get_metric head_size)
kv_block_size=$(get_metric kv_block_size)
query_group_size=$(get_metric query_group_size)
num_kv_heads=$(get_metric num_kv_heads)
num_seqs=$(get_metric num_seqs)
seq_len=$(get_metric seq_len)
kv_compute_block_size=$(get_metric kv_compute_block_size)
partition_size=$(get_metric partition_size)
dtype=$(get_metric dtype)
num_q_heads=$(get_metric num_q_heads)
num_partitions=$(get_metric num_partitions)
sum_seq_lens=$(get_metric sum_seq_lens)
total_live_parts=$(get_metric total_live_parts)
bytes_q=$(get_metric bytes_q)
bytes_kv_useful=$(get_metric bytes_kv_useful)
bytes_kv_executed=$(get_metric bytes_kv_executed)
bytes_tmp_out=$(get_metric bytes_tmp_out)
bytes_lse=$(get_metric bytes_lse)
bytes_seq_lens=$(get_metric bytes_seq_lens)
bytes_out_final=$(get_metric bytes_out_final)
bytes_main_useful=$(get_metric bytes_main_useful)
bytes_main_executed=$(get_metric bytes_main_executed)
bytes_reduce_in=$(get_metric bytes_reduce_in)
bytes_reduce_total=$(get_metric bytes_reduce_total)
bytes_combined_useful=$(get_metric bytes_combined_useful)
bytes_combined_executed=$(get_metric bytes_combined_executed)
total_flops=$(get_metric total_flops)

if [[ -z "$TAG" ]]; then
  echo "Error: could not parse METRIC lines from launcher output"
  cat "$LAUNCHER_LOG"
  exit 1
fi

# --- 2) Play: replay on the model ---
echo ""
echo "=== Step 2: Replay on FFM model ==="
# am_env.sh references $LD_PRELOAD which ffmlite_env.sh just unset; relax -u for the source.
set +u
source "$TRITON_GFX1250_MODEL_PATH/am_env.sh"
set -u
export DtifFbBaseLocation=0x200000000

# Pick the largest .cap file
CAP_FILE=""
largest_size=0
for f in pa_decode*.cap; do
  [[ -f "$f" ]] || continue
  fsize=$(stat --format="%s" "$f")
  if (( fsize > largest_size )); then
    largest_size=$fsize
    CAP_FILE="$f"
  fi
done

if [[ -z "$CAP_FILE" ]]; then
  echo "Error: no pa_decode*.cap files found"
  exit 1
fi
echo "Using cap file: $CAP_FILE ($largest_size bytes)"

"$TRITON_GFX1250_MODEL_PATH/tools/roccap/bin/roccap" play \
  -r "0x200000000-0xF00000000" "./$CAP_FILE"

# --- 3) Parse draw.log: extract per-dispatch durations ---
echo ""
echo "=== Step 3: Parse timing ==="
if [[ ! -f "$DRAW_LOG" ]]; then
  echo "Error: draw.log not found at $DRAW_LOG"
  exit 1
fi

# Launcher dispatches main first, reduce second — we tag durations by index.
total_ps=0
n_dispatches=0
main_ps=0
reduce_ps=0
cur_start=""
while IFS= read -r line; do
  if [[ "$line" =~ Time:([0-9]+)\ DrawId: ]]; then
    cur_start="${BASH_REMATCH[1]}"
  elif [[ "$line" =~ Time:([0-9]+)\ DrawDone: ]]; then
    if [[ -n "$cur_start" ]]; then
      dur=$(( ${BASH_REMATCH[1]} - cur_start ))
      total_ps=$(( total_ps + dur ))
      if (( n_dispatches == 0 )); then
        main_ps=$dur
      elif (( n_dispatches == 1 )); then
        reduce_ps=$dur
      fi
      n_dispatches=$(( n_dispatches + 1 ))
      cur_start=""
    fi
  fi
done < "$DRAW_LOG"

if (( n_dispatches == 0 )); then
  echo "Error: no DrawId/DrawDone pairs found in $DRAW_LOG"
  exit 1
fi
if (( n_dispatches != 2 )); then
  echo "Warning: expected 2 dispatches (main+reduce), got $n_dispatches"
fi

main_us=$(echo "scale=2; $main_ps / 1000000" | bc)
reduce_us=$(echo "scale=2; $reduce_ps / 1000000" | bc)
total_us=$(echo "scale=2; $total_ps / 1000000" | bc)

# --- 4) Compute bandwidth and TFLOPS (bytes / picoseconds = TB/s) ---
# tb() and pct() guard against $time being zero or empty.
tb() {
  local bytes="$1" ps="$2"
  if [[ -z "$ps" || "$ps" == "0" ]]; then echo "0"; return; fi
  echo "scale=4; $bytes / $ps" | bc
}
pct() {
  local tbps="$1"
  echo "scale=2; $tbps * 100 / $HBM_PEAK_TBPS" | bc
}

# Per-kernel BWs
bw_main_useful=$(tb "$bytes_main_useful" "$main_ps")
bw_main_executed=$(tb "$bytes_main_executed" "$main_ps")
bw_main_kv_useful=$(tb "$bytes_kv_useful" "$main_ps")
bw_main_kv_executed=$(tb "$bytes_kv_executed" "$main_ps")
util_main_useful=$(pct "$bw_main_useful")
util_main_executed=$(pct "$bw_main_executed")
util_main_kv_useful=$(pct "$bw_main_kv_useful")
util_main_kv_executed=$(pct "$bw_main_kv_executed")

bw_reduce_total=$(tb "$bytes_reduce_total" "$reduce_ps")
util_reduce_total=$(pct "$bw_reduce_total")

# Combined BWs
bw_combined_useful=$(tb "$bytes_combined_useful" "$total_ps")
bw_combined_executed=$(tb "$bytes_combined_executed" "$total_ps")
bw_combined_kv_useful=$(tb "$bytes_kv_useful" "$total_ps")
bw_combined_kv_executed=$(tb "$bytes_kv_executed" "$total_ps")
util_combined_useful=$(pct "$bw_combined_useful")
util_combined_executed=$(pct "$bw_combined_executed")
util_combined_kv_useful=$(pct "$bw_combined_kv_useful")
util_combined_kv_executed=$(pct "$bw_combined_kv_executed")

tflops_main=$(echo "scale=2; $total_flops / $main_ps" | bc)
tflops_combined=$(echo "scale=2; $total_flops / $total_ps" | bc)

print_stats() {
  cat <<EOF
FlyDSL pa_decode (main+reduce) Benchmark
config: head=$head_size kvb=$kv_block_size qg=$query_group_size nkv=$num_kv_heads
        nseqs=$num_seqs seq_len=$seq_len kv_compute=$kv_compute_block_size partition=$partition_size
        dtype=$dtype

num_q_heads=$num_q_heads  num_partitions=$num_partitions
sum_seq_lens=$sum_seq_lens  total_live_parts=$total_live_parts
Dispatches captured: $n_dispatches

HBM peak: $HBM_PEAK_TBPS TB/s (AM)

============================= Main kernel =============================
Time (us): $main_us
Time (ps): $main_ps

Bytes (useful):    $bytes_main_useful
Bytes (executed):  $bytes_main_executed
  Q:               $bytes_q
  KV (useful):     $bytes_kv_useful
  KV (executed):   $bytes_kv_executed
  tmp_out (f32):   $bytes_tmp_out
  LSE (f32):       $bytes_lse

Bandwidth (useful total):  $bw_main_useful TB/s   ($util_main_useful% of HBM peak)
Bandwidth (exec total):    $bw_main_executed TB/s   ($util_main_executed% of HBM peak)
Bandwidth (KV useful):     $bw_main_kv_useful TB/s   ($util_main_kv_useful% of HBM peak)
Bandwidth (KV executed):   $bw_main_kv_executed TB/s   ($util_main_kv_executed% of HBM peak)

Total FLOPs: $total_flops
TFLOPS:      $tflops_main

============================ Reduce kernel ============================
Time (us): $reduce_us
Time (ps): $reduce_ps

Bytes (total):     $bytes_reduce_total
  tmp_out (read):  $bytes_tmp_out
  LSE (read):      $bytes_lse
  seq_lens (read): $bytes_seq_lens
  out (write):     $bytes_out_final

Bandwidth (total): $bw_reduce_total TB/s   ($util_reduce_total% of HBM peak)

======================= Combined (main + reduce) ======================
Time (us): $total_us
Time (ps): $total_ps

Bandwidth (useful total):  $bw_combined_useful TB/s   ($util_combined_useful% of HBM peak)
Bandwidth (exec total):    $bw_combined_executed TB/s   ($util_combined_executed% of HBM peak)
Bandwidth (KV useful):     $bw_combined_kv_useful TB/s   ($util_combined_kv_useful% of HBM peak)
Bandwidth (KV executed):   $bw_combined_kv_executed TB/s   ($util_combined_kv_executed% of HBM peak)

Total FLOPs: $total_flops
TFLOPS:      $tflops_combined
EOF
}

echo ""
echo "=== Results ==="
print_stats

STATS_FILE="stats_flydsl_pa_decode_${TAG}.txt"
print_stats > "$STATS_FILE"
echo ""
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
    python3 "$SCRIPT_DIR/gen_perfetto.py" wgp0.txt "itrace_flydsl_pa_decode_${TAG}.json"
    echo "Perfetto trace written to itrace_flydsl_pa_decode_${TAG}.json"
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

"$TRITON_GFX1250_MODEL_PATH/ffm-lite/sp3disasm" "./$ISA_BIN" pa_decode.sp3
echo "SP3 disassembly written to pa_decode.sp3"

"$TRITON_GFX1250_MODEL_PATH/tools/rcv/amtool" "rcv_flydsl_pa_decode_${TAG}/" *.mon pa_decode.sp3
echo "amtool output written to rcv_flydsl_pa_decode_${TAG}/"

# --- 7) Pack traces ---
echo ""
echo "=== Packing traces ==="
PACK_LIST=("rcv_flydsl_pa_decode_${TAG}/" "$STATS_FILE")
if [[ -f "itrace_flydsl_pa_decode_${TAG}.json" ]]; then
  PACK_LIST+=("itrace_flydsl_pa_decode_${TAG}.json")
fi
if [[ -d "$FLYDSL_DEBUG_DIR" ]]; then
  PACK_LIST+=("$FLYDSL_DEBUG_DIR")
  # Also stage a local copy of the debug dumps for easy inspection.
  LOCAL_DEBUG_DIR="flydsl_debug_pa_decode_${TAG}"
  rm -rf "$LOCAL_DEBUG_DIR"
  cp -r "$FLYDSL_DEBUG_DIR" "$LOCAL_DEBUG_DIR"
  echo "Debug dumps copied to ./$LOCAL_DEBUG_DIR/"
fi

tar czf "traces_flydsl_pa_decode_${TAG}.tar.gz" "${PACK_LIST[@]}"
echo "Traces packed into traces_flydsl_pa_decode_${TAG}.tar.gz"
