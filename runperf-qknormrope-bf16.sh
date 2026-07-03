#!/usr/bin/env bash
# Perf harness for flydsl_qk_norm_rope_quant -- BF16 (no-quant) variant.
# gfx1250, DeepSeek-V4-Flash shape: H=64, D=512, RD=64.
#
# BF16 uses the --no-quant path (output bf16, no fp8 pack / block scales).
#
# Two modes:
#   default        run_perftest sweep -> us / GB/s / %peak table
#   TRACE=1        rocprofv3 kernel-trace -> per-T min/median dispatch time.
#                  Use this for small T (decode, e.g. T=64): the kernel is only
#                  ~6us there, so run_perftest's torch-profiler timing is
#                  dominated by profiler/launch noise. kernel-trace min is the
#                  true hardware dispatch duration.
#
# Usage:
#   ./runperf-qknormrope-bf16.sh                       # sweep, T=64
#   SWEEP_T="64 256 16384" ./runperf-qknormrope-bf16.sh
#   TRACE=1 ./runperf-qknormrope-bf16.sh               # kernel-trace min, T=64
#   TRACE=1 SWEEP_T="64 256 1024" ./runperf-qknormrope-bf16.sh
export HIP_VISIBLE_DEVICES=1
set -euo pipefail

# --- tunables (override via env) --------------------------------------------
TEST="${TEST:-op_tests/test_flydsl_qk_norm_rope_quant.py}"
H="${H:-64}"                      # num Q heads per rank (= num_attention_heads / TP)
D="${D:-512}"                     # head_dim (MVP: 512 only)
RD="${RD:-64}"                    # rope_head_dim
SWEEP_T="${SWEEP_T:-64}"          # T values (default: token=64 decode point)
TRACE="${TRACE:-0}"              # 1 -> kernel-trace min/median instead of sweep
OUTDIR="${OUTDIR:-/tmp/prof_qknorm_bf16}"

COMMON="--H ${H} --D ${D} --RD ${RD} --no-quant"

echo "config: H=${H} D=${D} RD=${RD} dtype=BF16 (no-quant)  T=${SWEEP_T}  TRACE=${TRACE}"

if [[ "$TRACE" == "0" ]]; then
    python "$TEST" -T ${SWEEP_T} $COMMON
    exit 0
fi

# --- kernel-trace mode: one dispatch DB per T, report min/median ------------
for T in $SWEEP_T; do
    d="${OUTDIR}/T${T}"
    rm -rf "$d"
    rocprofv3 --kernel-trace -d "$d" \
        -- python "$TEST" -T ${T} $COMMON >/dev/null 2>&1
    DB=$(ls "$d"/*/*.db 2>/dev/null | head -1)
    if [[ -z "$DB" ]]; then echo "T=${T}: no trace db"; continue; fi
    T_VAL=$T H_VAL=$H D_VAL=$D python3 - "$DB" <<'PY'
import sqlite3, sys, os, statistics
db = sys.argv[1]
T = int(os.environ["T_VAL"]); H = int(os.environ["H_VAL"]); D = int(os.environ["D_VAL"])
c = sqlite3.connect(db)
tabs = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
def tb(p): return [t for t in tabs if t.startswith(p)][0]
ks, kd = tb("rocpd_info_kernel_symbol"), tb("rocpd_kernel_dispatch")
q = (f"SELECT (d.end-d.start)/1000.0 FROM {kd} d JOIN {ks} s ON d.kernel_id=s.id "
     f"WHERE s.display_name LIKE '%H{H}%D{D}%' ORDER BY d.start")
ts = [r[0] for r in c.execute(q)]
if not ts:
    print(f"T={T}: no matching dispatches"); sys.exit()
# bf16 traffic: Q in + KV in + Q out + KV out (all bf16, 2B)
by = T*H*D*2 + T*D*2 + T*H*D*2 + T*D*2
mn = min(ts)
gbps = by/(mn*1e-6)/1e9
print(f"T={T:6d}  n={len(ts):3d}  min={mn:8.2f}us  p50={statistics.median(ts):8.2f}us  "
      f"max={max(ts):8.2f}us  |  min-based {gbps:6.0f} GB/s")
PY
done
