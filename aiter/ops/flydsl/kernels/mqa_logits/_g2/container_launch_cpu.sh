#!/bin/bash
# ROUND-LAUNCH no-device container entrypoint (derived from container_nosync_cpu.sh).
# Modes: cpu (all CPU gates + test_launch_cpu + host-overhead bench) | isa (base vs final ISA identity).
set -euo pipefail
test ! -e /dev/kfd; test ! -e /dev/dri
unset QSA_IMPORT_PACKAGE QSA_PAD_BUCKET QSA_PAD_INT32_GUARD QSA_STRICT_I32 QSA_FP8_CVT QSA_BF16_R16_TDM_COLS QSA_FP8_K_LOAD QSA_PERSIST_COLS QSA_BF16_R64 QSA_BF16_PERSIST QSA_BF16_K_LOAD QSA_G2_ACCESS
G2=aiter/ops/flydsl/kernels/mqa_logits/_g2
KERNEL=/source/aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py
BASEK=/base/aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py
sha256sum "$KERNEL" "$BASEK" > /results/kernel.sha256
rc=0
run() { local name=$1; shift; set +e; "$@" > /results/$name.log 2>&1; local s=$?; set -e; echo "$name exit=$s" | tee -a /results/gate-status.txt; tail -3 /results/$name.log; [ $s -eq 0 ] || rc=1; }
if [[ $1 == cpu ]]; then
 : > /results/gate-status.txt
 export QSA_BF16_K_LOAD=raw QSA_BF16_R64=0 QSA_BF16_PERSIST=0
 run contract python3 -B "$G2/test_contract_cpu.py"
 run address python3 -B "$G2/audit_addressing_cpu.py" --json /results/address.json
 run q-lds python3 -B "$G2/test_q_lds_cpu.py"
 QSA_TRACE_ROWS_PER_BLOCK=16 run trace16 python3 -B "$G2/trace_check_cpu.py"
 QSA_TRACE_ROWS_PER_BLOCK=32 run trace32 python3 -B "$G2/trace_check_cpu.py"
 run k-tdm python3 -B "$G2/test_k_tdm_cpu.py" -v
 run routing python3 -B "$G2/test_bf16_routing_cpu.py" -v
 run fp8-k-tdm python3 -B "$G2/test_fp8_k_tdm_cpu.py" -v
 run bm64 python3 -B "$G2/test_bf16_bm64_cpu.py" -v
 run persist python3 -B "$G2/test_persist_cpu.py" -v
 run fp8-cvt python3 -B "$G2/test_fp8_cvt_cpu.py" -v
 unset QSA_BF16_K_LOAD QSA_BF16_R64 QSA_BF16_PERSIST
 run pad-bucket python3 -B "$G2/test_pad_bucket_cpu.py" -v
 run pad-bucket-trace python3 -B "$G2/test_pad_bucket_trace_cpu.py" -v
 run nosync-i32 python3 -B "$G2/test_nosync_i32_cpu.py" -v
 run launch python3 -B "$G2/test_launch_cpu.py" -v
 run bench-final python3 -B "$G2/launch_overhead_cpu.py" --json /results/bench-final.json
 sha256sum "$G2"/*.py > /results/test-source.sha256
 echo "ALL_CPU_GATES_RC=$rc"; exit $rc
fi
if [[ $1 == isa ]]; then
 one() {
  local phase=$1 dtype=$2 rows=$3 requests=$4 cols=$5 meta=$6 src=$KERNEL
  [[ $phase == base ]] && src=$BASEK
  local label=$dtype-r$rows-n$requests-c$cols-$meta
  mkdir -p /results/$phase
  python3 -B "/source/$G2/compile_nosync_dispatch_cpu.py" --source "$src" --results /results/$phase --case "$label" --dtype "$dtype" --shape "$rows" "$requests" "$cols" $((cols / 4)) --pad unset --meta "$meta" > /results/$phase/$label.log 2>&1 && tail -1 /results/$phase/$label.log || { echo "FAIL $phase $label"; tail -30 /results/$phase/$label.log; return 1; }
 }
 export -f one; export KERNEL BASEK G2
 for dtype in bf16 fp8; do
  for shape in '64 16 32768' '16 16 32768' '512 2 32768'; do
   for pm in 'base i32' 'base qp64' 'final i32' 'final qp64'; do
    read -r phase meta <<< "$pm"; echo "$phase $dtype $shape $meta"
   done
  done
 done | xargs -P 8 -L 1 bash -c 'one "$@"' _ | tee /results/isa-run.txt
 if grep -q '^FAIL' /results/isa-run.txt; then exit 1; fi
 python3 - <<'PY'
import glob, json
rows = []
for p in sorted(glob.glob('/results/base/*/dispatch.json')):
    b = json.load(open(p)); f = json.load(open(p.replace('/base/', '/final/')))
    same = b['isa']['sha256'] == f['isa']['sha256'] and b['kernel_name'] == f['kernel_name'] and b['grid'] == f['grid']
    rows.append((b['case'], b['kernel_name'], b['isa']['sha256'][:16], f['isa']['sha256'][:16], same))
    print('ISA', 'IDENTICAL' if same else 'DIFFERENT', *rows[-1][:4])
assert rows and all(r[4] for r in rows), 'ISA mismatch'
print('ALL_LAUNCH_ISA_IDENTICAL', len(rows))
PY
fi
