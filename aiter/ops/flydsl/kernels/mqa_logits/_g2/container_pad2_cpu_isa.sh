#!/bin/bash
set -euo pipefail
test ! -e /dev/kfd; test ! -e /dev/dri
unset QSA_IMPORT_PACKAGE QSA_PAD_BUCKET QSA_PAD_INT32_GUARD QSA_FP8_CVT QSA_BF16_R16_TDM_COLS QSA_FP8_K_LOAD QSA_PERSIST_COLS QSA_BF16_R64 QSA_BF16_PERSIST QSA_BF16_K_LOAD QSA_G2_ACCESS
G2=aiter/ops/flydsl/kernels/mqa_logits/_g2
KERNEL=/source/aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py
python3 -B - <<'PY' > /results/versions.json
import importlib.metadata as m, json, platform, os, torch
assert m.version('flydsl') == '0.3.3.dev886'
print(json.dumps({'python':platform.python_version(),'flydsl':m.version('flydsl'),'torch':torch.__version__,'torch_hip':torch.version.hip,'no_kfd':not os.path.exists('/dev/kfd'),'no_dri':not os.path.exists('/dev/dri'),'COMPILE_ONLY':os.getenv('COMPILE_ONLY')},indent=2))
PY
sha256sum "$KERNEL" > /results/kernel.sha256
sha256sum "$QSA_GOLDEN_DIR/qsa_golden.py" "$QSA_GOLDEN_DIR/qsa_shapes.py" > /results/golden.sha256
rc=0
run() { local name=$1; shift; set +e; "$@" > /results/$name.log 2>&1; local s=$?; set -e; echo "$name exit=$s" | tee -a /results/gate-status.txt; tail -3 /results/$name.log; [ $s -eq 0 ] || rc=1; }
if [[ $1 == cpu ]]; then
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
 # New PAD tests require real default routing, not raw stock trace settings.
 unset QSA_BF16_K_LOAD QSA_BF16_R64 QSA_BF16_PERSIST
 run pad-bucket python3 -B "$G2/test_pad_bucket_cpu.py" -v
 run pad-bucket-trace python3 -B "$G2/test_pad_bucket_trace_cpu.py" -v
 sha256sum "$G2"/*.py > /results/test-source.sha256
 echo "ALL_CPU_GATES_RC=$rc"; exit $rc
fi
if [[ $1 == native ]]; then
 for phase in base 1; do
  if [[ $phase == base ]]; then
   SRC=/base/aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py; PAD=unset
  else SRC=$KERNEL; PAD=1; fi
  for dtype in bf16 fp8; do
   for cols in 32768 65536; do
    for shape in '64 16' '16 4' '512 1' '512 2' '512 4'; do
     read -r rows requests <<< "$shape"
     width=$((cols / 4))
     label=$dtype-native-r$rows-n$requests-c$cols-w$width
     mkdir -p /results/$phase
     python3 -B /tools/compile_pad2_dispatch_cpu.py --source "$SRC" --results /results/$phase --case "$label" --dtype "$dtype" --shape "$rows" "$requests" "$cols" "$width" --pad "$PAD" > /results/$phase/$label.log 2>&1 || { tail -70 /results/$phase/$label.log; exit 1; }
     tail -1 /results/$phase/$label.log
    done
   done
  done
 done
 echo ALL_NATIVE_BENCHMARK_DISPATCH_COMPILE_PASS
 exit 0
fi
matrix() {
 local pad=$1
 while read -r case dtype rows requests cols width; do
  [[ -n $case ]] || continue
  local dest=/results/$pad
  mkdir -p "$dest"
  python3 -B /tools/compile_pad2_dispatch_cpu.py --source "$KERNEL" --results "$dest" --case "$case" --dtype "$dtype" --shape "$rows" "$requests" "$cols" "$width" --pad "$pad" > "$dest/$case.log" 2>&1 || { tail -70 "$dest/$case.log"; return 1; }
  tail -1 "$dest/$case.log"
 done <<'CASES'
bf16-exact-r16-n1 bf16 16 1 32 8
bf16-exact-r16-n2 bf16 16 2 32 8
bf16-exact-r32-n1 bf16 32 1 512 128
bf16-exact-r32-n2 bf16 64 2 512 128
fp8-exact-r16-n1 fp8 16 1 64 16
fp8-exact-r16-n2 fp8 16 2 64 16
fp8-exact-r32-n1 fp8 32 1 64 16
fp8-exact-r32-n2 fp8 64 2 64 16
bf16-irregular-n1 bf16 37 1 67 17
bf16-irregular-n3 bf16 37 3 67 17
fp8-irregular-n1 fp8 37 1 67 17
fp8-irregular-n3 fp8 37 3 67 17
CASES
}
if [[ $1 == base ]]; then matrix unset; else matrix unset; matrix 0; matrix 1; fi
echo ALL_REAL_DISPATCH_COMPILE_PASS
