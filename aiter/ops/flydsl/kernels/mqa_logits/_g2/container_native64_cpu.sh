#!/bin/bash
# ROUND-NATIVE64 no-device container entrypoint (derived from container_nosync_cpu.sh).
# Modes: cpu (all CPU gates + native64) | isa (base vs final ISA, every specialization).
set -euo pipefail
test ! -e /dev/kfd; test ! -e /dev/dri
unset QSA_IMPORT_PACKAGE QSA_PAD_BUCKET QSA_PAD_INT32_GUARD QSA_STRICT_I32 QSA_FP8_CVT QSA_BF16_R16_TDM_COLS QSA_FP8_K_LOAD QSA_PERSIST_COLS QSA_BF16_R64 QSA_BF16_PERSIST QSA_BF16_K_LOAD QSA_G2_ACCESS QSA_BF16_PATH QSA_FP8_PATH
G2=aiter/ops/flydsl/kernels/mqa_logits/_g2
KERNEL=/source/aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py
BASEK=/base/aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py
python3 -B - <<'PY' > /results/versions.json
import importlib.metadata as m, json, platform, os, torch
print(json.dumps({'python':platform.python_version(),'flydsl':m.version('flydsl'),'torch':torch.__version__,'torch_hip':torch.version.hip,'no_kfd':not os.path.exists('/dev/kfd'),'no_dri':not os.path.exists('/dev/dri'),'COMPILE_ONLY':os.getenv('COMPILE_ONLY')},indent=2))
PY
sha256sum "$KERNEL" "$BASEK" > /results/kernel.sha256
sha256sum "$QSA_GOLDEN_DIR/qsa_golden.py" "$QSA_GOLDEN_DIR/qsa_shapes.py" > /results/golden.sha256
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
 run native64 python3 -B "$G2/test_native64_cpu.py" -v
 unset QSA_BF16_K_LOAD QSA_BF16_R64 QSA_BF16_PERSIST
 run pad-bucket python3 -B "$G2/test_pad_bucket_cpu.py" -v
 run pad-bucket-trace python3 -B "$G2/test_pad_bucket_trace_cpu.py" -v
 run nosync-i32 python3 -B "$G2/test_nosync_i32_cpu.py" -v
 run launch python3 -B "$G2/test_launch_cpu.py" -v
 sha256sum "$G2"/*.py > /results/test-source.sha256
 echo "ALL_CPU_GATES_RC=$rc"; exit $rc
fi
if [[ $1 == isa ]]; then
 # spec|dtype|rows,requests,cols|env (comma-separated; no spaces: xargs splits)
 cat > /results/specs.txt <<'EOF'
bf16-r16-tdm|bf16|16,4,4096|
bf16-r16-raw|bf16|16,4,4096|QSA_BF16_K_LOAD=raw
bf16-r32-tdm|bf16|512,1,4096|QSA_BF16_PERSIST=0
bf16-r32-persist8|bf16|512,1,4096|
bf16-r32-raw|bf16|512,1,4096|QSA_BF16_K_LOAD=raw
bf16-r64-tdm|bf16|512,1,4096|QSA_BF16_R64=1
fp8-r16-tdm|fp8|16,4,4096|
fp8-r16-raw|fp8|16,4,4096|QSA_FP8_K_LOAD=raw
fp8-r32-tdm|fp8|512,1,4096|
fp8-r32-raw|fp8|512,1,4096|QSA_FP8_K_LOAD=raw
bf16-scalar|bf16|16,4,4096|QSA_BF16_PATH=scalar
fp8-scalar|fp8|16,4,4096|QSA_FP8_PATH=scalar
EOF
 one() {
  local phase=$1 meta=$2 line=$3 src=$KERNEL
  IFS='|' read -r spec dtype shape envs <<< "$line"; shape=${shape//,/ }; envs=${envs//,/ }
  [[ $phase == base ]] && src=$BASEK
  local label=$spec-$meta
  mkdir -p /results/$phase
  env $envs python3 -B "/source/$G2/compile_native64_isa_cpu.py" --source "$src" --results /results/$phase --case "$label" --dtype "$dtype" --shape $shape --meta "$meta" > /results/$phase/$label.log 2>&1 && tail -1 /results/$phase/$label.log || { echo "FAIL $phase $label"; tail -30 /results/$phase/$label.log; return 1; }
 }
 export -f one; export KERNEL BASEK G2
 while read -r line; do
  for pm in 'base i32' 'final i32' 'final qp64'; do echo "$pm $line"; done
 done < /results/specs.txt | xargs -P 12 -L 1 bash -c 'one "$1" "$2" "$3"' _ | tee /results/isa-run.txt
 if grep -q '^FAIL' /results/isa-run.txt; then exit 1; fi
 [[ $(grep -c NATIVE64_DISPATCH_PASS /results/isa-run.txt) == 36 ]] || exit 1
 python3 -B "/source/$G2/summarize_native64_isa_cpu.py" /results > /results/isa-summary.txt
 cat /results/isa-summary.txt
fi
