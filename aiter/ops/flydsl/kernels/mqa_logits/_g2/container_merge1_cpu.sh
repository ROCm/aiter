#!/bin/bash
# ROUND-MERGE1 no-device container entrypoint (derived from container_native64_cpu.sh).
# Modes: cpu-default | cpu-optout (all CPU gates; optout = QSA_DRAFT_PATH=0 QSA_R16_NOBAR=0)
#        isa (NATIVE64 vs merge opt-out, merge default, DRAFT-branch opt-in vs merge default).
# Mounts: /source = merge, /base = NATIVE64 60b9aade, /draft = DRAFT f6a54cc9.
set -euo pipefail
test ! -e /dev/kfd; test ! -e /dev/dri
unset QSA_IMPORT_PACKAGE QSA_PAD_BUCKET QSA_PAD_INT32_GUARD QSA_STRICT_I32 QSA_FP8_CVT QSA_BF16_R16_TDM_COLS QSA_FP8_K_LOAD QSA_PERSIST_COLS QSA_BF16_R64 QSA_BF16_PERSIST QSA_BF16_K_LOAD QSA_G2_ACCESS QSA_BF16_PATH QSA_FP8_PATH QSA_DRAFT_PATH QSA_R16_NOBAR QSA_DRAFT_K_LOAD QSA_DRAFT_COLS
G2=aiter/ops/flydsl/kernels/mqa_logits/_g2
KREL=aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py
KERNEL=/source/$KREL
BASEK=/base/$KREL
DRAFTK=/draft/$KREL
MODE=$1
OUT=/results/$MODE
mkdir -p "$OUT"
python3 -B - <<'PY' > "$OUT/versions.json"
import importlib.metadata as m, json, platform, os, torch
print(json.dumps({'python':platform.python_version(),'flydsl':m.version('flydsl'),'torch':torch.__version__,'torch_hip':torch.version.hip,'no_kfd':not os.path.exists('/dev/kfd'),'no_dri':not os.path.exists('/dev/dri'),'COMPILE_ONLY':os.getenv('COMPILE_ONLY')},indent=2))
PY
sha256sum "$KERNEL" "$BASEK" "$DRAFTK" > "$OUT/kernel.sha256"
rc=0
run() { local name=$1; shift; set +e; "$@" > "$OUT/$name.log" 2>&1; local s=$?; set -e; echo "$name exit=$s" | tee -a "$OUT/gate-status.txt"; tail -3 "$OUT/$name.log"; [ $s -eq 0 ] || rc=1; }
if [[ $MODE == cpu-* ]]; then
 : > "$OUT/gate-status.txt"
 if [[ $MODE == cpu-optout ]]; then export QSA_DRAFT_PATH=0 QSA_R16_NOBAR=0; fi
 env | grep '^QSA_' | sort > "$OUT/env.txt" || true
 export QSA_BF16_K_LOAD=raw QSA_BF16_R64=0 QSA_BF16_PERSIST=0
 run contract python3 -B "$G2/test_contract_cpu.py"
 run address python3 -B "$G2/audit_addressing_cpu.py" --json "$OUT/address.json"
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
 run draft python3 -B "$G2/test_draft_cpu.py" -v
 run pad-bucket python3 -B "$G2/test_pad_bucket_cpu.py" -v
 run pad-bucket-trace python3 -B "$G2/test_pad_bucket_trace_cpu.py" -v
 run nosync-i32 python3 -B "$G2/test_nosync_i32_cpu.py" -v
 run launch python3 -B "$G2/test_launch_cpu.py" -v
 sha256sum "$G2"/*.py > "$OUT/test-source.sha256"
 echo "ALL_CPU_GATES_RC=$rc"; exit $rc
fi
if [[ $MODE == isa ]]; then
 # spec|dtype|rows,requests,cols|env (comma-separated; no spaces: xargs splits)
 cat > "$OUT/specs.txt" <<'EOF'
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
 cat > "$OUT/draft-specs.txt" <<'EOF'
draft-bf16-tdm|bf16|16,16,4096|
draft-fp8-tdm|fp8|16,16,4096|
draft-bf16-raw|bf16|16,16,4096|QSA_DRAFT_K_LOAD=raw
draft-fp8-raw|fp8|16,16,4096|QSA_DRAFT_K_LOAD=raw
nobar-bf16-r16-tdm|bf16|16,4,4096|
EOF
 # phase: n64 (NATIVE64 source) | optout (merge, QSA_DRAFT_PATH=0 QSA_R16_NOBAR=0)
 #        dflt (merge, default env) | draftbr (DRAFT source, QSA_DRAFT_PATH=1 QSA_R16_NOBAR=1)
 one() {
  local phase=$1 meta=$2 line=$3 src=$KERNEL extra=
  IFS='|' read -r spec dtype shape envs <<< "$line"; shape=${shape//,/ }; envs=${envs//,/ }
  case $phase in
   n64) src=$BASEK ;;
   optout) extra="QSA_DRAFT_PATH=0 QSA_R16_NOBAR=0" ;;
   draftbr) src=$DRAFTK; extra="QSA_DRAFT_PATH=1 QSA_R16_NOBAR=1" ;;
  esac
  local label=$spec-$meta
  mkdir -p "$OUT/$phase"
  env $extra $envs python3 -B "/source/$G2/compile_native64_isa_cpu.py" --source "$src" --results "$OUT/$phase" --case "$label" --dtype "$dtype" --shape $shape --meta "$meta" > "$OUT/$phase/$label.log" 2>&1 && tail -1 "$OUT/$phase/$label.log" || { echo "FAIL $phase $label"; tail -30 "$OUT/$phase/$label.log"; return 1; }
 }
 export -f one; export KERNEL BASEK DRAFTK G2 OUT
 {
  while read -r line; do
   for pm in 'n64 i32' 'n64 qp64' 'optout i32' 'optout qp64' 'dflt i32' 'dflt qp64'; do echo "$pm $line"; done
  done < "$OUT/specs.txt"
  while read -r line; do
   for pm in 'draftbr i32' 'dflt i32' 'dflt qp64'; do echo "$pm $line"; done
  done < "$OUT/draft-specs.txt"
 } | xargs -P "${ISA_JOBS:-16}" -L 1 bash -c 'one "$1" "$2" "$3"' _ | tee "$OUT/isa-run.txt"
 if grep -q '^FAIL' "$OUT/isa-run.txt"; then exit 1; fi
 [[ $(grep -c NATIVE64_DISPATCH_PASS "$OUT/isa-run.txt") == 87 ]] || exit 1
 s=0; python3 -B "/source/$G2/summarize_merge1_isa_cpu.py" "$OUT" > "$OUT/isa-summary.txt" || s=$?
 cat "$OUT/isa-summary.txt"; exit $s
fi
echo "unknown mode $MODE"; exit 2
