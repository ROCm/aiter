#!/bin/bash
# ROUND-DECG/DECG2 no-device container entrypoint (derived from container_merge1_cpu.sh).
# Modes: cpu-default (17 CPU gates + decg gate, default env = QSA_DECODE_GROUP unset = 'all' since DECG2)
#        cpu-off     (same 18 gates with QSA_DECODE_GROUP=0 exported = previous c45b24e0 routing)
#        isa (DECG2: base c45b24e0 == final QSA_DECODE_GROUP=0; final default == prev bbe25c33
#             QSA_DECODE_GROUP=all; bf16 COLS=64 == prev; fp8 COLS=64 -> w32 kernel).
# Mounts: /source = final, /base = champion c45b24e0, /prev = opt-in DECG bbe25c33.
set -euo pipefail
test ! -e /dev/kfd; test ! -e /dev/dri
unset QSA_IMPORT_PACKAGE QSA_PAD_BUCKET QSA_PAD_INT32_GUARD QSA_STRICT_I32 QSA_FP8_CVT QSA_BF16_R16_TDM_COLS QSA_FP8_K_LOAD QSA_PERSIST_COLS QSA_BF16_R64 QSA_BF16_PERSIST QSA_BF16_K_LOAD QSA_G2_ACCESS QSA_BF16_PATH QSA_FP8_PATH QSA_DRAFT_PATH QSA_R16_NOBAR QSA_DRAFT_K_LOAD QSA_DRAFT_COLS QSA_DECODE_GROUP QSA_DECODE_GROUP_COLS
G2=aiter/ops/flydsl/kernels/mqa_logits/_g2
KREL=aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py
KERNEL=/source/$KREL
BASEK=/base/$KREL
MODE=$1
OUT=/results/$MODE
mkdir -p "$OUT"
python3 -B - <<'PY' > "$OUT/versions.json"
import importlib.metadata as m, json, platform, os, torch
print(json.dumps({'python':platform.python_version(),'flydsl':m.version('flydsl'),'torch':torch.__version__,'torch_hip':torch.version.hip,'no_kfd':not os.path.exists('/dev/kfd'),'no_dri':not os.path.exists('/dev/dri'),'COMPILE_ONLY':os.getenv('COMPILE_ONLY')},indent=2))
PY
sha256sum "$KERNEL" "$BASEK" > "$OUT/kernel.sha256"
rc=0
run() { local name=$1; shift; set +e; "$@" > "$OUT/$name.log" 2>&1; local s=$?; set -e; echo "$name exit=$s" | tee -a "$OUT/gate-status.txt"; tail -3 "$OUT/$name.log"; [ $s -eq 0 ] || rc=1; }
if [[ $MODE == cpu-* ]]; then
 : > "$OUT/gate-status.txt"
 if [[ $MODE == cpu-off ]]; then export QSA_DECODE_GROUP=0; fi
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
 run decg python3 -B "$G2/test_decg_cpu.py" -v
 sha256sum "$G2"/*.py > "$OUT/test-source.sha256"
 echo "ALL_CPU_GATES_RC=$rc"; exit $rc
fi
if [[ $MODE == isa ]]; then
 PREVK=/prev/$KREL
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
draft-bf16-tdm|bf16|16,16,4096|
draft-fp8-tdm|fp8|16,16,4096|
draft-bf16-raw|bf16|16,16,4096|QSA_DRAFT_K_LOAD=raw
draft-fp8-raw|fp8|16,16,4096|QSA_DRAFT_K_LOAD=raw
decode-bf16-64x16|bf16|64,16,4096|
decode-fp8-64x16|fp8|64,16,4096|
EOF
 # extra (phase spec): COLS=64 variants
 cat > "$OUT/extra-specs.txt" <<'EOF'
dflt|cols64-bf16|bf16|64,16,4096|QSA_DECODE_GROUP_COLS=64
prev|cols64-bf16|bf16|64,16,4096|QSA_DECODE_GROUP=all,QSA_DECODE_GROUP_COLS=64
dflt|cols64-fp8|fp8|64,16,4096|QSA_DECODE_GROUP_COLS=64
EOF
 # phase: base (c45b24e0, default env) | off (final, QSA_DECODE_GROUP=0)
 #        prev (bbe25c33, QSA_DECODE_GROUP=all) | dflt (final, default env)
 one() {
  local phase=$1 meta=$2 line=$3 src=$KERNEL penv=
  IFS='|' read -r spec dtype shape envs <<< "$line"; shape=${shape//,/ }; envs=${envs//,/ }
  case $phase in base) src=$BASEK ;; prev) src=$PREVK; penv=QSA_DECODE_GROUP=all ;; off) penv=QSA_DECODE_GROUP=0 ;; esac
  local label=$spec-$meta
  mkdir -p "$OUT/$phase"
  env $penv $envs python3 -B "/source/$G2/compile_native64_isa_cpu.py" --source "$src" --results "$OUT/$phase" --case "$label" --dtype "$dtype" --shape $shape --meta "$meta" > "$OUT/$phase/$label.log" 2>&1 && tail -1 "$OUT/$phase/$label.log" || { echo "FAIL $phase $label"; tail -30 "$OUT/$phase/$label.log"; return 1; }
 }
 export -f one; export KERNEL BASEK PREVK G2 OUT
 {
  while read -r line; do
   for ph in base off prev dflt; do for m in i32 qp64; do echo "$ph $m $line"; done; done
  done < "$OUT/specs.txt"
  while IFS='|' read -r ph rest; do
   for m in i32 qp64; do echo "$ph $m $rest"; done
  done < "$OUT/extra-specs.txt"
 } | xargs -P "${ISA_JOBS:-16}" -L 1 bash -c 'one "$1" "$2" "$3"' _ | tee "$OUT/isa-run.txt"
 if grep -q '^FAIL' "$OUT/isa-run.txt"; then exit 1; fi
 [[ $(grep -c NATIVE64_DISPATCH_PASS "$OUT/isa-run.txt") == 150 ]] || exit 1
 sha256sum "$KERNEL" "$BASEK" "$PREVK" > "$OUT/kernel.sha256"
 s=0; python3 -B "/source/$G2/summarize_decg2_isa_cpu.py" "$OUT" > "$OUT/isa-summary.txt" || s=$?
 cat "$OUT/isa-summary.txt"; exit $s
fi
echo "unknown mode $MODE"; exit 2
