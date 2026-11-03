#!/bin/bash
# ROUND-W64FIX2 no-device container entrypoint.
# Modes: cpu-default (19 CPU gates, default env)
#        cpu-legacy  (same 19 gates, QSA_DECODE_GROUP=0 QSA_DRAFT_PATH=0 QSA_R16_NOBAR=0 exported)
#        isa         real-wrapper compiles of final vs base a78c285c:
#                    18 specs x {i32,qp64} default env; legacy env (4 specs); COLS=64 fp8/bf16;
#                    QSA_FP8_CVT_ASM=1 opt-out (4 pk8 specs + COLS=64 fp8); then
#                    summarize_w64fix2_isa_cpu.py: identity of every non-pk8 kernel, pk8
#                    kernel deltas, and the zero-tolerance WMMA hazard scan of every
#                    default / legacy / COLS=64 kernel.
# Mounts: /source = final (opt-w64fix), /base = champion a78c285c.
set -euo pipefail
test ! -e /dev/kfd; test ! -e /dev/dri
MODE=$1
unset QSA_IMPORT_PACKAGE QSA_PAD_BUCKET QSA_PAD_INT32_GUARD QSA_STRICT_I32 QSA_FP8_CVT QSA_BF16_R16_TDM_COLS QSA_FP8_K_LOAD QSA_PERSIST_COLS QSA_BF16_R64 QSA_BF16_PERSIST QSA_BF16_K_LOAD QSA_G2_ACCESS QSA_BF16_PATH QSA_FP8_PATH QSA_DRAFT_PATH QSA_R16_NOBAR QSA_DRAFT_K_LOAD QSA_DRAFT_COLS QSA_DECODE_GROUP QSA_DECODE_GROUP_COLS QSA_DECG_FP8_W64_DEBUG QSA_FP8_CVT_ASM
G2=aiter/ops/flydsl/kernels/mqa_logits/_g2
KREL=aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py
KERNEL=/source/$KREL
BASEK=/base/$KREL
OUT=/results/$MODE
mkdir -p "$OUT"
sha256sum "$KERNEL" "$BASEK" > "$OUT/kernel.sha256"
python3 -B - <<'PY' > "$OUT/versions.json"
import importlib.metadata as m, json, platform, os, torch
print(json.dumps({'python':platform.python_version(),'flydsl':m.version('flydsl'),'torch':torch.__version__,'torch_hip':torch.version.hip,'no_kfd':not os.path.exists('/dev/kfd'),'no_dri':not os.path.exists('/dev/dri'),'COMPILE_ONLY':os.getenv('COMPILE_ONLY')},indent=2))
PY
rc=0
run() { local name=$1; shift; set +e; "$@" > "$OUT/$name.log" 2>&1; local s=$?; set -e; echo "$name exit=$s" | tee -a "$OUT/gate-status.txt"; tail -3 "$OUT/$name.log"; [ $s -eq 0 ] || rc=1; }
if [[ $MODE == cpu-default || $MODE == cpu-legacy ]]; then
 : > "$OUT/gate-status.txt"
 if [[ $MODE == cpu-legacy ]]; then export QSA_DECODE_GROUP=0 QSA_DRAFT_PATH=0 QSA_R16_NOBAR=0; fi
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
 run wmma-hazard python3 -B "$G2/test_wmma_hazard_scan_cpu.py" -v
 sha256sum "$G2"/*.py > "$OUT/test-source.sha256"
 echo "ALL_CPU_GATES_RC=$rc gates=$(wc -l < "$OUT/gate-status.txt")"; exit $rc
fi
if [[ $MODE != isa ]]; then echo "unknown mode $MODE"; exit 2; fi
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
# phase|spec|dtype|shape|env ; phases: base (a78c285c source) or any other = final source
cat > "$OUT/extra-specs.txt" <<'EOF'
base|legacy-decode-bf16|bf16|64,16,4096|QSA_DECODE_GROUP=0,QSA_DRAFT_PATH=0,QSA_R16_NOBAR=0
legacy|legacy-decode-bf16|bf16|64,16,4096|QSA_DECODE_GROUP=0,QSA_DRAFT_PATH=0,QSA_R16_NOBAR=0
base|legacy-decode-fp8|fp8|64,16,4096|QSA_DECODE_GROUP=0,QSA_DRAFT_PATH=0,QSA_R16_NOBAR=0
legacy|legacy-decode-fp8|fp8|64,16,4096|QSA_DECODE_GROUP=0,QSA_DRAFT_PATH=0,QSA_R16_NOBAR=0
base|legacy-draft-bf16|bf16|16,16,4096|QSA_DECODE_GROUP=0,QSA_DRAFT_PATH=0,QSA_R16_NOBAR=0
legacy|legacy-draft-bf16|bf16|16,16,4096|QSA_DECODE_GROUP=0,QSA_DRAFT_PATH=0,QSA_R16_NOBAR=0
base|legacy-draft-fp8|fp8|16,16,4096|QSA_DECODE_GROUP=0,QSA_DRAFT_PATH=0,QSA_R16_NOBAR=0
legacy|legacy-draft-fp8|fp8|16,16,4096|QSA_DECODE_GROUP=0,QSA_DRAFT_PATH=0,QSA_R16_NOBAR=0
base|cols64-bf16|bf16|64,16,4096|QSA_DECODE_GROUP_COLS=64
cols64|cols64-bf16|bf16|64,16,4096|QSA_DECODE_GROUP_COLS=64
cols64|cols64-fp8|fp8|64,16,4096|QSA_DECODE_GROUP_COLS=64
asm|fp8-r32-tdm|fp8|512,1,4096|QSA_FP8_CVT_ASM=1
asm|draft-fp8-tdm|fp8|16,16,4096|QSA_FP8_CVT_ASM=1
asm|draft-fp8-raw|fp8|16,16,4096|QSA_FP8_CVT_ASM=1,QSA_DRAFT_K_LOAD=raw
asm|decode-fp8-64x16|fp8|64,16,4096|QSA_FP8_CVT_ASM=1
asm|cols64-fp8|fp8|64,16,4096|QSA_FP8_CVT_ASM=1,QSA_DECODE_GROUP_COLS=64
EOF
one() {
 local phase=$1 meta=$2 line=$3 src=$KERNEL
 IFS='|' read -r spec dtype shape envs <<< "$line"; shape=${shape//,/ }; envs=${envs//,/ }
 [[ $phase == base ]] && src=$BASEK
 local label=$spec-$meta
 mkdir -p "$OUT/$phase"
 env $envs python3 -B "/source/$G2/compile_native64_isa_cpu.py" --source "$src" --results "$OUT/$phase" --case "$label" --dtype "$dtype" --shape $shape --meta "$meta" > "$OUT/$phase/$label.log" 2>&1 && tail -1 "$OUT/$phase/$label.log" || { echo "FAIL $phase $label"; tail -30 "$OUT/$phase/$label.log"; return 1; }
}
export -f one; export KERNEL BASEK G2 OUT
{
 while read -r line; do for ph in base dflt; do for m in i32 qp64; do echo "$ph $m $line"; done; done; done < "$OUT/specs.txt"
 while IFS='|' read -r ph rest; do for m in i32 qp64; do echo "$ph $m $rest"; done; done < "$OUT/extra-specs.txt"
} | xargs -P "${ISA_JOBS:-16}" -L 1 bash -c 'one "$1" "$2" "$3"' _ | tee "$OUT/isa-run.txt"
if grep -q '^FAIL' "$OUT/isa-run.txt"; then exit 1; fi
[[ $(grep -c NATIVE64_DISPATCH_PASS "$OUT/isa-run.txt") == 104 ]] || { echo "expected 104 compiles"; exit 1; }
s=0; python3 -B "/source/$G2/summarize_w64fix2_isa_cpu.py" "$OUT" > "$OUT/isa-summary.txt" || s=$?
cat "$OUT/isa-summary.txt"; exit $s
