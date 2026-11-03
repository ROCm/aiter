#!/bin/bash
# ROUND-W64FIX no-device container entrypoint.
# Modes: cpu-default (the 18 DECG2 CPU gates, default env; delegates to container_decg_cpu.sh)
#        isa  (a) default env final == base a78c285c for 18 specs x {i32, qp64};
#             (b) fp8 w64 bisect variants (QSA_DECODE_GROUP_COLS=64 QSA_DECG_FP8_W64_DEBUG=v)
#                 for v in asis nop4 intrin tdmwait x {i32, qp64} + the WMMA hazard scan,
#                 and the shipping fp8/bf16 w32 + bf16 w64 kernels (scan only).
# Mounts: /source = final (opt-w64fix), /base = champion a78c285c.
set -euo pipefail
test ! -e /dev/kfd; test ! -e /dev/dri
MODE=$1
G2=aiter/ops/flydsl/kernels/mqa_logits/_g2
if [[ $MODE == cpu-* ]]; then
 exec bash "/source/$G2/container_decg_cpu.sh" "$MODE"
fi
unset QSA_IMPORT_PACKAGE QSA_PAD_BUCKET QSA_PAD_INT32_GUARD QSA_STRICT_I32 QSA_FP8_CVT QSA_BF16_R16_TDM_COLS QSA_FP8_K_LOAD QSA_PERSIST_COLS QSA_BF16_R64 QSA_BF16_PERSIST QSA_BF16_K_LOAD QSA_G2_ACCESS QSA_BF16_PATH QSA_FP8_PATH QSA_DRAFT_PATH QSA_R16_NOBAR QSA_DRAFT_K_LOAD QSA_DRAFT_COLS QSA_DECODE_GROUP QSA_DECODE_GROUP_COLS QSA_DECG_FP8_W64_DEBUG
KREL=aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py
KERNEL=/source/$KREL
BASEK=/base/$KREL
OUT=/results/$MODE
mkdir -p "$OUT"
sha256sum "$KERNEL" "$BASEK" > "$OUT/kernel.sha256"
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
cat > "$OUT/extra-specs.txt" <<'EOF'
var|w64-asis|fp8|64,16,4096|QSA_DECODE_GROUP_COLS=64,QSA_DECG_FP8_W64_DEBUG=asis
var|w64-nop4|fp8|64,16,4096|QSA_DECODE_GROUP_COLS=64,QSA_DECG_FP8_W64_DEBUG=nop4
var|w64-intrin|fp8|64,16,4096|QSA_DECODE_GROUP_COLS=64,QSA_DECG_FP8_W64_DEBUG=intrin
var|w64-tdmwait|fp8|64,16,4096|QSA_DECODE_GROUP_COLS=64,QSA_DECG_FP8_W64_DEBUG=tdmwait
var|bf16-w64|bf16|64,16,4096|QSA_DECODE_GROUP_COLS=64
dflt|cols64-fp8-off|fp8|64,16,4096|QSA_DECODE_GROUP_COLS=64
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
[[ $(grep -c NATIVE64_DISPATCH_PASS "$OUT/isa-run.txt") == 84 ]] || exit 1
s=0; python3 -B "/source/$G2/summarize_w64fix_isa_cpu.py" "$OUT" > "$OUT/isa-summary.txt" || s=$?
cat "$OUT/isa-summary.txt"; exit $s
