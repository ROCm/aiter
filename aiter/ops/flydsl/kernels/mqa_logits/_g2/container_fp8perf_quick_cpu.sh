#!/bin/bash
# ROUND-FP8PERF quick no-device ISA loop: compile listed specs from /source (final) or
# /base (5fb8740e) and print instr / VGPR / v_nop / s_delay_alu / hazard scan per kernel.
# Usage (inside container): container_fp8perf_quick_cpu.sh OUTDIR < lines "phase|label|dtype|shape|env"
set -euo pipefail
test ! -e /dev/kfd; test ! -e /dev/dri
unset QSA_IMPORT_PACKAGE QSA_PAD_BUCKET QSA_PAD_INT32_GUARD QSA_STRICT_I32 QSA_FP8_CVT QSA_BF16_R16_TDM_COLS QSA_FP8_K_LOAD QSA_PERSIST_COLS QSA_BF16_R64 QSA_BF16_PERSIST QSA_BF16_K_LOAD QSA_G2_ACCESS QSA_BF16_PATH QSA_FP8_PATH QSA_DRAFT_PATH QSA_R16_NOBAR QSA_DRAFT_K_LOAD QSA_DRAFT_COLS QSA_DECODE_GROUP QSA_DECODE_GROUP_COLS QSA_FP8_CVT_ASM QSA_DRAFT_FP8_SCHED QSA_DECG_FP8_SCHED
G2=aiter/ops/flydsl/kernels/mqa_logits/_g2
KREL=aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py
OUT=$1; mkdir -p "$OUT"
one() {
 IFS='|' read -r phase label dtype shape envs <<< "$1"; shape=${shape//,/ }; envs=${envs//,/ }
 local src=/source/$KREL; [[ $phase == base ]] && src=/base/$KREL
 mkdir -p "$OUT/$phase"
 if ! env $envs python3 -B "/source/$G2/compile_native64_isa_cpu.py" --source "$src" --results "$OUT/$phase" --case "$label" --dtype "$dtype" --shape $shape --meta "${META:-i32}" > "$OUT/$phase/$label.log" 2>&1; then
  echo "FAIL $phase $label"; tail -20 "$OUT/$phase/$label.log"; return 0; fi
 local isa; isa=$(ls "$OUT/$phase/$label"/isa/*/21_final_isa.s)
 python3 -B - "$isa" "$OUT/$phase/$label/dispatch.json" <<'PY'
import json, re, sys
sys.path.insert(0, '/source/aiter/ops/flydsl/kernels/mqa_logits/_g2')
import wmma_hazard_scan_cpu as W
isa, disp = sys.argv[1], json.load(open(sys.argv[2]))
t = open(isa).read().splitlines()
ins = [l.strip() for l in t if re.match(r'^\s*[a-z_][a-z0-9_.]*(\s|$)', l) and not l.strip().startswith('.')]
c = lambda p: sum(1 for l in ins if l.split()[0] == p)
s = W.scan(isa)
print(f"{disp['case']:<28} {disp['kernel_name'][-40:]:>40} instr={len(ins)} vgpr={disp['resources']['next_free_vgpr']} "
      f"sgpr={disp['resources']['next_free_sgpr']} lds={disp['resources']['group_segment_fixed_size']} "
      f"v_nop={c('v_nop')} s_delay_alu={c('s_delay_alu')} s_nop={c('s_nop')} wmma={s['wmma']} hazards={s['violations']} "
      f"wait_loadcnt={c('s_wait_loadcnt')} sha={disp['isa']['sha256'][:12]}")
PY
}
export -f one; export OUT KREL G2
xargs -d '\n' -P "${ISA_JOBS:-16}" -I{} bash -c 'one "$1"' _ {}
