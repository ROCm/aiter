#!/bin/bash
# No-device container entrypoint for the QSA index scorer integration (Stage 2, CPU only).
# Mounts: /source = this tree (ro), /base = champion tree (ro; 82c4dec5, or a78c285c for the bf16
#         identity audit), /results (rw), golden at $QSA_GOLDEN_DIR (ro).
# Modes: cpu (stage2 tests + gen_profile verify + the 19 _g2 CPU gates of champion 82c4dec5)
#        isa (API-dispatched vs direct /base kernel ISA; ISA_ALL_ROWS=1 all rows, ISA_DTYPE=bf16|fp8)
#        isa-bf16-a78 (= isa with ISA_DTYPE=bf16 against /base = a78c285c; output dir isa-bf16-a78)
set -euo pipefail
test ! -e /dev/kfd; test ! -e /dev/dri
unset QSA_IMPORT_PACKAGE QSA_PAD_BUCKET QSA_PAD_INT32_GUARD QSA_STRICT_I32 QSA_FP8_CVT QSA_BF16_R16_TDM_COLS \
  QSA_FP8_K_LOAD QSA_PERSIST_COLS QSA_BF16_R64 QSA_BF16_PERSIST QSA_BF16_K_LOAD QSA_G2_ACCESS QSA_BF16_PATH \
  QSA_FP8_PATH QSA_DRAFT_PATH QSA_R16_NOBAR QSA_DRAFT_K_LOAD QSA_DRAFT_COLS QSA_DECODE_GROUP \
  QSA_DECODE_GROUP_COLS QSA_DECG_FP8_W64_DEBUG QSA_FP8_CVT_ASM QSA_DRAFT_FP8_SCHED QSA_DECG_FP8_SCHED \
  AITER_QSA_INDEX_SCORER_PROFILE
G2=aiter/ops/flydsl/kernels/mqa_logits/_g2
KREL=aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py
MODE=$1
OUT=/results/$MODE
mkdir -p "$OUT"
export TMPDIR=$OUT/tmp; mkdir -p "$TMPDIR"
python3 -B - <<'PY' > "$OUT/versions.json"
import importlib.metadata as m, json, platform, os, torch
print(json.dumps({'python': platform.python_version(), 'flydsl': m.version('flydsl'), 'torch': torch.__version__,
                  'no_kfd': not os.path.exists('/dev/kfd'), 'no_dri': not os.path.exists('/dev/dri')}, indent=2))
PY
sha256sum "/source/$KREL" "/base/$KREL" > "$OUT/kernel.sha256"
if cmp -s "/source/$KREL" "/base/$KREL"; then echo "KERNEL_SOURCE_IDENTICAL_TO_BASE"
else echo "KERNEL_SOURCE_DIFFERS_FROM_BASE"; fi | tee "$OUT/kernel-identity.txt"
rc=0
run() { local name=$1; shift; set +e; "$@" > "$OUT/$name.log" 2>&1; local s=$?; set -e
        echo "$name exit=$s" | tee -a "$OUT/gate-status.txt"; tail -3 "$OUT/$name.log"; [ $s -eq 0 ] || rc=1; }
if [[ $MODE == cpu ]]; then
  : > "$OUT/gate-status.txt"
  cd /source
  run stage2-cpu python3 -B op_tests/test_qsa_index_scorer_cpu.py -v
  run stage2-gen-verify python3 -B aiter/ops/qsa_index_scorer/gen_profile.py verify
  # every _g2 CPU gate (same 19 gates/env as the champion 82c4dec5 container_fp8perf_cpu.sh cpu-default)
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
  sha256sum op_tests/test_qsa_index_scorer_cpu.py aiter/ops/qsa_index_scorer/*.py \
    aiter/configs/qsa_index_scorer/*.json "$G2"/*.py > "$OUT/test-source.sha256"
  echo "ALL_CPU_GATES_RC=$rc gates=$(grep -c exit= "$OUT/gate-status.txt")" | tee -a "$OUT/gate-status.txt"; exit $rc
fi
if [[ $MODE == isa-bf16-a78 ]]; then ISA_DTYPE=bf16; fi
if [[ $MODE == isa || $MODE == isa-bf16-a78 ]]; then
  cd /source
  export COMPILE_ONLY=1 FLYDSL_GPU_ARCH=gfx1250
  s=0; python3 -B op_tests/op_benchmarks/qsa_index_scorer/isa_audit_qsa_index_scorer.py \
    --base-source "/base/$KREL" --results "$OUT/audit" --jobs "${ISA_JOBS:-16}" ${ISA_ALL_ROWS:+--all-rows} \
    ${ISA_DTYPE:+--cache-dtype "$ISA_DTYPE"} \
    > "$OUT/isa-audit.log" 2>&1 || s=$?
  tail -40 "$OUT/isa-audit.log"; exit $s
fi
echo "unknown mode $MODE"; exit 2
