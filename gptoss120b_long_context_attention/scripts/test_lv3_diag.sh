#!/bin/bash
# LOOP_VARIANT 3 was ~18% slower than 2.  Split it: B only (folded fp8 scale) vs A only (lazy rescale),
# same session as LOOP_VARIANT 2 / 3.  fp8 like the TP1 server, BLOCK_M 256, MIXED (3 decode rows), accuracy on.
#   needs: /dockerx/gpt_oss/opt/attention/aiter/unified_attention_2d_gfx1250.py (the LOOP_VARIANT 3 file)
#          /dockerx/gpt_oss/opt/attention/make_lv3_diag.py
cd /dockerx/gpt_oss/opt/attention
export PYTHONPATH=$HOME/draftbase_1111/aiter
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
KF=aiter/unified_attention_2d_gfx1250.py
LOG=lv3_logs/diag_$(date +%m%d_%H%M); mkdir -p "$LOG"
python3 make_lv3_diag.py --src "$KF" --out lv3_diag 2>&1 | tee "$LOG/make.txt" || exit 1
B="python3 bench_prefill_attention.py --fp8"

echo "== LV2";                 $B --kernel-file "$KF" --cfg '{"BLOCK_M": 256, "LOOP_VARIANT": 2}' 2>&1 | tee "$LOG/lv2.txt"
echo "== LV3 B only (fold)";   $B --kernel-file lv3_diag/ua2d_foldonly.py --cfg '{"BLOCK_M": 256, "LOOP_VARIANT": 3}' 2>&1 | tee "$LOG/foldonly.txt"
echo "== LV3 A only (lazy)";   $B --kernel-file lv3_diag/ua2d_lazyonly.py --cfg '{"BLOCK_M": 256, "LOOP_VARIANT": 3}' 2>&1 | tee "$LOG/lazyonly.txt"
echo "== LV3 A+B";             $B --kernel-file "$KF" --cfg '{"BLOCK_M": 256, "LOOP_VARIANT": 3}' 2>&1 | tee "$LOG/lv3.txt"
echo "== LV2 again (noise)";   $B --kernel-file "$KF" --cfg '{"BLOCK_M": 256, "LOOP_VARIANT": 2}' 2>&1 | tee "$LOG/lv2_again.txt"
echo "logs in $LOG"
