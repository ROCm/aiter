#!/bin/bash
# Check the final LOOP_VARIANT 3 (variant 2 + fp8 P scale folded into the exponent) against LOOP_VARIANT 2,
# same session, BEFORE installing it.  It is loaded with --kernel-file (this process only).
# fp8 like the TP1 server, BLOCK_M 256, MIXED (3 decode rows), chunks 1-4, accuracy on.
#   needs: /dockerx/gpt_oss/opt/attention/aiter/unified_attention_2d_gfx1250.py  (the changed kernel file)
# Measured 2026-11-18: LV3 -10.5% (2 buffers), -17% (3 buffers, now in the table) vs LV2, identical rel_err.
# Logs go to ./lv3_logs/<date>_<time>/
cd /dockerx/gpt_oss/opt/attention
export PYTHONPATH=$HOME/draftbase_1111/aiter
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
KF=aiter/unified_attention_2d_gfx1250.py
grep -q "def attention_loop_fp8_folded" "$KF" || { echo "ERROR: $KF is not the final LOOP_VARIANT 3 file"; exit 1; }
LOG=lv3_logs/$(date +%m%d_%H%M); mkdir -p "$LOG"
B="python3 bench_prefill_attention.py --kernel-file $KF"

echo "== LV2 (fp8)";               $B --fp8 --cfg '{"BLOCK_M": 256, "LOOP_VARIANT": 2}' 2>&1 | tee "$LOG/lv2_fp8.txt"
echo "== LV3 (fp8)";               $B --fp8 --cfg '{"BLOCK_M": 256, "LOOP_VARIANT": 3}' 2>&1 | tee "$LOG/lv3_fp8.txt"
echo "== LV2 (fp8) again, noise";  $B --fp8 --cfg '{"BLOCK_M": 256, "LOOP_VARIANT": 2}' 2>&1 | tee "$LOG/lv2_fp8_again.txt"
echo "== LV3 (fp8, NUM_BUFFERS 3)"; $B --fp8 --cfg '{"BLOCK_M": 256, "LOOP_VARIANT": 3, "NUM_BUFFERS": 3}' 2>&1 | tee "$LOG/lv3_fp8_nb3.txt"
echo "== LV3 (bf16: same as LV2, no fold) -- compile + accuracy"
$B --cfg '{"BLOCK_M": 256, "LOOP_VARIANT": 3}' --prefix 0,32768 2>&1 | tee "$LOG/lv3_bf16.txt"
echo "logs in $LOG"
