#!/bin/bash
# 在容器 hyg_fyd1 内跑：用自编译（带 ATT-DEBUG instrumentation）的 rocprofv3 抓 ATT trace。
# 用法（在 host 上）：
#   docker exec hyg_fyd1 bash /data/yanguahe/code/wk_sp1/USER_RUN_att.sh [模式]
# 模式：
#   nosig  (默认) —— 关掉 rocprofv3 signal handler，让真实 abort/异常栈暴露
#   default       —— 原样跑（signal handler 开着）
#
# 日志固定写到： /data/yanguahe/code/wk_sp1/att_logs/att_run_<模式>.log
# 结果目录：     /data/yanguahe/code/wk_sp1/att_logs/att_out_<模式>/
# 跑完 AI 会自己 ssh 来取，无需手工贴。
set -uo pipefail
MODE="${1:-nosig}"

cd /data/yanguahe/code/wk_sp1/aiter
source /data/yanguahe/code/wk_sp1/rocprof_env.sh
export HIP_VISIBLE_DEVICES=0
ROCPROF=/data/yanguahe/code/wk_sp1/rocprof-install/bin/rocprofv3

EXTRA=()
if [ "$MODE" = "nosig" ]; then
  EXTRA+=(--disable-signal-handlers true)
fi

# 用下载的新版 decoder (0.1.5, Mar13)，测其是否支持 gfx1250 解码。
# 置空 DECODER_DIR 则用容器自带 Mar8 旧版。
DECODER_DIR="${DECODER_DIR:-/data/yanguahe/code/wk_sp1/decoder_new}"
if [ -n "$DECODER_DIR" ] && [ -f "$DECODER_DIR/librocprof-trace-decoder.so" ]; then
  EXTRA+=(--att-library-path "$DECODER_DIR")
fi

LOGDIR=/data/yanguahe/code/wk_sp1/att_logs
mkdir -p "$LOGDIR"
OUT="$LOGDIR/att_out_${MODE}"
rm -rf "$OUT"; mkdir -p "$OUT"
LOG="$LOGDIR/att_run_${MODE}.log"

{
  echo "== mode=$MODE  rocprofv3=$ROCPROF  $(date) =="
  echo "== cmd: rocprofv3 --att ${EXTRA[*]} --kernel-include-regex fmha_fwd_kernel_0 =="
  "$ROCPROF" --att "${EXTRA[@]}" --log-level info -d "$OUT" \
    --kernel-include-regex "fmha_fwd_kernel_0" -- \
    python op_tests/test_mha_flydsl_varlen.py --causal true --return_lse true -b 1 -nh 32 -sq 8192 -sk 8192 --random-value false --warmup 5 --repeat 20
  echo "== rocprofv3 exit=$? =="
  echo "== 结果文件 =="
  find "$OUT" -type f 2>/dev/null | head -40
} > "$LOG" 2>&1

echo "DONE mode=$MODE  log=$LOG  exit_recorded_in_log"
tail -5 "$LOG"
