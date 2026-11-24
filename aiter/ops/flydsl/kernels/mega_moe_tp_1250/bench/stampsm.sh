#!/bin/bash
# 4-GPU in-kernel phase stamps (MEGA_TS=1: the kernel's own timer, not a profiler) for a model / mode
# usage: stampsm.sh <model> <mode ar|ag_rs> <tag> "<tokens>" ["<MEGA_CFG>"]
MODEL=$1; MODE=$2; TAG=$3; TOK=$4; CFG=$5
cd /home/binding2/moe_1026/aiter
export PYTHONPATH=/home/binding2/moe_1026/pydeps/flydsl053:$PWD MEGA_TS=1
case $MODEL in
  dsv41)  SH="--model-dim 5120 --experts 384 --topk 6 --inter-total 2304" ;;
  m31)    SH="--model-dim 6144 --experts 129 --topk 5 --inter-total 3072 --shared-expert" ;;
  kimi31) SH="--model-dim 3584 --experts 896 --topk 16 --inter-total 3072 --act situv2" ;;
esac
MM=""; [ "$MODE" = "ar" ] && MM="--ar"
timeout 60 bash ../scripts/gpu_free.sh 0 1 2 3 >/dev/null || { echo BUSY; exit 2; }
MEGA_CFG="$CFG" bash /tmp/mmtp/crun.sh ../data_1114/raw/$TAG.dirty timeout 300 torchrun --nproc_per_node=4 --master_port=29803 ../scripts/run_mm.py $MM $SH --tokens $TOK --check --stamps $MM_EXTRA > ../data_1114/raw/$TAG.log 2>&1
echo rc=$?
pkill -f "[r]un_mm.py"; sleep 2
grep -A1 "^T=" ../data_1114/raw/$TAG.log | grep -v "^--" | sed 's/hist=-[0-9.]*\/-[0-9.]* //; s/y1=-[^ ]* y1wait=-[^ ]* //'
