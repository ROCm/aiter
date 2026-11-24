#!/bin/bash
# single-GPU (TP1) mega check with a TP4 per-rank shape: tp1g.sh <model> <mode ar|ag_rs> "<tokens>" ["<MEGA_CFG>"]
MODEL=$1; MODE=$2; TOK=$3; CFG=$4
cd /home/binding2/moe_1026/aiter
export PYTHONPATH=/home/binding2/moe_1026/pydeps/flydsl053:$PWD
case $MODEL in
  dsv41)  SH="--model-dim 5120 --experts 384 --topk 6 --inter-total 576" ;;
  m31)    SH="--model-dim 6144 --experts 129 --topk 5 --inter-total 768 --shared-expert" ;;
  kimi31) SH="--model-dim 3584 --experts 896 --topk 16 --inter-total 768 --act situv2" ;;
  *) echo "unknown model"; exit 1 ;;
esac
MM=""; [ "$MODE" = "ar" ] && MM="--ar"
ok=0; for _w in $(seq 180); do timeout 60 bash ../scripts/gpu_free.sh 0 >/dev/null && { ok=1; break; }; sleep 5; done
[ $ok = 1 ] || { echo GPU BUSY; exit 2; }
DM0=$(bash /tmp/mmtp/dmf.sh)
for T in $TOK; do
  HIP_VISIBLE_DEVICES=0 MEGA_CFG="$CFG" bash /tmp/mmtp/crun.sh /tmp/mmtp/tp1g.dirty timeout 300 torchrun --nproc_per_node=1 --master_port=29810 ../scripts/run_mm.py $MM \
    $SH --tokens $T --check $MM_EXTRA > /tmp/mmtp/tp1g.log 2>&1
  rc=$?
  echo "$MODEL $MODE T=$T rc=$rc $(grep '^T=' /tmp/mmtp/tp1g.log | head -1)$([ -s /tmp/mmtp/tp1g.dirty ] && echo ' DIRTY(another process on the GPU)')"
  if [ $rc -ne 0 ]; then grep -m3 -E "Error|error:|Fault" /tmp/mmtp/tp1g.log | grep -v NUMA | cut -c1-200; exit 1; fi
  grep -q "err=0 " /tmp/mmtp/tp1g.log || { echo "STOP: kernel err"; exit 1; }
  d=$(grep -m1 '^T=' /tmp/mmtp/tp1g.log | sed -n 's/.*logits_diff=\([^ ]*\).*/\1/p')
  python3 -c "import sys; sys.exit(0 if float('$d') < 1e-3 else 1)" || { echo "STOP: WRONG RESULT logits_diff=$d"; exit 1; }
  DM1=$(bash /tmp/mmtp/dmf.sh)
  [ "$DM1" != "$DM0" ] && { echo "STOP: DMESG FAULT ($DM0 -> $DM1) at $MODEL $MODE T=$T [$CFG]"; exit 3; }
  sleep 2
done
