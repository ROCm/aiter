#!/bin/bash
# usage: sweepm.sh <model> <mode ar|ag_rs> <tag> "<tokens>" cfg1 cfg2 ...   (cfg "" = default); mega only
# a run another job overlapped (crun.sh) is repeated (up to 3 tries) and tagged DIRTY if still overlapped
MODEL=$1; MODE=$2; TAG=$3; TOKS=$4; shift 4
cd /home/binding2/moe_1026/aiter
export PYTHONPATH=/home/binding2/moe_1026/pydeps/flydsl053:$PWD
S=../scripts; R=../data_1114/raw
case $MODEL in
  dsv41)  SH="--model-dim 5120 --experts 384 --topk 6 --inter-total 2304" ;;
  m31)    SH="--model-dim 6144 --experts 129 --topk 5 --inter-total 3072 --shared-expert" ;;
  kimi31) SH="--model-dim 3584 --experts 896 --topk 16 --inter-total 3072 --act situv2" ;;
esac
MM=""; [ "$MODE" = "ar" ] && MM="--ar"
i=0
DM0=$(bash /tmp/mmtp/dmf.sh)
for C in "$@"; do
  i=$((i+1))
  for try in 1 2 3; do
    bash /tmp/mmtp/gwait.sh ${GWAIT:-30} || { echo GPU BUSY; exit 2; }
    MEGA_CFG="$C" bash /tmp/mmtp/crun.sh $R/${TAG}_$i.dirty timeout 600 torchrun --nproc_per_node=4 --master_port=29803 $S/run_mm.py $MM $SH --tokens $TOKS --check --bench $MM_EXTRA > $R/${TAG}_$i.log 2>&1
    rc=$?
    pkill -f "[r]un_mm.py"; sleep 2
    [ -s $R/${TAG}_$i.dirty ] || break
  done
  echo "cfg[$C] rc=$rc$([ -s $R/${TAG}_$i.dirty ] && echo ' DIRTY')"; grep "^T=" $R/${TAG}_$i.log | sed 's/rel_l2=[^ ]* //' | awk '{ for (k=1;k<=NF;k++) if ($k ~ /^logits_diff=/) { split($k,a,"="); if (a[2]+0 > 1e-3) $0 = $0 "  <<< WRONG RESULT" } print }'
  [ $rc -ne 0 ] && grep -m2 -E "Error|error:" $R/${TAG}_$i.log | grep -v NUMA | cut -c1-200
  DM1=$(bash /tmp/mmtp/dmf.sh)
  [ "$DM1" != "$DM0" ] && { echo "STOP: DMESG FAULT ($DM0 -> $DM1) at cfg[$C]"; exit 3; }
done
