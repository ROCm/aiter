#!/bin/bash
# usage: bench_1120.sh <model: dsv41|m31|kimi31> <mode: ar|ag_rs> <tag> ["<tokens>"] [base|mega|both]
# baseline (latest upstream aiter fused_moe + comm: RCCL / custom AR (Lamport) / Gluon AR, custom AG / RS;
# the fastest counts) then mega, same batch; one GPU job at a time
MODEL=$1; MODE=$2; TAG=$3
TOK=${4:-"8 16 32 64 128 256 16384 32768"}; WHAT=${5:-both}
cd /home/binding2/moe_1026/aiter
export PYTHONPATH=/home/binding2/moe_1026/pydeps/flydsl053:$PWD
# the merged upstream tuned grouped table (scripts/mmtp/gen_fixed_csv.py): no model_configs scan / rewrite
export AITER_CONFIG_GROUPED_FMOE=/home/binding2/moe_1026/data_1114/tuned_grouped_fmoe_fixed_1124.csv
S=../scripts; R=../data_1114/raw
case $MODEL in
  dsv41)  SH="--model-dim 5120 --experts 384 --topk 6 --inter-total 2304" ;;
  m31)    SH="--model-dim 6144 --experts 129 --topk 5 --inter-total 3072 --shared-expert" ;;
  kimi31) SH="--model-dim 3584 --experts 896 --topk 16 --inter-total 3072 --act situv2" ;;
  *) echo "unknown model"; exit 1 ;;
esac
MM=""; [ "$MODE" = "ar" ] && MM="--ar"
DM0=$(bash /tmp/mmtp/dmf.sh)
# a run another job overlapped (crun.sh: a second process on a GPU) is repeated, up to 4 tries
if [ $WHAT != mega ]; then
  for try in 1 2 3 4; do
    bash /tmp/mmtp/gwait.sh ${GWAIT:-60} || exit 2
    bash /tmp/mmtp/crun.sh $R/${TAG}_base.dirty timeout 2400 torchrun --nproc_per_node=4 --master_port=29802 $S/baseline_moe_ar.py $SH --mode $MODE --tokens $TOK > $R/${TAG}_base.log 2>&1; rc=$?
    pkill -f "[b]aseline_moe_ar.py"; sleep 3
    [ -s $R/${TAG}_base.dirty ] || break
    echo "base try $try CONTAMINATED: $(head -1 $R/${TAG}_base.dirty)"
  done
  echo base rc=$rc$([ -s $R/${TAG}_base.dirty ] && echo " DIRTY")
fi
if [ $WHAT != base ]; then
  for try in 1 2 3 4; do
    bash /tmp/mmtp/gwait.sh ${GWAIT:-60} || exit 2
    bash /tmp/mmtp/crun.sh $R/${TAG}_mega.dirty timeout 2400 torchrun --nproc_per_node=4 --master_port=29803 $S/run_mm.py $MM $SH --tokens $TOK --check --bench > $R/${TAG}_mega.log 2>&1; rc=$?
    pkill -f "[r]un_mm.py"; sleep 2
    [ -s $R/${TAG}_mega.dirty ] || break
    echo "mega try $try CONTAMINATED: $(head -1 $R/${TAG}_mega.dirty)"
  done
  echo mega rc=$rc$([ -s $R/${TAG}_mega.dirty ] && echo " DIRTY")
fi
grep -m3 -E "Error|error:|unavailable|failed" $R/${TAG}_base.log $R/${TAG}_mega.log 2>/dev/null | grep -v NUMA | cut -c1-200
DM1=$(bash /tmp/mmtp/dmf.sh)
[ "$DM1" != "$DM0" ] && { echo "STOP: DMESG FAULT ($DM0 -> $DM1) at $MODEL $MODE"; exit 3; }
python3 /tmp/mmtp/tab_1120.py $MODEL $MODE $R/${TAG}_base.log $R/${TAG}_mega.log
