#!/bin/bash
# bench_sh8_base.sh <mode ar|ag_rs> <tag> ["<tokens>"]: M3.1 A8W8-shared baselines on 4 GPUs:
# sep (upstream main: routed fused_moe E128/top-4 + the shared expert's own A8W8 chain) for every token count,
# fuse (branch binding_moe_m3.1_opt_gfx1250: fused_moe(shared_w*)) up to 512 tokens
MODE=$1; TAG=$2; TOK=${3:-"16 32 64 128 256 512 16384 32768"}
cd /home/binding2/moe_1026/aiter
export PYTHONPATH=/home/binding2/moe_1026/pydeps/flydsl053:$PWD
export AITER_CONFIG_GROUPED_FMOE=/home/binding2/moe_1026/data_1114/tuned_grouped_fmoe_fixed_1124b.csv
S=../scripts; R=../data_1114/raw
SH="--model-dim 6144 --experts 129 --topk 5 --inter-total 3072 --shared-expert"
DM0=$(bash /tmp/mmtp/dmf.sh)
for V in sep fuse; do
  TK="$TOK"; [ $V = fuse ] && TK=$(for t in $TOK; do [ $t -le 512 ] && echo -n "$t "; done)
  [ -z "$TK" ] && continue
  for try in 1 2 3 4; do
    bash /tmp/mmtp/gwait.sh ${GWAIT:-60} || exit 2
    bash /tmp/mmtp/crun.sh $R/${TAG}_${V}.dirty timeout 2400 torchrun --nproc_per_node=4 --master_port=29802 $S/baseline_m31_sh8.py $SH --mode $MODE --variant $V --tokens $TK > $R/${TAG}_${V}.log 2>&1; rc=$?
    sleep 3
    [ -s $R/${TAG}_${V}.dirty ] || break
    echo "$V try $try CONTAMINATED: $(head -1 $R/${TAG}_${V}.dirty)"
  done
  echo "$V rc=$rc$([ -s $R/${TAG}_${V}.dirty ] && echo ' DIRTY')"
  grep -a "^T=" $R/${TAG}_${V}.log | cut -c1-210
  DM1=$(bash /tmp/mmtp/dmf.sh); [ "$DM1" != "$DM0" ] && { echo "STOP: DMESG FAULT ($DM0 -> $DM1)"; exit 3; }
done
