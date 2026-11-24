#!/bin/bash
# offline compile (no GPU): AR defaults (hist batching) and ag_rs LL with the W1 prefetch variants
cd /home/binding2/moe_1026/aiter
export FLYDSL_GPU_ARCH=gfx1250 PYTHONPATH=/home/binding2/moe_1026/pydeps/flydsl053:$PWD
fail=0
for MODEL in dsv41 m31 kimi31; do
  case $MODEL in
    dsv41)  E="HID=5120 EXP=384 TOPK=6 INTER_TOTAL=2304 ACT=swiglu" ;;
    m31)    E="HID=6144 EXP=129 TOPK=5 INTER_TOTAL=3072 ACT=swiglu" ;;
    kimi31) E="HID=3584 EXP=896 TOPK=16 INTER_TOTAL=3072 ACT=situv2" ;;
  esac
  for T in ${TOKS:-8 64 256}; do
    for CFG in "ar=1,rs_pull=0,rs_wt=1,rs_ll=$([ $T -le 64 ] && echo 1 || echo 0)" "ag_ll=1,rs_ll=1" "ag_ll=1,rs_ll=1,wpf=16" "ag_ll=1,rs_ll=1,wpf=24,wpf_e=1" ${XCFG}; do
      r=$(env $E MEGA_CFG="$CFG" timeout 900 python ../scripts/compile_mm.py $T 2>&1 | grep -v NUMA | grep -E "^compiled|Error|error" | head -1 | cut -c1-70)
      echo "$MODEL $T [$CFG]: $r"
      echo "$r" | grep -q "^compiled" || fail=1
    done
  done
done
echo "FAIL=$fail"
