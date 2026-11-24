#!/bin/bash
# offline compile (no GPU) of mega for the 3 models x {ag_rs, ar} at the target token counts
cd /home/binding2/moe_1026/aiter
export FLYDSL_GPU_ARCH=gfx1250 PYTHONPATH=/home/binding2/moe_1026/pydeps/flydsl053:$PWD
TOKS=${TOKS:-"8 16 32 64 128 256 16384 32768"}
fail=0
for MODEL in ${MODELS:-dsv41 m31 kimi31}; do
  case $MODEL in
    dsv41)  E="HID=5120 EXP=384 TOPK=6 INTER_TOTAL=2304 ACT=swiglu" ;;
    m31)    E="HID=6144 EXP=129 TOPK=5 INTER_TOTAL=3072 ACT=swiglu" ;;
    kimi31) E="HID=3584 EXP=896 TOPK=16 INTER_TOTAL=3072 ACT=situv2" ;;
  esac
  for MODE in ag_rs ar; do
    for T in $TOKS; do
      if [ $MODE = ar ]; then
        CFG="ar=1,rs_pull=0,rs_wt=1,ar_two=$([ $T -gt 512 ] && echo 1 || echo 0),q_place=$([ $T -gt 256 ] && echo 1 || echo 0)"
      else
        CFG=""
      fi
      r=$(env $E MEGA_CFG="$CFG${EXTRA:+,$EXTRA}" timeout 900 python ../scripts/compile_mm.py $T 2>&1 | grep -v NUMA | grep -E "^compiled|Error" | head -1 | cut -c1-90)
      echo "$MODEL $MODE $T: $r"
      echo "$r" | grep -q "^compiled" || fail=1
    done
  done
done
echo "FAIL=$fail"
