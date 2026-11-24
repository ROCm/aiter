#!/bin/bash
# r234: sh8 with the tuned shared tile (32x128x256 b3; big: b_th=0): 1-GPU validation of all 8 token counts,
# both modes, then the 4-GPU table
cd /home/binding2/moe_1026/aiter
export PYTHONPATH=/home/binding2/moe_1026/pydeps/flydsl053:$PWD
L=/tmp/mmtp/r234_chain.log
ALLT="16 32 64 128 256 512 16384 32768"
one() { for t in $3; do
    C=$(python3 /tmp/mmtp/rowcfg.py $1 $2 $t 2>/dev/null | tail -1); X="${C:+$C,}$4"; X=${X%,}
    for try in 1 2 3; do
      o=$(bash /tmp/mmtp/tp1g.sh $1 $2 "$t" "$X" | grep -E "T=|STOP|rror|BUSY" | cut -c1-125)
      echo "$o" | grep -q "DIRTY" || break
      sleep 20
    done
    echo "$o" | sed "s|$| [$MM_EXTRA $4]|"
  done; }
export MM_EXTRA="--sh8" GWAIT=20
echo "== 1-GPU $(date +%T)"
one m31 ag_rs "$ALLT"; one m31 ar "$ALLT"
if grep -qE "STOP|rc=[1-9]|BUSY|DIRTY" $L; then echo "1-GPU FAILED"; exit 1; fi
echo "== 4-GPU $(date +%T)"
for r in 1 2; do for MO in ag_rs ar; do
  echo "#### m31 $MO sh8 #$r $(date +%T)"
  bash /tmp/mmtp/sweepm.sh m31 $MO r234sh8_${MO}_$r "$ALLT" "" | sed 's/logits_diff=[^ ]* //'
done; done
echo "ALL DONE $(date +%T) dmesg=$(bash /tmp/mmtp/dmf.sh)"
