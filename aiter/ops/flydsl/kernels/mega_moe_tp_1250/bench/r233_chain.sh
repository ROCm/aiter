#!/bin/bash
# r233: shared-tile (s1 / s2) shapes for the A8W8 shared expert at 16 / 32 / 64 tokens, AR first
cd /home/binding2/moe_1026/aiter
export PYTHONPATH=/home/binding2/moe_1026/pydeps/flydsl053:$PWD
L=/tmp/mmtp/r233_chain.log
V=("s1.num_buffers=4" "s1.num_buffers=4,s2.num_buffers=4" "s1.tile_n=128,s1.num_buffers=6" "s1.tile_n=128,s1.tile_k=256,s1.num_buffers=3" "s1.tile_m=16,s1.m_warp=1,s1.n_warp=4,s2.tile_m=16,s2.m_warp=1,s2.n_warp=4,s1.num_buffers=4" "s1.tile_n=128,s1.tile_k=256,s1.num_buffers=3,s2.tile_n=128,s2.num_buffers=4")
one() { for t in $3; do
    C=$(python3 /tmp/mmtp/rowcfg.py $1 $2 $t 2>/dev/null | tail -1); X="${C:+$C,}$4"; X=${X%,}
    bash /tmp/mmtp/tp1g.sh $1 $2 "$t" "$X" | grep -E "T=|STOP|rror|BUSY" | cut -c1-110 | sed "s|$| [$4]|"
  done; }
export MM_EXTRA="--sh8" GWAIT=20
MO=${1:-ar}; TK=${2:-"16 32 64"}
echo "== 1-GPU $MO $(date +%T)"
for v in "${V[@]}"; do one m31 $MO "$TK" "$v"; done
grep -E "STOP|rc=[1-9]|BUSY|DIRTY" $L | sed 's/.*\[\(.*\)\]$/\1/' | sort -u > /tmp/mmtp/r233_fail.txt
echo "failed variants: $(tr '\n' ';' < /tmp/mmtp/r233_fail.txt)"
CF=(); for v in "${V[@]}"; do grep -qxF "$v" /tmp/mmtp/r233_fail.txt || CF+=("$v"); done
echo "== 4-GPU $MO $(date +%T)"
bash /tmp/mmtp/sweepm.sh m31 $MO r233_$MO "$TK" "" "${CF[@]}" "" "${CF[@]}" | sed 's/logits_diff=[^ ]* //'
echo "ALL DONE $(date +%T) dmesg=$(bash /tmp/mmtp/dmf.sh)"
