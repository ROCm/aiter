#!/bin/bash
# ab2.sh LOG "BASE_ENV" "VARIANT_ENV" bench_agentx-args : A/B on ONE GPU set under one lock
# (base then variant, back to back; sets differ by ~0.5 us/layer so never compare across sets).
# Prints "<LOG>_a" / "<LOG>_b" result lines. GPUSET / ROOT as bax2.sh.
NAME=$1; EA=$2; EB=$3; shift 3
if [ "${GPUSET:-1}" = "0" ] && ! $(dirname $0)/set0_idle.sh; then echo "GPUSET=0: GPUs 0-3 busy (another user?); using set 1" >&2; GPUSET=1; DEVS=4,5,6,7; PORT=29700; GI=4; fi
LOCK=/home/vigokhal/agentx/lock_set${GPUSET:-1}
mkdir -p /home/vigokhal/agentx
D=$(dirname $0)
flock $LOCK bash -c "NOLOCK=1 $D/bax2.sh ${NAME}_a \"$EA\" $*; NOLOCK=1 $D/bax2.sh ${NAME}_b \"$EB\" $*"
