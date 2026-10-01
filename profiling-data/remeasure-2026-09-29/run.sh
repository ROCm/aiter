#!/bin/bash
# usage: run.sh <timeout> <tag> args...
T=$1; TAG=$2; shift 2
W=/home/jograner/projects/aiter/unified-attention-gemma4
cd $W/profiling-data && gpu-idle-check 2 && ENABLE_CK=0 HIP_VISIBLE_DEVICES=2 FLYDSL_RUNTIME_ENABLE_CACHE=0 PYTHONPATH=$W:$W/profiling-data timeout $T python3 remeasure.py "$@" --output /tmp/ua_repro/remeasure/$TAG.csv > /tmp/ua_repro/remeasure/$TAG.log 2>&1
echo rc=$?
