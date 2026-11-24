#!/bin/bash
# crun.sh <flagfile> <cmd...>: run cmd while sampling the GPUs' process lists every 3 s. Another job's
# process next to ours on any GPU (more than CRUN_MAXP=1 processes) -> a line in <flagfile> (the run is
# contaminated) and, unless CRUN_ABORT=0, cmd gets SIGTERM at once: a persistent kernel next to another
# job only times out and risks driver eviction failures (11-24 incidents)
F=$1; shift
: > $F
"$@" &
C=$!
( while kill -0 $C 2>/dev/null; do
    for g in 0 1 2 3; do
      n=$(timeout 10 amd-smi process -g $g 2>/dev/null | grep -c "PID:")
      if [ "${n:-0}" -gt ${CRUN_MAXP:-1} ]; then
        echo "$(date +%T) gpu$g procs=$n" >> $F
        [ "${CRUN_ABORT:-1}" = 1 ] && kill -TERM $C 2>/dev/null
      fi
    done
    sleep 3
  done ) &
S=$!
wait $C; rc=$?
kill $S 2>/dev/null; wait $S 2>/dev/null
exit $rc
