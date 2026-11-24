#!/bin/bash
# stop a chain script (by name) without touching a GPU process: kill the chain's bash, then let the
# current tp1g / sweepm child finish its run (it exits on its own)
N=$1
for p in $(ps -eo pid,cmd | grep "bash /tmp/mmtp/$N" | grep -v grep | awk '{print $1}'); do kill $p; done
for i in $(seq 120); do
  ps -eo cmd | grep -E "run_mm.py|baseline_moe_ar.py" | grep -v grep >/dev/null || break
  sleep 2
done
for p in $(ps -eo pid,cmd | grep -E "bash /tmp/mmtp/(tp1g|sweepm|gwait|bench_1120|crun)" | grep -v grep | awk '{print $1}'); do kill $p 2>/dev/null; done
sleep 1
ps -eo pid,etime,cmd | grep -E "mmtp|run_mm|torchrun" | grep -v -E "grep|gwatch|killchain" | cut -c1-100
echo "stopped $N"
