#!/bin/bash
set -u
cd /work/c16-monokernel
worker=$1
case $worker in
 0) jobs=(qkv_cache attention output);;
 1) jobs=(mhc routing shared identity);;
 2) jobs=(whole_ffn);;
 3) jobs=(compress);;
esac
for batch in 8 16; do
 for task in "${jobs[@]}"; do
  if [ "$task" = compress ]; then
   for c in 0 1 2; do
    BATCH=$batch python native_compress.py "$c" > "run-c${batch}-compress${c}.log" 2>&1
    echo $? > "run-c${batch}-compress${c}.exit"
   done
  else
   splits=8; if [ "$batch" = 16 ]; then splits=4; fi
   BATCH=$batch SPLITS=$splits python "native_${task}.py" > "run-c${batch}-${task}.log" 2>&1
   echo $? > "run-c${batch}-${task}.exit"
  fi
 done
done
