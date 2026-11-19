#!/bin/bash
# Where does the prefill attention time go?  Times the installed kernel's LOOP_VARIANT 2 loop with parts of
# the softmax removed (make_ablations.py).  Outputs of the ablations are WRONG on purpose (no accuracy check);
# only the times matter.  fp8 like the TP1 server, BLOCK_M 256, pure prefill (no decode rows), chunks 2-4.
# "base" runs first and last: the difference between the two shows the run-to-run noise.
# Logs go to ./ablate_logs/<date>_<time>/
cd /dockerx/gpt_oss/opt/attention
export PYTHONPATH=$HOME/draftbase_1111/aiter
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
LOG=ablate_logs/$(date +%m%d_%H%M); mkdir -p "$LOG"
python3 make_ablations.py --out ablate 2>&1 | tee "$LOG/make.txt" || exit 1

for v in base noexp norescale noexp_norescale base; do
  echo "== $v"
  python3 bench_prefill_attention.py --fp8 --cfg '{"BLOCK_M": 256}' --decode-rows 0 --no-check \
      --prefix 16384,32768,49152 --kernel-file "ablate/ua2d_$v.py" 2>&1 | tee -a "$LOG/$v.txt"
done
echo "logs in $LOG"
