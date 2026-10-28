#!/bin/bash
# torchrun --no-python bench/ktrace_wrap.sh script args: each rank under rocprofv3 kernel trace,
# CSVs to bench/logs/ktrace/$KT_NAME/r$RANK
exec /usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel/bin/rocprofv3 --kernel-trace -d /home/vigokhal/m3_monokernel/bench/logs/ktrace/$KT_NAME/r$RANK -o kt --output-format csv -- python3 "$@" > /home/vigokhal/m3_monokernel/bench/logs/ktrace_r$RANK.txt 2>&1
