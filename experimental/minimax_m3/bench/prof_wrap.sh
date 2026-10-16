#!/bin/bash
# torchrun --no-python bench/prof_wrap.sh script args: each rank under rocprofv3 counter collection
# (PMC="..." counters), CSVs to bench/logs/prof/r$RANK
exec /usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel/bin/rocprofv3 --pmc $PMC -d /home/vigokhal/m3_monokernel/bench/logs/prof/r$RANK -o pmc --output-format csv -- python3 "$@" > /home/vigokhal/m3_monokernel/bench/logs/prof_r$RANK.txt 2>&1
