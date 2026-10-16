#!/bin/bash
# torchrun --no-python bench/gdb_wrap.sh script args: each rank under rocgdb (batch); the
# faulting wave's pc / instructions go to bench/logs/gdb_r$RANK.txt
exec rocgdb -batch -ex "set pagination off" -ex "set confirm off" -ex run -ex "info threads" -ex "x/6i \$pc-24" \
  -ex "x/4i \$pc" -ex "info registers pc" -ex "bt 5" --args python3 "$@" > /home/vigokhal/m3_monokernel/bench/logs/gdb_r$RANK.txt 2>&1
