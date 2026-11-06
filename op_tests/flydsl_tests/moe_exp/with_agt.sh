#!/bin/bash
# Run a command with agt_internal's power-management log on, inside the caller's gpu-lock turn.
# usage: with_agt.sh <out.csv> <period ms> <command ...>
#   AGT_INTERNAL  agt_internal binary
#   AGT_ARGS      parameter selection (default -unilogallgroups)
# The logger stops through its stop file (-unilogstopcheck reads terminate.txt in its working directory, a private one
# here), never by a signal; -unilogcount bounds it at 10 minutes in case that fails.
set -u
out=$1; period=$2; shift 2
AGT=${AGT_INTERNAL:-/tdata/lixun/moe_exp/tools/agt_4.1.174/agt_internal}
ARGS=${AGT_ARGS:--unilogallgroups}
dir=$(mktemp -d "$(dirname "$out")/.agt.XXXXXX")
( cd "$dir" && exec sudo -n "$AGT" -i=0 -unilog=PM $ARGS -unilogperiod="$period" -unilogcount=$((600000 / period)) \
    -unilogoutput="$out" -unilogstopcheck -unilognoesckey -unilogexitonerror </dev/null >"$out.log" 2>&1 ) &
agt=$!
# wait for the first samples, so the logger is running before the command starts
for _ in $(seq 1 100); do
  [ "$(wc -l <"$out" 2>/dev/null || echo 0)" -ge 5 ] && break
  sleep 0.1
done
"$@"
rc=$?
sleep 0.5
touch "$dir/terminate.txt"
wait "$agt"
echo "agt rc=$? samples=$(($(wc -l <"$out") - 1))" >>"$out.log"
sudo -n chown "$(id -u):$(id -g)" "$out" "$out.log"
sudo -n rm -rf "$dir"
exit $rc
