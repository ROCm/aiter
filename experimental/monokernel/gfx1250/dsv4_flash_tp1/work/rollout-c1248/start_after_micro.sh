#!/bin/bash
set -euo pipefail
while [ ! -f /work/rollout-c1248/micro-approved ] || [ ! -f /work/rollout-c1248/micro-c1.done ] || [ ! -f /work/rollout-c1248/micro-c2.done ] || [ ! -f /work/rollout-c1248/micro-c4.done ] || [ ! -f /work/rollout-c1248/micro-c8.done ]; do sleep 2; done
exec bash /work/rollout-c1248/server.sh "$1"
