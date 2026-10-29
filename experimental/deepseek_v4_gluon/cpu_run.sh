#!/usr/bin/env bash
# CPU-only (no GPU devices) run in the patched-Triton image: compile probes, layout checks.
# Usage: ./cpu_run.sh -m deepseek_v4_gluon.tests.probe_m1_loads 1 16 48
IMG=${IMG:-vigokhal/vllm-nightly-tritonfix:00b6c4c5-v2}  # triton_patch/build.sh
X=$(cd "$(dirname "$0")/.." && pwd)  # the experimental/ dir
exec docker run --rm -v "$X":/x:ro \
  -e PYTHONPATH=/x -e TRITON_CACHE_DIR=/tmp/tc -e TRITON_FRONT_END_DEBUGGING=1 -w /x \
  --entrypoint python3 "$IMG" "$@"
