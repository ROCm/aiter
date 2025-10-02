#!/bin/bash
set -euo pipefail

rocm-smi
rocm-smi --showpids
echo 'Cleaning up ROCm processes...'
# Kill all processes using the ROCm GPUs by extracting their PIDs from rocm-smi output
rocm-smi --showpids | awk '{if($1 ~ /^[0-9]+$/) print $1}' | xargs -r sudo kill -9 || true
rocm-smi
rocm-smi --showpids