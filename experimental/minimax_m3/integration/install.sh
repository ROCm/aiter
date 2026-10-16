#!/bin/bash
# Link the M3 mono package into the container's vLLM (idempotent). Run inside ix_dsv41.
set -e
SRC=/home/vigokhal/m3_monokernel
V=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
ln -sfn "$SRC/m3mono" "$V/models/minimax_m3/amd/mono"
python3 -c "import vllm.models.minimax_m3.amd.mono.config as c; print('installed', c.__file__, 'VLLM_CACHE', c.VLLM_CACHE)" 2>&1 | grep -v "WARNING\|aiter\] import"
