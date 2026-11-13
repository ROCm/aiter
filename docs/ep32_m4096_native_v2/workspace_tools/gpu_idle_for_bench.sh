#!/usr/bin/env bash
set -euo pipefail
GPU_HIP=${1:?HIP number required}
CC=/root/workspace/workflow_34/moonshot_ep32_moe_v2/cc
if [ "${EP32_IGNORE_RESERVED_VRAM:-0}" != 1 ]; then
    exec "$CC/opt_gemm1/gpu_idle.sh" "$GPU_HIP" 10 45
fi

# User explicitly allowed the observed 25.7 GB retained allocation on 2026-11-12.
# Still run the requested gate, then require ten seconds of zero gfx/umc.
"$CC/opt_gemm1/gpu_idle.sh" "$GPU_HIP" 1 2 || true
SMI_MAP=(3 0 2 1 7 4 6 5)
SMI=${SMI_MAP[$GPU_HIP]}
START=$(date +%s)
SINCE=
while true; do
    UTIL=$(amd-smi metric -g "$SMI" -u --csv | awk -F, 'NR==2{print $2, $3}')
    VRAM=$(amd-smi metric -g "$SMI" -m --csv | awk -F, 'NR==2{print $3}')
    read -r GFX UMC <<< "$UTIL"
    [ -n "$GFX" ] && [ -n "$UMC" ] && [ -n "$VRAM" ] || exit 3
    NOW=$(date +%s)
    if [ "$GFX" = 0 ] && [ "$UMC" = 0 ]; then
        [ -n "$SINCE" ] || SINCE=$NOW
        if [ $((NOW - SINCE)) -ge 10 ]; then
            echo "[gpu_idle] HIP$GPU_HIP/SMI$SMI gfx=0 umc=0 for 10s; VRAM=${VRAM}MB ignored per user instruction" >&2
            exit 0
        fi
    else
        SINCE=
    fi
    [ $((NOW - START)) -lt 45 ] || exit 1
    sleep 1
done
