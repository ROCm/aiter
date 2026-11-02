#!/bin/bash
set -euo pipefail
cp /work/round2/originals/gemm_mxfp8.py /app/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/gemm/basic/gemm_mxfp8.py
cp /work/round2/originals/gemm_afp8wfp8.py /app/aiter/aiter/ops/triton/gemm/basic/gemm_afp8wfp8.py
cp /work/round2/originals/configs/*.json /app/aiter/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp8wfp8_preshuffled/
cp /work/round2/candidate/linear.py /app/ATOM/atom/model_ops/linear.py
cp /work/round2/candidate/dsv4_hybrid.py /app/ATOM/atom/model_ops/dsv4_hybrid.py
cp /work/round2/candidate/dsv4_attention_configs.json /app/ATOM/atom/model_ops/dsv4_attention_configs.json
python -m py_compile /app/ATOM/atom/model_ops/linear.py /app/ATOM/atom/model_ops/dsv4_hybrid.py
