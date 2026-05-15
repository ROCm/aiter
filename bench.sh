#!/bin/bash
# MoE TDM→LDS BW microbenchmark — DeepSeek V3 TP1

# FFM (single launch, no timing, small problem size)
# source $GFX1250_MODEL_PATH/ffmlite_env.sh && export PYTHONPATH=/workspace/ffm/triton/python:/workspace/ffm/FlyDSL:${PYTHONPATH} && export HSA_MODEL_NUM_THREADS=8 FLYDSL_REBUILD_KERNELS=1 && python -m kernels.bench_moe_lds_bw --experts 1 --model_dim 512 --inter_dim 256 --tile_n 128 --tile_k 256 --num_stages 2 --dtype fp8

# Real HW (full size, prints BW GB/s)
export PYTHONPATH=/workspace/ffm/triton/python:/workspace/ffm/FlyDSL:${PYTHONPATH} && python -m kernels.bench_moe_lds_bw --experts 256 --model_dim 7168 --inter_dim 2048 --tile_n 128 --tile_k 256 --num_stages 2 --dtype fp8
