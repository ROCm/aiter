#!/bin/bash
seq_q_l=256
seq_kv_l=256
num_heads_q=8
num_heads_k=1
head_size=128
bs=1
window_size=0
block_size=128
waves_per_eu=1
num_warps=4
block_m=128
num_buffers=3
causal=1
loop_variant=3
remove_indirect_access=1
shuffled_kv_cache=0
# export LLIR_REMOVE_DS_WAIT_0="loop"
# export LLIR_REMOVE_BARRIER=1
rm -rf ~/.triton/cache
export PRINT_IRS=1
source $TRITON_GFX1250_MODEL_PATH/ffmlite_env.sh


python3 run_kernel.py \
--causal $causal \
--loop_variant $loop_variant \
--num_buffers $num_buffers \
--remove_indirect_access $remove_indirect_access \
--block_m $block_m \
--num_warps $num_warps \
--waves_per_eu $waves_per_eu \
--head_size $head_size \
--block_size $block_size \
--num_heads_q $num_heads_q \
--num_heads_k $num_heads_k \
--bs $bs \
--seq_q_l $seq_q_l \
--seq_kv_l $seq_kv_l \
--shuffled_kv_cache $shuffled_kv_cache