
seq_q_l=256
seq_kv_l=256
num_heads_q=8
num_heads_k=1
bs=1
window_size=0
block_size=128
head_size=128
waves_per_eu=1
num_warps=4
block_m=128
remove_indirect_access=1
num_buffers=3
loop_variant=1
causal=1
export PRINT_IRS=1
rm -r ~/.triton/cache/
# maxnum: v_max_num_f32_e32
# maximum: v_maximum -> nan propagating, hence no self max
# remove s_wait_dscnt 0s from the loop, inserted with barrier for tdm asyn wait
export LLIR_REMOVE_DS_WAIT_0="loop" 
# # # # # # If no TDM load wait around
# export LLIR_REMOVE_BARRIER=1
source $TRITON_GFX1250_MODEL_PATH/ffmlite_env.sh
python3 run_kernel.py --causal $causal --loop_variant $loop_variant --num_buffers $num_buffers --remove_indirect_access $remove_indirect_access --block_m $block_m --num_warps $num_warps --waves_per_eu $waves_per_eu --head_size $head_size  --block_size $block_size --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $bs --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l