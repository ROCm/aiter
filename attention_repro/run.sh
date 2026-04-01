
seq_q_l=512
seq_kv_l=512
num_heads_q=8
num_heads_k=1
bs=1
window_size=0
block_size=128
head_size=128
waves_per_eu=1
num_warps=4
block_m=128
remove_indirect_access=0
export PRINT_IRS=1
# maxnum: v_max_num_f32_e32
# maximum: v_maximum -> nan propagating, hence no self max
# requires using: https://github.amd.com/GFX-IP-Arch/triton/tree/cagri/fma_fix
# for the pk_fma and maximum hacks
# 1 means chain reduction to produce max3/maximum3
export LLIR_TRANSFORM_MAX="maximum 1"
source $TRITON_GFX1250_MODEL_PATH/ffmlite_env.sh
python3 run_kernel.py --remove_indirect_access $remove_indirect_access --block_m $block_m --num_warps $num_warps --waves_per_eu $waves_per_eu --head_size $head_size  --block_size $block_size --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $bs --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l