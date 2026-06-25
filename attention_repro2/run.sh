
seq_q_l=1
seq_kv_l=8192
num_heads_q=64
num_heads_k=8
bs=128
prefill_cnt=-1
decode_cnt=-1
window_size=0
block_size=64
tile_size=-1
head_size=64
waves_per_eu=-1
num_warps=-1
q_fp8=1
kv_fp8=1
block_m=-1
causal=1
remove_indirect=0
loop_variant=-1
num_buffer=-1
shuffled_kv_cache=1
num_splits=2
repeat=500
warmup=50
export PRINT_IRS=1
export USE_3D=0
#export AITER_TRITON_ONLY=1
python3 run_kernel.py --shuffled_kv_cache $shuffled_kv_cache --loop_variant $loop_variant --num_buffers $num_buffer --remove_indirect_access $remove_indirect  --causal $causal --block_m $block_m --q_fp8 $q_fp8 --kv_fp8 $kv_fp8 --num_warps $num_warps --waves_per_eu $waves_per_eu --tile_size $tile_size --window_size $window_size --head_size $head_size --prefill_cnt $prefill_cnt --decode_cnt $decode_cnt  --block_size $block_size --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $bs --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --repeat $repeat --num_splits $num_splits

#rocprofv3 --att -i att_input.json -d my_trace_bs_128_64_8_8k_decode_fp8_loop_2 -- python run_kernel.py --window_size $window_size --head_size $head_size --prefill_cnt $prefill_cnt --decode_cnt $decode_cnt  --block_size $block_size --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $bs --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --path $path --repeat $repeat
# python collect_results.py --window_size $window_size --head_size $head_size --prefill_cnt $prefill_cnt --decode_cnt $decode_cnt --block_size $block_size --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $bs --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --path $path --repeat 500 --kernel_names kernel_unified_attention_3d reduce_segments kernel_unified_attention_2d
