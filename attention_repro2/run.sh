
seq_q_l=128
seq_kv_l=128
num_heads_q=8
num_heads_k=1
bs=1
prefill_cnt=-1
decode_cnt=-1
window_size=0
block_size=128
tile_size=128
head_size=64
waves_per_eu=1
num_warps=4
q_fp8=0
kv_fp8=0
block_m=128
causal=1
remove_indirect=0
loop_variant=2
num_buffer=-1
shuffled_kv_cache=1
num_splits=1
repeat=0
export PRINT_IRS=1
python3 run_kernel.py --shuffled_kv_cache $shuffled_kv_cache --loop_variant $loop_variant --num_buffers $num_buffer --remove_indirect_access $remove_indirect  --causal $causal --block_m $block_m --q_fp8 $q_fp8 --kv_fp8 $kv_fp8 --num_warps $num_warps --waves_per_eu $waves_per_eu --tile_size $tile_size --window_size $window_size --head_size $head_size --prefill_cnt $prefill_cnt --decode_cnt $decode_cnt  --block_size $block_size --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $bs --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --repeat $repeat --num_splits $num_splits

# rocprofv3  --kernel-trace --output-format csv -o results -- python run_kernel_realistic.py --window_size $window_size --head_size $head_size --prefill_cnt $prefill_cnt --decode_cnt $decode_cnt  --block_size $block_size --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $bs --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --path $path --repeat $repeat
# python collect_results.py --window_size $window_size --head_size $head_size --prefill_cnt $prefill_cnt --decode_cnt $decode_cnt --block_size $block_size --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $bs --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --path $path --repeat 500 --kernel_names kernel_unified_attention_3d reduce_segments kernel_unified_attention_2d
