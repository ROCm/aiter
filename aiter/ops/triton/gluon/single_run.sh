
path="test_run"
seq_q_l=512
seq_kv_l=512
num_heads_q=64
num_heads_k=8
bs=1
prefill_cnt=-1
decode_cnt=-1
window_size=0
block_size=64
head_size=64
repeat=200
export HIP_VISIBLE_DEVICES=0
export TRITON_HIP_USE_ASYNC_COPY=1
export PRINT_IRS=1
python3 run_kernel_realistic_gfx_1250.py --window_size $window_size --head_size $head_size --prefill_cnt $prefill_cnt --decode_cnt $decode_cnt  --block_size $block_size --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $bs --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --path $path --repeat $repeat

# rocprofv3  --kernel-trace --output-format csv -o results -- python run_kernel_realistic.py --window_size $window_size --head_size $head_size --prefill_cnt $prefill_cnt --decode_cnt $decode_cnt  --block_size $block_size --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $bs --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --path $path --repeat $repeat
# python collect_results.py --window_size $window_size --head_size $head_size --prefill_cnt $prefill_cnt --decode_cnt $decode_cnt --block_size $block_size --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $bs --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --path $path --repeat 500 --kernel_names kernel_unified_attention_3d reduce_segments kernel_unified_attention_2d
