
path="test_run"
seq_q_l=$SEQ_Q_L
seq_kv_l=$SEQ_KV_L
num_heads_q=$NUM_HEADS_Q
num_heads_k=$NUM_HEADS_K
bs=$BS
prefill_cnt=-1
decode_cnt=-1
window_size=$WINDOW_SIZE
block_size=$BLOCK_SIZE
head_size=$HEAD_SIZE
use_tdm=$USE_TDM
num_kv_blocks=$NUM_KV_BLOCKS
waves_per_eu=$WAVES_PER_EU
num_warps=$NUM_WARPS
q_fp8=$Q_FP8
kv_fp8=$KV_FP8
block_m=$BLOCK_M
repeat=1
python3 run_kernel.py --block_m $block_m --q_fp8 $q_fp8 --kv_fp8 $kv_fp8 --num_warps $num_warps --waves_per_eu $waves_per_eu --window_size $window_size --head_size $head_size --prefill_cnt $prefill_cnt --decode_cnt $decode_cnt  --block_size $block_size --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $bs --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --path $path --repeat $repeat --use_tdm $use_tdm --num_kv_blocks $num_kv_blocks

# rocprofv3  --kernel-trace --output-format csv -o results -- python run_kernel_realistic.py --window_size $window_size --head_size $head_size --prefill_cnt $prefill_cnt --decode_cnt $decode_cnt  --block_size $block_size --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $bs --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --path $path --repeat $repeat
# python collect_results.py --window_size $window_size --head_size $head_size --prefill_cnt $prefill_cnt --decode_cnt $decode_cnt --block_size $block_size --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $bs --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --path $path --repeat 500 --kernel_names kernel_unified_attention_3d reduce_segments kernel_unified_attention_2d
