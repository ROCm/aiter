
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
waves_per_eu=$WAVES_PER_EU
num_warps=$NUM_WARPS
tile_size=$TILE_SIZE
q_fp8=$Q_FP8
kv_fp8=$KV_FP8
block_m=$BLOCK_M
causal=$CAUSAL
remove_indirect=$REMOVE_INDIRECT
num_buffer=$NUM_BUFFER
loop_variant=$loop_variant
shuffled_kv_cache=$SHUFFLED_KV_CACHE
num_splits=$NUM_SPLITS
repeat=200
python3 run_kernel.py --shuffled_kv_cache $shuffled_kv_cache --loop_variant $loop_variant --num_buffers $num_buffer --remove_indirect_access $remove_indirect  --causal $causal --block_m $block_m --q_fp8 $q_fp8 --kv_fp8 $kv_fp8 --num_warps $num_warps --waves_per_eu $waves_per_eu --tile_size $tile_size --window_size $window_size --head_size $head_size --prefill_cnt $prefill_cnt --decode_cnt $decode_cnt  --block_size $block_size --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $bs --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --repeat $repeat --use_tdm $use_tdm --num_splits $num_splits
