
seq_q_l=512
seq_kv_l=512
num_heads_q=8
num_heads_k=1
bs=1
prefill_cnt=-1
decode_cnt=-1
window_size=0
block_size=64
head_size=64
use_tdm=1
num_kv_blocks=1
waves_per_eu=1
num_warps=4
q_fp8=0
kv_fp8=0
block_m=128
repeat=1
export PRINT_IRS=1
python3 run_kernel.py --block_m $block_m --q_fp8 $q_fp8 --kv_fp8 $kv_fp8 --num_warps $num_warps --waves_per_eu $waves_per_eu --window_size $window_size --head_size $head_size --prefill_cnt $prefill_cnt --decode_cnt $decode_cnt  --block_size $block_size --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $bs --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --repeat $repeat --use_tdm $use_tdm --num_kv_blocks $num_kv_blocks