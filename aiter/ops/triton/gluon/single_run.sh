#export AMDGCN_USE_BUFFER_OPS=1

path="9_23_modif_vllm_attn_block_64_rebased"
num_heads_q=64
num_heads_k=8
bs=128
block_size=64
window_size=0
prefill_cnt=-1
decode_cnt=-1
head_size=64
seq_q_l=1
seq_kv_l=4096
head_pairs=("1 8" "8 64")
# for window_size in 0 128; do
#     for pair in "${head_pairs[@]}"; do
#         for bs in 1 4 8 16 32 64 128; do
#             read num_heads_k num_heads_q <<< "$pair"
#             rocprofv2 -o results python run_kernel_realistic.py --window_size $window_size --head_size $head_size  --prefill_cnt $prefill_cnt --decode_cnt $decode_cnt  --block_size $block_size --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $bs --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --path $path --repeat 500
#             python collect_results.py --block_size $block_size --window_size $window_size --head_size $head_size  --prefill_cnt $prefill_cnt --decode_cnt $decode_cnt --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $bs --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --path $path --repeat 500 --kernel_names kernel_unified_attention_3d reduce_segments kernel_unified_attention_2d
#         done
#     done
# done

# seq_q_l=4096
# seq_kv_l=4096
# for window_size in 0 128; do
#     for pair in "${head_pairs[@]}"; do
#         for bs in 1 2; do
#             read num_heads_k num_heads_q <<< "$pair"
#             rocprofv2 -o results python run_kernel_realistic.py --window_size $window_size --prefill_cnt $prefill_cnt --decode_cnt $decode_cnt  --block_size $block_size --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $bs --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --path $path --repeat 500
#             python collect_results.py --block_size $block_size --window_size $window_size --prefill_cnt $prefill_cnt --decode_cnt $decode_cnt --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $bs --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --path $path --repeat 500 --kernel_names kernel_unified_attention_3d reduce_segments kernel_unified_attention_2d
#         done
#     done
# done

# seq_q_l=1
# seq_kv_l=1024
# head_pairs=("1 8" "8 64")
# for window_size in 0 128; do
#     for pair in "${head_pairs[@]}"; do
#         for bs in 1 4 8 16 32 64 128; do
#             read num_heads_k num_heads_q <<< "$pair"
#             rocprofv2 -o results python run_kernel_realistic.py --window_size $window_size --head_size $head_size  --prefill_cnt $prefill_cnt --decode_cnt $decode_cnt  --block_size $block_size --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $bs --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --path $path --repeat 500
#             python collect_results.py --block_size $block_size --window_size $window_size --head_size $head_size  --prefill_cnt $prefill_cnt --decode_cnt $decode_cnt --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $bs --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --path $path --repeat 500 --kernel_names kernel_unified_attention_3d reduce_segments kernel_unified_attention_2d
#         done
#     done
# done

# seq_q_l=1024
# seq_kv_l=1024
# for window_size in 0 128; do
#     for pair in "${head_pairs[@]}"; do
#         for bs in 1 2 4 8; do
#             read num_heads_k num_heads_q <<< "$pair"
#             rocprofv2 -o results python run_kernel_realistic.py --window_size $window_size --prefill_cnt $prefill_cnt --decode_cnt $decode_cnt  --block_size $block_size --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $bs --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --path $path --repeat 500
#             python collect_results.py --block_size $block_size --window_size $window_size --prefill_cnt $prefill_cnt --decode_cnt $decode_cnt --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $bs --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --path $path --repeat 500 --kernel_names kernel_unified_attention_3d reduce_segments kernel_unified_attention_2d
#         done
#     done
# done


path="test_run"
seq_q_l=8192
seq_kv_l=8192
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
#python run_kernel_realistic.py --window_size $window_size --head_size $head_size --prefill_cnt $prefill_cnt --decode_cnt $decode_cnt  --block_size $block_size --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $bs --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --path $path --repeat $repeat

rocprofv3  --kernel-trace --output-format csv -o results -- python run_kernel_realistic.py --window_size $window_size --head_size $head_size --prefill_cnt $prefill_cnt --decode_cnt $decode_cnt  --block_size $block_size --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $bs --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --path $path --repeat $repeat
python collect_results.py --window_size $window_size --head_size $head_size --prefill_cnt $prefill_cnt --decode_cnt $decode_cnt --block_size $block_size --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $bs --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --path $path --repeat 500 --kernel_names kernel_unified_attention_3d reduce_segments kernel_unified_attention_2d
