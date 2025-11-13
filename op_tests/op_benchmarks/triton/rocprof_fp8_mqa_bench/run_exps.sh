#!/usr/bin/env bash
set -euo pipefail

# Base multiplier for "K" values
k=1024

# Convert values like "1K", "2K", "24K" to integers using k
parse_k() {
  local v="$1"
  if [[ "$v" =~ ^([0-9]+)K$ ]]; then
    echo $(( ${BASH_REMATCH[1]} * k ))
  else
    echo "$v"
  fi
}

# List of q_len and kv_len pairs (using K-suffixed values)
pairs=(
  "1K 2K"
  "2K 2K"
  "1K 4K"
  "2K 4K"
  "3K 4K"
  "4K 4K"
  "1K 8K"
  "2K 8K"
  "3K 8K"
  "4K 8K"
  "8K 8K"
  "2K 16K"
  "3K 16K"
  "4K 16K"
  "8K 16K"
  "2K 32K"
  "3K 32K"
  "4K 32K"
  "2K 64K"
  "3K 64K"
  "8K 64K"
  "8K 56K"
  "8K 48K"
  "8K 40K"
  "8K 32K"
  "8K 24K"
  "8K 16K"
)

path="bench_out"

num_heads_q=64
head_dim=128
repeat=20
for pair in "${pairs[@]}"; do
    read -r q_len kv_len <<< "$pair"

    seq_q_l=$(parse_k "$q_len")
    seq_kv_l=$(parse_k "$kv_len")
    echo $seq_q_l $seq_kv_l

    rocprofv2 -o results python3 run_kernel.py --head_dim $head_dim --num_heads_q $num_heads_q --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --repeat $repeat
    python3 collect_results.py --head_dim $head_dim --num_heads_q $num_heads_q --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --path $path --repeat $repeat --kernel_names fp8_mqa_logits_kernel
done