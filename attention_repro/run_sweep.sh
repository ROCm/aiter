#!/bin/bash
set -u

seq_q_l=256
seq_kv_l=256
num_heads_q=8
num_heads_k=1
bs=1
window_size=0
block_size=128
waves_per_eu=1
num_warps=4
block_m=128
num_buffers=3

export PRINT_IRS=1
source $TRITON_GFX1250_MODEL_PATH/ffmlite_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# # Stash any stray IRs from prior runs so they don't get miscategorized
# shopt -s nullglob
# if compgen -G "unif_attention_2d_*.txt" >/dev/null || compgen -G "unified_attention_2d_*.txt" >/dev/null; then
#   mkdir -p _prev_irs
#   mv unif_attention_2d_*.txt _prev_irs/ 2>/dev/null || true
#   mv unified_attention_2d_*.txt _prev_irs/ 2>/dev/null || true
# fi

LOG_FILE="sweep.log"
: > "$LOG_FILE"

total=0
for head_size in 64 128; do
  for loop_variant in 3; do
    for causal in 0 1; do
      for remove_indirect_access in 0 1; do
        for ds_wait in none; do
          total=$((total+1))
          out_dir="new_head_size_${head_size}/loop_${loop_variant}/causal_${causal}/rm_pg_${remove_indirect_access}/ds_wait_rm_${ds_wait}"
          mkdir -p "$out_dir"

          export LLIR_REMOVE_DS_WAIT_0="$ds_wait"
          rm -rf ~/.triton/cache/

          echo "=== [$total/72] head_size=$head_size loop=$loop_variant causal=$causal rm_pg=$remove_indirect_access ds_wait=$ds_wait ===" | tee -a "$LOG_FILE"

          python3 run_kernel.py \
            --causal $causal \
            --loop_variant $loop_variant \
            --num_buffers $num_buffers \
            --remove_indirect_access $remove_indirect_access \
            --block_m $block_m \
            --num_warps $num_warps \
            --waves_per_eu $waves_per_eu \
            --head_size $head_size \
            --block_size $block_size \
            --num_heads_q $num_heads_q \
            --num_heads_k $num_heads_k \
            --bs $bs \
            --seq_q_l $seq_q_l \
            --seq_kv_l $seq_kv_l >>"$LOG_FILE" 2>&1

          # Move all produced IRs into the target folder
          moved=0
          if compgen -G "unif_attention_2d_*.txt" >/dev/null; then
            mv unif_attention_2d_*.txt "$out_dir/"
            moved=1
          fi
          if compgen -G "unified_attention_2d_*.txt" >/dev/null; then
            mv unified_attention_2d_*.txt "$out_dir/"
            moved=1
          fi
          if [ $moved -eq 1 ]; then
            echo "    -> moved IRs to $out_dir" | tee -a "$LOG_FILE"
          else
            echo "    !! no IRs produced for this combo" | tee -a "$LOG_FILE"
          fi
        done
      done
    done
  done
done

echo "Sweep done: $total combinations. Log: $LOG_FILE"
