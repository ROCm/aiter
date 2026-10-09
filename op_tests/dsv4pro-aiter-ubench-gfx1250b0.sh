#!/bin/bash
# DSV4-Pro kernel microbenchmarks on gfx1250 B0
# Run inside container: bash /datahome/dsv4pro-aiter-ubench-gfx1250b0.sh
# Single GPU (HIP_VISIBLE_DEVICES=0) except EP dispatch/combine (4 GPUs)
# Generates per-test logs + summary report

set -uo pipefail

AITER_DIR="${AITER_DIR:-$HOME/aiter}"
MORI_DIR="${MORI_DIR:-$HOME/mori}"
RUN_NUMBER="${RUN_NUMBER:-0}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
HOST=$(hostname)
LOGDIR=/datahome/dsv4-ubench/run${RUN_NUMBER}-${HOST}-${TIMESTAMP}
SUMMARY=$LOGDIR/SUMMARY.txt
mkdir -p $LOGDIR

{
echo "=== DSV4-Pro kernel ubench run $RUN_NUMBER ==="
echo "Date: $(date)"
echo "Host: $HOST"
echo "AITER_DIR: $AITER_DIR"
echo "MORI_DIR: $MORI_DIR"
echo "Log dir: $LOGDIR"
echo "---"

echo "=== GPU info ==="
AMDGPU_ARCH_BIN=$(which amdgpu-arch || find ${ROCM_PATH:-/opt/rocm} -name amdgpu-arch -type f | head -1)
GPU_ARCH=$($AMDGPU_ARCH_BIN | head -1)
NUM_GPUS=$($AMDGPU_ARCH_BIN | wc -l)
echo "GPU arch: $GPU_ARCH"
echo "GPU count: $NUM_GPUS"
echo ""

echo "=== Install system deps ==="
apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y libpci-dev libibverbs-dev
echo ""

echo "=== Clone and build aiter from main ==="
cd $HOME
git clone https://github.com/ROCm/aiter.git
cd aiter
git submodule update --init --recursive
AITER_USE_SYSTEM_TRITON=1 ENABLE_CK=0 GPU_ARCHS="gfx1250" python3 -m pip install -e .
cd /
python -c "import aiter; print(f\"aiter: {aiter.__file__}\")"
echo ""

echo "=== Clone and build mori from main ==="
cd $HOME
git clone https://github.com/ROCm/mori.git
cd mori
python3 -m pip install -U packaging setuptools pybind11 Cython
python3 -m pip install --no-build-isolation -e .
cd /
python -c "import mori; print(f\"mori: {mori.__file__}\")"
echo ""

echo "=== pip packages (key) ==="
pip list | grep -iE "aiter|triton|torch|flydsl|mori"
echo ""

cd $AITER_DIR
export PYTHONPATH=${MORI_DIR}/python:${MORI_DIR}:${AITER_DIR}${PYTHONPATH:+:$PYTHONPATH}
export ENABLE_CK=0
export AITER_FORCE_GFX1250=1

echo "=== Setup ==="
echo "PYTHONPATH=$PYTHONPATH"
echo "ENABLE_CK=$ENABLE_CK"
echo "AITER_FORCE_GFX1250=$AITER_FORCE_GFX1250"
echo ""

echo "=== Running 10 ubench suites ==="
echo ""

echo "# DSV4-Pro kernel ubench summary — run $RUN_NUMBER" > $SUMMARY
echo "# Host: $HOST" >> $SUMMARY
echo "# GPU: $GPU_ARCH x $NUM_GPUS" >> $SUMMARY
echo "# Date: $(date)" >> $SUMMARY
echo "" >> $SUMMARY

# 1. MoE prefill/decode (flydsl, a8w4) — single GPU
echo "=== [1/10] MoE prefill/decode (flydsl, a8w4) ==="
HIP_VISIBLE_DEVICES=0 \
AITER_MOE_EXPERT_BALANCE=true \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
  python op_tests/flydsl_tests/test_flydsl_grouped_gemm.py \
    --no-check-aot-cache \
    --scenario kernel \
    --data-format a8w4 \
    --experts 96 \
    --topk 6 \
    --model-dim 7168 \
    --inter-dim 3072 \
    --tokens 1 16 64 256 1024 4096 16384 \
  2>&1 | tee $LOGDIR/01_moe_flydsl_a8w4.log || true
echo "[1/10] exit=$?" >> $SUMMARY
echo ""

# 2. MoE decode (Gluon, a4w4) — single GPU
echo "=== [2/10] MoE decode (Gluon, a4w4) ==="
HIP_VISIBLE_DEVICES=0 \
  python op_tests/op_benchmarks/triton/bench_moe_gemm_a4w4_cudagraph.py \
    --backend gluon \
    --shape 2880 5760 \
    --experts 128 4 \
  2>&1 | tee $LOGDIR/02_moe_gluon_a4w4.log || true
echo "[2/10] exit=$?" >> $SUMMARY
echo ""

# 3. MLA-v4 sparse prefill (asm, fp8) — single GPU
echo "=== [3/10] MLA-v4 sparse prefill (asm, fp8) ==="
HIP_VISIBLE_DEVICES=0 \
  python op_tests/test_pa_sparse_prefill.py \
    --prec fp8 \
    --backend asm \
    --h_q 128 \
  2>&1 | tee $LOGDIR/03_mla_v4_sparse_prefill_asm.log || true
echo "[3/10] exit=$?" >> $SUMMARY
echo ""

# 4. MLA-v3 decode (asm, fp8) — single GPU
echo "=== [4/10] MLA-v3 decode (asm, fp8) ==="
HIP_VISIBLE_DEVICES=0 \
  python op_tests/test_mla_decode_pagesize64.py \
    -n 128,1 \
    -b 1 16 64 512 \
    -c 1024 8192 \
  2>&1 | tee $LOGDIR/04_mla_v3_decode_asm.log || true
echo "[4/10] exit=$?" >> $SUMMARY
echo ""

# 5. MLA-v4 decode (Gluon, bf16) — single GPU
echo "=== [5/10] MLA-v4 decode (Gluon, bf16) ==="
HIP_VISIBLE_DEVICES=0 \
  python op_tests/bench_gfx1250_combo.py \
    --dsv4 --ops mla_v4_decode \
  2>&1 | tee $LOGDIR/05_mla_v4_decode_gluon.log || true
echo "[5/10] exit=$?" >> $SUMMARY
echo ""

# 6. MQA indexer (Gluon, bf16) — single GPU
echo "=== [6/10] MQA indexer (Gluon, bf16) ==="
for kv in 384 4608 10240; do
  echo "--- kv_length=$kv ---"
  HIP_VISIBLE_DEVICES=0 \
    python op_tests/op_benchmarks/triton/bench_deepgemm_attention.py \
      --batch 512 \
      --heads 64 \
      --index_dim 128 \
      -kv_length $kv \
      -mtp 0 \
      --kv_preshuffle \
      --blocksize 64 || true
done 2>&1 | tee $LOGDIR/06_mqa_indexer_gluon.log
echo "[6/10] done" >> $SUMMARY
echo ""

# 7. HCA compressed attention (flydsl, bf16) — single GPU
echo "=== [7/10] HCA compressed attention (flydsl, bf16) ==="
HIP_VISIBLE_DEVICES=0 \
  python op_tests/test_flydsl_compress_attn.py \
    -s hca_main \
  2>&1 | tee $LOGDIR/07_hca_flydsl.log || true
echo "[7/10] exit=$?" >> $SUMMARY
echo ""

# 8. BMM bf16 (Gluon) — single GPU
echo "=== [8/10] BMM bf16 (Gluon) ==="
for b in 4 8 16; do
  for m in 1 16 64 256 1024 4096 16384; do
    HIP_VISIBLE_DEVICES=0 \
      python op_tests/op_benchmarks/triton/bench_batched_gemm_bf16.py \
        --shape $b $m 1024 4096 \
        --metric time || true
  done
done 2>&1 | tee $LOGDIR/08_bmm_gluon.log
echo "[8/10] done" >> $SUMMARY
echo ""

# 9. BMM a8w8 (flydsl, fp8 x fp8) — single GPU
echo "=== [9/10] BMM a8w8 (flydsl, fp8 x fp8) ==="
(
  failed=0
  for b in 4 8 16; do
    for m in 1 16 64 256 1024 4096 16384; do
      HIP_VISIBLE_DEVICES=0 \
        python op_tests/test_flydsl_batched_gemm.py \
          -b $b \
          -s $m,1024,4096 \
          -d bf16 \
          -l mbn
      case_exit=$?
      if [ "$case_exit" -ne 0 ]; then
        echo "FAIL B=$b M=$m exit=$case_exit"
        failed=1
      fi
    done
  done
  exit "$failed"
) 2>&1 | tee $LOGDIR/09_bmm_flydsl_a8w8.log
echo "[9/10] exit=${PIPESTATUS[0]}" >> $SUMMARY
echo ""

# 10. EP dispatch/combine (hip, 4 GPUs) — requires mori + torchrun
echo "=== [10/10] EP dispatch/combine (hip, 4 GPUs) ==="
if [ -d "$MORI_DIR" ]; then
  cd $MORI_DIR
  MORI_SOCKET_IFNAME=lo \
  GLOO_SOCKET_IFNAME=lo \
  BACKENDS=hip \
  MORI_V2_KERNEL_BACKEND=hip \
  HIDDEN=7168 \
  TOPK=6 \
  EPR=96 \
  SWEEP=64,128,256,512,1024,2048,4096,8192,16384 \
  ITERS=200 \
  WARMUP=10 \
  MODES=eager,graph \
  COMBINE_IN=inplace \
  CHECK=1 \
  DATA_INIT=norm \
  DISP=fp4 \
    torchrun --standalone --nproc_per_node=$NUM_GPUS \
      tests/python/ops/dispatch_combine_v2/bench_ep.py \
    2>&1 | tee $LOGDIR/10_ep_dispatch_combine.log || true
  echo "[10/10] exit=$?" >> $SUMMARY
  cd $AITER_DIR
else
  echo "SKIP: $MORI_DIR not found"
  echo "[10/10] SKIP" >> $SUMMARY
fi
echo ""

echo "" >> $SUMMARY
echo "# Completed: $(date)" >> $SUMMARY

echo ""
echo "=========================================="
echo "=== UBENCH SUMMARY ==="
echo "=========================================="
cat $SUMMARY
echo ""
echo "Per-test logs: $LOGDIR/"
echo "Summary: $SUMMARY"
echo "Completed: $(date)"
} 2>&1 | tee $LOGDIR/full_run.log
