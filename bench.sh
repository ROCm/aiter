ulimit -c 0 && \
MORI_SOCKET_IFNAME=lo \
MORI_GPU_ARCHS=gfx950 \
MORI_SHMEM_HEAP_SIZE=40G \
HSA_COREDUMP_PATTERN=/dev/null \
PYTHONPATH="$PWD:${PYTHONPATH:-}" \
timeout 30m \
torchrun \
  --standalone \
  --nproc-per-node=8 \
  op_tests/multigpu_tests/bench_mega_moe_v2.py \
  --tokens 64 \
  --mtpr 64 \
  --iters 30 \
  --route uniform \
  --tp