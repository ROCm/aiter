# gfx1250 MegaMoE validation

Run from `/home/yashao/aiter` in a gfx1250 container with matching AITER,
FlyDSL, and Mori builds:

```bash
export ENABLE_CK=0
export AITER_FORCE_A8W4=1
export AITER_USE_GROUPED_GEMM=1
export AITER_BF16_FP8_MOE_BOUND=0

for mode in gather scatter scatter_fused; do
  torchrun --standalone --nproc_per_node=4 \
    op_tests/multigpu_tests/test_mega_moe.py \
    -q a8w4_mxfp4 -e 384 -k 6 -hd 7168 -id 3072 \
    -bs 128 --layers 1 --combine "${mode}" --acc_verify 1
done
```

Exercise fused-combine token buckets:

```bash
for tokens in 8 32 128 512 2048; do
  torchrun --standalone --nproc_per_node=4 \
    op_tests/multigpu_tests/test_mega_moe.py \
    -q a8w4_mxfp4 -e 384 -k 6 -hd 7168 -id 3072 \
    -bs "${tokens}" --layers 1 --combine scatter_fused --acc_verify 1
done
```

Run the multi-layer CUDA graph and collect kernel timings:

```bash
torchrun --standalone --nproc_per_node=4 \
  op_tests/multigpu_tests/test_mega_moe.py \
  -q a8w4_mxfp4 -e 384 -k 6 -hd 7168 -id 3072 \
  -bs 128 --layers 61 --combine scatter_fused \
  --acc_verify 1 --profile_table 1
```

Record `MEGA-CHECK`, per-layer time, and the profiler table for each run. This
workspace's available GPU is gfx942 and its installed Triton/FlyDSL extensions
do not match the checkout, so only syntax, Ruff, and import-boundary checks can
be completed locally.
