# FlyDSL `fused_moe_allreduce` (GLM-5 W8A8 MoE + TP all-reduce, gfx950)

This directory contains a FlyDSL monokernel for the whole GLM-5 decode MoE layer on one
tensor-parallel rank, including the TP all-reduce. It is a drop-in counterpart of TileRT's
`fused_moe_allreduce_w8a8_v4` (`glm5_fused_moe_allreduce_w8a8_v4_op`). It takes the same
argument order, the same packed weight layouts and the same outputs.

| file | contents |
|---|---|
| `fused_moe_allreduce_w8a8.py` | the kernel (`compile_fused_moe_allreduce(samples, proto)`) |
| `../../fused_moe_allreduce.py` | host side: weight packers, `fused_moe_allreduce_w8a8(...)` (TileRT op signature), `FusedMoeAllreduceW8A8` module |
| `op_tests/multigpu_tests/test_flydsl_fuse_moe_allreduce.py` | accuracy + benchmark against a torch golden and the TileRT code object; `-q 8` = A8W8, `-q 4` = A4W4 |

## Shapes

- hidden 6144; 256 routed experts plus 1 shared expert; top-8 routing.
- TP8: every rank holds a 256-wide intermediate shard of each expert.
- Weights are FP8 E4M3 with 128x128 block scales. Activations and intermediates are
  dynamically quantized to FP8 per 128 elements.
- Each launch takes S = 1, 2 or 4 tokens (decode). The grid is always 256 workgroups x 512 threads,
  with one workgroup per CU, all resident at the same time.

## Fused ops

One launch replaces the following chain, which is normally 7-8 kernels plus a collective:

| # | op | details |
|---|---|---|
| 1 | RMSNorm | `norm = bf16((gamma * x) * rsqrt(mean(x^2) + 1e-5))` |
| 2 | Router GEMV | `logits = norm @ Wr^T`, Wr `[256, 6144]` bf16 |
| 3 | Routing / top-k | `sigmoid(logits) + bias` picks the top-8 experts (ties go to the lower id); `probs = score * 2.5 / sum(top-8 scores)` |
| 4 | Activation quant | `norm` to FP8, per-128 scale `max(amax, 1e-4) / 448` |
| 5 | Up/Gate GEMM (W8A8) | 9 slots (slot 0 = shared expert, slots 1..8 = routed), each `[512, 6144]` FP8; `mfma_scale_f32_16x16x128_f8f6f4` with block dequant |
| 6 | SiLU(gate) * up | gives `hidden_mid` bf16 `[S, 9, 256]` |
| 7 | Mid quant | per-128 FP8 |
| 8 | Down GEMM (W8A8) | `[6144, 256]` FP8 per slot; slot outputs are weighted by `probs` (shared expert weight = 1), summed, and rounded to a bf16 partial |
| 9 | TP all-reduce + residual | peer partials are summed in fp32 in rank order, then `+ residual`, giving `out` bf16 |

Outputs (same as TileRT): `norm_hidden`, `scores` (the raw router logits), `probs`, `indices`,
`hidden_mid`, `out`.

## Performance vs TileRT

Setup: 8x MI355X (gfx950), TP8, proto 0, TileRT KI=8. One process drives all 8 GPUs with
peer access, which is how TileRT is deployed. Timing uses CUDA graphs with all ranks replaying
together, and each number is the best of several rounds.

Both kernels are decode kernels limited to S <= 4 tokens per launch. So a batch of T tokens runs
as `ceil(T/4)` back-to-back S=4 launches (T < 4 runs as a single S=T launch). The same schedule is
used for both kernels, inside one graph.

### Token sweep (a fresh graph each round; each launch works on its own slice of T distinct tokens)

| tokens | launches | FlyDSL (us) | TileRT (us) | FlyDSL / TileRT | FlyDSL us/token |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 17.58 | 17.45 | 1.008 | 17.58 |
| 2 | 1 | 23.91 | 23.35 | 1.024 | 11.96 |
| 4 | 1 | 37.35 | 36.28 | 1.030 | 9.34 |
| 16 | 4 | 151.56 | 149.41 | 1.014 | 9.47 |
| 64 | 16 | 614.57 | 610.36 | 1.007 | 9.60 |
| 256 | 64 | 2480.03 | 2450.56 | 1.012 | 9.69 |
| 1024 | 256 | 9774.27 | 9662.23 | 1.012 | 9.55 |
| 2048 | 512 | 19086.82 | 18945.74 | 1.007 | 9.32 |

### Single launch, unit-test harness (`test_flydsl_fuse_moe_allreduce.py`, graph of 50 launches on the same tokens)

| tokens (S) | FlyDSL (us) | TileRT (us) | FlyDSL / TileRT |
|---:|---:|---:|---:|
| 1 | 16.22 | 16.19 | 1.002 |
| 2 | 22.20 | 22.60 | 0.982 |
| 4 | 35.72 | 36.03 | 0.991 |

Summary: FlyDSL is within about ±3% of TileRT at every size. Run-to-run noise on this machine is
about 3-4%, and one harness puts FlyDSL slightly ahead while the other puts it slightly behind.
Throughput levels off at about 9.3-9.6 us per token once there are 4 or more tokens per launch.
At that point the layer is bound by weight streaming: every token reads the router plus 9 experts'
shards, about 45 MB per token (about 180 MB per S=4 launch), which comes to about 5 TB/s.

### Accuracy

- Compared with an fp32 torch golden (TP8, rtol = atol = 2e-2), `out` has rel-L2 of 2.1e-3 to
  2.5e-3. TileRT measures 3.2e-3 at S=1.
- `norm_hidden` and `indices` match exactly. `scores` and `probs` have rel-L2 below 1e-5.
- On a single GPU, `out` and `hidden_mid` are bit-identical to TileRT. With 8 GPUs they differ by
  at most 1 bf16 ulp.

## A4W4 variant (`fused_moe_allreduce_a4w4.py`)

This variant has the same fusion, arguments and outputs as W8A8, but uses MXFP4 for the expert
weights and activations:

- **Format:** fp4 e2m1 values packed two per byte, with one e8m0 scale per 32 elements along K.
  The scale is rounded up (`ceil_pow2(amax / 6)`), which is aiter's `MX_DEFAULT_ROUND_MODE`.
- **Quantized in the kernel:** the normed activations and the SiLU mids, per 32 elements.
- **Unchanged:** the router GEMV stays in bf16.
- **Scales in hardware:** `mfma_scale_f32_16x16x128_f8f6f4` runs in fp4 mode and applies the
  per-32 scales itself, so there is no separate block-dequant multiply.
- **Traffic:** every token streams about 25 MB instead of about 45 MB.
- **Host API:** `aiter.ops.flydsl.fused_moe_allreduce_a4w4`
  - `FusedMoeAllreduceA4W4`, with the same interface as `FusedMoeAllreduceW8A8`;
  - `fused_moe_allreduce_a4w4(...)`;
  - the packers `pack_up_gate_a4w4` and `pack_down_a4w4`.
- **Logical weights:**
  - `ug_w [257, 512, 3072]` u8 with `ug_scales [257, 512, 192]` e8m0;
  - `down_w [257, 6144, 128]` u8 with `down_scales [257, 6144, 8]` e8m0.

Results from `op_tests/multigpu_tests/test_flydsl_fuse_moe_allreduce.py -q 4 8` (the `perf (us)` table it
prints): 8x MI355X, TP8, CUDA graph, TileRT KI=8. Both FlyDSL variants are built from the same master
weights.

| tokens (S) | proto | TileRT A8W8 (us) | FlyDSL A8W8 (us) | FlyDSL A4W4 (us) | TileRT / FlyDSL A8W8 | FlyDSL A8W8 / A4W4 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 16.19 | 16.33 | 14.43 | 0.99 | 1.13 |
| 1 | 1 | 17.02 | 17.15 | 14.59 | 0.99 | 1.18 |
| 2 | 0 | 22.48 | 22.47 | 16.35 | 1.00 | 1.37 |
| 2 | 1 | 22.60 | 22.61 | 16.51 | 1.00 | 1.37 |
| 4 | 0 | 34.71 | 35.56 | 22.87 | 0.98 | 1.55 |
| 4 | 1 | 34.97 | 35.08 | 22.64 | 1.00 | 1.55 |

A ratio above 1 means the kernel after the slash is faster. Run-to-run noise on this machine is about 3-4% (an earlier
run measured A4W4 at 13.2 us for S=1).

Accuracy:

- Against an fp32 golden that reproduces the MXFP4 quantization, `out` rel-L2 is at most 5e-5 and
  `hidden_mid` is exact.
- Against the unquantized bf16 model, the MoE output rel-L2 is about 0.30 for A4W4 and about 0.06
  for A8W8 (on random Gaussian weights). This is the precision cost of 4-bit activations and
  weights.

## Usage

```python
from aiter.ops.flydsl.fused_moe_allreduce import FusedMoeAllreduceW8A8

moes = FusedMoeAllreduceW8A8.peer_group([torch.device("cuda", i) for i in range(8)])
for moe, w in zip(moes, per_rank_weights):   # TP8 shards, see the host module docstring
    moe.load_weights(**w)                      # router_w, gamma, bias, ug_w, ug_scales, down_w, down_scales
norm, scores, mid, probs, indices, out = moes[r](hidden, residual)   # hidden: [S, 6144] bf16, S in {1, 2, 4}
```

Under `torchrun`, construct `FusedMoeAllreduceW8A8(rank=r, world_size=8)` once per process. It
exchanges an uncached symmetric buffer over HIP IPC.

```bash
# accuracy + perf vs TileRT (single process, all GPUs)
python op_tests/multigpu_tests/test_flydsl_fuse_moe_allreduce.py -s 1 2 4 --proto 0 1
# A4W4 and A8W8 side by side
python op_tests/multigpu_tests/test_flydsl_fuse_moe_allreduce.py -q 4 8
# accuracy only
python op_tests/multigpu_tests/test_flydsl_fuse_moe_allreduce.py --no-perf
```

Notes for this platform:
- Use gloo process groups; RCCL collectives hang on the test box.
- Multi-process HIP IPC exchange is much slower than single-process peer access, for both
  kernels.
- Clear `~/.flydsl/cache` after editing the kernel.
