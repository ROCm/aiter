"""Minimal single-dispatch driver for ATT profiling of mhc_fused_post_pre_gemm_sqrsum.

No reference, no checkAllclose, no perf loop -- just build the tensors and issue the
fused kernel N times (default 1), so rocprofv3 -i inputs.json has exactly the dispatches
it wants to trace and nothing else. Usage:

    ROCR_VISIBLE_DEVICES=1 rocprofv3 -i ../inputs.json -d ./att-logs -- \
        python3 op_tests/mhc_att_min.py -m 2048 -n 7168
"""

import argparse

import torch

import aiter
from aiter import dtypes

p = argparse.ArgumentParser()
p.add_argument("-m", type=int, default=2048)
p.add_argument("-n", "--hidden_size", type=int, default=7168)
p.add_argument("--hc_mult", type=int, default=4)
p.add_argument("--iters", type=int, default=1)
p.add_argument("--res_w_preshuffle_bf16", action="store_true", default=True)
p.add_argument(
    "--no_res_w_preshuffle_bf16", dest="res_w_preshuffle_bf16", action="store_false"
)
p.add_argument("--fuse_rmsnorm", action="store_true", default=True)
p.add_argument("--warmup", type=int, default=200, help="--bench: warm-up dispatches before timing")
p.add_argument(
    "--bench",
    action="store_true",
    help="Time mhc_fused_post_pre with run_perftest and print 'BENCH us=<x>' instead of "
    "issuing --iters plain dispatches. Skips the torch reference and every checkAllclose, "
    "which is what makes test_mhc.py ~50s per case; use it for A/B once correctness is "
    "already established elsewhere.",
)
a = p.parse_args()

m, hidden_size, hc_mult = a.m, a.hidden_size, a.hc_mult
hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
hc_hidden_size = hc_mult * hidden_size

torch.set_default_device("cuda")
layer_input = torch.randn(m, hidden_size, dtype=dtypes.bf16)
residual_in = torch.randn(m, hc_mult, hidden_size, dtype=dtypes.bf16)
post_layer_mix = torch.randn(m, hc_mult, 1, dtype=dtypes.fp32)
comb_res_mix = torch.randn(m, hc_mult, hc_mult, dtype=dtypes.fp32)
fn = torch.randn(hc_mult3, hc_hidden_size, dtype=dtypes.fp32)
hc_scale = torch.randn((3,), dtype=dtypes.fp32) * 0.1
hc_base = torch.randn((hc_mult3,), dtype=dtypes.fp32) * 0.1
norm_weight = torch.randn(hidden_size, dtype=dtypes.bf16) if a.fuse_rmsnorm else None

kwargs = dict(
    rms_eps=1e-6,
    hc_pre_eps=1e-6,
    hc_sinkhorn_eps=1e-6,
    hc_post_mult_value=2.0,
    sinkhorn_repeat=20,
)
if a.fuse_rmsnorm:
    kwargs["norm_weight"] = norm_weight
    kwargs["norm_eps"] = 1e-6

pack_flag = 1 if a.res_w_preshuffle_bf16 else 0
if pack_flag:
    from aiter.ops.mhc import MHC_RES_SHUFFLE, mhc_pre_convert_fn, mhc_res_shuffle

    fn_gemm = torch.empty(hc_mult3, hc_hidden_size, dtype=torch.int32)
    mhc_pre_convert_fn(fn_gemm, fn)
    # The flag also switches residual_in/next_residual to the pre-shuffled layout.
    if MHC_RES_SHUFFLE:
        residual_in = mhc_res_shuffle(residual_in)
else:
    fn_gemm = fn

call = dict(
    force_fused=True,
    is_res_w_preshuffle_bf16=pack_flag,
    **kwargs,
)
args_pos = (
    layer_input,
    residual_in,
    post_layer_mix,
    comb_res_mix,
    fn_gemm,
    hc_scale,
    hc_base,
)

if a.bench:
    from aiter.test_common import run_perftest

    # Cold-start clocks make a bare run_perftest ~4x noisier than the same timing
    # taken inside test_mhc.py (which happens to warm the GPU with the reference
    # computation first): measured +-1.6% vs +-0.4% across repeats of one binary.
    # Spin first so the A/B compares steady-state clocks.
    for _ in range(a.warmup):
        aiter.mhc_fused_post_pre(*args_pos, **call)
    torch.cuda.synchronize()

    _, us = run_perftest(aiter.mhc_fused_post_pre, *args_pos, **call)
    print(f"BENCH us={us:.4f} m={m} hidden={hidden_size} preshuffle={pack_flag}")
else:
    for _ in range(a.iters):
        aiter.mhc_fused_post_pre(*args_pos, **call)
    torch.cuda.synchronize()
    print(f"done: m={m} hidden={hidden_size} preshuffle={pack_flag} iters={a.iters}")
