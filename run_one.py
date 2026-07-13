"""Time the batched a8w4 TDM kernel at an explicit tile_m (for 256x256x256 tuning).
Usage: run256.py tile_m tile_n tile_k nb mw nw [ref]
"""
import sys
import torch
import aiter
from aiter.utility import fp4_utils
from aiter.ops.shuffle import shuffle_weight_gfx1250, shuffle_scale_n32k4
from aiter.ops.flydsl.kernels.mxfp4_preshuffle_gfx1250_tdm import launch_gemm_a8w4_tdm
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

torch.set_default_device("cuda")
SG = 32
B, M, N, K = 1, 8192, 8192, 8192
FLOP = 2.0 * B * M * N * K
tflops = lambda us: FLOP / (us * 1e-6) / 1e12


def pad32(s):
    r = s.shape[0]
    return s if r % 32 == 0 else torch.cat([s, torch.zeros((32 - r % 32, s.shape[1]), dtype=s.dtype)], 0)


def time_graph(fn, iters=50):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(5):
        g.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000.0 / (5 * iters)


tm, tn, tk, nb, mw, nw = (int(v) for v in sys.argv[1:7])
want_ref = "ref" in sys.argv[7:]
torch.manual_seed(5)
qf = aiter.get_triton_quant(aiter.QuantType.per_1x32)
c0, s0 = qf(torch.randn(N, K, dtype=torch.bfloat16), shuffle=False)
w = c0.view(torch.uint8)[None]; w_scales = s0.view(torch.uint8)[None]
x = (torch.randn(B, M, K) * 4.0).to(torch.float8_e4m3fn).view(torch.uint8)
x_scales = torch.randint(124, 130, (B, M, K // SG), dtype=torch.uint8)
w_sh = shuffle_weight_gfx1250(w[0].contiguous()).reshape(B * (N // 16), (K // 2) * 16)
sb = shuffle_scale_n32k4(pad32(w_scales[0]).unsqueeze(0), experts_cnt=1).view(torch.uint8).reshape(-1)
sa = shuffle_scale_n32k4(pad32(x_scales[0]).unsqueeze(0), experts_cnt=1).view(torch.uint8).reshape(-1)
a = x.contiguous()
out = torch.empty((B, M, N), dtype=torch.bfloat16)
wtm, wtn = tm // mw, tn // nw


def run():
    launch_gemm_a8w4_tdm(out, ptr_arg(a), ptr_arg(w_sh), sa.view(torch.int32),
                         sb.view(torch.int32), M, torch.cuda.current_stream(),
                         N, K, tm, tn, tk, mw, nw, wtm, wtn, 0, B, 0, nb, 0, 0)


mm = -1.0
if want_ref:
    run(); torch.cuda.synchronize()
    x_f32 = x.view(torch.float8_e4m3fn).to(torch.float32)
    xs = fp4_utils.e8m0_to_f32(x_scales.repeat_interleave(SG, -1))
    ws = fp4_utils.e8m0_to_f32(w_scales.repeat_interleave(SG, -1))
    ref = torch.bmm(x_f32 * xs, (fp4_utils.mxfp4_to_f32(w) * ws).transpose(1, 2)).to(torch.bfloat16)
    d = (out.float() - ref.float()).abs()
    mm = (d > (1e-2 + 1e-2 * ref.float().abs())).float().mean().item()
us = time_graph(run)
print(f"{tm}x{tn}x{tk} nb{nb} w{mw}x{nw}: {us:.1f} us  {tflops(us):.1f} TF  mism {mm:.3f}", flush=True)
