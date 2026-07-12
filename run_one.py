"""Run the batched a8w4 TDM kernel once at a fixed config (for profiling / quick iterate).
Usage: run_one.py B M N K tile_n tile_k nb [iters]
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


def pad32(s):
    r = s.shape[0]
    return s if r % 32 == 0 else torch.cat([s, torch.zeros((32 - r % 32, s.shape[1]), dtype=s.dtype)], 0)


def gen(B, M, N, K, want_ref):
    torch.manual_seed(5)
    qf = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    wc = [qf(torch.randn(N, K, dtype=torch.bfloat16), shuffle=False) for _ in range(B)]
    w = torch.stack([c.view(torch.uint8) for c, _ in wc])
    w_scales = torch.stack([s.view(torch.uint8) for _, s in wc])
    x = (torch.randn(B, M, K) * 4.0).to(torch.float8_e4m3fn).view(torch.uint8)
    x_scales = torch.randint(124, 130, (B, M, K // SG), dtype=torch.uint8)
    w_sh = torch.cat([shuffle_weight_gfx1250(w[b].contiguous()).reshape(-1) for b in range(B)]).reshape(B * (N // 16), (K // 2) * 16)
    sb = torch.cat([shuffle_scale_n32k4(pad32(w_scales[b]).unsqueeze(0), experts_cnt=1).view(torch.uint8).reshape(-1) for b in range(B)])
    sa_list = [shuffle_scale_n32k4(pad32(x_scales[b]).unsqueeze(0), experts_cnt=1).view(torch.uint8).reshape((M + 31) // 32, -1) for b in range(B)]
    a = torch.stack([x[b].contiguous() for b in range(B)], 0).contiguous()
    sa = torch.cat([s.reshape(-1) for s in sa_list])
    ref = None
    if want_ref:
        x_f32 = x.view(torch.float8_e4m3fn).to(torch.float32)
        xs = fp4_utils.e8m0_to_f32(x_scales.repeat_interleave(SG, -1))
        ws = fp4_utils.e8m0_to_f32(w_scales.repeat_interleave(SG, -1))
        ref = torch.bmm(x_f32 * xs, (fp4_utils.mxfp4_to_f32(w) * ws).transpose(1, 2)).to(torch.bfloat16)
    return a, w_sh, sa, sb, ref


def time_graph(fn, iters=50):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    g.replay(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(5):
        g.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000.0 / (5 * iters)


def main():
    B, M, N, K = (int(v) for v in sys.argv[1:5])
    tile_n = int(sys.argv[5]); tile_k = int(sys.argv[6]); nb = int(sys.argv[7])
    l2pf = int(sys.argv[8]) if len(sys.argv) > 8 and sys.argv[8].isdigit() else 0
    # optional mw= nw= overrides
    mw_ov = nw_ov = None
    for tok in sys.argv[8:]:
        if tok.startswith("mw="):
            mw_ov = int(tok[3:])
        elif tok.startswith("nw="):
            nw_ov = int(tok[3:])
    prof = "prof" in sys.argv[8:]
    noref = "noref" in sys.argv[8:]
    a, w_sh, sa, sb, ref = gen(B, M, N, K, want_ref=not (prof or noref))
    out = torch.empty((B, M, N), dtype=torch.bfloat16)
    tile_m = 64 if M <= 64 else 128
    m_warp = mw_ov if mw_ov else ((tile_m // 16) if tile_m <= 64 else (tile_m // 32))
    wtm = tile_m // m_warp
    n_warp = nw_ov if nw_ov else max(1, tile_n // 64)
    wtn = tile_n // n_warp
    flops = 2 * B * M * N * K

    def run():
        launch_gemm_a8w4_tdm(out, ptr_arg(a), ptr_arg(w_sh), sa.view(torch.int32), sb.view(torch.int32),
                             M, torch.cuda.current_stream(), N, K, tile_m, tile_n, tile_k,
                             m_warp, n_warp, wtm, wtn, 0, B, 0, nb, 0)

    if prof:
        run(); torch.cuda.synchronize()
        for _ in range(20):
            run()
        torch.cuda.synchronize()
        print("prof done")
        return

    us = time_graph(run)
    mm = -1.0
    if ref is not None:
        d = (out.float() - ref.float()).abs()
        mm = (d > (1e-2 + 1e-2 * ref.float().abs())).float().mean().item()
    print(f"{B}x{M}x{N}x{K} tn{tile_n} tk{tile_k} nb{nb} mw{m_warp} nw{n_warp} l2pf{l2pf}: {us:.1f} us  {flops/(us*1e-6)/1e12:.1f} TFLOPS  mism {mm:.3f}")


if __name__ == "__main__":
    main()
