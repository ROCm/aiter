# Standalone repro (torch + AITER only, random weights): AITER Gluon moe_gemm_a4w4 GEMM1 on gfx1250
# gives wrong output when rows are gathered in-kernel (gather_indx) and routing picks block_m 128.
#   python repro_gluon_a4w4_gather_bm128.py
# Shape: 256 experts, EP2 (rank 0 holds 128), top-8, hidden 2304, intermediate 2304, SwiGLU, MXFP4 W4A4.
import torch
import triton
from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight
from aiter.ops.triton.moe.moe_op_gemm_a4w4 import moe_gemm_a4w4, mxfp4_quant
from aiter.ops.triton.moe.moe_routing.routing import (
    _compute_expt_data_internal,
    EpScatterGeometry,
    ep_sort_routing,
    ExptData,
    RoutingData,
)
from aiter.ops.triton.moe.reduce import EpCombineScatter
from aiter.ops.triton.utils.shuffle import moe_weight_decode_view
 
E, EP, RANK, TOPK, D, I = 256, 2, 0, 8, 2304, 2304
E_LOCAL = E // EP
dev = "cuda"
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], device=dev)
 
 
def dequant(q, s):
    """[..., K/2] packed e2m1 (low nibble first) + [..., K/32] e8m0 -> fp32 [..., K]"""
    v = torch.stack((E2M1[(q & 0xF).long()], E2M1[(q >> 4).long()]), -1).flatten(-2)
    return v * torch.exp2(s.float() - 127.0).repeat_interleave(32, -1)
 
 
def mx_weight(n, k, seed):
    """random [E_LOCAL, n, k] -> (kernel weight, kernel scale, fp32 dequant ref)"""
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn(E_LOCAL, n, k, generator=g, device=dev, dtype=torch.bfloat16) / k**0.5
    q, s = mxfp4_quant(w.reshape(-1, k))
    q, s = q.view(E_LOCAL, n, k // 2), s.view(E_LOCAL, n, k // 32)
    ref = dequant(q, s)
    wk = moe_weight_decode_view(moe_shuffle_weight(q.contiguous()).contiguous())
    sk = moe_shuffle_scale(s.contiguous()).transpose(-1, -2)
    return wk, sk, ref
 
 
def routing(ids, wts, block_m_cap):
    M = ids.shape[0]
    n_gates = M * TOPK
    emap = torch.full((E + 1,), -1, dtype=torch.int32, device=dev)
    emap[RANK * E_LOCAL : (RANK + 1) * E_LOCAL] = torch.arange(E_LOCAL, dtype=torch.int32, device=dev)
    block_m = max(16, min(triton.next_power_of_2(max(1, n_gates // E)), block_m_cap))
    bufs = _compute_expt_data_internal(E_LOCAL, n_gates, block_m, dev)
    geo = EpScatterGeometry(src_token_map=torch.arange(M, dtype=torch.int32, device=dev),
                            max_tokens_per_rank=M, peer_rows=n_gates)
    hist, gidx, sidx, gscal, gvalid, dst = ep_sort_routing(
        wts, ids, emap, E_LOCAL, torch.full((1,), M, dtype=torch.int32, device=dev), M, TOPK, n_gates, bufs,
        ep_scatter_geometry=geo)
    h = hist[:E_LOCAL]
    return RoutingData(block_m, gscal, h, E_LOCAL, TOPK, ExptData(h, bufs[0], bufs[1], bufs[2])), gidx, sidx, gvalid, dst
 
 
def moe(x, ids, wts, w1, w2, block_m_cap, in_kernel_gather):
    M = x.shape[0]
    rd, gidx, sidx, gvalid, dst = routing(ids, wts, block_m_cap)
    xq, xs = mxfp4_quant(x)
    if in_kernel_gather:
        a, a_s, gi = xq, xs, gidx
    else:
        rows = gidx.long().clamp(0, M * TOPK - 1) // TOPK
        a, a_s, gi = xq[rows], xs[rows], None
    y1, y1s = moe_gemm_a4w4(a, w1[0], a_s, w1[1], None, rd, gather_indx=gi, swizzle_mx_scale="GFX1250_SCALE",
                            preshuffle_weights=True, out_mx_quant=True, apply_swiglu=True, alpha=1.0, limit=1e4,
                            swiglu_add_residual=False)
    raw = torch.zeros(M * TOPK, D, dtype=torch.bfloat16, device=dev)
    moe_gemm_a4w4(y1, w2[0], y1s, w2[1], None, rd, scatter_indx=sidx, swizzle_mx_scale="GFX1250_SCALE",
                  preshuffle_weights=True, gate_valid=gvalid, ep_scatter=EpCombineScatter(out=raw, dst_row=dst))
    return raw.view(M, TOPK, D), rd.block_m
 
 
def reference(x, ids, w1_ref, w2_ref):
    """fp32 per-slot reference with the same activation and GEMM1-output MXFP4 requant as the kernel"""
    lo = RANK * E_LOCAL
    loc = (ids >= lo) & (ids < lo + E_LOCAL)
    t, k = loc.nonzero(as_tuple=True)
    e = (ids[t, k] - lo).long()
    xq, xs = mxfp4_quant(x)
    xd = dequant(xq, xs)
    ref = torch.empty(t.numel(), D, device=dev)
    for ex in e.unique().tolist():
        sel = (e == ex).nonzero(as_tuple=True)[0]
        h = xd[t[sel]] @ w1_ref[ex].T
        a = h[:, ::2] * torch.sigmoid(h[:, ::2]) * h[:, 1::2]  # kernel convention: gate = even, up = odd columns
        aq, asc = mxfp4_quant(a.to(torch.bfloat16))
        ref[sel] = dequant(aq, asc) @ w2_ref[ex].T
    return t, k, ref
 
 
def main():
    import importlib.metadata as md
 
    print("triton", md.version("triton"), "| aiter", md.version("amd-aiter"), "|", torch.cuda.get_device_properties(0).gcnArchName)
    w1k, s1k, w1r = mx_weight(2 * I, D, 1)
    w2k, s2k, w2r = mx_weight(D, I, 2)
    g = torch.Generator(device=dev).manual_seed(0)
    for M in (1024, 2048, 4096, 8192):
        x = torch.randn(M, D, generator=g, device=dev, dtype=torch.bfloat16)
        vals, ids = torch.topk(torch.randn(M, E, generator=g, device=dev), TOPK, dim=-1)
        ids, wts = ids.to(torch.int32).contiguous(), torch.softmax(vals, -1).contiguous()
        t, k, ref = reference(x, ids, w1r, w2r)
        for ikg in (True, False):
            out, bm = moe(x, ids, wts, (w1k, s1k), (w2k, s2k), 128, ikg)
            got = out[t, k].float().flatten()
            cos = torch.nn.functional.cosine_similarity(got, ref.flatten(), dim=0).item()
            print(f"M={M:5d} block_m={bm:3d} {'in-kernel gather' if ikg else 'pre-gathered    '}  cos={cos:.6f}  "
                  f"{'OK' if cos > 0.999 else 'WRONG'}")
 
 
 
 
if __name__ == "__main__":
    main()
