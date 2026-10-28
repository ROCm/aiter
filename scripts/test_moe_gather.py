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
 
 
def routing_torch(ids, wts, block_m_cap):
    """Pure-torch equivalent of routing() for reference/debugging."""
    M = ids.shape[0]
    n_gates = M * TOPK
    block_m = max(16, min(triton.next_power_of_2(max(1, n_gates // E)), block_m_cap))
    lo = RANK * E_LOCAL

    flat_ids = ids.reshape(-1)
    flat_wts = wts.reshape(-1)
    gate_indices = torch.arange(n_gates, device=dev, dtype=torch.int32)

    local_eid = (flat_ids - lo).int()
    valid = (local_eid >= 0) & (local_eid < E_LOCAL)

    # per-expert histogram
    hist = torch.zeros(E_LOCAL, dtype=torch.int32, device=dev)
    if valid.any():
        hist.scatter_add_(0, local_eid[valid].long(), torch.ones(valid.sum(), dtype=torch.int32, device=dev))

    # sort valid gates by local expert (stable preserves intra-expert order)
    valid_gate_idx = gate_indices[valid]
    valid_expert = local_eid[valid]
    order = torch.argsort(valid_expert.long(), stable=True)
    sorted_gate_idx = valid_gate_idx[order]
    sorted_wts = flat_wts[valid.nonzero(as_tuple=True)[0][order]]

    # pad each expert to a multiple of block_m
    padded_hist = ((hist + block_m - 1) // block_m * block_m).int()
    total_padded = padded_hist.sum().item()

    gidx = torch.zeros(total_padded, dtype=torch.int32, device=dev)
    gscal = torch.zeros(total_padded, device=dev, dtype=flat_wts.dtype)
    gvalid_out = torch.zeros(total_padded, dtype=torch.bool, device=dev)

    src_off, dst_off = 0, 0
    for e in range(E_LOCAL):
        cnt = hist[e].item()
        pad = padded_hist[e].item()
        if cnt > 0:
            gidx[dst_off:dst_off + cnt] = sorted_gate_idx[src_off:src_off + cnt]
            gscal[dst_off:dst_off + cnt] = sorted_wts[src_off:src_off + cnt]
            gvalid_out[dst_off:dst_off + cnt] = True
            if pad > cnt:
                gidx[dst_off + cnt:dst_off + pad] = sorted_gate_idx[src_off + cnt - 1]
        src_off += cnt
        dst_off += pad

    sidx = gidx.clone()
    dst = gidx.clone()

    # build ExptData buffers
    token_offs_raw = torch.zeros(E_LOCAL + 1, dtype=torch.int32, device=dev)
    token_offs_pad = torch.zeros(E_LOCAL + 1, dtype=torch.int32, device=dev)
    token_offs_raw[1:] = torch.cumsum(hist, 0)
    token_offs_pad[1:] = torch.cumsum(padded_hist, 0)
    token_offs_raw = token_offs_raw[:E_LOCAL]
    token_offs_pad = token_offs_pad[:E_LOCAL]

    n_blocks_per_expert = padded_hist // block_m
    total_blocks = n_blocks_per_expert.sum().item()
    block_pid_map = torch.empty(total_blocks, dtype=torch.int32, device=dev)
    blk_off = 0
    for e in range(E_LOCAL):
        n_blk = n_blocks_per_expert[e].item()
        for b in range(n_blk):
            block_pid_map[blk_off] = (e & 0xFFFF) | ((b & 0xFFFF) << 16)
            blk_off += 1

    expt_data = ExptData(hist, token_offs_raw, token_offs_pad, block_pid_map)
    rd = RoutingData(block_m, gscal, hist, E_LOCAL, TOPK, expt_data)
    return rd, gidx, sidx, gvalid_out, dst


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
 
 
 
 
_bench_weights = {}


def _get_bench_weights():
    if not _bench_weights:
        w1k, s1k, w1r = mx_weight(2 * I, D, 1)
        w2k, s2k, w2r = mx_weight(D, I, 2)
        _bench_weights["w1"] = (w1k, s1k)
        _bench_weights["w2"] = (w2k, s2k)
    return _bench_weights["w1"], _bench_weights["w2"]


configs = [
    triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[1024, 2048, 4096, 8192],
        line_arg="provider",
        line_vals=["in_kernel_gather", "pre_gathered"],
        line_names=["In-Kernel Gather", "Pre-Gathered"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="ms",
        plot_name="moe-gather-performance",
        args={"block_m_cap": 128},
    ),
]


@triton.testing.perf_report(configs)
def bench_moe_gather(M, provider, block_m_cap):
    w1, w2 = _get_bench_weights()
    g = torch.Generator(device=dev).manual_seed(M)
    x = torch.randn(M, D, generator=g, device=dev, dtype=torch.bfloat16)
    vals, ids = torch.topk(torch.randn(M, E, generator=g, device=dev), TOPK, dim=-1)
    ids = ids.to(torch.int32).contiguous()
    wts = torch.softmax(vals, -1).contiguous()
    in_kernel_gather = provider == "in_kernel_gather"
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: moe(x, ids, wts, w1, w2, block_m_cap, in_kernel_gather),
        quantiles=quantiles, warmup=50, rep=100,
    )
    return ms, max_ms, min_ms


if __name__ == "__main__":
    import sys
    if "--benchmark" in sys.argv:
        bench_moe_gather.run(show_plots=False, print_data=True)
    else:
        main()
