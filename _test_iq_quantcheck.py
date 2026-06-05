"""Pure-torch check: does the FlyDSL-epilogue quant formula (used by inline
quant) reproduce mxfp4_moe_quant's a_quant / a_scale exactly?

Replicates the kernel's per-32 amax -> e8m0 -> f32_to_e2m1 in torch and diffs
against the HIP mxfp4 quant output.
"""
import sys
import torch
import aiter
from aiter import dtypes
from bench_up_moe_v1 import KIMI, build_inputs


def _bits_u(t_f32):
    return t_f32.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF


def _f32_to_e2m1_torch(scaled):
    # all bit-ops in int64 (unsigned 32-bit semantics)
    qx = _bits_u(scaled)
    s = qx & 0x80000000
    qx_abs = qx & 0x7FFFFFFF
    denormal_mask = qx_abs < 0x3F800000
    normal_mask = (qx_abs < 0x40C00000) & (qx_abs >= 0x3F800000)
    denorm_f32 = (
        qx_abs.to(torch.int32).view(torch.float32)
        + torch.tensor(0x4A800000, dtype=torch.int32, device=qx.device).view(torch.float32)
    )
    denormal_x = (_bits_u(denorm_f32) - 0x4A800000) & 0xFFFFFFFF
    mant_odd = (qx_abs >> 22) & 1
    normal_x = ((qx_abs + 0xC11FFFFF + mant_odd) & 0xFFFFFFFF) >> 22
    e2m1 = torch.where(normal_mask, normal_x, torch.full_like(normal_x, 0x7))
    e2m1 = torch.where(denormal_mask, denormal_x, e2m1)
    return (((s >> 28) | e2m1) & 0xF).to(torch.uint8)


def quant_like_kernel(hidden):
    # hidden [M, K] bf16 -> per-32 amax -> e8m0 -> fp4 (K-contiguous packing)
    M, K = hidden.shape
    f = hidden.float()
    g = f.reshape(M, K // 32, 32)
    amax = g.abs().amax(dim=-1)  # [M, K/32]
    max_bits = _bits_u(amax)
    # mxfp4 moe_sort_quant: bexp = ((amax_f32bits + 0x200000) >> 23) & 0xFF
    exp_field = ((max_bits + 0x200000) >> 23) & 0xFF
    e8m0_biased = torch.clamp(exp_field - 2, min=0, max=254)  # headroom=2
    quant_exp = 254 - e8m0_biased
    quant_scale = (quant_exp << 23).to(torch.int32).view(torch.float32)  # [M, K/32]
    scaled = g * quant_scale.unsqueeze(-1)
    fp4 = _f32_to_e2m1_torch(scaled).reshape(M, K)
    lo = fp4[:, 0::2]
    hi = fp4[:, 1::2]
    packed = (lo | (hi << 4)).to(torch.uint8)  # [M, K/2]
    e8m0 = e8m0_biased.to(torch.uint8)  # [M, K/32]
    return packed, e8m0


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    device = torch.device("cuda")
    shape = KIMI
    hidden, topk_ids, topk_weight = build_inputs(shape, M, device)
    NE = shape.NE
    D_HIDDEN = shape.H
    D_INTER = shape.INTER
    topk = topk_ids.shape[1]
    BM = 32

    a_quant = torch.empty((M, D_HIDDEN // 2), device=device, dtype=torch.uint8)
    a_scale = torch.empty((M, D_HIDDEN // 32), device=device, dtype=torch.uint8)
    out_buf = torch.empty((M, D_HIDDEN), dtype=dtypes.bf16, device=device)
    aiter.mxfp4_moe_quant(
        a_input=hidden, a_quant=a_quant, a_scale=a_scale,
        bf16_zero_out=out_buf, NE=NE, TOPK=topk, D_HIDDEN=D_HIDDEN, MB=BM,
    )
    torch.cuda.synchronize()

    my_q, my_s = quant_like_kernel(hidden)

    q_match = (my_q == a_quant).float().mean().item()
    s_match = (my_s == a_scale).float().mean().item()
    print(f"M={M}  a_quant match: {q_match*100:.3f}%   a_scale match: {s_match*100:.3f}%")
    # show a few mismatched scales
    diff = (my_s != a_scale)
    if diff.any():
        idx = diff.nonzero()[:5]
        for r in idx:
            i, j = r.tolist()
            print(f"  scale[{i},{j}] mine={int(my_s[i,j])} mxfp4={int(a_scale[i,j])}")
    # value distribution of a_scale vs mine
    print("  my_s mean/min/max:", float(my_s.float().mean()), int(my_s.min()), int(my_s.max()))
    print("  mx_s mean/min/max:", float(a_scale.float().mean()), int(a_scale.min()), int(a_scale.max()))


if __name__ == "__main__":
    main()
