"""Parity check: send-side per-token MX quant vs the existing recv-side quant.

Validates two things needed to move quantization ahead of dispatch:
  1. the E8M0/payload math matches the quantizer the GEMM is already fed by, and
  2. the WMMA scale-preshuffle formula, which dispatch will have to apply itself
     once the scale travels over the wire (the permutation is reproduced here in
     torch and checked against the kernel's output).
"""
import torch

from aiter.ops.flydsl.kernels.per_token_mx_quant import per_1x32_mx_quant
from aiter.ops.flydsl.moe_kernels import flydsl_moe_fused_quant_preshuffle

M, N, WMMA_REP = 128, 7168, 4
DEV = "cuda"


def preshuffle_index(m, n, wmma_rep):
    """Return idx[slot, mx_block] -> byte offset in the preshuffled scale buffer."""
    rows_per_tile = wmma_rep * 16
    mx_per_row = n // 32
    scale_dwords_per_row = mx_per_row // 4

    slot = torch.arange(m).view(m, 1)
    mx_block = torch.arange(mx_per_row).view(1, mx_per_row)

    scale_tile = slot // rows_per_tile
    wmma_row = (slot % rows_per_tile) // 16
    row_lane16 = slot % 16
    out_row = scale_tile * 16 + row_lane16

    scale_dword = mx_block // 4
    byte_in_dword = mx_block % 4
    dst_dword = (
        out_row * (scale_dwords_per_row * wmma_rep) + scale_dword * wmma_rep + wmma_row
    )
    return (dst_dword * 4 + byte_in_dword).expand(m, mx_per_row)


torch.manual_seed(0)
x = torch.randn(M, N, dtype=torch.bfloat16, device=DEV)

payload_send, scale_send = per_1x32_mx_quant(x, quant_mode="fp8")
payload_recv, scale_recv = flydsl_moe_fused_quant_preshuffle(
    x.reshape(1, M, N), 1, M, wmma_rep=WMMA_REP, quant_mode="fp8", masked_m=None
)

p_send = payload_send.view(torch.uint8).reshape(M, N)
p_recv = payload_recv.view(torch.uint8).reshape(M, N)
same = (p_send == p_recv).float().mean().item()
print(f"payload bytes identical : {same * 100:.4f}%")

idx = preshuffle_index(M, N, WMMA_REP).to(DEV)
flat = scale_recv.view(torch.uint8).reshape(-1)
scale_recv_linear = flat[idx.reshape(-1)].reshape(M, N // 32)
s_match = (scale_send == scale_recv_linear).float().mean().item()
print(f"scale bytes identical   : {s_match * 100:.4f}%   (after un-preshuffling)")

if same < 1.0 or s_match < 1.0:
    bad = (p_send != p_recv).nonzero()[:5]
    print(f"first payload mismatches: {bad.tolist()}")
    bads = (scale_send != scale_recv_linear).nonzero()[:5]
    print(f"first scale mismatches  : {bads.tolist()}")
    raise SystemExit("FAIL: send-side quant diverged from the recv-side quant")
print("PASS")
