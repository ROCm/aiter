"""Re-tune XCD remap (xcd_c x xcd_w) for the D256 uniform shapes AT threads=512.

The threads=512 win (commit 3a62cc2) was measured on top of xcd_c/w tuned at
threads=256. Doubling the warps changes occupancy and L2 access cadence, so the
XCD sweet spot may have moved. This re-sweeps it. cos-gate + do_bench cold-L2.
Run inside the jdbba-flydsl container.
"""
from __future__ import annotations

import itertools

import torch
import triton

import flydsl.compiler as flyc
from aiter.ops.flydsl.jagged_dense_bmm_dispatch_v2 import jagged_dense_bmm_dispatched
from aiter.ops.flydsl.kernels.jagged_dense_bmm_gen import jagged_dense_bmm, BLOCK_M as _BLOCK_M

SHAPES = [(120, 256, 256), (1024, 256, 256)]
MI = 7680
XCD_C = [1, 16, 32, 60, 120, 240]
XCD_W = [4, 8, 16]
THREADS = 512


def make_inputs(B, D, Kout, Mi, device="cuda"):
    torch.manual_seed(0)
    N, K = Kout, D
    so = torch.arange(0, (B + 1) * Mi, Mi, dtype=torch.int32, device=device)
    L = int(so[-1].item())
    jagged = torch.randn(max(L, 1), K, dtype=torch.bfloat16, device=device)
    dense = torch.randn(B, K, N, dtype=torch.bfloat16, device=device)
    bias = torch.randn(B, N, dtype=torch.bfloat16, device=device)
    dense_tall = dense.transpose(1, 2).reshape(B * N, K).contiguous()
    bias_flat = bias.reshape(B * N).contiguous()
    return jagged, dense, bias, dense_tall, bias_flat, so, L, N, K


def torch_ref(jagged, dense, bias, so, N):
    L = jagged.shape[0]
    out = torch.zeros((L, N), dtype=torch.bfloat16, device=jagged.device)
    for b in range(dense.shape[0]):
        s, e = int(so[b]), int(so[b + 1])
        if e > s:
            out[s:e] = (jagged[s:e].float() @ dense[b].float() + bias[b].float()[None, :]).to(torch.bfloat16)
    return out


def run(B, D, Kout, xcd_c, xcd_w, threads):
    jagged, dense, bias, dense_tall, bias_flat, so, L, N, K = make_inputs(B, D, Kout, MI)
    out = torch.zeros(L + _BLOCK_M, N, dtype=torch.bfloat16, device="cuda")
    tA = flyc.from_dlpack(jagged).mark_layout_dynamic(leading_dim=1, divisibility=8)
    tC = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1, divisibility=8)

    def fn():
        jagged_dense_bmm(
            tC, tA, dense_tall, bias_flat, so, B, MI,
            stream=torch.cuda.current_stream(), uniform_seqlen=True,
            xcd_c=xcd_c, xcd_w=xcd_w, use_mfma_k32=False, threads=threads,
        )

    fn()
    torch.cuda.synchronize()
    ref = torch_ref(jagged, dense, bias, so, N)
    cos = torch.nn.functional.cosine_similarity(
        ref.float().flatten(), out[:L].float().flatten(), dim=0
    ).item()
    if cos < 0.999:
        return None, cos
    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return ms, cos


def main():
    for (B, D, Kout) in SHAPES:
        print(f"\n=== B{B}_D{D} threads={THREADS} ===")
        best = (None, None, 1e9)
        for xcd_c, xcd_w in itertools.product(XCD_C, XCD_W):
            ms, cos = run(B, D, Kout, xcd_c, xcd_w, THREADS)
            if ms is None:
                print(f"  c={xcd_c:4d} w={xcd_w:3d}  SKIP cos={cos:.4f}")
                continue
            mark = ""
            if ms < best[2]:
                best = (xcd_c, xcd_w, ms)
                mark = "  <-- best"
            print(f"  c={xcd_c:4d} w={xcd_w:3d}  {ms:.4f} ms  cos={cos:.4f}{mark}")
        print(f"  WINNER B{B}_D{D}: c={best[0]} w={best[1]}  {best[2]:.4f} ms")


if __name__ == "__main__":
    main()
