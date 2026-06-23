"""Validate the gfx1250 gluon dKV-intermediate kernel (M3) vs the Triton
`_bwd_compute_dkv_intermediate`. Kernel-vs-kernel (no torch GEMM) -> direct GPU compare.

Run on a gfx1250 box:  python test_dkv_interm_gluon_gfx1250.py
"""
import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx1250.attention.dsa_bwd_dkv_interm import (
    sparse_mla_bwd_dkv_interm_gl_gfx1250,
)
from aiter.ops.triton._triton_kernels.attention._dsa_bwd_gather import (
    _bwd_compute_dkv_intermediate,
)


def main():
    T, H, D_V, D_ROPE = 128, 64, 512, 64
    D_QK = D_V + D_ROPE
    R_CHUNK, TILE_K_G, BH_G = 128, 64, 32
    torch.manual_seed(0)
    q = torch.randn(T, H, D_QK, dtype=torch.bfloat16, device="cuda")
    do = torch.randn(T, H, D_V, dtype=torch.bfloat16, device="cuda")
    dS = torch.randn(T, H, R_CHUNK, dtype=torch.bfloat16, device="cuda")
    P = torch.randn(T, H, R_CHUNK, dtype=torch.bfloat16, device="cuda")

    interm_g = sparse_mla_bwd_dkv_interm_gl_gfx1250(
        q, do, dS, P, R_CHUNK, kv_lora_rank=D_V, BLOCK_H=BH_G, TILE_K=TILE_K_G)
    torch.cuda.synchronize()

    q_t = q.transpose(1, 2).contiguous()
    do_t = do.transpose(1, 2).contiguous()
    interm_t = torch.empty(T, R_CHUNK, D_QK, dtype=torch.bfloat16, device="cuda")
    chunk_topk = torch.zeros(T, R_CHUNK, dtype=torch.int32, device="cuda")
    bh_t = 64
    _bwd_compute_dkv_intermediate[(T,)](
        q_t, do_t, dS, P, chunk_topk, interm_t,
        q_t.stride(0), do_t.stride(0),
        dS.stride(0), dS.stride(1),
        R_CHUNK, interm_t.stride(0), interm_t.stride(1), H,
        TOPK=R_CHUNK, TILE_K=64, BLOCK_H=bh_t, NUM_HG=triton.cdiv(H, bh_t),
        D_V=D_V, D_ROPE=D_ROPE, num_warps=4, num_stages=1,
    )
    torch.cuda.synchronize()

    a = interm_g.float().cpu(); b = interm_t.float().cpu()
    rel = (a - b).norm() / (b.norm() + 1e-9)
    print(f"interm: max_abs={(a-b).abs().max():.3e} rel-vs-Triton={rel:.3e}")
    assert rel < 5e-3, "gfx1250 gluon dkv-interm mismatch"
    print("PASS")


if __name__ == "__main__":
    main()
