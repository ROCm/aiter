"""Unit test for FP8-direct path of `unified_attention_sparse_mla`.

V4 stores compressed KV as per-tile (FP8 + E8M0 block scales) and currently
vllm dequantizes to BF16 before invoking this kernel. The KV_FP8 branch in
`_kernel_unified_attention_sparse_mla_2d` skips that dequant: it loads FP8
and the e8m0 scale byte inside the MFMA prologue, dequantizes to BF16, then
runs the standard MFMA. The kernel then matches the BF16 path that operates
on the host-dequantized values (within BF16 round-off).

This test:
  1. Builds a random BF16 KV cache.
  2. Quantizes each 64-element tile to (FP8 E4M3FNUZ values, E8M0 scale byte).
  3. Builds a "host-dequantized" BF16 version of the cache.
  4. Runs the BF16 path on (BF16 KV) and the FP8 path on (FP8 KV + scales).
  5. Asserts the two outputs are close.

Mimics V4 shapes (head_dim=512, no rope).
"""

import math
import pytest
import torch

from aiter.ops.triton.attention.unified_attention_sparse_mla import (
    unified_attention_sparse_mla,
)


FP8_DTYPE = torch.float8_e4m3fnuz  # ROCm fp8 dtype
FP8_MAX = torch.finfo(FP8_DTYPE).max  # 240.0 for e4m3fnuz
FP8_TILE = 64


def quantize_to_fp8_e8m0(
    x_bf16: torch.Tensor, fp8_tile: int = FP8_TILE
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-tile MX-style quantization.

    Args:
      x_bf16: [..., head_dim] BF16. head_dim must be divisible by fp8_tile.

    Returns:
      fp8_vals  : same shape as x_bf16, FP8 dtype.
      scale_byte: same shape with last dim = head_dim // fp8_tile, uint8 (e8m0
                  byte representation).
      x_dequant : same shape as x_bf16, BF16 — the lossy round-trip
                  (fp8.to(bf16) * 2^(byte-127)).
    """
    assert x_bf16.dtype == torch.bfloat16
    assert x_bf16.shape[-1] % fp8_tile == 0
    head_dim = x_bf16.shape[-1]
    num_tiles = head_dim // fp8_tile

    x = x_bf16.to(torch.float32)
    tiled = x.reshape(*x.shape[:-1], num_tiles, fp8_tile)

    amax = tiled.abs().amax(dim=-1).clamp_min(1e-30)  # [..., num_tiles]
    # e8m0 unbiased exponent: pick smallest e such that 2^e * FP8_MAX >= amax,
    # i.e. e >= log2(amax / FP8_MAX). ceil it. clamp into [-127, 127].
    exp_unbiased = torch.ceil(torch.log2(amax / FP8_MAX)).clamp_(-127, 127)
    scale_fp32 = torch.exp2(exp_unbiased)  # [..., num_tiles]
    inv_scale = torch.exp2(-exp_unbiased)

    # Quantize to fp8.
    qx_fp32 = tiled * inv_scale.unsqueeze(-1)
    qx_fp32 = qx_fp32.clamp_(-FP8_MAX, FP8_MAX)
    fp8_tile_vals = qx_fp32.to(FP8_DTYPE)
    fp8_vals = fp8_tile_vals.reshape_as(x_bf16)

    # Round-trip back to BF16 (mimics _dequantize_blocked_k_cache).
    deq = fp8_tile_vals.to(torch.bfloat16) * scale_fp32.to(torch.bfloat16).unsqueeze(-1)
    x_dequant = deq.reshape_as(x_bf16)

    # E8M0 byte = unbiased_exp + 127, as uint8.
    scale_byte = (exp_unbiased + 127).to(torch.uint8)

    return fp8_vals, scale_byte, x_dequant


def _make_v4_inputs(
    batch: int,
    s_q: int,
    topk: int,
    kv_pool: int,
    seed: int = 0,
):
    """V4-shape inputs: head_dim=512, no rope, single h_kv=1, FP4-like blocks."""
    torch.manual_seed(seed)
    device = "cuda"
    h_q = 128
    head_dim = 512
    block_size = 64

    # Pad kv_pool to block_size.
    kv_pool = ((kv_pool + block_size - 1) // block_size) * block_size
    num_blocks = kv_pool // block_size

    # bf16 KV: small magnitude so quantization is well-conditioned.
    kv_bf16 = (
        torch.randn(num_blocks, block_size, 1, head_dim, dtype=torch.float32, device=device)
        * 0.5
    ).to(torch.bfloat16)
    # Bring out wider dynamic range in some tiles to exercise the e8m0 scale.
    kv_bf16 = kv_bf16 * (1.0 + torch.rand(num_blocks, block_size, 1, 1, device=device) * 8.0).to(torch.bfloat16)

    # Quantize → fp8 + e8m0 + lossy bf16 round-trip.
    fp8_vals, scale_byte, kv_dequant = quantize_to_fp8_e8m0(kv_bf16.squeeze(2))
    fp8_vals = fp8_vals.unsqueeze(2)         # [num_blocks, block_size, 1, head_dim]
    scale_byte = scale_byte.unsqueeze(2)     # [num_blocks, block_size, 1, num_tiles]
    kv_dequant = kv_dequant.unsqueeze(2)     # [num_blocks, block_size, 1, head_dim]

    # Q in bf16, varlen.
    q = (
        torch.randn(batch * s_q, h_q, head_dim, dtype=torch.float32, device=device) * 0.5
    ).to(torch.bfloat16)

    # Topk indices: pick random valid slot ids per (batch_token).
    topk_indices = torch.empty((batch * s_q, topk), dtype=torch.int32, device=device)
    for b in range(batch):
        for q_local in range(s_q):
            idx = b * s_q + q_local
            sampled = torch.randperm(kv_pool, device=device)[:topk]
            topk_indices[idx] = sampled.int()

    cu_seqlens_q = torch.arange(0, batch + 1, dtype=torch.int32, device=device) * s_q
    seqused_k = torch.full((batch,), kv_pool, dtype=torch.int32, device=device)
    sm_scale = head_dim ** -0.5

    # Strip h_kv dim for scales view (kernel expects [num_blocks, block_size, num_tiles]).
    scale_byte_kernel = scale_byte.squeeze(2).contiguous()

    return {
        "q": q,
        "kv_bf16": kv_bf16,
        "kv_dequant": kv_dequant,
        "kv_fp8": fp8_vals,
        "kv_scales": scale_byte_kernel,
        "topk_indices": topk_indices,
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_k": seqused_k,
        "sm_scale": sm_scale,
        "head_dim": head_dim,
        "kv_pool": kv_pool,
        "h_q": h_q,
        "max_q": s_q,
        "block_size": block_size,
        "batch": batch,
    }


def _run_kernel(d, kv_input, kv_scales=None, return_lse=False):
    """Run the kernel with a given KV input (bf16 or fp8 view)."""
    block_table = torch.zeros((d["batch"], 1), dtype=torch.int32, device=d["q"].device)
    out = torch.empty((d["q"].shape[0], d["h_q"], d["head_dim"]),
                      dtype=torch.bfloat16, device=d["q"].device)
    lse = unified_attention_sparse_mla(
        d["q"],
        kv_input,
        out,
        d["cu_seqlens_q"],
        d["max_q"],
        d["seqused_k"],
        d["kv_pool"],
        d["sm_scale"],
        d["topk_indices"],
        block_table,
        d["head_dim"],
        attn_sink=None,
        return_lse=return_lse,
        kv_scales=kv_scales,
    )
    return out, lse


# ---- tests ----------------------------------------------------------------


@pytest.mark.parametrize("topk", [256, 512, 1024])
@pytest.mark.parametrize("batch,s_q", [(1, 1), (4, 1), (16, 1), (1, 128), (1, 512)])
def test_fp8_matches_bf16(topk, batch, s_q):
    """FP8-direct kernel ≈ BF16 kernel on host-dequantized cache.

    Same numerical inputs (modulo bf16-vs-fp32 multiply order) → outputs must
    agree within bf16 round-off (atol=2e-2, rtol=1e-2 — same as the existing
    sparse-mla bf16 test bumped slightly for the multiply-order delta).
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    d = _make_v4_inputs(batch=batch, s_q=s_q, topk=topk, kv_pool=4096)

    # BF16 reference: the kernel sees the host-dequantized cache.
    out_bf16, _ = _run_kernel(d, d["kv_dequant"])
    # FP8 path: kernel dequants in-MFMA-prologue.
    out_fp8, _ = _run_kernel(d, d["kv_fp8"], kv_scales=d["kv_scales"])

    assert out_fp8.shape == out_bf16.shape
    assert out_fp8.dtype == torch.bfloat16
    assert torch.isfinite(out_fp8).all(), "FP8 output has NaN/Inf"

    diff = (out_fp8.float() - out_bf16.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    print(f"[topk={topk} b={batch} s_q={s_q}] "
          f"max_abs={max_abs:.4e} mean_abs={mean_abs:.4e}")

    torch.testing.assert_close(
        out_fp8, out_bf16, atol=2e-2, rtol=1e-2,
    )


def test_fp8_matches_bf16_with_lse():
    """LSE output must also agree (used by two-pool merge)."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    d = _make_v4_inputs(batch=2, s_q=1, topk=512, kv_pool=4096)
    out_bf16, lse_bf16 = _run_kernel(d, d["kv_dequant"], return_lse=True)
    out_fp8,  lse_fp8  = _run_kernel(d, d["kv_fp8"], kv_scales=d["kv_scales"], return_lse=True)

    assert lse_bf16 is not None and lse_fp8 is not None
    torch.testing.assert_close(out_fp8, out_bf16, atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(lse_fp8, lse_bf16, atol=5e-3, rtol=5e-3)


def test_fp8_invariant_under_random_neg1_padding():
    """-1 padded slots in topk must be ignored regardless of FP8 path."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    d = _make_v4_inputs(batch=4, s_q=1, topk=256, kv_pool=4096)
    # Random 25% -1 padding.
    mask = torch.rand_like(d["topk_indices"], dtype=torch.float32) < 0.25
    d["topk_indices"] = torch.where(mask, torch.full_like(d["topk_indices"], -1),
                                    d["topk_indices"])
    out_bf16, _ = _run_kernel(d, d["kv_dequant"])
    out_fp8,  _ = _run_kernel(d, d["kv_fp8"], kv_scales=d["kv_scales"])
    assert torch.isfinite(out_fp8).all(), "FP8 output has NaN/Inf with -1 padding"
    torch.testing.assert_close(out_fp8, out_bf16, atol=2e-2, rtol=1e-2)


# ---- V4 split-storage tests (448 FP8 nope + 64 BF16 rope + 7 e8m0 scales) -----


V4_FP8_DIM = 448
V4_BF16_DIM = 64
V4_HEAD_DIM = V4_FP8_DIM + V4_BF16_DIM  # 512
V4_NUM_TILES = V4_FP8_DIM // FP8_TILE   # 7


def _make_v4_split_inputs(batch: int, s_q: int, topk: int, kv_pool: int, seed: int = 0):
    """Build inputs matching V4's actual cache layout.

    Per-token byte layout in cache (matches `_dequantize_blocked_k_cache`):
      [V4_FP8_DIM bytes FP8 nope | 2*V4_BF16_DIM bytes BF16 rope | 8 e8m0 scale bytes]

    Returns:
      q             : [batch*s_q, h_q, V4_HEAD_DIM] BF16
      kv_fp8        : [num_blocks, block_size, 1, V4_FP8_DIM] FP8 view
      k_bf16        : [num_blocks, block_size, 1, V4_BF16_DIM] BF16 view
      kv_scales     : [num_blocks, block_size, V4_NUM_TILES] uint8 view
      kv_dequant    : [num_blocks, block_size, 1, V4_HEAD_DIM] BF16 (host-dequant
                      reference: fp8.to(bf16)*scale concatenated with the bf16 part)
      ... plus topk_indices, cu_seqlens_q, seqused_k, sm_scale, etc.
    """
    torch.manual_seed(seed)
    device = "cuda"
    h_q = 128
    block_size = 64
    kv_pool = ((kv_pool + block_size - 1) // block_size) * block_size
    num_blocks = kv_pool // block_size

    # Generate logical bf16 KV [num_blocks, block_size, V4_HEAD_DIM]; first
    # V4_FP8_DIM channels become FP8, last V4_BF16_DIM stay BF16.
    kv_full_bf16 = (
        torch.randn(num_blocks, block_size, V4_HEAD_DIM, dtype=torch.float32, device=device)
        * 0.5
    ).to(torch.bfloat16)
    # Wider per-row dynamic range to exercise scales.
    kv_full_bf16 = kv_full_bf16 * (
        1.0 + torch.rand(num_blocks, block_size, 1, device=device) * 8.0
    ).to(torch.bfloat16)

    nope_bf16 = kv_full_bf16[..., :V4_FP8_DIM].contiguous()
    rope_bf16 = kv_full_bf16[..., V4_FP8_DIM:].contiguous()  # [num_blocks, block_size, V4_BF16_DIM]

    # Quantize the nope part to FP8 + e8m0.
    fp8_nope, scale_byte, nope_dequant = quantize_to_fp8_e8m0(nope_bf16)
    # fp8_nope shape: [num_blocks, block_size, V4_FP8_DIM]; same for nope_dequant.
    # scale_byte shape: [num_blocks, block_size, V4_NUM_TILES].

    # Build kernel views (add the singleton h_kv dim).
    kv_fp8_view = fp8_nope.unsqueeze(2)               # [B, blk, 1, V4_FP8_DIM]
    k_bf16_view = rope_bf16.unsqueeze(2)              # [B, blk, 1, V4_BF16_DIM]
    v_bf16_view = rope_bf16.unsqueeze(2)              # V == K for V4
    scales_view = scale_byte.contiguous()             # [B, blk, V4_NUM_TILES]

    # Host-dequant reference: full BF16 kv with the same FP8 round-trip applied.
    kv_dequant_full = torch.cat([nope_dequant, rope_bf16], dim=-1).unsqueeze(2)

    # Q: contiguous BF16 over the full V4_HEAD_DIM.
    q = (
        torch.randn(batch * s_q, h_q, V4_HEAD_DIM, dtype=torch.float32, device=device) * 0.5
    ).to(torch.bfloat16)

    topk_indices = torch.empty((batch * s_q, topk), dtype=torch.int32, device=device)
    for b in range(batch):
        for q_local in range(s_q):
            idx = b * s_q + q_local
            sampled = torch.randperm(kv_pool, device=device)[:topk]
            topk_indices[idx] = sampled.int()

    cu_seqlens_q = torch.arange(0, batch + 1, dtype=torch.int32, device=device) * s_q
    seqused_k = torch.full((batch,), kv_pool, dtype=torch.int32, device=device)
    sm_scale = V4_HEAD_DIM ** -0.5

    return {
        "q": q,
        "kv_fp8": kv_fp8_view,
        "k_bf16": k_bf16_view,
        "v_bf16": v_bf16_view,
        "kv_scales": scales_view,
        "kv_dequant": kv_dequant_full,
        "topk_indices": topk_indices,
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_k": seqused_k,
        "sm_scale": sm_scale,
        "head_dim": V4_HEAD_DIM,
        "fp8_dim": V4_FP8_DIM,
        "bf16_dim": V4_BF16_DIM,
        "kv_pool": kv_pool,
        "h_q": h_q,
        "max_q": s_q,
        "block_size": 64,
        "batch": batch,
    }


def _run_kernel_split(d, return_lse=False):
    """Run the kernel with V4 split storage inputs."""
    block_table = torch.zeros((d["batch"], 1), dtype=torch.int32, device=d["q"].device)
    out = torch.empty((d["q"].shape[0], d["h_q"], d["head_dim"]),
                      dtype=torch.bfloat16, device=d["q"].device)
    lse = unified_attention_sparse_mla(
        d["q"],
        d["kv_fp8"],
        out,
        d["cu_seqlens_q"],
        d["max_q"],
        d["seqused_k"],
        d["kv_pool"],
        d["sm_scale"],
        d["topk_indices"],
        block_table,
        d["head_dim"],         # kv_lora_rank = rounded-up power-of-2 channel count
        attn_sink=None,
        return_lse=return_lse,
        kv_scales=d["kv_scales"],
        k_bf16=d["k_bf16"],
        v_bf16=d["v_bf16"],
        bf16_head_dim=d["bf16_dim"],
        fp8_segment_dim=d["fp8_dim"],
    )
    return out, lse


def _run_kernel_bf16_full(d, return_lse=False):
    """Reference: run the kernel on the host-dequantized BF16 kv (single-stride path)."""
    block_table = torch.zeros((d["batch"], 1), dtype=torch.int32, device=d["q"].device)
    out = torch.empty((d["q"].shape[0], d["h_q"], d["head_dim"]),
                      dtype=torch.bfloat16, device=d["q"].device)
    lse = unified_attention_sparse_mla(
        d["q"],
        d["kv_dequant"],
        out,
        d["cu_seqlens_q"],
        d["max_q"],
        d["seqused_k"],
        d["kv_pool"],
        d["sm_scale"],
        d["topk_indices"],
        block_table,
        d["head_dim"],
        attn_sink=None,
        return_lse=return_lse,
    )
    return out, lse


@pytest.mark.parametrize("topk", [256, 512, 1024])
@pytest.mark.parametrize("batch,s_q", [(1, 1), (4, 1), (16, 1), (1, 128), (1, 512)])
def test_fp8_split_matches_bf16_v4(topk, batch, s_q):
    """V4 split storage path matches the BF16 reference (atol=2e-2, rtol=1e-2)."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    d = _make_v4_split_inputs(batch=batch, s_q=s_q, topk=topk, kv_pool=4096)
    out_split, _ = _run_kernel_split(d)
    out_bf16,  _ = _run_kernel_bf16_full(d)
    assert out_split.shape == (batch * s_q, 128, V4_HEAD_DIM)
    assert torch.isfinite(out_split).all()
    diff = (out_split.float() - out_bf16.float()).abs()
    print(f"[V4-split topk={topk} b={batch} s_q={s_q}] "
          f"max_abs={diff.max().item():.4e} mean_abs={diff.mean().item():.4e}")
    torch.testing.assert_close(out_split, out_bf16, atol=2e-2, rtol=1e-2)


def test_fp8_split_v4_with_lse():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    d = _make_v4_split_inputs(batch=2, s_q=1, topk=512, kv_pool=4096)
    out_split, lse_split = _run_kernel_split(d, return_lse=True)
    out_bf16,  lse_bf16  = _run_kernel_bf16_full(d, return_lse=True)
    torch.testing.assert_close(out_split, out_bf16, atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(lse_split, lse_bf16, atol=5e-3, rtol=5e-3)


if __name__ == "__main__":
    # Smoke run.
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    print("=== Smoke: V4 split FP8 vs BF16 (topk=512, b=4, s_q=1) ===")
    d = _make_v4_split_inputs(batch=4, s_q=1, topk=512, kv_pool=4096)
    out_split, _ = _run_kernel_split(d)
    out_bf16,  _ = _run_kernel_bf16_full(d)
    diff = (out_split.float() - out_bf16.float()).abs()
    print(f"max_abs_diff = {diff.max().item():.4e}")
    print(f"mean_abs_diff = {diff.mean().item():.4e}")
    print(f"finite: {torch.isfinite(out_split).all().item()}")
