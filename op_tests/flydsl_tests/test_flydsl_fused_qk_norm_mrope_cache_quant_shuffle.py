# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and perf sweep for the FlyDSL and HIP fused QK-norm / 3D-MRoPE / cache ops
for the feature subset implemented for FlyDSL
"""

import argparse
import itertools

import pandas as pd
import torch
from torch import Tensor

import aiter
from aiter import dtypes, per_tensor_quant
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import (
    flydsl_fused_qk_norm_mrope_3d_cache_pts_quant_shuffle,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

# The FlyDSL wrapper dispatches wave64 / wave32 internally. Positive allow-list
# so an unknown card does not try to JIT an unvalidated kernel.
SUPPORTED_GFX = ["gfx942", "gfx950", "gfx1250"]

EPS = 1e-6
MAX_POSITIONS = 4096
MROPE_SECTIONS = {
    64: [12, 10, 10],
    128: [24, 20, 20],
    256: [48, 40, 40],
}
# Same tolerances the HIP test uses against this reference.
_RTOL = 1e-2
_ATOL = 0.05


def rms_norm_forward(x: Tensor, weight: Tensor, eps: float):
    input_dtype = x.dtype
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(input_dtype)
    return weight * x


def gemma_rms_norm_forward(x: Tensor, weight: Tensor, eps: float):
    input_dtype = x.dtype
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(input_dtype)
    return (1.0 + weight) * x


def apply_interleaved_rope(x: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
    """Apply interleaved MRoPE to 3D rotary embeddings.

    Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
    interleaved [THTHWHTHW...TT], preserving frequency continuity.
    """
    x_t = x[0].clone()
    x_t[..., 1 : mrope_section[1] * 3 : 3] = x[1, ..., 1 : mrope_section[1] * 3 : 3]
    x_t[..., 2 : mrope_section[2] * 3 : 3] = x[2, ..., 2 : mrope_section[2] * 3 : 3]
    return x_t


def apply_rotary_emb_torch(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    is_neox_style: bool,
) -> Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


def apply_rotary_emb_dispatch(
    x: Tensor, cos: Tensor, sin: Tensor, is_neox_style: bool, rotary_dim: int = 0
) -> Tensor:
    """Apply rotary embeddings. If rotary_dim > 0 and < head_size, only the
    first rotary_dim elements are rotated; the rest pass through unchanged."""
    head_size = x.shape[-1]
    rd = rotary_dim if rotary_dim > 0 else head_size
    if rd < head_size:
        x_rot = apply_rotary_emb_torch(x[..., :rd], cos, sin, is_neox_style)
        return torch.cat((x_rot, x[..., rd:]), dim=-1)
    return apply_rotary_emb_torch(x, cos, sin, is_neox_style)


def set_kv_cache_shuffle_layout(
    k_quantized: Tensor,  # [num_tokens, num_kv_heads, head_size] - already quantized
    v_quantized: Tensor,  # [num_tokens, num_kv_heads, head_size] - already quantized
    k_cache: Tensor,  # [num_blocks, ...], packed or dim-0-strided shuffle layout
    v_cache: Tensor,  # [num_blocks, ...], packed or dim-0-strided shuffle layout
    kv_loc: Tensor,  # [num_tokens]
    block_size: int,
    x: int,
):
    """Set KV cache with shuffle layout using torch indexing.

    This is reference-only code; keep it separate from the timed candidates.
    - Key shuffle: [num_blocks, num_kv_heads, head_size // x, block_size, x]
    - Value shuffle: [num_blocks, num_kv_heads, block_size // x, head_size, x]
    Dim 0 may be strided. Each block is still packed.
    """
    _, num_kv_heads, head_size = k_quantized.shape
    valid = kv_loc >= 0
    if not valid.any():
        return

    slots = kv_loc[valid]
    block_ids = slots // block_size
    block_offsets = slots % block_size
    num_blocks = k_cache.shape[0]

    # Reinterpret the packed per-block storage in the two physical shuffle
    # layouts. Using explicit strides also handles a dim-0-strided cache view.
    k_view = k_cache.as_strided(
        (num_blocks, num_kv_heads, head_size // x, block_size, x),
        (
            k_cache.stride(0),
            head_size * block_size,
            block_size * x,
            x,
            1,
        ),
    )
    v_view = v_cache.as_strided(
        (num_blocks, num_kv_heads, block_size // x, head_size, x),
        (
            v_cache.stride(0),
            head_size * block_size,
            head_size * x,
            x,
            1,
        ),
    )
    k_view[block_ids, :, :, block_offsets, :] = k_quantized[valid].view(
        -1, num_kv_heads, head_size // x, x
    )
    v_view[block_ids, :, block_offsets // x, :, block_offsets % x] = v_quantized[valid]


def run_torch(
    qkv: Tensor,  # contiguous (num_tokens * (num_heads_q + num_heads_k + num_heads_v) * head_size)
    qw: Tensor,  #  contiguous (head_size)
    kw: Tensor,  #  contiguous (head_size)
    cos_sin: Tensor,  # contiguous (max_positions * head_size)
    positions: Tensor,  # (3, num_tokens) or flat 3 * num_tokens; both strides are honored
    num_tokens: int,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_size: int,
    is_neox_style: bool,
    mrope_section: list[int],
    is_interleaved: bool,
    eps: float,
    q_out: Tensor,
    k_cache: Tensor,  # shuffle layout; dim 0 may be strided, each block is packed
    v_cache: Tensor,  # shuffle layout; dim 0 may be strided, each block is packed
    kv_loc: Tensor,  # contiguous (num_tokens)
    k_scale: float,
    v_scale: float,
    is_mrope: bool,
    k_out: Tensor = None,  # Optional output buffer for k
    v_out: Tensor = None,  # Optional output buffer for v
    return_kv: bool = False,  # Whether to return k_out and v_out
    use_shuffle_layout: bool = False,  # Whether to use shuffle layout
    page_size: int = 0,  # Page size (block_size) for shuffle layout
    rotary_dim: int = 0,  # Partial rotary dim (0 = full rotary = head_size)
    gemma_norm: bool = False,
):
    """Torch reference lifted from the HIP kernel test. Not timed."""
    rotary_dim_ = rotary_dim if rotary_dim > 0 else head_size
    q_size = num_heads_q * head_size
    k_size = num_heads_k * head_size
    v_size = num_heads_v * head_size
    qkv = qkv.view(num_tokens, q_size + k_size + v_size)
    q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)

    q_by_head = q.view(num_tokens, num_heads_q, head_size)
    norm_fn = gemma_rms_norm_forward if gemma_norm else rms_norm_forward
    q_by_head = norm_fn(q_by_head, qw, eps)
    q = q_by_head.view(q.shape)

    k_by_head = k.view(num_tokens, num_heads_k, head_size)
    k_by_head = norm_fn(k_by_head, kw, eps)
    k = k_by_head.view(k.shape)

    # Infer max_positions from cos_sin shape
    cos_sin_dim = rotary_dim_
    max_positions = (
        cos_sin.shape[0] // cos_sin_dim if cos_sin.ndim == 1 else cos_sin.shape[0]
    )
    cos_sin = cos_sin.view(max_positions, cos_sin_dim)
    if is_mrope:
        # reshape keeps a strided (3, num_tokens) view, such as storage[:, ::2],
        # and still accepts a contiguous flat buffer of 3 * num_tokens ids.
        positions = positions.reshape(3, num_tokens)
    cos_sin = cos_sin[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)

    if is_mrope:
        if is_interleaved:
            cos = apply_interleaved_rope(cos, mrope_section)
            sin = apply_interleaved_rope(sin, mrope_section)
        else:
            cos = torch.cat(
                [m[i] for i, m in enumerate(cos.split(mrope_section, dim=-1))],
                dim=-1,
            )
            sin = torch.cat(
                [m[i] for i, m in enumerate(sin.split(mrope_section, dim=-1))],
                dim=-1,
            )

    q_shape = q.shape
    q = q.view(num_tokens, -1, head_size)
    q = apply_rotary_emb_dispatch(q, cos, sin, is_neox_style, rotary_dim_)
    q = q.reshape(q_shape)

    k_shape = k.shape
    k = k.view(num_tokens, -1, head_size)
    k = apply_rotary_emb_dispatch(k, cos, sin, is_neox_style, rotary_dim_)
    k = k.reshape(k_shape)

    # Quantize k and v for cache storage
    # Reshape k and v to [num_tokens, num_heads, head_size] before quantization
    k_for_quant = k.view(num_tokens, num_heads_k, head_size)
    v_for_quant = v.view(num_tokens, num_heads_v, head_size)
    # Use the actual k_scale and v_scale parameters, and ensure quant_dtype matches kv_cache_dtype
    kv_cache_dtype = k_cache.dtype
    qkv_dtype = qkv.dtype

    # When kv_cache_dtype == qkv_dtype, kernel directly stores without quantization
    # Only quantize when types differ (e.g., fp8)
    if kv_cache_dtype == qkv_dtype:
        k_quantized = k_for_quant.to(kv_cache_dtype)
        v_quantized = v_for_quant.to(kv_cache_dtype)
    else:
        k_quantized, _ = per_tensor_quant(
            k_for_quant,
            scale=torch.tensor(k_scale, device=k_for_quant.device),
            quant_dtype=kv_cache_dtype,
        )
        v_quantized, _ = per_tensor_quant(
            v_for_quant,
            scale=torch.tensor(v_scale, device=v_for_quant.device),
            quant_dtype=kv_cache_dtype,
        )

    # Store k and v to cache using kv_loc indexing
    if use_shuffle_layout:
        # Calculate x for shuffle layout: x = 16 // k_cache.element_size()
        x = (
            16
            // torch.empty(
                0, dtype=kv_cache_dtype, device=k_cache.device
            ).element_size()
        )
        # Use shuffle layout implementation (k_quantized and v_quantized are already quantized)
        set_kv_cache_shuffle_layout(
            k_quantized,
            v_quantized,
            k_cache,
            v_cache,
            kv_loc,
            page_size,
            x,
        )
    else:
        # Normal layout: [num_slots, num_kv_heads, head_size]
        k_cache[kv_loc] = k_quantized
        v_cache[kv_loc] = v_quantized
    # q_out shape is [num_tokens, num_heads_q, head_size]
    # q shape after reshape is [num_tokens, q_size] where q_size = num_heads_q * head_size
    q_out.copy_(q.view(num_tokens, num_heads_q, head_size))

    # Flat outputs are in token order. Tokens with negative slots are skipped,
    # matching both kernels, so their preinitialized output values are preserved.
    if return_kv and k_out is not None and v_out is not None:
        valid_slots = kv_loc >= 0
        k_out[valid_slots] = k_quantized[valid_slots]
        v_out[valid_slots] = v_quantized[valid_slots]


def _check(name, ref, out, msg, rtol=_RTOL, atol=_ATOL, tol_err_ratio=0.05):
    err = checkAllclose(
        ref.to(dtypes.fp32),
        out.to(dtypes.fp32),
        rtol=rtol,
        atol=atol,
        tol_err_ratio=tol_err_ratio,
        msg=f"{name}: {msg}",
    )
    assert err <= tol_err_ratio, f"{name}: {msg} mismatch ratio {err}"
    return err


@benchmark()
def test_fused_qk_norm_mrope_cache_quant_shuffle(
    num_tokens,
    num_q_heads,
    num_kv_heads,
    head_size,
    cache_dtype,
    page_size,
    interleaved,
    slot_pattern,
    strided_positions,
    strided_caches,
    gemma_norm,
    return_kv,
):
    torch.manual_seed(0)
    num_blocks = max(4, (num_tokens + page_size - 1) // page_size)
    x = 16 // torch.empty((), dtype=cache_dtype).element_size()
    sections = MROPE_SECTIONS[head_size]
    total_heads = num_q_heads + 2 * num_kv_heads

    qkv = torch.randn(num_tokens, total_heads * head_size, dtype=dtypes.bf16)
    qw = torch.randn(head_size, dtype=dtypes.bf16)
    kw = torch.randn(head_size, dtype=dtypes.bf16)
    cos_sin = torch.randn(MAX_POSITIONS, head_size, dtype=dtypes.bf16) * 0.25
    positions_storage = torch.randint(
        0,
        MAX_POSITIONS,
        (3, num_tokens * (2 if strided_positions else 1)),
        dtype=torch.int64,
    )
    # Non-contiguous positions: the kernel honors both strides. The reference
    # only needs the index values, so it receives a contiguous copy.
    positions = positions_storage[:, ::2] if strided_positions else positions_storage

    if slot_pattern == "aligned":
        slots = torch.arange(num_tokens, dtype=torch.int64)
    elif slot_pattern == "random":
        slots = torch.randperm(num_blocks * page_size, dtype=torch.int64)[:num_tokens]
    elif slot_pattern == "negative":
        slots = torch.randperm(num_blocks * page_size, dtype=torch.int64)[:num_tokens]
        slots[0] = -1
    else:
        raise ValueError(f"unknown slot pattern: {slot_pattern}")

    # Match the model's physical shuffle layouts:
    # K=[blocks, heads, D/x, page, x], V=[blocks, heads, page/x, D, x].
    k_cache_shape = (num_blocks, num_kv_heads, head_size // x, page_size, x)
    v_cache_shape = (num_blocks, num_kv_heads, page_size // x, head_size, x)
    initial_k_cache = torch.randn(k_cache_shape, dtype=dtypes.bf16).to(cache_dtype)
    initial_v_cache = torch.randn(v_cache_shape, dtype=dtypes.bf16).to(cache_dtype)
    k_scale = torch.tensor(1.5, dtype=torch.float32)
    v_scale = torch.tensor(2.0, dtype=torch.float32)

    def alloc_caches(strided):
        if not strided:
            return initial_k_cache.clone(), initial_v_cache.clone()
        # Model the common vLLM cache allocation [num_blocks, 2, ...].
        # Slicing out K/V doubles dim-0's stride while preserving packed blocks.
        k_storage = torch.empty((num_blocks, 2, *k_cache_shape[1:]), dtype=cache_dtype)
        v_storage = torch.empty((num_blocks, 2, *v_cache_shape[1:]), dtype=cache_dtype)
        k_cache = k_storage[:, 0]
        v_cache = v_storage[:, 1]
        k_cache.copy_(initial_k_cache)
        v_cache.copy_(initial_v_cache)
        return k_cache, v_cache

    def alloc_outs(strided):
        q_out = torch.empty(num_tokens, num_q_heads, head_size, dtype=dtypes.bf16)
        k_cache, v_cache = alloc_caches(strided)
        if return_kv:
            # Kernels intentionally leave negative-slot rows untouched.
            k_out = torch.zeros(num_tokens, num_kv_heads, head_size, dtype=cache_dtype)
            v_out = torch.zeros(num_tokens, num_kv_heads, head_size, dtype=cache_dtype)
        else:
            k_out = None
            v_out = None
        return q_out, k_cache, v_cache, k_out, v_out

    q_ref, k_cache_ref, v_cache_ref, k_out_ref, v_out_ref = alloc_outs(strided_caches)
    run_torch(
        qkv,
        qw,
        kw,
        cos_sin,
        positions.contiguous(),
        num_tokens,
        num_q_heads,
        num_kv_heads,
        num_kv_heads,
        head_size,
        True,
        sections,
        interleaved,
        EPS,
        q_ref,
        k_cache_ref,
        v_cache_ref,
        slots,
        float(k_scale),
        float(v_scale),
        True,
        k_out_ref,
        v_out_ref,
        return_kv,
        True,
        page_size,
        head_size,
        gemma_norm,
    )
    torch.cuda.synchronize()

    q_fly, k_fly, v_fly, k_out_fly, v_out_fly = alloc_outs(strided_caches)
    q_hip, k_hip, v_hip, k_out_hip, v_out_hip = alloc_outs(strided_caches)

    def _launch(fn, q_out, k_cache, v_cache, k_out, v_out):
        fn(
            qkv,
            qw,
            kw,
            cos_sin,
            positions,
            num_tokens,
            num_q_heads,
            num_kv_heads,
            num_kv_heads,
            head_size,
            True,
            sections,
            interleaved,
            EPS,
            q_out,
            k_cache,
            v_cache,
            slots,
            k_scale,
            v_scale,
            k_out,
            v_out,
            return_kv,
            True,
            page_size,
            x,
            head_size,
            gemma_norm,
        )

    candidates = {
        "flydsl": lambda: _launch(
            flydsl_fused_qk_norm_mrope_3d_cache_pts_quant_shuffle,
            q_fly,
            k_fly,
            v_fly,
            k_out_fly,
            v_out_fly,
        ),
        "hip": lambda: _launch(
            aiter.fused_qk_norm_mrope_3d_cache_pts_quant_shuffle,
            q_hip,
            k_hip,
            v_hip,
            k_out_hip,
            v_out_hip,
        ),
    }
    outputs = {
        "flydsl": (q_fly, k_fly, v_fly, k_out_fly, v_out_fly),
        "hip": (q_hip, k_hip, v_hip, k_out_hip, v_out_hip),
    }

    # RMSNorm ~5 FLOP/elem and NEOX RoPE ~3 FLOP/elem on Q and K.
    # Per-tensor quant is one div/elem on K and V when the cache is not bf16.
    norm_elems = num_tokens * (num_q_heads + num_kv_heads) * head_size
    quant_elems = (
        0 if cache_dtype == dtypes.bf16 else num_tokens * 2 * num_kv_heads * head_size
    )
    flops = 5 * norm_elems + 3 * norm_elems + quant_elems
    cache_bytes = k_fly.element_size()
    nbytes = (
        qkv.nbytes
        + qw.nbytes
        + kw.nbytes
        + 3 * num_tokens * head_size * cos_sin.element_size()
        + positions.numel() * positions.element_size()
        + slots.nbytes
        + k_scale.nbytes
        + v_scale.nbytes
        + q_fly.nbytes
        + 2 * num_tokens * num_kv_heads * head_size * cache_bytes
    )
    if return_kv:
        nbytes += 2 * num_tokens * num_kv_heads * head_size * cache_bytes

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        _, us = run_perftest(fn)
        q_out, k_cache, v_cache, k_out, v_out = outputs[name]
        err = _check(name, q_ref, q_out, "q_out")
        err = max(
            err,
            _check(name, k_cache_ref, k_cache, "k_cache"),
        )
        err = max(
            err,
            _check(name, v_cache_ref, v_cache, "v_cache"),
        )
        if return_kv:
            err = max(
                err,
                _check(name, k_out_ref, k_out, "k_out"),
            )
            err = max(
                err,
                _check(name, v_out_ref, v_out, "v_out"),
            )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6 if us > 0 else 0
        ret[f"{name} TB/s"] = nbytes / us / 1e6 if us > 0 else 0
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "fused_qk_norm_mrope_cache_quant_shuffle unsupported on %s; skipping",
            get_gfx(),
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL fused QK-norm / 3D-MRoPE / shuffle-cache sweep",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.fp8],
        help="""KV cache dtype. qkv stays bf16.
    e.g.: -d fp8
          -d bf16""",
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        nargs="*",
        default=[63, 64, 256, 4096, 10885, 20317, 29136, 30584, 32768],
        help="""Token count.
    e.g.: -t 64 256""",
    )
    parser.add_argument("--num-q-heads", type=int, nargs="*", default=[64])
    parser.add_argument("--num-kv-heads", type=int, nargs="*", default=[4])
    parser.add_argument(
        "--head-sizes",
        type=int,
        nargs="*",
        choices=list(MROPE_SECTIONS),
        default=[128],
    )
    parser.add_argument(
        "--page-sizes",
        type=int,
        nargs="*",
        default=[64],
    )
    parser.add_argument(
        "--interleaved",
        type=dtypes.str2bool,
        nargs="*",
        default=[True],
    )
    parser.add_argument(
        "--slot-patterns",
        nargs="*",
        choices=["aligned", "random", "negative"],
        default=["aligned", "random"],
    )
    parser.add_argument(
        "--strided-positions",
        type=dtypes.str2bool,
        nargs="*",
        default=[False],
    )
    parser.add_argument(
        "--strided-caches",
        type=dtypes.str2bool,
        nargs="*",
        default=[False],
    )
    parser.add_argument(
        "--gemma-norm",
        type=dtypes.str2bool,
        nargs="*",
        default=[False],
    )
    parser.add_argument(
        "--return-kv",
        type=dtypes.str2bool,
        nargs="*",
        default=[False],
    )
    args = parser.parse_args()

    df = []
    for (
        cache_dtype,
        num_tokens,
        num_q_heads,
        num_kv_heads,
        head_size,
        page_size,
        interleaved,
        slot_pattern,
        strided_positions,
        strided_caches,
        gemma_norm,
        return_kv,
    ) in itertools.product(
        args.dtype,
        args.tokens,
        args.num_q_heads,
        args.num_kv_heads,
        args.head_sizes,
        args.page_sizes,
        args.interleaved,
        args.slot_patterns,
        args.strided_positions,
        args.strided_caches,
        args.gemma_norm,
        args.return_kv,
    ):
        df.append(
            test_fused_qk_norm_mrope_cache_quant_shuffle(
                num_tokens,
                num_q_heads,
                num_kv_heads,
                head_size,
                cache_dtype,
                page_size,
                interleaved,
                slot_pattern,
                strided_positions,
                strided_caches,
                gemma_norm,
                return_kv,
            )
        )
    df = pd.DataFrame(df)
    aiter.logger.info(
        "fused_qk_norm_mrope_cache_quant_shuffle summary (markdown):\n%s",
        df.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
