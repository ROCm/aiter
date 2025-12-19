# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
import aiter
from aiter.test_common import checkAllclose, perftest, benchmark
from aiter import dtypes, per_tensor_quant
from typing import List


def rms_norm_forward(x: Tensor, weight: Tensor, eps: float):
    input_dtype = x.dtype
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(input_dtype)
    return weight * x


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
    x: Tensor, cos: Tensor, sin: Tensor, is_neox_style: bool
) -> Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    return apply_rotary_emb_torch(x, cos, sin, is_neox_style)


@perftest()
def run_torch_mrope_3d_rms(
    qkv: Tensor,  # contiguous (num_tokens * (num_heads_q + num_heads_k + num_heads_v) * head_size)
    qw: Tensor,  #  contiguous (head_size)
    kw: Tensor,  #  contiguous (head_size)
    cos_sin: Tensor,  # contiguous (max_positions * head_size)
    positions: Tensor,  # contiguous (3 * num_tokens) or (num_tokens)
    num_tokens: int,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_size: int,
    is_neox_style: bool,
    mrope_section: List[int],
    is_interleaved: bool,
    eps: float,
    is_mrope: bool,
):
    q_size = num_heads_q * head_size
    k_size = num_heads_k * head_size
    v_size = num_heads_v * head_size
    qkv = qkv.view(num_tokens, q_size + k_size + v_size)
    q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)

    q_by_head = q.view(num_tokens, num_heads_q, head_size)
    q_by_head = rms_norm_forward(q_by_head, qw, eps)
    q = q_by_head.view(q.shape)

    k_by_head = k.view(num_tokens, num_heads_k, head_size)
    k_by_head = rms_norm_forward(k_by_head, kw, eps)
    k = k_by_head.view(k.shape)

    cos_sin = cos_sin.view(max_positions, head_size)
    if is_mrope:
        positions = positions.view(3, num_tokens)
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
    q = apply_rotary_emb_dispatch(q, cos, sin, is_neox_style)
    q = q.reshape(q_shape)

    k_shape = k.shape
    k = k.view(num_tokens, -1, head_size)
    k = apply_rotary_emb_dispatch(k, cos, sin, is_neox_style)
    k = k.reshape(k_shape)

    return q, k, v


@perftest()
def run_aiter_mrope_3d_rms(
    qkv: Tensor,  # contiguous (num_tokens * (num_heads_q + num_heads_k + num_heads_v) * head_size)
    qw: Tensor,  #  contiguous (head_size)
    kw: Tensor,  #  contiguous (head_size)
    cos_sin: Tensor,  # contiguous (max_positions * head_size)
    positions: Tensor,  # contiguous (3 * num_tokens)
    num_tokens: int,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_size: int,
    is_neox_style: bool,
    mrope_section: List[int],
    is_interleaved: bool,
    eps: float,
    is_mrope: bool,
):
    qkv = qkv.clone()  # inplace op

    if is_mrope:
        aiter.fused_mrope_3d_rms(
            qkv,
            qw,
            kw,
            cos_sin,
            positions,
            num_tokens,
            num_heads_q,
            num_heads_k,
            num_heads_v,
            head_size,
            is_neox_style,
            mrope_section,
            is_interleaved,
            eps,
        )
    else:
        aiter.fused_rope_rms(
            qkv,
            qw,
            kw,
            cos_sin,
            positions,
            num_tokens,
            num_heads_q,
            num_heads_k,
            num_heads_v,
            head_size,
            is_neox_style,
            eps,
        )

    q_size = num_heads_q * head_size
    k_size = num_heads_k * head_size
    v_size = num_heads_v * head_size

    qkv = qkv.view(num_tokens, q_size + k_size + v_size)
    q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)
    return q, k, v


@benchmark()
def test_mrope_3d_rms(
    dtype,
    num_tokens,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    head_size,
    is_neox_style,
    mrope_section,
    is_interleaved,
    eps,
    is_mrope,
):
    qkv = torch.randn(
        (num_tokens, num_heads_q + num_heads_k + num_heads_v, head_size),
        dtype=dtype,
        device="cuda",
    )
    qw = torch.randn(head_size, dtype=dtype, device="cuda")
    kw = torch.randn(head_size, dtype=dtype, device="cuda")
    cos_sin = torch.randn((max_positions, head_size), dtype=dtype, device="cuda")
    if is_mrope:
        pos_shape = (3, num_tokens)
    else:
        pos_shape = (num_tokens,)
    positions = torch.randint(
        0, max_positions, pos_shape, dtype=torch.int64, device="cuda"
    )

    (q_ref, k_ref, v_ref), avg_torch = run_torch_mrope_3d_rms(
        qkv,
        qw,
        kw,
        cos_sin,
        positions,
        num_tokens,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_size,
        is_neox_style,
        mrope_section,
        is_interleaved,
        eps,
        is_mrope,
    )
    (q, k, v), avg_cu = run_aiter_mrope_3d_rms(
        qkv,
        qw,
        kw,
        cos_sin,
        positions,
        num_tokens,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_size,
        is_neox_style,
        mrope_section,
        is_interleaved,
        eps,
        is_mrope,
    )

    info = f"dtype:{dtype}, num_tokens:{num_tokens}, num_heads_q:{num_heads_q}, num_heads_k:{num_heads_k}, num_heads_v:{num_heads_v}, head_size:{head_size}, is_neox_style:{is_neox_style}"
    if is_mrope:
        info += f", mrope_section:{mrope_section}, is_interleaved:{is_interleaved}, eps:{eps}"
    msg = f"[perf] === {info} === torch avg: {avg_torch:<8.2f} us, cu avg: {avg_cu:<8.2f} us, uplift: {avg_torch/avg_cu-1:<5.1%}"
    checkAllclose(q_ref, q, msg="q", rtol=1e-2, atol=0.05)
    checkAllclose(k_ref, k, msg="k", rtol=1e-2, atol=0.05)
    checkAllclose(v_ref, v, msg=msg, rtol=1e-2, atol=0.05)


@perftest()
def run_torch_mrope_3d_rms_set_kv(
    qkv: Tensor,  # contiguous (num_tokens * (num_heads_q + num_heads_k + num_heads_v) * head_size)
    qw: Tensor,  #  contiguous (head_size)
    kw: Tensor,  #  contiguous (head_size)
    cos_sin: Tensor,  # contiguous (max_positions * head_size)
    positions: Tensor,  # contiguous (3 * num_tokens)
    num_tokens: int,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_size: int,
    is_neox_style: bool,
    mrope_section: List[int],
    is_interleaved: bool,
    eps: float,
    q_out: Tensor,
    k_cache: Tensor,  # contiguous (-1, num_heads_k, head_size)
    v_cache: Tensor,  # contiguous (-1, num_heads_v, head_size)
    kv_loc: Tensor,  # contiguous (num_tokens)
    k_scale: float,
    v_scale: float,
    is_mrope: bool,
    k_out: Tensor = None,  # Optional output buffer for k
    v_out: Tensor = None,  # Optional output buffer for v
    return_kv: bool = False,  # Whether to return k_out and v_out
):
    q_size = num_heads_q * head_size
    k_size = num_heads_k * head_size
    v_size = num_heads_v * head_size
    qkv = qkv.view(num_tokens, q_size + k_size + v_size)
    q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)

    q_by_head = q.view(num_tokens, num_heads_q, head_size)
    q_by_head = rms_norm_forward(q_by_head, qw, eps)
    q = q_by_head.view(q.shape)

    k_by_head = k.view(num_tokens, num_heads_k, head_size)
    k_by_head = rms_norm_forward(k_by_head, kw, eps)
    k = k_by_head.view(k.shape)

    cos_sin = cos_sin.view(max_positions, head_size)
    if is_mrope:
        positions = positions.view(3, num_tokens)
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
    q = apply_rotary_emb_dispatch(q, cos, sin, is_neox_style)
    q = q.reshape(q_shape)

    k_shape = k.shape
    k = k.view(num_tokens, -1, head_size)
    k = apply_rotary_emb_dispatch(k, cos, sin, is_neox_style)
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
        k_quantized, _ = per_tensor_quant(k_for_quant, scale=torch.tensor(k_scale, device=k_for_quant.device), quant_dtype=kv_cache_dtype)
        v_quantized, _ = per_tensor_quant(v_for_quant, scale=torch.tensor(v_scale, device=v_for_quant.device), quant_dtype=kv_cache_dtype)
    
    # Store k and v to cache using kv_loc indexing
    k_cache[kv_loc] = k_quantized
    v_cache[kv_loc] = v_quantized
    # q_out shape is [num_tokens, num_heads_q, head_size]
    # q shape after reshape is [num_tokens, q_size] where q_size = num_heads_q * head_size
    q_out.copy_(q.view(num_tokens, num_heads_q, head_size))
    
    # Return k_out and v_out if requested
    # k_out and v_out should match k_cache[kv_loc] and v_cache[kv_loc] respectively
    # In kernel: k_out is stored at token_id order, k_cache is stored at kv_loc[token_id] order
    # So k_out should equal k_quantized (token_id order), and k_cache[kv_loc] should also equal k_quantized
    if return_kv and k_out is not None and v_out is not None:
        # k_out and v_out are stored in token_id order, same as k_quantized and v_quantized
        k_out.copy_(k_quantized)
        v_out.copy_(v_quantized)
    
    return None


@perftest()
def run_fused_mrope_3d_rms_set_kv(
    qkv: Tensor,  # contiguous (num_tokens * (num_heads_q + num_heads_k + num_heads_v) * head_size)
    qw: Tensor,  #  contiguous (head_size)
    kw: Tensor,  #  contiguous (head_size)
    cos_sin: Tensor,  # contiguous (max_positions * head_size)
    positions: Tensor,  # contiguous (3 * num_tokens)
    num_tokens: int,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_size: int,
    is_neox_style: bool,
    mrope_section: List[int],
    is_interleaved: bool,
    eps: float,
    q_out: Tensor,
    k_cache: Tensor,  # contiguous (-1, num_heads_k, head_size)
    v_cache: Tensor,  # contiguous (-1, num_heads_v, head_size)
    kv_loc: Tensor,  # contiguous (num_tokens)
    k_scale: float,
    v_scale: float,
    is_mrope: bool,
    k_out: Tensor = None,  # Optional output buffer for k
    v_out: Tensor = None,  # Optional output buffer for v
    return_kv: bool = False,  # Whether to return k_out and v_out
):
    qkv = qkv.clone()  # inplace op
    if is_mrope:
        aiter.fused_mrope_3d_rms_set_kv(
            qkv,
            qw,
            kw,
            cos_sin,
            positions,
            num_tokens,
            num_heads_q,
            num_heads_k,
            num_heads_v,
            head_size,
            is_neox_style,
            mrope_section,
            is_interleaved,
            eps,
            q_out,
            k_cache,
            v_cache,
            kv_loc,
            k_scale,
            v_scale,
            k_out,
            v_out,
            return_kv,
        )
    else:
        # For non-mrope case, use fused_rope_rms_set_kv (now supports k_out/v_out)
        aiter.fused_rope_rms_set_kv(
            qkv,
            qw,
            kw,
            cos_sin,
            positions,
            num_tokens,
            num_heads_q,
            num_heads_k,
            num_heads_v,
            head_size,
            is_neox_style,
            eps,
            q_out,
            k_cache,
            v_cache,
            kv_loc,
            k_scale,
            v_scale,
            k_out,
            v_out,
            return_kv,
        )
    return None


@benchmark()
def test_mrope_3d_rms_set_kv(
    dtype,
    num_tokens,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    head_size,
    is_neox_style,
    mrope_section,
    is_interleaved,
    eps,
    is_mrope,
    kv_cache_dtype=None,  # Optional: specify KV cache dtype (e.g., torch.float8_e4m3fn)
    test_return_kv=False,  # Whether to test k_out and v_out return
):
    qkv = torch.randn(
        (num_tokens, num_heads_q + num_heads_k + num_heads_v, head_size),
        dtype=dtype,
        device="cuda",
    )
    qw = torch.randn(head_size, dtype=dtype, device="cuda")
    kw = torch.randn(head_size, dtype=dtype, device="cuda")
    cos_sin = torch.randn((max_positions, head_size), dtype=dtype, device="cuda")
    if is_mrope:
        pos_shape = (3, num_tokens)
    else:
        pos_shape = (num_tokens,)
    positions = torch.randint(
        0, max_positions, pos_shape, dtype=torch.int64, device="cuda"
    )

    q_out_ref = torch.empty(
        num_tokens, num_heads_q, head_size, dtype=dtype, device="cuda"
    )
    q_out = torch.empty(num_tokens, num_heads_q, head_size, dtype=dtype, device="cuda")
    
    # Determine KV cache dtype
    # Use the same logic as sglang: fp8_e4m3 maps to float8_e4m3fnuz on HIP, float8_e4m3fn on CUDA
    if kv_cache_dtype is None:
        # Use aiter's default FP8 dtype which matches the hardware (gfx942 -> fnuz, gfx950 -> fn)
        kv_cache_dtype = dtypes.fp8  # This will be torch.float8_e4m3fnuz on HIP (gfx942) or torch.float8_e4m3fn on CUDA/gfx950
    
    k_cache_ref = torch.rand(max_positions, num_heads_k, head_size, device="cuda").to(
        kv_cache_dtype
    )
    v_cache_ref = torch.rand(max_positions, num_heads_v, head_size, device="cuda").to(
        kv_cache_dtype
    )
    k_cache = k_cache_ref.clone()
    v_cache = v_cache_ref.clone()
    kv_loc = torch.randint(
        0, max_positions, (num_tokens,), dtype=torch.int64, device="cuda"
    )
    k_scale = 1.5
    v_scale = 2.0

    # Create k_out and v_out buffers if testing return_kv
    k_out_ref = None
    v_out_ref = None
    k_out = None
    v_out = None
    if test_return_kv:
        k_out_ref = torch.empty(num_tokens, num_heads_k, head_size, dtype=kv_cache_dtype, device="cuda")
        v_out_ref = torch.empty(num_tokens, num_heads_v, head_size, dtype=kv_cache_dtype, device="cuda")
        k_out = torch.empty(num_tokens, num_heads_k, head_size, dtype=kv_cache_dtype, device="cuda")
        v_out = torch.empty(num_tokens, num_heads_v, head_size, dtype=kv_cache_dtype, device="cuda")

    _, avg_torch = run_torch_mrope_3d_rms_set_kv(
        qkv,
        qw,
        kw,
        cos_sin,
        positions,
        num_tokens,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_size,
        is_neox_style,
        mrope_section,
        is_interleaved,
        eps,
        q_out_ref,
        k_cache_ref,
        v_cache_ref,
        kv_loc,
        k_scale,
        v_scale,
        is_mrope,
        k_out_ref,
        v_out_ref,
        test_return_kv,
    )
    _, avg_cu = run_fused_mrope_3d_rms_set_kv(
        qkv,
        qw,
        kw,
        cos_sin,
        positions,
        num_tokens,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_size,
        is_neox_style,
        mrope_section,
        is_interleaved,
        eps,
        q_out,
        k_cache,
        v_cache,
        kv_loc,
        k_scale,
        v_scale,
        is_mrope,
        k_out,
        v_out,
        test_return_kv,
    )

    info = f"dtype:{dtype}, kv_cache_dtype:{kv_cache_dtype}, num_tokens:{num_tokens}, num_heads_q:{num_heads_q}, num_heads_k:{num_heads_k}, num_heads_v:{num_heads_v}, head_size:{head_size}, is_neox_style:{is_neox_style}"
    if is_mrope:
        info += f", mrope_section:{mrope_section}, is_interleaved:{is_interleaved}, eps:{eps}"
    if test_return_kv:
        info += f", return_kv:{test_return_kv}"
    msg = f"[perf] === {info} === torch avg: {avg_torch:<8.2f} us, cu avg: {avg_cu:<8.2f} us, uplift: {avg_torch/avg_cu-1:<5.1%}"
    
    checkAllclose(q_out_ref, q_out, msg="q_out", rtol=1e-2, atol=0.05)
    checkAllclose(
        k_cache_ref[kv_loc].float(),
        k_cache[kv_loc].float(),
        msg="k_cache",
        rtol=1e-2,
        atol=0.05,
    )
    checkAllclose(
        v_cache_ref[kv_loc].float(),
        v_cache[kv_loc].float(),
        msg="v_cache",
        rtol=1e-2,
        atol=0.05,
    )
    
    # Verify k_out and v_out if return_kv is enabled
    if test_return_kv and k_out is not None and v_out is not None:
        checkAllclose(
            k_out_ref.float(),
            k_out.float(),
            msg="k_out",
            rtol=1e-2,
            atol=0.05,
        )
        checkAllclose(
            v_out_ref.float(),
            v_out.float(),
            msg="v_out",
            rtol=1e-2,
            atol=0.05,
        )


if __name__ == "__main__":

    print("\n\n================== test_rope_rms ==================\n\n")
    is_neox_styles = [True, False]
    num_tokens = [513, 1257, 127, 778, 10024, 3]
    num_heads = [32, 64]
    head_sizes = [64, 128, 256]
    max_positions = 10000
    dtype = torch.bfloat16
    for is_neox_style in is_neox_styles:
        for num_token in num_tokens:
            for num_head in num_heads:
                for i, head_size in enumerate(head_sizes):
                    test_mrope_3d_rms(
                        dtype,
                        num_token,
                        num_head,
                        num_head,
                        num_head,
                        head_size,
                        is_neox_style,
                        None,
                        None,
                        eps=1e-6,
                        is_mrope=False,
                    )

    print("\n\n================== test_mrope_3d_rms ==================\n\n")
    is_neox_styles = [True, False]
    num_tokens = [513, 1257, 127, 778, 10024, 3]
    num_heads = [32, 64]
    head_sizes = [64, 128, 256]
    mrope_sections = [[12, 10, 10], [24, 20, 20], [48, 40, 40]]
    is_interleaveds = [True, False]
    max_positions = 10000
    dtype = torch.bfloat16
    for is_neox_style in is_neox_styles:
        for num_token in num_tokens:
            for num_head in num_heads:
                for i, head_size in enumerate(head_sizes):
                    ms = mrope_sections[i]
                    for is_interleaved in is_interleaveds:
                        test_mrope_3d_rms(
                            dtype,
                            num_token,
                            num_head,
                            num_head,
                            num_head,
                            head_size,
                            is_neox_style,
                            ms,
                            is_interleaved,
                            eps=1e-6,
                            is_mrope=True,
                        )
    print("\n\n================== test_rope_rms_set_kv ==================\n\n")
    is_neox_styles = [True, False]
    num_tokens = [513, 1257, 127, 778, 10024, 3]
    num_heads = [32, 64]
    head_sizes = [64, 128, 256]
    max_positions = 10000
    dtype = torch.bfloat16
    kv_cache_dtypes = [torch.bfloat16, torch.float8_e4m3fn, torch.float8_e4m3fnuz]
    test_return_kv_flags = [True, False]
    
    for kv_cache_dtype in kv_cache_dtypes:
        for test_return_kv in test_return_kv_flags:
            for is_neox_style in is_neox_styles:
                for num_token in num_tokens:
                    for num_head in num_heads:
                        for i, head_size in enumerate(head_sizes):
                            test_mrope_3d_rms_set_kv(
                                dtype,
                                num_token,
                                num_head,
                                num_head,
                                num_head,
                                head_size,
                                is_neox_style,
                                None,
                                None,
                                eps=1e-6,
                                is_mrope=False,
                                kv_cache_dtype=kv_cache_dtype,
                                test_return_kv=test_return_kv,
                            )

    print("\n\n================== test_mrope_3d_rms_set_kv ==================\n\n")
    is_neox_styles = [True, False]
    num_tokens = [513, 1257, 127, 778, 10024, 3]
    num_heads = [32, 64]
    head_sizes = [64, 128, 256]
    mrope_sections = [[12, 10, 10], [24, 20, 20], [48, 40, 40]]
    is_interleaveds = [True, False]
    max_positions = 10000
    dtype = torch.bfloat16
    kv_cache_dtypes = [torch.bfloat16, torch.float8_e4m3fn, torch.float8_e4m3fnuz]
    test_return_kv_flags = [True, False]

    for kv_cache_dtype in kv_cache_dtypes:
        for test_return_kv in test_return_kv_flags:
            for is_neox_style in is_neox_styles:
                for num_token in num_tokens:
                    for num_head in num_heads:
                        for i, head_size in enumerate(head_sizes):
                            ms = mrope_sections[i]
                            for is_interleaved in is_interleaveds:
                                test_mrope_3d_rms_set_kv(
                                    dtype,
                                    num_token,
                                    num_head,
                                    num_head,
                                    num_head,
                                    head_size,
                                    is_neox_style,
                                    ms,
                                    is_interleaved,
                                    eps=1e-6,
                                    is_mrope=True,
                                    kv_cache_dtype=kv_cache_dtype,
                                    test_return_kv=test_return_kv,
                                )
 

    print("done")
