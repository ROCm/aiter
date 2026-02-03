# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
import aiter
from aiter.test_common import checkAllclose, perftest, benchmark
from aiter import per_tensor_quant
from aiter.utility import dtypes
from typing import List
import argparse
import triton
import triton.language as tl


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


@triton.jit
def set_kv_cache_shuffle_kernel(
    key_ptr,  # [num_tokens, num_kv_heads, head_size]
    value_ptr,  # [num_tokens, num_kv_heads, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size // x, block_size, x]
    value_cache_ptr,  # [num_blocks, num_kv_heads, block_size // x, head_size, x]
    slot_mapping_ptr,  # [num_tokens]
    x,
    k_stride0,
    v_stride0,
    block_size,
    head_size,
    num_kv_heads,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel to set KV cache with shuffle layout.

    This implements the same shuffle logic as reshape_and_cache_shuffle_kernel:
    - Key shuffle: [num_blocks, num_kv_heads, head_size // x, block_size, x]
    - Value shuffle: [num_blocks, num_kv_heads, block_size // x, head_size, x]
    """
    tid = tl.program_id(0)
    head_id = tl.program_id(1)
    offset = tl.arange(0, BLOCK_SIZE)
    src_offset_k = tid * k_stride0 + head_id * head_size
    src_offset_v = tid * v_stride0 + head_id * head_size
    slot_id = tl.load(slot_mapping_ptr + tid)
    if slot_id < 0:
        return
    block_id = slot_id // block_size
    block_offset = slot_id % block_size
    dst_offset = (
        block_id * num_kv_heads * head_size * block_size
        + head_id * head_size * block_size
    )
    dst_k_shuffle_offset = (
        dst_offset + offset // x * block_size * x + block_offset * x + offset % x
    )
    dst_v_shuffle_offset = (
        dst_offset + block_offset // x * head_size * x + offset * x + block_offset % x
    )
    k_val = tl.load(key_ptr + src_offset_k + offset)
    v_val = tl.load(value_ptr + src_offset_v + offset)
    tl.store(key_cache_ptr + dst_k_shuffle_offset, k_val)
    tl.store(value_cache_ptr + dst_v_shuffle_offset, v_val)


def set_kv_cache_shuffle_layout(
    k_quantized: Tensor,  # [num_tokens, num_kv_heads, head_size] - already quantized
    v_quantized: Tensor,  # [num_tokens, num_kv_heads, head_size] - already quantized
    k_cache: Tensor,  # [num_blocks, num_kv_heads, head_size // x, block_size, x]
    v_cache: Tensor,  # [num_blocks, num_kv_heads, block_size // x, head_size, x]
    kv_loc: Tensor,  # [num_tokens]
    block_size: int,
    x: int,
):
    """Set KV cache with shuffle layout using Triton kernel.

    This implements the same shuffle logic as the triton kernel:
    - Key shuffle: [num_blocks, num_kv_heads, head_size // x, block_size, x]
    - Value shuffle: [num_blocks, num_kv_heads, block_size // x, head_size, x]
    """
    num_tokens = kv_loc.shape[0]
    _, num_kv_heads, head_size = k_quantized.shape
    num_blocks = k_cache.shape[0]

    # Create template tensors for view_as
    k_cache_template = torch.empty(
        [num_blocks, num_kv_heads, head_size // x, block_size, x],
        dtype=k_cache.dtype,
        device="meta",
    )
    v_cache_template = torch.empty(
        [num_blocks, num_kv_heads, block_size // x, head_size, x],
        dtype=v_cache.dtype,
        device="meta",
    )
    new_key_cache = k_cache.view_as(k_cache_template)
    new_value_cache = v_cache.view_as(v_cache_template)

    grid = (num_tokens, num_kv_heads)
    set_kv_cache_shuffle_kernel[grid](
        k_quantized,
        v_quantized,
        new_key_cache,
        new_value_cache,
        kv_loc,
        x,
        k_quantized.stride(0),
        v_quantized.stride(0),
        block_size,
        head_size,
        num_kv_heads,
        BLOCK_SIZE=head_size,
    )


@perftest()
def run_torch_mrope_3d_rms_set_kv_shuffle(
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
    k_cache: Tensor,  # contiguous (-1, num_heads_k, head_size) or shuffle layout
    v_cache: Tensor,  # contiguous (-1, num_heads_v, head_size) or shuffle layout
    kv_loc: Tensor,  # contiguous (num_tokens)
    k_scale: float,
    v_scale: float,
    is_mrope: bool,
    k_out: Tensor = None,  # Optional output buffer for k
    v_out: Tensor = None,  # Optional output buffer for v
    return_kv: bool = False,  # Whether to return k_out and v_out
    use_shuffle_layout: bool = False,  # Whether to use shuffle layout
    page_size: int = 0,  # Page size (block_size) for shuffle layout
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

    # Infer max_positions from cos_sin shape
    max_positions = (
        cos_sin.shape[0] // head_size if cos_sin.ndim == 1 else cos_sin.shape[0]
    )
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
def run_fused_mrope_3d_rms_set_kv_shuffle(
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
    k_cache: Tensor,  # contiguous (-1, num_heads_k, head_size) or shuffle layout
    v_cache: Tensor,  # contiguous (-1, num_heads_v, head_size) or shuffle layout
    kv_loc: Tensor,  # contiguous (num_tokens)
    k_scale: float,
    v_scale: float,
    is_mrope: bool,
    k_out: Tensor = None,  # Optional output buffer for k
    v_out: Tensor = None,  # Optional output buffer for v
    return_kv: bool = False,  # Whether to return k_out and v_out
    use_shuffle_layout: bool = False,  # Whether to use shuffle layout
    page_size: int = 0,  # Page size (block_size) for shuffle layout
):
    # qkv = qkv.clone()  # inplace op
    # Calculate x for shuffle layout: x = 16 // k_cache.element_size()
    x = 0
    block_size = page_size
    if use_shuffle_layout:
        x = 16 // k_cache.element_size()

    if is_mrope:
        aiter.fused_qk_norm_mrope_3d_cache_quant_shuffle(
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
            torch.tensor(k_scale),
            torch.tensor(v_scale),
            k_out,
            v_out,
            return_kv,
            use_shuffle_layout,
            block_size,
            x,
        )
    else:
        raise NotImplementedError("not implemented")
    return None


@benchmark()
def test_mrope_3d_rms_set_kv_shuffle(
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
    use_shuffle_layout=False,  # Whether to test shuffle layout
    page_size=0,  # Page size (block_size) for shuffle layout
    max_positions=10000,
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
        kv_cache_dtype = (
            dtypes.fp8
        )  # This will be torch.float8_e4m3fnuz on HIP (gfx942) or torch.float8_e4m3fn on CUDA/gfx950

    # Create cache buffers based on layout type
    if use_shuffle_layout:
        # Calculate x: x = 16 // k_cache.element_size()
        x = 16 // torch.empty(0, dtype=kv_cache_dtype, device="cuda").element_size()
        num_blocks = (max_positions + page_size - 1) // page_size
        # Shuffle layout: key_cache [num_blocks, num_kv_heads, head_size // x, block_size, x]
        #                 value_cache [num_blocks, num_kv_heads, block_size // x, head_size, x]
        k_cache_ref = torch.rand(
            num_blocks, num_heads_k, head_size // x, page_size, x, device="cuda"
        ).to(kv_cache_dtype)
        v_cache_ref = torch.rand(
            num_blocks, num_heads_v, page_size // x, head_size, x, device="cuda"
        ).to(kv_cache_dtype)
        k_cache = k_cache_ref.clone()
        v_cache = v_cache_ref.clone()
        # For shuffle layout, we need to reshape to flat format for reference comparison
        # But we'll use the shuffle format directly for kernel call
        k_cache_ref_flat = k_cache_ref.view(
            num_blocks * page_size, num_heads_k, head_size
        )
        v_cache_ref_flat = v_cache_ref.view(
            num_blocks * page_size, num_heads_v, head_size
        )
    else:
        k_cache_ref = torch.rand(
            max_positions, num_heads_k, head_size, device="cuda"
        ).to(kv_cache_dtype)
        v_cache_ref = torch.rand(
            max_positions, num_heads_v, head_size, device="cuda"
        ).to(kv_cache_dtype)
        k_cache = k_cache_ref.clone()
        v_cache = v_cache_ref.clone()
        k_cache_ref_flat = k_cache_ref
        v_cache_ref_flat = v_cache_ref
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
        k_out_ref = torch.empty(
            num_tokens, num_heads_k, head_size, dtype=kv_cache_dtype, device="cuda"
        )
        v_out_ref = torch.empty(
            num_tokens, num_heads_v, head_size, dtype=kv_cache_dtype, device="cuda"
        )
        k_out = torch.empty(
            num_tokens, num_heads_k, head_size, dtype=kv_cache_dtype, device="cuda"
        )
        v_out = torch.empty(
            num_tokens, num_heads_v, head_size, dtype=kv_cache_dtype, device="cuda"
        )

    _, avg_torch = run_torch_mrope_3d_rms_set_kv_shuffle(
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
        use_shuffle_layout,
        page_size,
    )
    _, avg_cu = run_fused_mrope_3d_rms_set_kv_shuffle(
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
        use_shuffle_layout,
        page_size,
    )

    info = f"dtype:{dtype}, kv_cache_dtype:{kv_cache_dtype}, num_tokens:{num_tokens}, num_heads_q:{num_heads_q}, num_heads_k:{num_heads_k}, num_heads_v:{num_heads_v}, head_size:{head_size}, is_neox_style:{is_neox_style}"
    if is_mrope:
        info += f", mrope_section:{mrope_section}, is_interleaved:{is_interleaved}, eps:{eps}"
    if test_return_kv:
        info += f", return_kv:{test_return_kv}"
    if use_shuffle_layout:
        info += f", use_shuffle_layout:{use_shuffle_layout}, page_size:{page_size}"
    msg = f"[perf] === {info} === torch avg: {avg_torch:<8.2f} us, cu avg: {avg_cu:<8.2f} us, uplift: {avg_torch/avg_cu-1:<5.1%}"

    checkAllclose(q_out_ref, q_out, msg="q_out", rtol=1e-2, atol=0.05)
    print(msg, flush=True)
    # For shuffle layout, we need to reshape cache for comparison
    if use_shuffle_layout:
        # Reshape shuffle cache back to flat format for comparison
        k_cache_flat = k_cache.view(-1, num_heads_k, head_size)
        v_cache_flat = v_cache.view(-1, num_heads_v, head_size)
        checkAllclose(
            k_cache_ref_flat[kv_loc].float(),
            k_cache_flat[kv_loc].float(),
            msg="k_cache",
            rtol=1e-2,
            atol=0.05,
        )
        checkAllclose(
            v_cache_ref_flat[kv_loc].float(),
            v_cache_flat[kv_loc].float(),
            msg="v_cache",
            rtol=1e-2,
            atol=0.05,
        )
    else:
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


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-n",
    "--neox_style",
    type=dtypes.str2bool,
    nargs="*",
    default=[True, False],
    help="""Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    e.g.: -n true   # for Neox-style
          or -n false # for GPT-J-style""",
)
parser.add_argument(
    "-t",
    "--token",
    type=int,
    nargs="*",
    default=[1257, 127, 3],
    help="""Number of tokens.
    e.g.: -t 513""",
)
parser.add_argument(
    "-hd",
    "--head",
    type=int,
    nargs="*",
    default=[32, 64],
    help="""Number of heads.
    e.g.: -hd 32""",
)
parser.add_argument(
    "-hs",
    "--head_sizes",
    type=int,
    nargs="*",
    default=[64, 128, 256],
    help="""Head size.
    e.g.: -hs 64""",
)
parser.add_argument(
    "-m",
    "--max_positions",
    type=int,
    default=10000,
    help="""Max Positions.
    e.g.: -m 10000""",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    default="bf16",
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-i",
    "--is_interleaved",
    type=dtypes.str2bool,
    nargs="*",
    default=[True, False],
    help="""Whether to use the interleaved MRoPE.
    e.g.: -i true   # for interleaved MRoPE
          or -i false # for non-interleaved MRoPE""",
)

mrope_sections_dict = {64: [12, 10, 10], 128: [24, 20, 20], 256: [48, 40, 40]}

if __name__ == "__main__":
    args = parser.parse_args()
    kv_cache_dtypes = [torch.bfloat16, torch.float8_e4m3fn, torch.float8_e4m3fnuz]
    test_return_kv_flags = [True, False]
    use_shuffle_layouts = [True]  # Test both normal and shuffle layouts
    page_sizes = [16]  # Test two page sizes for shuffle layout
    for kv_cache_dtype in kv_cache_dtypes:
        for test_return_kv in test_return_kv_flags:
            for use_shuffle_layout in use_shuffle_layouts:
                # For shuffle layout, test both page sizes; for normal layout, skip page_size
                page_size_list = page_sizes if use_shuffle_layout else [0]
                for page_size in page_size_list:
                    for is_neox_style in args.neox_style:
                        for num_token in args.token:
                            for num_head in args.head:
                                for i, head_size in enumerate(args.head_sizes):
                                    for is_interleaved in args.is_interleaved:
                                        test_mrope_3d_rms_set_kv_shuffle(
                                            args.dtype,
                                            num_token,
                                            num_head,
                                            1,
                                            1,
                                            head_size,
                                            is_neox_style,
                                            mrope_sections_dict[head_size],
                                            is_interleaved,
                                            eps=1e-6,
                                            is_mrope=True,
                                            kv_cache_dtype=kv_cache_dtype,
                                            test_return_kv=test_return_kv,
                                            use_shuffle_layout=use_shuffle_layout,
                                            page_size=page_size,
                                            max_positions=args.max_positions,
                                        )
