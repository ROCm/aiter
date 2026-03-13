import functools
import torch
import triton
import aiter
from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention import (
    map_dims,
)
from aiter.ops.triton._triton_kernels.quant.sage_attention_quant import (
    perblock_quantize_kernel,
    sage_quant_v_kernel,
    sage_quant_kernel,
    _rot_k_only_kernel,
    _rot_q_kernel,
    _rotate_quantize_q_kernel,
    _rotate_mxfp_quantize_k_kernel,
    _compute_delta_s_kernel,
    sage_quant_v_bhsd_kernel,
    perblock_quantize_q_kernel,
    perblock_quantize_kernel
)
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp

@functools.lru_cache(maxsize=16)
def create_hadamard_matrix(block_size, device="cuda", dtype=torch.bfloat16):
    """
    Returns a Hadamard matrix of size block_size x block_size. Remember to normalize with sqrt(block_size) for it to be orthogonal.
    """
    assert (block_size & (block_size - 1)) == 0, "block_size must be power of 2"
    assert block_size > 0, "block_size must be positive"

    # Base case: H_1 = [1]
    if block_size == 1:
        return torch.ones(1, 1, device=device, dtype=dtype)

    # Recursive construction: H_{2n} = [H_n   H_n  ]
    #                                   [H_n  -H_n ]
    H_half = create_hadamard_matrix(block_size // 2, device=device, dtype=dtype)

    # Build the full matrix (unnormalized)
    H = torch.zeros(block_size, block_size, device=device, dtype=dtype)
    half = block_size // 2
    H[:half, :half] = H_half
    H[:half, half:] = H_half
    H[half:, :half] = H_half
    H[half:, half:] = -H_half

    # The unnormalized matrix satisfies H_unnorm @ H_unnorm.T = block_size * I
    # remember to divide by sqrt(block_size) to get orthogonal matrix
    return H

def unified_perblock_quantize_int8(
    q,
    BLOCK_SIZE_M,
    cu_seqlens,
    sm_scale,
    config,
    num_queries_per_kv,
    DTYPE_MAX: int = 256,
):  
    # q is expected to have layout thd
    d = q.shape[-1]
    hq = q.shape[1]
    Q_q = torch.empty((*q.shape[:-1], d // 2), dtype=torch.uint8, device=q.device)
    num_seqs = len(cu_seqlens)
    total_num_blocks = q.shape[0] // BLOCK_SIZE_M + num_seqs
    
    assert num_queries_per_kv is not None and config is not None
    num_heads = hq // num_queries_per_kv
    Q_descale = torch.empty(
        (total_num_blocks, num_heads), dtype=torch.uint8, device=q.device
    )
    perblock_quantize_q_kernel[(
                num_heads,
                total_num_blocks,
            )](
                q,
                Q_q,
                Q_descale,
                cu_seqlens,
                num_seqs,
                hq,
                num_queries_per_kv,
                q.stride(0),
                q.stride(1),
                Q_descale.stride(0),
                Q_descale.stride(1),
                sm_scale=sm_scale,
                **config,
                DTYPE_MAX=DTYPE_MAX,
            )
    return Q_q, Q_descale

"""
expected shapes
tensors (and quantized tensors) (bshd), (thd) or (num_blocks,block_size,h,d)
descales (b,cdiv(s, BLOCK_M),h,1), (b,cdiv(max_seqlen, BLOCK_M),h,1) or (num_blocks,cdiv(block_size, BLOCK_M), h,1)
"""
def perblock_quantize_int8(
    q,
    BLOCK_SIZE_M,
    cu_seqlens,
    layout="bhsd",
    sm_scale=None,
    DTYPE_MAX: int = 256,
):  
    d = q.shape[-1]
    if layout=="thd":
        b = len(cu_seqlens)
        h = q.shape[1]
        s = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item() 
        total_num_blocks = (s + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M * b * h
        stride_m = h * d
    elif layout=="bhsd":
        b,h,s,_ = q.shape
        total_num_blocks = (s + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M * b * h
        stride_m = d
    elif layout=="bshd":
        b,s,h,d = q.shape
        total_num_blocks = (s + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M * b * h
        stride_m = h * d
    else: # num_blocks,block_size,h,d = q.shape
        # cached layout can be thought of as
        b,s,h,d = q.shape
        total_num_blocks = (s + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M * b * h
        stride_m = h * d

    Q_q = torch.empty((*q.shape[:-1], d // 2), dtype=torch.uint8, device=q.device)
    
    # bshd, thd and cache layout can be dealt with one kernel that has stride_t = h*d and:
    # bhsd: cu_seqlens = s,2s,3s,...
    # thd: cu_seqlens = cu_seqlens
    # cache: cu_seqlens = block_size, 2 block_size, 3 block_size,...
    Q_descale = torch.empty(
        (total_num_blocks, h), dtype=torch.uint8, device=q.device
    )
    num_pid_m = (s + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid = (b, num_pid_m, h)

    perblock_quantize_kernel[grid](
        q, Q_q, Q_descale, s, cu_seqlens, stride_m, Q_descale.stride(0), Q_descale.stride(1), BLOCK_SIZE_M, d, sm_scale, DTYPE_MAX)
    
    return Q_q, Q_descale



"""
tensor shapes expected here

k.shape: (b,h,s,d) or (t,h,d) or (num_blocks, block_size, h, d)
K_q: (b,h,s,d//2) or (t,h,d//2) or (num_blocks, block_size, h, d//2)
K_descale: (b,h,s,d//32) or (t,h,d//32) or (num_blocks, block_size, h, d//32)
"""

def pertoken_rotate_quantize_mxfp4(
    k,
    BLOCK_SIZE_M,
    R,
    BLOCK_R,
    hadamard_rotation=False,
    sm_scale=None,
):  
    d = k.shape[-1]
    K_q = torch.empty((*k.shape[:-1], d // 2), dtype=torch.uint8, device=k.device)
    K_descale = torch.empty(
        (*k.shape[:-1], d // 32), dtype=torch.uint8, device=k.device
    )
    num_tokens = k.shape.numel() // d
    stride_t = k.stride(-2)
    stride_ts = K_descale.stride(-2)
    stride_tq = K_q.stride(-2)
    num_pid_k = (num_tokens + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_k = (num_pid_k,)
    _rotate_mxfp_quantize_k_kernel[grid_k](
        k,
        K_q,
        K_descale,
        R,
        stride_t,
        stride_tq,
        stride_ts,
        num_tokens,
        hadamard_rotation=hadamard_rotation,
        BLOCK_M=BLOCK_SIZE_M,
        BLOCK_R=BLOCK_R,
        D=d,
        sm_scale=sm_scale,
        num_warps=4,
        num_stages=5,
    )
    return K_q, K_descale

"""
v or v_q.shape: (b,h,s,d) or (t,h,d) or (num_blocks, block_size, h, d)
v_descale: (h,d)
"""

def perchannel_quantize_fp8(
    v,
    BLOCK_M,
    layout_k="bhsd",
    v_descale=None,
):  
    d = v.shape[-1]
    FP8_TYPE = aiter.dtypes.fp8
    FP8_MAX = torch.finfo(FP8_TYPE).max
    v_q = torch.empty_like(v, dtype=FP8_TYPE, device=v.device)

    if v_descale is None:
        if layout_k=="bhsd":
            reduce_dims = (0,2)
            stride_h = v.stride(1)
            num_heads = v.shape[1]
        elif layout_k=="bshd":
            reduce_dims = (0,1)
            stride_h = v.stride(2)
            num_heads = v.shape[2]
        elif layout_k=="thd":
            reduce_dims = (0)
            stride_h = v.stride(1)
            num_heads = v.shape[1]
        else: # (num_blocks, block_size, h, d)
            reduce_dims = (0,1)
            stride_h = v.stride(2)
            num_heads = v.shape[2]    
        v_descale = v.abs().amax(dim=reduce_dims).to(torch.float32) / FP8_MAX
        stride_sh = v_descale.stride(0)
    
    num_tokens = v.shape.numel() // (d*num_heads)
    stride_t = v.stride(-2)
    
    num_pid_v = (num_tokens + BLOCK_M - 1) // BLOCK_M
    grid_v = (num_pid_v, num_heads)
    sage_quant_v_kernel[grid_v](
        v,
        v_q,
        v_descale,
        stride_t,
        stride_h,
        stride_sh,
        num_tokens,
        D=d,
        BLOCK_M=BLOCK_M,
        num_stages=3,
        num_warps=8,
    )
    return v_q, v_descale

def sage_quant_v1(
    q,
    k,
    v,
    BLOCK_M,
    BLOCK_N,
    layout_q="bshd", # options: "bshd", "bhsd", "thd", "cache"
    layout_k="bshd", # same for v
    v_descale = None,
    cu_seqlens_q= None,
    cu_seqlens_k= None,
    config=None,
):
    d = q.shape[-1]
    sm_scale = d**-0.5 * 1.4426950408889634

    # this is pain in the ass as it groups tokens, so we have to consider the layouting
    if layout_q == "unified":
        q_q, q_descale = unified_perblock_quantize_int8(
            q,
            BLOCK_M,
            cu_seqlens_q,
            sm_scale=sm_scale,
            config=config,
            DTYPE_MAX=256
        )
    else:
        q_q, q_descale = perblock_quantize_int8(
            q,
            BLOCK_M,
            cu_seqlens_q,
            layout=layout_q,
            sm_scale=sm_scale,
            DTYPE_MAX=256
        )

    k_q, k_descale = perblock_quantize_int8(
        k,
        BLOCK_N,
        cu_seqlens_k,
        layout=layout_k,
        sm_scale=None,
        DTYPE_MAX=256
    )

    v_q, v_descale = perchannel_quantize_fp8(v, 256, layout_k=layout_k, v_descale=v_descale)

    return q_q, q_descale, k_q, k_descale, v_q, v_descale

def sage_quant_v2(
    q,
    k,
    v,
    BLOCK_M,
    BLOCK_N,
    hadamard_rotation=False,
    R=None,
    BLOCK_R=None,
    layout_k="bshd", # options: "bshd", "bhsd", "thd", "cache". same for v
    v_descale = None,
):
    
    if hadamard_rotation:
        if R is None:
            assert (
                BLOCK_R is not None
            ), "if using hadamard rotation, BLOCK_R (size of the hadamard matrix) must be provided."
            R = create_hadamard_matrix(BLOCK_R, device=q.device, dtype=q.dtype) / (
                BLOCK_R**0.5
            )
        else:
            BLOCK_R = R.shape[-1]
    
    d = q.shape[-1]
    sm_scale = d**-0.5 * 1.4426950408889634

    # this is easy as its per token
    q_q, q_descale = pertoken_rotate_quantize_mxfp4(
        q,
        R=R,
        BLOCK_R=BLOCK_R,
        BLOCK_SIZE_M=BLOCK_M,
        hadamard_rotation=hadamard_rotation,
        sm_scale=sm_scale,
    )
    k_q, k_descale = pertoken_rotate_quantize_mxfp4(
        k,
        R=R,
        BLOCK_R=BLOCK_R,
        BLOCK_SIZE_M=BLOCK_N,
        hadamard_rotation=hadamard_rotation,
        sm_scale=None, # do not apply sm scale to k tensor!
    )
    v_q, v_descale = perchannel_quantize_fp8(v, 256, layout_k=layout_k, v_descale=v_descale)
    
    return q_q, q_descale, k_q, k_descale, v_q, v_descale