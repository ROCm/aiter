import torch
import triton
import triton.language as tl
from torch import autograd
from enum import IntEnum

from typing import Optional, Tuple

class RotateStyle(IntEnum):
    NEOX = 0,
    GPTJ = 1

@triton.jit
def _rope_fwd_kernel_neox_nope(x_ptr: torch.Tensor,
                freqs_ptr: torch.Tensor,
                out_ptr: torch.Tensor,
                stride_x_s, stride_x_b, stride_x_h, stride_x_d,
                stride_freqs_s, stride_freqs_b, stride_freqs_h, stride_freqs_d,
                stride_out_s, stride_out_b, stride_out_h, stride_out_d,
                rotate_style: tl.constexpr,
                reuse_freqs_front_part: tl.constexpr,
                nope_first: tl.constexpr,
                SEQ_LEN: tl.constexpr,
                D_MODEL: tl.constexpr,
                D_MODEL_HALF: tl.constexpr,
):
    #Parallelize over batch and head. Handle 1 sequence per program
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    #Load freqs. Note: freqs is D_MODEL/2 or D_MODEL/4(nope+reuse_freqs_front_part), 
    #but freqs shape in here is D_MODEL which matches the shape of the final output.
    #We use mask to load 0s in the bottom half or top half(nope_first)
    freqs_base_offs = (stride_freqs_s * s + 
                    0 * stride_freqs_b + 
                    0 * stride_freqs_h)
    if nope_first:
        if reuse_freqs_front_part:
            freqs_offs = tl.arange(0, D_MODEL) - D_MODEL_HALF
            freqs_offs = tl.where((freqs_offs >= D_MODEL // 4) & (freqs_offs < D_MODEL_HALF), freqs_offs - D_MODEL // 4, freqs_offs).to(freqs_offs.dtype)
            freqs_mask = (freqs_offs >=0) & (freqs_offs <= D_MODEL // 4)
        else:
            freqs_offs = tl.arange(0, D_MODEL) - D_MODEL_HALF
            freqs_mask = (freqs_offs >= 0) & (freqs_offs < D_MODEL_HALF)    
    else:
        if reuse_freqs_front_part:
            freqs_offs = tl.arange(0, D_MODEL) 
            freqs_offs = tl.where((freqs_offs >= D_MODEL // 4) & (freqs_offs < D_MODEL_HALF), freqs_offs - D_MODEL_HALF // 2, freqs_offs).to(freqs_offs.dtype)
            freqs_mask = freqs_offs < D_MODEL_HALF // 2
        else:
            freqs_offs = tl.arange(0, D_MODEL) 
            freqs_mask = freqs_offs < D_MODEL_HALF
    freqs = tl.load(freqs_ptr + freqs_base_offs + freqs_offs, mask=freqs_mask)

    #Load X
    x_base_offs = stride_x_b * b + stride_x_s * s + stride_x_h * h
    x_offs = tl.arange(0, D_MODEL)
    if nope_first:
        x_mask = (x_offs >= D_MODEL_HALF) & (x_offs < D_MODEL)
    else:
        x_mask = x_offs < D_MODEL_HALF
    x = tl.load(x_ptr + x_base_offs + x_offs, mask=x_mask)
    
    #Load X rotated
    #rotate_style: NEOX
    if nope_first:
        x1_offs = tl.where((x_offs >= D_MODEL_HALF) & (x_offs < D_MODEL_HALF + D_MODEL_HALF/2), x_offs + D_MODEL_HALF/2, 0).to(x_offs.dtype)
        x2_offs = tl.where((x_offs >= D_MODEL_HALF + D_MODEL_HALF/2) & (x_offs < D_MODEL), x_offs - D_MODEL_HALF/2, 0).to(x_offs.dtype)
        x_rotated_offs = x1_offs + x2_offs
        x_rotated_mask = (x_rotated_offs >= D_MODEL_HALF) & (x_rotated_offs < D_MODEL)
    else:
        x1_offs = tl.where( x_offs < D_MODEL_HALF/2, x_offs + D_MODEL_HALF/2, 0).to(x_offs.dtype)
        x2_offs = tl.where((x_offs >= D_MODEL_HALF/2) & (x_offs < D_MODEL_HALF), x_offs - D_MODEL_HALF/2, 0).to(x_offs.dtype)
        x_rotated_offs = x1_offs + x2_offs
        x_rotated_mask = x_rotated_offs < D_MODEL_HALF
    x_rotated = tl.load(x_ptr + x_base_offs + x_rotated_offs, mask=x_rotated_mask)
    if nope_first:
        x_rotated = tl.where((x_offs >= D_MODEL_HALF) & (x_offs < D_MODEL_HALF + D_MODEL_HALF/2), -x_rotated, x_rotated)
    else:
        x_rotated = tl.where(x_offs < D_MODEL_HALF/2, -x_rotated, x_rotated)

    #compute cos(freqs)
    fc = tl.cos(freqs.to(tl.float32)) 

    #compute sin(freqs)
    fs = tl.sin(freqs.to(tl.float32)) 

    #compute output
    out = x * fc + x_rotated * fs

    #Load nope
    if nope_first:
        x_nope_mask = tl.where(x_offs < D_MODEL_HALF, 1, 0).to(x_rotated_mask.dtype)
    else:
        x_nope_mask = tl.where(x_offs >= D_MODEL_HALF, 1, 0).to(x_rotated_mask.dtype)
    x_nope = tl.load(x_ptr + x_base_offs + x_offs, mask=x_nope_mask)

    out = out + x_nope
    out = out.to(x_ptr.dtype.element_ty)

    #store output for this batch and head (s, 1, 1, d)
    tl.store(out_ptr + x_base_offs + x_offs, out, mask=x_offs < D_MODEL)

@triton.jit
def _rope_fwd_kernel_neox(x_ptr: torch.Tensor,
                freqs_ptr: torch.Tensor,
                out_ptr: torch.Tensor,
                stride_x_s, stride_x_b, stride_x_h, stride_x_d,
                stride_freqs_s, stride_freqs_b, stride_freqs_h, stride_freqs_d,
                stride_out_s, stride_out_b, stride_out_h, stride_out_d,
                rotate_style: tl.constexpr,
                reuse_freqs_front_part: tl.constexpr,
                SEQ_LEN: tl.constexpr,
                D_MODEL: tl.constexpr,
                D_MODEL_HALF: tl.constexpr
):
    #Parallelize over batch and head. Handle 1 sequence per program
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    #Load freqs for this batch and head (s, 1, 1, d)
    freqs_base_offs = (stride_freqs_s * s + 
                    0 * stride_freqs_b + 
                    0 * stride_freqs_h)

    if reuse_freqs_front_part:
        freqs_offs = tl.arange(0, D_MODEL)
        freqs_offs = tl.where((freqs_offs >= D_MODEL_HALF) & (freqs_offs < D_MODEL), freqs_offs-D_MODEL_HALF, freqs_offs).to(freqs_offs.dtype)
        freqs_mask = freqs_offs < D_MODEL
    else:
        freqs_offs = tl.arange(0, D_MODEL)
        freqs_mask = freqs_offs < D_MODEL
    freqs = tl.load(freqs_ptr + freqs_base_offs + freqs_offs, mask=freqs_mask)

    #Load X
    x_base_offs = stride_x_b * b + stride_x_s * s + stride_x_h * h
    x_offs = tl.arange(0, D_MODEL)
    x_mask = x_offs < D_MODEL
    x = tl.load(x_ptr + x_base_offs + x_offs, mask=x_mask)
    
    #Load X rotated
    #rotate_style: NEOX
    x_offs_rotated = tl.where(x_offs < D_MODEL_HALF, x_offs + D_MODEL_HALF, x_offs-D_MODEL_HALF).to(x_offs.dtype)
    x_rotated = tl.load(x_ptr + x_base_offs + x_offs_rotated, mask=x_mask)
    x_rotated = tl.where(x_offs < D_MODEL_HALF, -x_rotated, x_rotated)

    #compute cos(freqs)
    fc = tl.cos(freqs.to(tl.float32)) 

    #compute sin(freqs)
    fs = tl.sin(freqs.to(tl.float32)) 

    #compute output
    out = x * fc + x_rotated * fs

    out = out.to(x_ptr.dtype.element_ty)

    #store output for this batch and head (s, 1, 1, d)
    tl.store(out_ptr + x_base_offs + x_offs, out, mask=x_mask)

@triton.jit
def _rope_fwd_kernel_gptj_nope(x_ptr: torch.Tensor,
                freqs_ptr: torch.Tensor,
                out_ptr: torch.Tensor,
                stride_x_s, stride_x_b, stride_x_h, stride_x_d,
                stride_freqs_s, stride_freqs_b, stride_freqs_h, stride_freqs_d,
                stride_out_s, stride_out_b, stride_out_h, stride_out_d,
                rotate_style: tl.constexpr,
                reuse_freqs_front_part: tl.constexpr,
                nope_first: tl.constexpr,
                SEQ_LEN: tl.constexpr,
                D_MODEL: tl.constexpr,
                D_MODEL_HALF: tl.constexpr,
):
    #Parallelize over batch and head. Handle 1 sequence per program
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    #Load freqs for this batch and head (1, 1, 1, d)
    freqs_base_offs = (stride_freqs_s * s + 
                    0 * stride_freqs_b + 
                    0 * stride_freqs_h)
    if nope_first:
        if reuse_freqs_front_part:
            freqs_offs = tl.arange(0, D_MODEL) - D_MODEL_HALF
            freqs_offs = freqs_offs // 2
            freqs_mask = (freqs_offs >= 0) & (freqs_offs < D_MODEL // 4)
        else:
            freqs_offs = tl.arange(0, D_MODEL) - D_MODEL_HALF
            freqs_mask = (freqs_offs >= 0) & (freqs_offs < D_MODEL_HALF)
    else:
        if reuse_freqs_front_part:
            freqs_offs = tl.arange(0, D_MODEL) // 2
            freqs_mask = freqs_offs < D_MODEL // 4
        else:
            freqs_offs = tl.arange(0, D_MODEL)
            freqs_mask = freqs_offs < D_MODEL_HALF
    freqs = tl.load(freqs_ptr + freqs_base_offs + freqs_offs, mask=freqs_mask)

    #Load X [D_MODEL]
    x_base_offs = stride_x_b * b + stride_x_s * s + stride_x_h * h
    x_offs = tl.arange(0, D_MODEL)
    if nope_first:
        x_mask = (x_offs >= D_MODEL_HALF) & (x_offs < D_MODEL)
    else:
        x_mask = x_offs < D_MODEL_HALF
    x = tl.load(x_ptr + x_base_offs + x_offs, mask=x_mask)
    
    #Load rotated.
    #rotate_style:GPTJ
    #X1 = even idx of x, [D_MODEL/2]
    #X2 = odd idx of x, [D_MODEL/2]
    x_offs_rotated = tl.arange(0, D_MODEL_HALF) * 2
    if nope_first:
        x_mask_rotated = x_offs_rotated >= D_MODEL_HALF
    else:
        x_mask_rotated = x_offs_rotated < D_MODEL_HALF
    
    x1 = tl.load(x_ptr + x_base_offs + x_offs_rotated, mask=x_mask_rotated)
    x2 = tl.load(x_ptr + x_base_offs + (x_offs_rotated +1), mask=x_mask_rotated)
    x2 = -x2
    x_rotated = tl.interleave(x2, x1)

    #compute cos(freqs)
    fc = tl.cos(freqs.to(tl.float32)) 

    #compute sin(freqs)
    fs = tl.sin(freqs.to(tl.float32)) 

    #compute output
    out = x * fc + x_rotated * fs

    #Load nope
    if nope_first:
        x_nope_mask = tl.where(x_offs < D_MODEL_HALF, 1, 0).to(x_mask_rotated.dtype)
    else:
        x_nope_mask = tl.where(x_offs >= D_MODEL_HALF, 1, 0).to(x_mask_rotated.dtype)
    x_nope = tl.load(x_ptr + x_base_offs + x_offs, mask=x_nope_mask)

    out = out + x_nope
    out = out.to(x_ptr.dtype.element_ty)

    #store output for this batch and head (1, 1, 1, d)
    tl.store(out_ptr + x_base_offs + x_offs, out)

@triton.jit
def _rope_fwd_kernel_gptj(x_ptr: torch.Tensor,
                freqs_ptr: torch.Tensor,
                out_ptr: torch.Tensor,
                stride_x_s, stride_x_b, stride_x_h, stride_x_d,
                stride_freqs_s, stride_freqs_b, stride_freqs_h, stride_freqs_d,
                stride_out_s, stride_out_b, stride_out_h, stride_out_d,
                rotate_style: tl.constexpr,
                reuse_freqs_front_part: tl.constexpr,
                SEQ_LEN: tl.constexpr,
                D_MODEL: tl.constexpr,
                D_MODEL_HALF: tl.constexpr,
):
    #Parallelize over batch and head. Handle 1 sequence per program
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    #Load freqs for this batch and head (s, 1, 1, d)
    freqs_base_offs = (stride_freqs_s * s + 
                    0 * stride_freqs_b + 
                    0 * stride_freqs_h)
    if reuse_freqs_front_part:
        freqs_offs = tl.arange(0, D_MODEL) // 2
        freqs_mask = freqs_offs < D_MODEL_HALF
    else:
        freqs_offs = tl.arange(0, D_MODEL)
        freqs_mask = freqs_offs < D_MODEL
    freqs = tl.load(freqs_ptr + freqs_base_offs + freqs_offs, mask=freqs_mask)

    #Load X [D_MODEL]
    x_base_offs = stride_x_b * b + stride_x_s * s + stride_x_h * h
    x_offs = tl.arange(0, D_MODEL)
    x_mask = x_offs < D_MODEL
    x = tl.load(x_ptr + x_base_offs + x_offs, mask=x_mask)
    
    #Load rotated.
    #X1 = even idx of x, [D_MODEL/2]
    #X2 = odd idx of x, [D_MODEL/2]
    x_offs_rotated = tl.arange(0, D_MODEL_HALF) * 2
    x_mask_rotated = x_offs_rotated < D_MODEL
    x1 = tl.load(x_ptr + x_base_offs + x_offs_rotated, mask=x_mask_rotated)
    x2 = tl.load(x_ptr + x_base_offs + (x_offs_rotated +1), mask=x_mask_rotated)
    x2 = -x2
    x_rotated = tl.interleave(x2, x1)

    #compute cos(freqs)
    fc = tl.cos(freqs.to(tl.float32)) 

    #compute sin(freqs)
    fs = tl.sin(freqs.to(tl.float32)) 

    #compute output
    out = x * fc + x_rotated * fs
    out = out.to(x_ptr.dtype.element_ty)

    #store output for this batch and head (s, 1, 1, d)
    tl.store(out_ptr + x_base_offs + x_offs, out, mask=x_mask)

@triton.jit
def _rope_fwd_kernel_neox_thd(x_ptr: torch.Tensor,
                cu_seqlens_ptr: torch.Tensor,
                freqs_ptr: torch.Tensor,
                out_ptr: torch.Tensor,
                stride_x_t, stride_x_h, stride_x_d,
                stride_freqs_t, stride_freqs_b, stride_freqs_h, stride_freqs_d,
                stride_out_t, stride_out_h, stride_out_d,
                rotate_style: tl.constexpr,
                reuse_freqs_front_part: tl.constexpr,
                MAX_SEQ_LEN_POW2: tl.constexpr,
                D_MODEL: tl.constexpr,
                D_MODEL_HALF: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)


    s_start = tl.load(cu_seqlens_ptr + b)
    s_end = tl.load(cu_seqlens_ptr + b + 1)
    s = s_end - s_start

    #Load freqs for this batch and head (s, 1, 1, d)
    #Note: freqs_offs are not offset by s_start. They always start at 0 because 
    #sequence for each batch starts from 0
    offs_s = tl.arange(0, MAX_SEQ_LEN_POW2)
    mask_s = offs_s < s
    if reuse_freqs_front_part:
        freqs_offs_d = tl.arange(0, D_MODEL)
        freqs_offs_d = tl.where((freqs_offs_d >= D_MODEL_HALF) & (freqs_offs_d < D_MODEL), freqs_offs_d-D_MODEL_HALF, freqs_offs_d).to(freqs_offs_d.dtype)
        freqs_mask_d = freqs_offs_d < D_MODEL
    else:
        freqs_offs_d = tl.arange(0, D_MODEL)
        freqs_mask_d = freqs_offs_d < D_MODEL

    freqs_offs = stride_freqs_t * offs_s[:, None] + stride_freqs_d * freqs_offs_d[None, :]
    freqs_mask = (mask_s[:, None]) & (freqs_mask_d[None, :])
    freqs = tl.load(freqs_ptr + freqs_offs, mask=freqs_mask)

    #Load X
    x_offs_d = tl.arange(0, D_MODEL)
    x_offs =  (stride_x_t * s_start + stride_x_t * offs_s[:, None] +
                stride_x_h * h +
                stride_x_d * x_offs_d[None, :]
                )
    x_mask = (mask_s[:, None]) & (x_offs_d < D_MODEL)[None, :]
    x = tl.load(x_ptr + x_offs, mask=x_mask)

    #Load X rotated
    #rotate_style: NEOX
    x_offs_d_rotated = tl.where(x_offs_d < D_MODEL_HALF, x_offs_d + D_MODEL_HALF, x_offs_d-D_MODEL_HALF).to(x_offs_d.dtype)
    x_offs_rotated =  (stride_x_t * s_start + stride_x_t * offs_s[:, None] +
                stride_x_h * h +
                stride_x_d * x_offs_d_rotated[None, :]
                )
    x_rotated = tl.load(x_ptr + x_offs_rotated, mask=x_mask)
    x_rotated = tl.where(x_offs_d_rotated[None, :] >= D_MODEL_HALF, -x_rotated, x_rotated)

    #compute cos(freqs)
    fc = tl.cos(freqs.to(tl.float32))

    #compute sin(freqs)
    fs = tl.sin(freqs.to(tl.float32))

    #compute output
    out = x * fc + x_rotated * fs

    out = out.to(x_ptr.dtype.element_ty)

    #store output for this batch and head (s, 1, 1, d)
    tl.store(out_ptr +  x_offs, out, mask=x_mask)

@triton.jit
def _rope_fwd_kernel_neox_nope_thd(x_ptr: torch.Tensor,
                cu_seqlens_ptr: torch.Tensor,
                freqs_ptr: torch.Tensor,
                out_ptr: torch.Tensor,
                stride_x_t, stride_x_h, stride_x_d,
                stride_freqs_t, stride_freqs_b, stride_freqs_h, stride_freqs_d,
                stride_out_t, stride_out_h, stride_out_d,
                rotate_style: tl.constexpr,
                reuse_freqs_front_part: tl.constexpr,
                nope_first: tl.constexpr,
                MAX_SEQ_LEN_POW2: tl.constexpr,
                D_MODEL: tl.constexpr,
                D_MODEL_HALF: tl.constexpr
):
    #Parallelize over batch and head. Handle 1 sequence per program
    b = tl.program_id(0)
    h = tl.program_id(1)

    s_start = tl.load(cu_seqlens_ptr + b)
    s_end = tl.load(cu_seqlens_ptr + b + 1)
    s = s_end - s_start

    #Load freqs. Note: freqs is D_MODEL/2 or D_MODEL/4(nope+reuse_freqs_front_part), 
    #but freqs shape in here is D_MODEL which matches the shape of the final output.
    #We use mask to load 0s in the bottom half or top half(nope_first)
    offs_s = tl.arange(0, MAX_SEQ_LEN_POW2)
    if nope_first:
        if reuse_freqs_front_part:
            freqs_offs_d = tl.arange(0, D_MODEL) - D_MODEL_HALF
            freqs_offs_d = tl.where((freqs_offs_d >= D_MODEL // 4) & (freqs_offs_d < D_MODEL_HALF), freqs_offs_d - D_MODEL // 4, freqs_offs_d).to(freqs_offs_d.dtype)
            freqs_mask_d = (freqs_offs_d >=0) & (freqs_offs_d <= D_MODEL // 4)
        else:
            freqs_offs_d = tl.arange(0, D_MODEL) - D_MODEL_HALF
            freqs_mask_d = (freqs_offs_d >= 0) & (freqs_offs_d < D_MODEL_HALF)
    else:
        if reuse_freqs_front_part:
            freqs_offs_d = tl.arange(0, D_MODEL) 
            freqs_offs_d = tl.where((freqs_offs_d >= D_MODEL // 4) & (freqs_offs_d < D_MODEL_HALF), freqs_offs_d - D_MODEL_HALF // 2, freqs_offs_d).to(freqs_offs_d.dtype)
            freqs_mask_d = freqs_offs_d < D_MODEL_HALF // 2
        else:
            freqs_offs_d = tl.arange(0, D_MODEL) 
            freqs_mask_d = freqs_offs_d < D_MODEL_HALF
    freqs_offs = stride_freqs_t * offs_s[:, None] + stride_freqs_d * freqs_offs_d[None, :]
    mask_s = offs_s < s
    freqs_mask = (mask_s[:, None]) & freqs_mask_d[None, :]
    freqs = tl.load(freqs_ptr + freqs_offs, mask=freqs_mask)

    #Load X
    x_base_offs = stride_x_t * s_start + stride_x_h * h
    x_offs_d = tl.arange(0, D_MODEL)
    if nope_first:
        x_mask_d = (x_offs_d >= D_MODEL_HALF) & (x_offs_d < D_MODEL)
    else:
        x_mask_d = x_offs_d < D_MODEL_HALF
    x_mask = mask_s[:, None] & x_mask_d[None, :]
    x_offs = stride_x_t * offs_s[:, None] + stride_x_d * x_offs_d[None, :]
    x = tl.load(x_ptr + x_base_offs + x_offs, mask=x_mask)

    #Load X rotated
    #rotate_style: NEOX
    if nope_first:
        x1_offs_d = tl.where((x_offs_d >= D_MODEL_HALF) & (x_offs_d < D_MODEL_HALF + D_MODEL_HALF/2), x_offs_d + D_MODEL_HALF/2, 0).to(x_offs_d.dtype)
        x2_offs_d = tl.where((x_offs_d >= D_MODEL_HALF + D_MODEL_HALF/2) & (x_offs_d < D_MODEL), x_offs_d - D_MODEL_HALF/2, 0).to(x_offs_d.dtype)
        x_rotated_offs_d = x1_offs_d + x2_offs_d
        x_rotated_mask_d = (x_rotated_offs_d >= D_MODEL_HALF) & (x_rotated_offs_d < D_MODEL)
    else:
        x1_offs_d = tl.where( x_offs_d < D_MODEL_HALF/2, x_offs_d + D_MODEL_HALF/2, 0).to(x_offs_d.dtype)
        x2_offs_d = tl.where((x_offs_d >= D_MODEL_HALF/2) & (x_offs_d < D_MODEL_HALF), x_offs_d - D_MODEL_HALF/2, 0).to(x_offs_d.dtype)
        x_rotated_offs_d = x1_offs_d + x2_offs_d
        x_rotated_mask_d = x_rotated_offs_d < D_MODEL_HALF
    x_rotated_offs = stride_x_t * offs_s[:, None] + stride_x_d * x_rotated_offs_d[None, :]
    x_rotated_mask = mask_s[:, None] & x_rotated_mask_d[None, :]
    x_rotated = tl.load(x_ptr + x_base_offs + x_rotated_offs, mask=x_rotated_mask)
    if nope_first:
        x_rotated = tl.where((x_offs_d >= D_MODEL_HALF) & (x_offs_d < D_MODEL_HALF + D_MODEL_HALF/2), -x_rotated, x_rotated)
    else:
        x_rotated = tl.where(x_offs_d < D_MODEL_HALF/2, -x_rotated, x_rotated)

    #compute cos(freqs)
    fc = tl.cos(freqs.to(tl.float32)) 

    #compute sin(freqs)
    fs = tl.sin(freqs.to(tl.float32)) 

    #compute output
    out = x * fc + x_rotated * fs

    #Load nope
    if nope_first:
        x_nope_mask_d = tl.where(x_offs_d < D_MODEL_HALF, 1, 0).to(x_rotated_mask.dtype)
    else:
        x_nope_mask_d = tl.where(x_offs_d >= D_MODEL_HALF, 1, 0).to(x_rotated_mask.dtype)
    x_nope_mask = mask_s[:, None] & x_nope_mask_d[None, :]
    x_nope = tl.load(x_ptr + x_base_offs + x_offs, mask=x_nope_mask)

    out = out + x_nope
    out = out.to(x_ptr.dtype.element_ty)

    #store output for this batch and head (s, 1, 1, d)
    out_mask = mask_s[:, None] & (x_offs_d < D_MODEL)[None, :]
    tl.store(out_ptr + x_base_offs + x_offs, out, mask=out_mask)

@triton.jit
def _rope_fwd_kernel_gptj_nope_thd(x_ptr: torch.Tensor,
                cu_seqlens_ptr: torch.Tensor,
                freqs_ptr: torch.Tensor,
                out_ptr: torch.Tensor,
                stride_x_t, stride_x_h, stride_x_d,
                stride_freqs_t, stride_freqs_b, stride_freqs_h, stride_freqs_d,
                stride_out_t, stride_out_h, stride_out_d,
                rotate_style: tl.constexpr,
                reuse_freqs_front_part: tl.constexpr,
                nope_first: tl.constexpr,
                MAX_SEQ_LEN_POW2: tl.constexpr,
                D_MODEL: tl.constexpr,
                D_MODEL_HALF: tl.constexpr,
):
    #Parallelize over batch and head. Handle 1 sequence per program
    b = tl.program_id(0)
    h = tl.program_id(1)

    s_start = tl.load(cu_seqlens_ptr + b)
    s_end = tl.load(cu_seqlens_ptr + b + 1)
    s = s_end - s_start

    offs_s = tl.arange(0, MAX_SEQ_LEN_POW2)
    mask_s = offs_s < s

    #Load freqs for this batch and head (s, 1, 1, d)
    if nope_first:
        if reuse_freqs_front_part:
            freqs_offs_d = tl.arange(0, D_MODEL) - D_MODEL_HALF
            freqs_offs_d = freqs_offs_d // 2
            freqs_mask_d = (freqs_offs_d >= 0) & (freqs_offs_d < D_MODEL // 4)
        else:
            freqs_offs_d = tl.arange(0, D_MODEL) - D_MODEL_HALF
            freqs_mask_d = (freqs_offs_d >= 0) & (freqs_offs_d < D_MODEL_HALF)
    else:
        if reuse_freqs_front_part:
            freqs_offs_d = tl.arange(0, D_MODEL) // 2
            freqs_mask_d = freqs_offs_d < D_MODEL // 4
        else:
            freqs_offs_d = tl.arange(0, D_MODEL)
            freqs_mask_d = freqs_offs_d < D_MODEL_HALF
    freqs_offs = stride_freqs_t * offs_s[:, None] + stride_freqs_d * freqs_offs_d[None, :]
    freqs_mask = (mask_s[:, None]) & (freqs_mask_d[None, :])
    freqs = tl.load(freqs_ptr + freqs_offs, mask=freqs_mask)

    #Load X [D_MODEL]
    x_offs_d = tl.arange(0, D_MODEL)
    x_offs = (stride_x_t * s_start + stride_x_t * offs_s[:, None] + 
            stride_x_h * h +
            stride_x_d * x_offs_d[None, :]
            )
    if nope_first:
        x_mask_d = (x_offs_d >= D_MODEL_HALF) & (x_offs_d < D_MODEL)
    else:
        x_mask_d = x_offs_d < D_MODEL_HALF
    x_mask = (mask_s[:, None]) & (x_mask_d[None, :])
    x = tl.load(x_ptr + x_offs, mask=x_mask)
    
    #Load rotated.
    #rotate_style:GPTJ
    #X1 = even idx of x, [D_MODEL/2]
    #X2 = odd idx of x, [D_MODEL/2]
    x_offs_rotated_d = tl.arange(0, D_MODEL_HALF) * 2
    if nope_first:
        x_mask_rotated_d = x_offs_rotated_d >= D_MODEL_HALF
    else:
        x_mask_rotated_d = x_offs_rotated_d < D_MODEL_HALF
    x_offs_rotated_base = (stride_x_t * s_start + stride_x_t * offs_s[:, None] + 
            stride_x_h * h)
    x1_offs = x_offs_rotated_base + stride_x_d * x_offs_rotated_d[None, :]
    x2_offs = x_offs_rotated_base + stride_x_d * (x_offs_rotated_d + 1)[None, :]
    x_mask_rotated = (mask_s[:, None]) & (x_mask_rotated_d)[None, :]
    x1 = tl.load(x_ptr + x1_offs, mask=x_mask_rotated)
    x2 = tl.load(x_ptr + x2_offs, mask=x_mask_rotated)
    x2 = -x2
    x_rotated = tl.interleave(x2, x1)

    #compute cos(freqs)
    fc = tl.cos(freqs.to(tl.float32)) 

    #compute sin(freqs)
    fs = tl.sin(freqs.to(tl.float32)) 

    #compute output
    out = x * fc + x_rotated * fs

    #Load nope
    if nope_first:
        x_nope_mask_d = tl.where(x_offs_d < D_MODEL_HALF, 1, 0).to(x_mask_rotated_d.dtype)
    else:
        x_nope_mask_d = tl.where(x_offs_d >= D_MODEL_HALF, 1, 0).to(x_mask_rotated_d.dtype)
    x_nope_mask = (mask_s[:, None]) & (x_nope_mask_d)[None, :]
    x_nope = tl.load(x_ptr + x_offs, mask=x_nope_mask)

    out = out + x_nope
    out = out.to(x_ptr.dtype.element_ty)

    #store output for this batch and head (1, 1, 1, d)
    out_mask = mask_s[:, None] & (x_offs_d < D_MODEL)[None, :]
    tl.store(out_ptr + x_offs, out, mask=out_mask)

@triton.jit
def _rope_fwd_kernel_gptj_thd(x_ptr: torch.Tensor,
                cu_seqlens_ptr: torch.Tensor,
                freqs_ptr: torch.Tensor,
                out_ptr: torch.Tensor,
                stride_x_t, stride_x_h, stride_x_d,
                stride_freqs_t, stride_freqs_b, stride_freqs_h, stride_freqs_d,
                stride_out_t, stride_out_h, stride_out_d,
                rotate_style: tl.constexpr,
                reuse_freqs_front_part: tl.constexpr,
                MAX_SEQ_LEN_POW2: tl.constexpr,
                D_MODEL: tl.constexpr,
                D_MODEL_HALF: tl.constexpr,
):
    #Parallelize over batch and head. Handle 1 sequence per program
    b = tl.program_id(0)
    h = tl.program_id(1)

    s_start = tl.load(cu_seqlens_ptr + b)
    s_end = tl.load(cu_seqlens_ptr + b + 1)
    s = s_end - s_start

    #Load freqs for this batch and head (s, 1, 1, d)
    offs_s = tl.arange(0, MAX_SEQ_LEN_POW2)
    mask_s = offs_s < s
    if reuse_freqs_front_part:
        freqs_offs_d = tl.arange(0, D_MODEL) // 2
        freqs_mask_d = freqs_offs_d < D_MODEL_HALF
    else:
        freqs_offs_d = tl.arange(0, D_MODEL)
        freqs_mask_d = freqs_offs_d < D_MODEL
    freqs_offs = stride_freqs_t * offs_s[:, None] + stride_freqs_d * freqs_offs_d[None, :]
    freqs_mask = (mask_s[:, None]) & (freqs_mask_d[None, :])
    freqs = tl.load(freqs_ptr + freqs_offs, mask=freqs_mask)

    #Load X [D_MODEL]
    x_offs_d = tl.arange(0, D_MODEL)
    x_offs = (stride_x_t * s_start + stride_x_t * offs_s[:, None] + 
            stride_x_h * h +
            stride_x_d * x_offs_d[None, :]
            )
    x_mask = (mask_s[:, None]) & (x_offs_d < D_MODEL)[None, :]
    x = tl.load(x_ptr + x_offs, mask=x_mask)

    #Load rotated.
    #X1 = even idx of x, [D_MODEL/2]
    #X2 = odd idx of x, [D_MODEL/2]
    x_offs_d_rotated = tl.arange(0, D_MODEL_HALF) * 2
    x_mask_d_rotated = x_offs_d_rotated < D_MODEL
    x_offs_rotated_base = (stride_x_t * s_start + stride_x_t * offs_s[:, None] + 
            stride_x_h * h)
    x1_offs = x_offs_rotated_base + stride_x_d * x_offs_d_rotated[None, :]
    x2_offs = x_offs_rotated_base + stride_x_d * (x_offs_d_rotated + 1)[None, :]
    x_mask_rotated = (mask_s[:, None]) & (x_mask_d_rotated)[None, :]
    x1 = tl.load(x_ptr + x1_offs, mask=x_mask_rotated)
    x2 = tl.load(x_ptr + x2_offs , mask=x_mask_rotated)
    x2 = -x2
    x_rotated = tl.interleave(x2, x1)

    #compute cos(freqs)
    fc = tl.cos(freqs.to(tl.float32)) 

    #compute sin(freqs)
    fs = tl.sin(freqs.to(tl.float32)) 

    #compute output
    out = x * fc + x_rotated * fs
    out = out.to(x_ptr.dtype.element_ty)

    #store output for this batch and head (s, 1, 1, d)
    tl.store(out_ptr + x_offs, out, mask=x_mask)

@triton.jit
def _rope_fwd_kernel_neox_nope_cached(x_ptr: torch.Tensor,
                cos_ptr: torch.Tensor,
                sin_ptr: torch.Tensor,
                out_ptr: torch.Tensor,
                stride_x_s, stride_x_b, stride_x_h, stride_x_d,
                stride_cos_s, stride_cos_b, stride_cos_h, stride_cos_d,
                stride_out_s, stride_out_b, stride_out_h, stride_out_d,
                rotate_style: tl.constexpr,
                reuse_freqs_front_part: tl.constexpr,
                nope_first: tl.constexpr,
                SEQ_LEN: tl.constexpr,
                D_MODEL: tl.constexpr,
                D_MODEL_HALF: tl.constexpr,
):
    #Parallelize over batch and head. Handle 1 sequence per program
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    #Load freqs. Note: freqs is D_MODEL/2 or D_MODEL/4(nope+reuse_freqs_front_part), 
    #but freqs shape in here is D_MODEL which matches the shape of the final output.
    #We use mask to load 0s in the bottom half or top half(nope_first)
    cos_base_offs = (stride_cos_s * s + 
                    0 * stride_cos_b + 
                    0 * stride_cos_h)
    if nope_first:
        if reuse_freqs_front_part:
            cos_offs = tl.arange(0, D_MODEL) - D_MODEL_HALF
            cos_offs = tl.where((cos_offs >= D_MODEL // 4) & (cos_offs < D_MODEL_HALF), cos_offs - D_MODEL // 4, cos_offs).to(cos_offs.dtype)
            cos_mask = (cos_offs >=0) & (cos_offs <= D_MODEL // 4)
        else:
            cos_offs = tl.arange(0, D_MODEL) - D_MODEL_HALF
            cos_mask = (cos_offs >= 0) & (cos_offs < D_MODEL_HALF)    
    else:
        if reuse_freqs_front_part:
            cos_offs = tl.arange(0, D_MODEL) 
            cos_offs = tl.where((cos_offs >= D_MODEL // 4) & (cos_offs < D_MODEL_HALF), cos_offs - D_MODEL_HALF // 2, cos_offs).to(cos_offs.dtype)
            cos_mask = cos_offs < D_MODEL_HALF // 2
        else:
            cos_offs = tl.arange(0, D_MODEL) 
            cos_mask = cos_offs < D_MODEL_HALF
    cos = tl.load(cos_ptr + cos_base_offs + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_base_offs + cos_offs, mask=cos_mask)

    #Load X
    x_base_offs = stride_x_b * b + stride_x_s * s + stride_x_h * h
    x_offs = tl.arange(0, D_MODEL)
    if nope_first:
        x_mask = (x_offs >= D_MODEL_HALF) & (x_offs < D_MODEL)
    else:
        x_mask = x_offs < D_MODEL_HALF
    x = tl.load(x_ptr + x_base_offs + x_offs, mask=x_mask)
    
    #Load X rotated
    #rotate_style: NEOX
    if nope_first:
        x1_offs = tl.where((x_offs >= D_MODEL_HALF) & (x_offs < D_MODEL_HALF + D_MODEL_HALF/2), x_offs + D_MODEL_HALF/2, 0).to(x_offs.dtype)
        x2_offs = tl.where((x_offs >= D_MODEL_HALF + D_MODEL_HALF/2) & (x_offs < D_MODEL), x_offs - D_MODEL_HALF/2, 0).to(x_offs.dtype)
        x_rotated_offs = x1_offs + x2_offs
        x_rotated_mask = (x_rotated_offs >= D_MODEL_HALF) & (x_rotated_offs < D_MODEL)
    else:
        x1_offs = tl.where( x_offs < D_MODEL_HALF/2, x_offs + D_MODEL_HALF/2, 0).to(x_offs.dtype)
        x2_offs = tl.where((x_offs >= D_MODEL_HALF/2) & (x_offs < D_MODEL_HALF), x_offs - D_MODEL_HALF/2, 0).to(x_offs.dtype)
        x_rotated_offs = x1_offs + x2_offs
        x_rotated_mask = x_rotated_offs < D_MODEL_HALF
    x_rotated = tl.load(x_ptr + x_base_offs + x_rotated_offs, mask=x_rotated_mask)
    if nope_first:
        x_rotated = tl.where((x_offs >= D_MODEL_HALF) & (x_offs < D_MODEL_HALF + D_MODEL_HALF/2), -x_rotated, x_rotated)
    else:
        x_rotated = tl.where(x_offs < D_MODEL_HALF/2, -x_rotated, x_rotated)

    #compute output
    out = x * cos + x_rotated * sin

    #Load nope
    if nope_first:
        x_nope_mask = tl.where(x_offs < D_MODEL_HALF, 1, 0).to(x_rotated_mask.dtype)
    else:
        x_nope_mask = tl.where(x_offs >= D_MODEL_HALF, 1, 0).to(x_rotated_mask.dtype)
    x_nope = tl.load(x_ptr + x_base_offs + x_offs, mask=x_nope_mask)

    out = out + x_nope
    out = out.to(x_ptr.dtype.element_ty)

    #store output for this batch and head (s, 1, 1, d)
    tl.store(out_ptr + x_base_offs + x_offs, out, mask=x_offs < D_MODEL)

@triton.jit
def _rope_fwd_kernel_neox_cached(x_ptr: torch.Tensor,
                cos_ptr: torch.Tensor,
                sin_ptr: torch.Tensor,
                out_ptr: torch.Tensor,
                stride_x_s, stride_x_b, stride_x_h, stride_x_d,
                stride_cos_s, stride_cos_b, stride_cos_h, stride_cos_d,
                stride_out_s, stride_out_b, stride_out_h, stride_out_d,
                rotate_style: tl.constexpr,
                reuse_freqs_front_part: tl.constexpr,
                SEQ_LEN: tl.constexpr,
                D_MODEL: tl.constexpr,
                D_MODEL_HALF: tl.constexpr
):
    #Parallelize over batch and head. Handle 1 sequence per program
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    #Load cos for this batch and head (s, 1, 1, d)
    cos_base_offs = (stride_cos_s * s + 
                    0 * stride_cos_b + 
                    0 * stride_cos_h)

    if reuse_freqs_front_part:
        cos_offs = tl.arange(0, D_MODEL)
        cos_offs = tl.where((cos_offs >= D_MODEL_HALF) & (cos_offs < D_MODEL), cos_offs-D_MODEL_HALF, cos_offs).to(cos_offs.dtype)
        cos_mask = cos_offs < D_MODEL
    else:
        cos_offs = tl.arange(0, D_MODEL)
        cos_mask = cos_offs < D_MODEL
    cos = tl.load(cos_ptr + cos_base_offs + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_base_offs + cos_offs, mask=cos_mask)

    #Load X
    x_base_offs = stride_x_b * b + stride_x_s * s + stride_x_h * h
    x_offs = tl.arange(0, D_MODEL)
    x_mask = x_offs < D_MODEL
    x = tl.load(x_ptr + x_base_offs + x_offs, mask=x_mask)
    
    #Load X rotated
    #rotate_style: NEOX
    x_offs_rotated = tl.where(x_offs < D_MODEL_HALF, x_offs + D_MODEL_HALF, x_offs-D_MODEL_HALF).to(x_offs.dtype)
    x_rotated = tl.load(x_ptr + x_base_offs + x_offs_rotated, mask=x_mask)
    x_rotated = tl.where(x_offs < D_MODEL_HALF, -x_rotated, x_rotated)

    #compute output
    out = x * cos + x_rotated * sin

    out = out.to(x_ptr.dtype.element_ty)

    #store output for this batch and head (s, 1, 1, d)
    tl.store(out_ptr + x_base_offs + x_offs, out, mask=x_mask)

@triton.jit
def _rope_fwd_kernel_gptj_nope_cached(x_ptr: torch.Tensor,
                cos_ptr: torch.Tensor,
                sin_ptr: torch.Tensor,
                out_ptr: torch.Tensor,
                stride_x_s, stride_x_b, stride_x_h, stride_x_d,
                stride_cos_s, stride_cos_b, stride_cos_h, stride_cos_d,
                stride_out_s, stride_out_b, stride_out_h, stride_out_d,
                rotate_style: tl.constexpr,
                reuse_freqs_front_part: tl.constexpr,
                nope_first: tl.constexpr,
                SEQ_LEN: tl.constexpr,
                D_MODEL: tl.constexpr,
                D_MODEL_HALF: tl.constexpr,
):
    #Parallelize over batch and head. Handle 1 sequence per program
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    #Load cos for this batch and head (1, 1, 1, d)
    cos_base_offs = (stride_cos_s * s + 
                    0 * stride_cos_b + 
                    0 * stride_cos_h)
    if nope_first:
        if reuse_freqs_front_part:
            cos_offs = tl.arange(0, D_MODEL) - D_MODEL_HALF
            cos_offs = cos_offs // 2
            cos_mask = (cos_offs >= 0) & (cos_offs < D_MODEL // 4)
        else:
            cos_offs = tl.arange(0, D_MODEL) - D_MODEL_HALF
            cos_mask = (cos_offs >= 0) & (cos_offs < D_MODEL_HALF)
    else:
        if reuse_freqs_front_part:
            cos_offs = tl.arange(0, D_MODEL) // 2
            cos_mask = cos_offs < D_MODEL // 4
        else:
            cos_offs = tl.arange(0, D_MODEL)
            cos_mask = cos_offs < D_MODEL_HALF
    cos = tl.load(cos_ptr + cos_base_offs + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_base_offs + cos_offs, mask=cos_mask)

    #Load X [D_MODEL]
    x_base_offs = stride_x_b * b + stride_x_s * s + stride_x_h * h
    x_offs = tl.arange(0, D_MODEL)
    if nope_first:
        x_mask = (x_offs >= D_MODEL_HALF) & (x_offs < D_MODEL)
    else:
        x_mask = x_offs < D_MODEL_HALF
    x = tl.load(x_ptr + x_base_offs + x_offs, mask=x_mask)
    
    #Load rotated.
    #rotate_style:GPTJ
    #X1 = even idx of x, [D_MODEL/2]
    #X2 = odd idx of x, [D_MODEL/2]
    x_offs_rotated = tl.arange(0, D_MODEL_HALF) * 2
    if nope_first:
        x_mask_rotated = x_offs_rotated >= D_MODEL_HALF
    else:
        x_mask_rotated = x_offs_rotated < D_MODEL_HALF
    
    x1 = tl.load(x_ptr + x_base_offs + x_offs_rotated, mask=x_mask_rotated)
    x2 = tl.load(x_ptr + x_base_offs + (x_offs_rotated +1), mask=x_mask_rotated)
    x2 = -x2
    x_rotated = tl.interleave(x2, x1)

    #compute output
    out = x * cos + x_rotated * sin

    #Load nope
    if nope_first:
        x_nope_mask = tl.where(x_offs < D_MODEL_HALF, 1, 0).to(x_mask_rotated.dtype)
    else:
        x_nope_mask = tl.where(x_offs >= D_MODEL_HALF, 1, 0).to(x_mask_rotated.dtype)
    x_nope = tl.load(x_ptr + x_base_offs + x_offs, mask=x_nope_mask)

    out = out + x_nope
    out = out.to(x_ptr.dtype.element_ty)

    #store output for this batch and head (1, 1, 1, d)
    tl.store(out_ptr + x_base_offs + x_offs, out)

@triton.jit
def _rope_fwd_kernel_gptj_cached(x_ptr: torch.Tensor,
                cos_ptr: torch.Tensor,
                sin_ptr: torch.Tensor,
                out_ptr: torch.Tensor,
                stride_x_s, stride_x_b, stride_x_h, stride_x_d,
                stride_cos_s, stride_cos_b, stride_cos_h, stride_cos_d,
                stride_out_s, stride_out_b, stride_out_h, stride_out_d,
                rotate_style: tl.constexpr,
                reuse_freqs_front_part: tl.constexpr,
                SEQ_LEN: tl.constexpr,
                D_MODEL: tl.constexpr,
                D_MODEL_HALF: tl.constexpr,
):
    #Parallelize over batch and head. Handle 1 sequence per program
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    #Load cos for this batch and head (s, 1, 1, d)
    cos_base_offs = (stride_cos_s * s + 
                    0 * stride_cos_b + 
                    0 * stride_cos_h)
    if reuse_freqs_front_part:
        cos_offs = tl.arange(0, D_MODEL) // 2
        cos_mask = cos_offs < D_MODEL_HALF
    else:
        cos_offs = tl.arange(0, D_MODEL)
        cos_mask = cos_offs < D_MODEL
    cos = tl.load(cos_ptr + cos_base_offs + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_base_offs + cos_offs, mask=cos_mask)

    #Load X [D_MODEL]
    x_base_offs = stride_x_b * b + stride_x_s * s + stride_x_h * h
    x_offs = tl.arange(0, D_MODEL)
    x_mask = x_offs < D_MODEL
    x = tl.load(x_ptr + x_base_offs + x_offs, mask=x_mask)
    
    #Load rotated.
    #X1 = even idx of x, [D_MODEL/2]
    #X2 = odd idx of x, [D_MODEL/2]
    x_offs_rotated = tl.arange(0, D_MODEL_HALF) * 2
    x_mask_rotated = x_offs_rotated < D_MODEL
    x1 = tl.load(x_ptr + x_base_offs + x_offs_rotated, mask=x_mask_rotated)
    x2 = tl.load(x_ptr + x_base_offs + (x_offs_rotated +1), mask=x_mask_rotated)
    x2 = -x2
    x_rotated = tl.interleave(x2, x1)

    #compute output
    out = x * cos + x_rotated * sin
    out = out.to(x_ptr.dtype.element_ty)

    #store output for this batch and head (s, 1, 1, d)
    tl.store(out_ptr + x_base_offs + x_offs, out, mask=x_mask)

@triton.jit
def _rope_fwd_kernel_neox_nope_cached_position(x_ptr: torch.Tensor,
                cos_ptr: torch.Tensor,
                sin_ptr: torch.Tensor,
                pos_ptr: torch.Tensor,
                out_ptr: torch.Tensor,
                stride_x_s, stride_x_b, stride_x_h, stride_x_d,
                stride_cos_s, stride_cos_b, stride_cos_h, stride_cos_d,
                stride_pos_s, stride_pos_b,
                stride_out_s, stride_out_b, stride_out_h, stride_out_d,
                reuse_freqs_front_part: tl.constexpr,
                nope_first: tl.constexpr,
                SEQ_LEN: tl.constexpr,
                D_MODEL: tl.constexpr,
                D_MODEL_HALF: tl.constexpr,
):
    #Parallelize over batch and head. Handle 1 sequence per program
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    #Load freqs. Note: freqs is D_MODEL/2 or D_MODEL/4(nope+reuse_freqs_front_part), 
    #but freqs shape in here is D_MODEL which matches the shape of the final output.
    #We use mask to load 0s in the bottom half or top half(nope_first)
    pos_offs = stride_pos_s * s + stride_pos_b * b
    pos = tl.load(pos_ptr + pos_offs)
    cos_base_offs = (stride_cos_s * pos + 
                    0 * stride_cos_b + 
                    0 * stride_cos_h)

    if nope_first:
        if reuse_freqs_front_part:
            cos_offs = tl.arange(0, D_MODEL) - D_MODEL_HALF
            cos_offs = tl.where((cos_offs >= D_MODEL // 4) & (cos_offs < D_MODEL_HALF), cos_offs - D_MODEL // 4, cos_offs).to(cos_offs.dtype)
            cos_mask = (cos_offs >=0) & (cos_offs <= D_MODEL // 4)
        else:
            cos_offs = tl.arange(0, D_MODEL) - D_MODEL_HALF
            cos_mask = (cos_offs >= 0) & (cos_offs < D_MODEL_HALF)    
    else:
        if reuse_freqs_front_part:
            cos_offs = tl.arange(0, D_MODEL) 
            cos_offs = tl.where((cos_offs >= D_MODEL // 4) & (cos_offs < D_MODEL_HALF), cos_offs - D_MODEL_HALF // 2, cos_offs).to(cos_offs.dtype)
            cos_mask = cos_offs < D_MODEL_HALF // 2
        else:
            cos_offs = tl.arange(0, D_MODEL) 
            cos_mask = cos_offs < D_MODEL_HALF
    cos = tl.load(cos_ptr + cos_base_offs + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_base_offs + cos_offs, mask=cos_mask)

    #Load X
    x_base_offs = stride_x_b * b + stride_x_s * s + stride_x_h * h
    x_offs = tl.arange(0, D_MODEL)
    if nope_first:
        x_mask = (x_offs >= D_MODEL_HALF) & (x_offs < D_MODEL)
    else:
        x_mask = x_offs < D_MODEL_HALF
    x = tl.load(x_ptr + x_base_offs + x_offs, mask=x_mask)
    
    #Load X rotated
    #rotate_style: NEOX
    if nope_first:
        x1_offs = tl.where((x_offs >= D_MODEL_HALF) & (x_offs < D_MODEL_HALF + D_MODEL_HALF/2), x_offs + D_MODEL_HALF/2, 0).to(x_offs.dtype)
        x2_offs = tl.where((x_offs >= D_MODEL_HALF + D_MODEL_HALF/2) & (x_offs < D_MODEL), x_offs - D_MODEL_HALF/2, 0).to(x_offs.dtype)
        x_rotated_offs = x1_offs + x2_offs
        x_rotated_mask = (x_rotated_offs >= D_MODEL_HALF) & (x_rotated_offs < D_MODEL)
    else:
        x1_offs = tl.where( x_offs < D_MODEL_HALF/2, x_offs + D_MODEL_HALF/2, 0).to(x_offs.dtype)
        x2_offs = tl.where((x_offs >= D_MODEL_HALF/2) & (x_offs < D_MODEL_HALF), x_offs - D_MODEL_HALF/2, 0).to(x_offs.dtype)
        x_rotated_offs = x1_offs + x2_offs
        x_rotated_mask = x_rotated_offs < D_MODEL_HALF
    x_rotated = tl.load(x_ptr + x_base_offs + x_rotated_offs, mask=x_rotated_mask)
    if nope_first:
        x_rotated = tl.where((x_offs >= D_MODEL_HALF) & (x_offs < D_MODEL_HALF + D_MODEL_HALF/2), -x_rotated, x_rotated)
    else:
        x_rotated = tl.where(x_offs < D_MODEL_HALF/2, -x_rotated, x_rotated)

    #compute output
    out = x * cos + x_rotated * sin

    #Load nope
    if nope_first:
        x_nope_mask = tl.where(x_offs < D_MODEL_HALF, 1, 0).to(x_rotated_mask.dtype)
    else:
        x_nope_mask = tl.where(x_offs >= D_MODEL_HALF, 1, 0).to(x_rotated_mask.dtype)
    x_nope = tl.load(x_ptr + x_base_offs + x_offs, mask=x_nope_mask)

    out = out + x_nope
    out = out.to(x_ptr.dtype.element_ty)

    #store output for this batch and head (s, 1, 1, d)
    tl.store(out_ptr + x_base_offs + x_offs, out, mask=x_offs < D_MODEL)

@triton.jit
def _rope_fwd_kernel_neox_cached_position(x_ptr: torch.Tensor,
                cos_ptr: torch.Tensor,
                sin_ptr: torch.Tensor,
                pos_ptr: torch.Tensor,
                out_ptr: torch.Tensor,
                stride_x_s, stride_x_b, stride_x_h, stride_x_d,
                stride_cos_s, stride_cos_b, stride_cos_h, stride_cos_d,
                stride_pos_s, stride_pos_b,
                stride_out_s, stride_out_b, stride_out_h, stride_out_d,
                reuse_freqs_front_part: tl.constexpr,
                SEQ_LEN: tl.constexpr,
                D_MODEL: tl.constexpr,
                D_MODEL_HALF: tl.constexpr
):
    #Parallelize over batch and head. Handle 1 sequence per program
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    #Load cos for this batch and head (s, 1, 1, d)
    pos_offs = stride_pos_s * s + stride_pos_b * b
    pos = tl.load(pos_ptr + pos_offs)
    cos_base_offs = (stride_cos_s * pos + 
                    0 * stride_cos_b + 
                    0 * stride_cos_h)

    if reuse_freqs_front_part:
        cos_offs = tl.arange(0, D_MODEL)
        cos_offs = tl.where((cos_offs >= D_MODEL_HALF) & (cos_offs < D_MODEL), cos_offs-D_MODEL_HALF, cos_offs).to(cos_offs.dtype)
        cos_mask = cos_offs < D_MODEL
    else:
        cos_offs = tl.arange(0, D_MODEL)
        cos_mask = cos_offs < D_MODEL
    cos = tl.load(cos_ptr + cos_base_offs + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_base_offs + cos_offs, mask=cos_mask)


    #Load X
    x_base_offs = stride_x_b * b + stride_x_s * s + stride_x_h * h
    x_offs = tl.arange(0, D_MODEL)
    x_mask = x_offs < D_MODEL
    x = tl.load(x_ptr + x_base_offs + x_offs, mask=x_mask)
    
    #Load X rotated
    #rotate_style: NEOX
    x_offs_rotated = tl.where(x_offs < D_MODEL_HALF, x_offs + D_MODEL_HALF, x_offs-D_MODEL_HALF).to(x_offs.dtype)
    x_rotated = tl.load(x_ptr + x_base_offs + x_offs_rotated, mask=x_mask)
    x_rotated = tl.where(x_offs < D_MODEL_HALF, -x_rotated, x_rotated)

    #compute output
    out = x * cos + x_rotated * sin

    out = out.to(x_ptr.dtype.element_ty)

    #store output for this batch and head (s, 1, 1, d)
    tl.store(out_ptr + x_base_offs + x_offs, out, mask=x_mask)

@triton.jit
def _rope_fwd_kernel_gptj_nope_cached_position(x_ptr: torch.Tensor,
                cos_ptr: torch.Tensor,
                sin_ptr: torch.Tensor,
                pos_ptr: torch.Tensor,
                out_ptr: torch.Tensor,
                stride_x_s, stride_x_b, stride_x_h, stride_x_d,
                stride_cos_s, stride_cos_b, stride_cos_h, stride_cos_d,
                stride_pos_s, stride_pos_b,
                stride_out_s, stride_out_b, stride_out_h, stride_out_d,
                reuse_freqs_front_part: tl.constexpr,
                nope_first: tl.constexpr,
                SEQ_LEN: tl.constexpr,
                D_MODEL: tl.constexpr,
                D_MODEL_HALF: tl.constexpr,
):
    #Parallelize over batch and head. Handle 1 sequence per program
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    #Load cos for this batch and head (1, 1, 1, d)
    pos_offs = stride_pos_s * s + stride_pos_b * b
    pos = tl.load(pos_ptr + pos_offs)
    cos_base_offs = (stride_cos_s * pos + 
                    0 * stride_cos_b + 
                    0 * stride_cos_h)
    if nope_first:
        if reuse_freqs_front_part:
            cos_offs = tl.arange(0, D_MODEL) - D_MODEL_HALF
            cos_offs = cos_offs // 2
            cos_mask = (cos_offs >= 0) & (cos_offs < D_MODEL // 4)
        else:
            cos_offs = tl.arange(0, D_MODEL) - D_MODEL_HALF
            cos_mask = (cos_offs >= 0) & (cos_offs < D_MODEL_HALF)
    else:
        if reuse_freqs_front_part:
            cos_offs = tl.arange(0, D_MODEL) // 2
            cos_mask = cos_offs < D_MODEL // 4
        else:
            cos_offs = tl.arange(0, D_MODEL)
            cos_mask = cos_offs < D_MODEL_HALF
    cos = tl.load(cos_ptr + cos_base_offs + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_base_offs + cos_offs, mask=cos_mask)

    #Load X [D_MODEL]
    x_base_offs = stride_x_b * b + stride_x_s * s + stride_x_h * h
    x_offs = tl.arange(0, D_MODEL)
    if nope_first:
        x_mask = (x_offs >= D_MODEL_HALF) & (x_offs < D_MODEL)
    else:
        x_mask = x_offs < D_MODEL_HALF
    x = tl.load(x_ptr + x_base_offs + x_offs, mask=x_mask)
    
    #Load rotated.
    #rotate_style:GPTJ
    #X1 = even idx of x, [D_MODEL/2]
    #X2 = odd idx of x, [D_MODEL/2]
    x_offs_rotated = tl.arange(0, D_MODEL_HALF) * 2
    if nope_first:
        x_mask_rotated = x_offs_rotated >= D_MODEL_HALF
    else:
        x_mask_rotated = x_offs_rotated < D_MODEL_HALF
    
    x1 = tl.load(x_ptr + x_base_offs + x_offs_rotated, mask=x_mask_rotated)
    x2 = tl.load(x_ptr + x_base_offs + (x_offs_rotated +1), mask=x_mask_rotated)
    x2 = -x2
    x_rotated = tl.interleave(x2, x1)

    #compute output
    out = x * cos + x_rotated * sin

    #Load nope
    if nope_first:
        x_nope_mask = tl.where(x_offs < D_MODEL_HALF, 1, 0).to(x_mask_rotated.dtype)
    else:
        x_nope_mask = tl.where(x_offs >= D_MODEL_HALF, 1, 0).to(x_mask_rotated.dtype)
    x_nope = tl.load(x_ptr + x_base_offs + x_offs, mask=x_nope_mask)

    out = out + x_nope
    out = out.to(x_ptr.dtype.element_ty)

    #store output for this batch and head (1, 1, 1, d)
    tl.store(out_ptr + x_base_offs + x_offs, out)

@triton.jit
def _rope_fwd_kernel_gptj_cached_position(x_ptr: torch.Tensor,
                cos_ptr: torch.Tensor,
                sin_ptr: torch.Tensor,
                pos_ptr: torch.Tensor,
                out_ptr: torch.Tensor,
                stride_x_s, stride_x_b, stride_x_h, stride_x_d,
                stride_cos_s, stride_cos_b, stride_cos_h, stride_cos_d,
                stride_pos_s, stride_pos_b,
                stride_out_s, stride_out_b, stride_out_h, stride_out_d,
                reuse_freqs_front_part: tl.constexpr,
                SEQ_LEN: tl.constexpr,
                D_MODEL: tl.constexpr,
                D_MODEL_HALF: tl.constexpr,
):
    #Parallelize over batch and head. Handle 1 sequence per program
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    #Load cos for this batch and head (s, 1, 1, d)
    pos_offs = stride_pos_s * s + stride_pos_b * b
    pos = tl.load(pos_ptr + pos_offs)
    cos_base_offs = (stride_cos_s * pos + 
                    0 * stride_cos_b + 
                    0 * stride_cos_h)
    if reuse_freqs_front_part:
        cos_offs = tl.arange(0, D_MODEL) // 2
        cos_mask = cos_offs < D_MODEL_HALF
    else:
        cos_offs = tl.arange(0, D_MODEL)
        cos_mask = cos_offs < D_MODEL
    cos = tl.load(cos_ptr + cos_base_offs + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_base_offs + cos_offs, mask=cos_mask)

    #Load X [D_MODEL]
    x_base_offs = stride_x_b * b + stride_x_s * s + stride_x_h * h
    x_offs = tl.arange(0, D_MODEL)
    x_mask = x_offs < D_MODEL
    x = tl.load(x_ptr + x_base_offs + x_offs, mask=x_mask)
    
    #Load rotated.
    #X1 = even idx of x, [D_MODEL/2]
    #X2 = odd idx of x, [D_MODEL/2]
    x_offs_rotated = tl.arange(0, D_MODEL_HALF) * 2
    x_mask_rotated = x_offs_rotated < D_MODEL
    x1 = tl.load(x_ptr + x_base_offs + x_offs_rotated, mask=x_mask_rotated)
    x2 = tl.load(x_ptr + x_base_offs + (x_offs_rotated +1), mask=x_mask_rotated)
    x2 = -x2
    x_rotated = tl.interleave(x2, x1)

    #compute output
    out = x * cos + x_rotated * sin
    out = out.to(x_ptr.dtype.element_ty)

    #store output for this batch and head (s, 1, 1, d)
    tl.store(out_ptr + x_base_offs + x_offs, out, mask=x_mask)

@triton.jit
def _rope_fwd_kernel_neox_nope_cached_position_off(x_ptr: torch.Tensor,
                cos_ptr: torch.Tensor,
                sin_ptr: torch.Tensor,
                pos_ptr: torch.Tensor,
                off_ptr: torch.Tensor,
                out_ptr: torch.Tensor,
                stride_x_s, stride_x_b, stride_x_h, stride_x_d,
                stride_cos_s, stride_cos_b, stride_cos_h, stride_cos_d,
                stride_pos_s, stride_pos_b,
                stride_out_s, stride_out_b, stride_out_h, stride_out_d,
                reuse_freqs_front_part: tl.constexpr,
                nope_first: tl.constexpr,
                SEQ_LEN: tl.constexpr,
                D_MODEL: tl.constexpr,
                D_MODEL_HALF: tl.constexpr,
):
    #Parallelize over batch and head. Handle 1 sequence per program
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    #Load freqs. Note: freqs is D_MODEL/2 or D_MODEL/4(nope+reuse_freqs_front_part), 
    #but freqs shape in here is D_MODEL which matches the shape of the final output.
    #We use mask to load 0s in the bottom half or top half(nope_first)
    pos_offs = stride_pos_s * s + stride_pos_b * b
    pos = tl.load(pos_ptr + pos_offs)
    offset = tl.load(off_ptr + pos_offs)
    cos_base_offs = (stride_cos_s * (pos + offset) +
                    0 * stride_cos_b + 
                    0 * stride_cos_h)

    if nope_first:
        if reuse_freqs_front_part:
            cos_offs = tl.arange(0, D_MODEL) - D_MODEL_HALF
            cos_offs = tl.where((cos_offs >= D_MODEL // 4) & (cos_offs < D_MODEL_HALF), cos_offs - D_MODEL // 4, cos_offs).to(cos_offs.dtype)
            cos_mask = (cos_offs >=0) & (cos_offs <= D_MODEL // 4)
        else:
            cos_offs = tl.arange(0, D_MODEL) - D_MODEL_HALF
            cos_mask = (cos_offs >= 0) & (cos_offs < D_MODEL_HALF)    
    else:
        if reuse_freqs_front_part:
            cos_offs = tl.arange(0, D_MODEL) 
            cos_offs = tl.where((cos_offs >= D_MODEL // 4) & (cos_offs < D_MODEL_HALF), cos_offs - D_MODEL_HALF // 2, cos_offs).to(cos_offs.dtype)
            cos_mask = cos_offs < D_MODEL_HALF // 2
        else:
            cos_offs = tl.arange(0, D_MODEL) 
            cos_mask = cos_offs < D_MODEL_HALF
    cos = tl.load(cos_ptr + cos_base_offs + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_base_offs + cos_offs, mask=cos_mask)

    #Load X
    x_base_offs = stride_x_b * b + stride_x_s * s + stride_x_h * h
    x_offs = tl.arange(0, D_MODEL)
    if nope_first:
        x_mask = (x_offs >= D_MODEL_HALF) & (x_offs < D_MODEL)
    else:
        x_mask = x_offs < D_MODEL_HALF
    x = tl.load(x_ptr + x_base_offs + x_offs, mask=x_mask)
    
    #Load X rotated
    #rotate_style: NEOX
    if nope_first:
        x1_offs = tl.where((x_offs >= D_MODEL_HALF) & (x_offs < D_MODEL_HALF + D_MODEL_HALF/2), x_offs + D_MODEL_HALF/2, 0).to(x_offs.dtype)
        x2_offs = tl.where((x_offs >= D_MODEL_HALF + D_MODEL_HALF/2) & (x_offs < D_MODEL), x_offs - D_MODEL_HALF/2, 0).to(x_offs.dtype)
        x_rotated_offs = x1_offs + x2_offs
        x_rotated_mask = (x_rotated_offs >= D_MODEL_HALF) & (x_rotated_offs < D_MODEL)
    else:
        x1_offs = tl.where( x_offs < D_MODEL_HALF/2, x_offs + D_MODEL_HALF/2, 0).to(x_offs.dtype)
        x2_offs = tl.where((x_offs >= D_MODEL_HALF/2) & (x_offs < D_MODEL_HALF), x_offs - D_MODEL_HALF/2, 0).to(x_offs.dtype)
        x_rotated_offs = x1_offs + x2_offs
        x_rotated_mask = x_rotated_offs < D_MODEL_HALF
    x_rotated = tl.load(x_ptr + x_base_offs + x_rotated_offs, mask=x_rotated_mask)
    if nope_first:
        x_rotated = tl.where((x_offs >= D_MODEL_HALF) & (x_offs < D_MODEL_HALF + D_MODEL_HALF/2), -x_rotated, x_rotated)
    else:
        x_rotated = tl.where(x_offs < D_MODEL_HALF/2, -x_rotated, x_rotated)

    #compute output
    out = x * cos + x_rotated * sin

    #Load nope
    if nope_first:
        x_nope_mask = tl.where(x_offs < D_MODEL_HALF, 1, 0).to(x_rotated_mask.dtype)
    else:
        x_nope_mask = tl.where(x_offs >= D_MODEL_HALF, 1, 0).to(x_rotated_mask.dtype)
    x_nope = tl.load(x_ptr + x_base_offs + x_offs, mask=x_nope_mask)

    out = out + x_nope
    out = out.to(x_ptr.dtype.element_ty)

    #store output for this batch and head (s, 1, 1, d)
    tl.store(out_ptr + x_base_offs + x_offs, out, mask=x_offs < D_MODEL)

@triton.jit
def _rope_fwd_kernel_neox_cached_position_off(x_ptr: torch.Tensor,
                cos_ptr: torch.Tensor,
                sin_ptr: torch.Tensor,
                pos_ptr: torch.Tensor,
                off_ptr: torch.Tensor,
                out_ptr: torch.Tensor,
                stride_x_s, stride_x_b, stride_x_h, stride_x_d,
                stride_cos_s, stride_cos_b, stride_cos_h, stride_cos_d,
                stride_pos_s, stride_pos_b,
                stride_out_s, stride_out_b, stride_out_h, stride_out_d,
                reuse_freqs_front_part: tl.constexpr,
                SEQ_LEN: tl.constexpr,
                D_MODEL: tl.constexpr,
                D_MODEL_HALF: tl.constexpr
):
    #Parallelize over batch and head. Handle 1 sequence per program
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    #Load cos for this batch and head (s, 1, 1, d)
    pos_offs = stride_pos_s * s + stride_pos_b * b
    pos = tl.load(pos_ptr + pos_offs)
    offset = tl.load(off_ptr + pos_offs)
    cos_base_offs = (stride_cos_s * (pos + offset) +
                    0 * stride_cos_b + 
                    0 * stride_cos_h)

    if reuse_freqs_front_part:
        cos_offs = tl.arange(0, D_MODEL)
        cos_offs = tl.where((cos_offs >= D_MODEL_HALF) & (cos_offs < D_MODEL), cos_offs-D_MODEL_HALF, cos_offs).to(cos_offs.dtype)
        cos_mask = cos_offs < D_MODEL
    else:
        cos_offs = tl.arange(0, D_MODEL)
        cos_mask = cos_offs < D_MODEL
    cos = tl.load(cos_ptr + cos_base_offs + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_base_offs + cos_offs, mask=cos_mask)


    #Load X
    x_base_offs = stride_x_b * b + stride_x_s * s + stride_x_h * h
    x_offs = tl.arange(0, D_MODEL)
    x_mask = x_offs < D_MODEL
    x = tl.load(x_ptr + x_base_offs + x_offs, mask=x_mask)
    
    #Load X rotated
    #rotate_style: NEOX
    x_offs_rotated = tl.where(x_offs < D_MODEL_HALF, x_offs + D_MODEL_HALF, x_offs-D_MODEL_HALF).to(x_offs.dtype)
    x_rotated = tl.load(x_ptr + x_base_offs + x_offs_rotated, mask=x_mask)
    x_rotated = tl.where(x_offs < D_MODEL_HALF, -x_rotated, x_rotated)

    #compute output
    out = x * cos + x_rotated * sin

    out = out.to(x_ptr.dtype.element_ty)

    #store output for this batch and head (s, 1, 1, d)
    tl.store(out_ptr + x_base_offs + x_offs, out, mask=x_mask)

@triton.jit
def _rope_fwd_kernel_gptj_nope_cached_position_off(x_ptr: torch.Tensor,
                cos_ptr: torch.Tensor,
                sin_ptr: torch.Tensor,
                pos_ptr: torch.Tensor,
                off_ptr: torch.Tensor,
                out_ptr: torch.Tensor,
                stride_x_s, stride_x_b, stride_x_h, stride_x_d,
                stride_cos_s, stride_cos_b, stride_cos_h, stride_cos_d,
                stride_pos_s, stride_pos_b,
                stride_out_s, stride_out_b, stride_out_h, stride_out_d,
                reuse_freqs_front_part: tl.constexpr,
                nope_first: tl.constexpr,
                SEQ_LEN: tl.constexpr,
                D_MODEL: tl.constexpr,
                D_MODEL_HALF: tl.constexpr,
):
    #Parallelize over batch and head. Handle 1 sequence per program
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    #Load cos for this batch and head (1, 1, 1, d)
    pos_offs = stride_pos_s * s + stride_pos_b * b
    pos = tl.load(pos_ptr + pos_offs)
    offset = tl.load(off_ptr + pos_offs)
    cos_base_offs = (stride_cos_s * (pos + offset) +
                    0 * stride_cos_b + 
                    0 * stride_cos_h)

    if nope_first:
        if reuse_freqs_front_part:
            cos_offs = tl.arange(0, D_MODEL) - D_MODEL_HALF
            cos_offs = cos_offs // 2
            cos_mask = (cos_offs >= 0) & (cos_offs < D_MODEL // 4)
        else:
            cos_offs = tl.arange(0, D_MODEL) - D_MODEL_HALF
            cos_mask = (cos_offs >= 0) & (cos_offs < D_MODEL_HALF)
    else:
        if reuse_freqs_front_part:
            cos_offs = tl.arange(0, D_MODEL) // 2
            cos_mask = cos_offs < D_MODEL // 4
        else:
            cos_offs = tl.arange(0, D_MODEL)
            cos_mask = cos_offs < D_MODEL_HALF
    cos = tl.load(cos_ptr + cos_base_offs + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_base_offs + cos_offs, mask=cos_mask)

    #Load X [D_MODEL]
    x_base_offs = stride_x_b * b + stride_x_s * s + stride_x_h * h
    x_offs = tl.arange(0, D_MODEL)
    if nope_first:
        x_mask = (x_offs >= D_MODEL_HALF) & (x_offs < D_MODEL)
    else:
        x_mask = x_offs < D_MODEL_HALF
    x = tl.load(x_ptr + x_base_offs + x_offs, mask=x_mask)
    
    #Load rotated.
    #rotate_style:GPTJ
    #X1 = even idx of x, [D_MODEL/2]
    #X2 = odd idx of x, [D_MODEL/2]
    x_offs_rotated = tl.arange(0, D_MODEL_HALF) * 2
    if nope_first:
        x_mask_rotated = x_offs_rotated >= D_MODEL_HALF
    else:
        x_mask_rotated = x_offs_rotated < D_MODEL_HALF
    
    x1 = tl.load(x_ptr + x_base_offs + x_offs_rotated, mask=x_mask_rotated)
    x2 = tl.load(x_ptr + x_base_offs + (x_offs_rotated +1), mask=x_mask_rotated)
    x2 = -x2
    x_rotated = tl.interleave(x2, x1)

    #compute output
    out = x * cos + x_rotated * sin

    #Load nope
    if nope_first:
        x_nope_mask = tl.where(x_offs < D_MODEL_HALF, 1, 0).to(x_mask_rotated.dtype)
    else:
        x_nope_mask = tl.where(x_offs >= D_MODEL_HALF, 1, 0).to(x_mask_rotated.dtype)
    x_nope = tl.load(x_ptr + x_base_offs + x_offs, mask=x_nope_mask)

    out = out + x_nope
    out = out.to(x_ptr.dtype.element_ty)

    #store output for this batch and head (1, 1, 1, d)
    tl.store(out_ptr + x_base_offs + x_offs, out)

@triton.jit
def _rope_fwd_kernel_gptj_cached_position_off(x_ptr: torch.Tensor,
                cos_ptr: torch.Tensor,
                sin_ptr: torch.Tensor,
                pos_ptr: torch.Tensor,
                off_ptr: torch.Tensor,
                out_ptr: torch.Tensor,
                stride_x_s, stride_x_b, stride_x_h, stride_x_d,
                stride_cos_s, stride_cos_b, stride_cos_h, stride_cos_d,
                stride_pos_s, stride_pos_b,
                stride_out_s, stride_out_b, stride_out_h, stride_out_d,
                reuse_freqs_front_part: tl.constexpr,
                SEQ_LEN: tl.constexpr,
                D_MODEL: tl.constexpr,
                D_MODEL_HALF: tl.constexpr,
):
    #Parallelize over batch and head. Handle 1 sequence per program
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    #Load cos for this batch and head (s, 1, 1, d)
    pos_offs = stride_pos_s * s + stride_pos_b * b
    pos = tl.load(pos_ptr + pos_offs)
    offset = tl.load(off_ptr + pos_offs)
    cos_base_offs = (stride_cos_s * (pos + offset) +
                    0 * stride_cos_b + 
                    0 * stride_cos_h)

    if reuse_freqs_front_part:
        cos_offs = tl.arange(0, D_MODEL) // 2
        cos_mask = cos_offs < D_MODEL_HALF
    else:
        cos_offs = tl.arange(0, D_MODEL)
        cos_mask = cos_offs < D_MODEL
    cos = tl.load(cos_ptr + cos_base_offs + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_base_offs + cos_offs, mask=cos_mask)

    #Load X [D_MODEL]
    x_base_offs = stride_x_b * b + stride_x_s * s + stride_x_h * h
    x_offs = tl.arange(0, D_MODEL)
    x_mask = x_offs < D_MODEL
    x = tl.load(x_ptr + x_base_offs + x_offs, mask=x_mask)
    
    #Load rotated.
    #X1 = even idx of x, [D_MODEL/2]
    #X2 = odd idx of x, [D_MODEL/2]
    x_offs_rotated = tl.arange(0, D_MODEL_HALF) * 2
    x_mask_rotated = x_offs_rotated < D_MODEL
    x1 = tl.load(x_ptr + x_base_offs + x_offs_rotated, mask=x_mask_rotated)
    x2 = tl.load(x_ptr + x_base_offs + (x_offs_rotated +1), mask=x_mask_rotated)
    x2 = -x2
    x_rotated = tl.interleave(x2, x1)

    #compute output
    out = x * cos + x_rotated * sin
    out = out.to(x_ptr.dtype.element_ty)

    #store output for this batch and head (s, 1, 1, d)
    tl.store(out_ptr + x_base_offs + x_offs, out, mask=x_mask)

@triton.jit
def _rope_fwd_kernel_gptj_cached_thd_position_2c(x_ptr: torch.Tensor,
                y_ptr: torch.Tensor,
                cos_ptr: torch.Tensor,
                sin_ptr: torch.Tensor,
                pos_ptr: torch.Tensor,
                out_x_ptr: torch.Tensor,
                out_y_ptr: torch.Tensor,
                stride_x_t, stride_x_h, stride_x_d,
                stride_cos_t, stride_cos_d,
                stride_pos_t, 
                stride_out_t, stride_out_h, stride_out_d,
                reuse_freqs_front_part: tl.constexpr,
                SEQ_LEN, #: tl.constexpr,  # assume SEQ_LEN is a variable
                D_MODEL: tl.constexpr,
                D_MODEL_HALF: tl.constexpr,
                SPLIT_SEQ_SIZE: tl.constexpr,
):
    #Parallelize over head. Handle 1 sequence per program
    h = tl.program_id(0)
    s = tl.program_id(1)

    #Load cos for this head
    pos_offs = s * SPLIT_SEQ_SIZE + tl.arange(0, SPLIT_SEQ_SIZE)
    pos_mask = pos_offs < SEQ_LEN
    pos = tl.load(pos_ptr + pos_offs, mask=pos_mask)
    cos_offs_t = pos

    if reuse_freqs_front_part:
        cos_offs_d = tl.arange(0, D_MODEL) // 2
        cos_mask_d = cos_offs_d < D_MODEL_HALF
    else:
        cos_offs_d = tl.arange(0, D_MODEL)
        cos_mask_d = cos_offs_d < D_MODEL

    cos_mask = (cos_offs_t < SEQ_LEN)[:, None] & (cos_mask_d)[None, :]
    cos_offs = (stride_cos_t * cos_offs_t[:, None] +
                stride_cos_d * cos_offs_d[None, :])
    cos = tl.load(cos_ptr + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_offs, mask=cos_mask)

    #Load X [SEQ_LEN, D_MODEL]
    x_offs_t = s * SPLIT_SEQ_SIZE + tl.arange(0, SPLIT_SEQ_SIZE)
    x_offs_d = tl.arange(0, D_MODEL)
    x_offs = (x_offs_t[:, None] * stride_x_t + 
            stride_x_h*h +
            x_offs_d[None, :] * stride_x_d)
    x_mask = (x_offs_t < SEQ_LEN)[:, None] &  (x_offs_d < D_MODEL)[None, :]
    #tl.device_print("x_offs_t", x_offs_t)
    #tl.device_print("x_offs_d", x_offs_d)
    #tl.device_print("x_offs", x_offs)

    # x = tl.load(x_ptr + x_offs, mask=x_mask)
    # y = tl.load(y_ptr + x_offs, mask=x_mask)
    
    #Load rotated.
    #X1 = even idx of x, [D_MODEL/2]
    #X2 = odd idx of x, [D_MODEL/2]
    x1_offs_d_rotated = tl.arange(0, D_MODEL_HALF) * 2
    x1_offs_rotated = (x_offs_t[:, None] * stride_x_t + 
            stride_x_h*h +
            x1_offs_d_rotated[None, :] * stride_x_d)
    x1_mask_rotated = (x_offs_t < SEQ_LEN)[:, None] &  (x1_offs_d_rotated < D_MODEL)[None, :]
    x1 = tl.load(x_ptr + x1_offs_rotated, mask=x1_mask_rotated)
    y1 = tl.load(y_ptr + x1_offs_rotated, mask=x1_mask_rotated)

    x2_offs_rotated = (x_offs_t[:, None] * stride_x_t + 
            stride_x_h*h +
            (x1_offs_d_rotated + 1)[None, :] * stride_x_d)
    x2_mask_rotated = (x_offs_t < SEQ_LEN)[:, None] &  ((x1_offs_d_rotated + 1) < D_MODEL)[None, :]
    x2 = tl.load(x_ptr + (x2_offs_rotated), mask=x2_mask_rotated)
    y2 = tl.load(y_ptr + (x2_offs_rotated), mask=x2_mask_rotated)
    x = tl.interleave(x1, x2)
    y = tl.interleave(y1, y2)
    x2 = -x2
    x_rotated = tl.interleave(x2, x1)
    y2 = -y2
    y_rotated = tl.interleave(y2, y1)

    #compute output
    out_x = x * cos + x_rotated * sin
    out_x = out_x.to(x_ptr.dtype.element_ty)
    out_y = y * cos + y_rotated * sin
    out_y = out_y.to(y_ptr.dtype.element_ty)

    #store output for this batch and head (s, 1, 1, d)
    tl.store(out_x_ptr + x_offs, out_x, mask=x_mask)
    tl.store(out_y_ptr + x_offs, out_y, mask=x_mask)
   


@triton.jit
def _rope_fwd_2d_kernel_neox(x_ptr: torch.Tensor,
                            cos_h_ptr, 
                            sin_h_ptr, 
                            cos_w_ptr, 
                            sin_w_ptr, 
                            out_ptr,
                            stride_x_b, stride_x_wh, stride_x_h, stride_x_d,
                            stride_cos_h_b, stride_cos_h_ht, stride_cos_h_h, stride_cos_h_d,
                            stride_cos_w_b, stride_cos_w_w, stride_cos_w_h, stride_cos_w_d,
                            WH: tl.constexpr,
                            HEIGHT: tl.constexpr,
                            WEIGHT: tl.constexpr,
                            D_MODEL: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    #load cos_h [HT, D_MODEL]
    offs_wh = tl.arange(0, WH)
    offs_cos_h_h = offs_wh // WEIGHT
    offs_d = tl.arange(0, D_MODEL)
    offs_cos_h = ( stride_cos_h_h*offs_cos_h_h[:, None] +
                stride_cos_h_d*offs_d[None, :])
    mask_cos_h = (offs_d < D_MODEL // 2)
    cos_h = tl.load(cos_h_ptr + offs_cos_h, mask=mask_cos_h[None,:])

    #load sin_h
    sin_h = tl.load(sin_h_ptr + offs_cos_h, mask=mask_cos_h[None,:])

    #load cos_w
    offs_cos_w_w = offs_wh % WEIGHT
    offs_cos_w_d = offs_d - D_MODEL // 2
    offs_cos_w = (stride_cos_w_w*offs_cos_w_w[:, None] +
                stride_cos_w_d*offs_cos_w_d[None, :])
    mask_cos_w = (offs_cos_w_d >= 0) & (offs_cos_w_d < D_MODEL // 2)
    cos_w = tl.load(cos_w_ptr + offs_cos_w, mask=mask_cos_w[None,:])

    #load sin_w
    sin_w = tl.load(sin_w_ptr + offs_cos_w, mask=mask_cos_w[None,:])

    #load x
    offs_wh = tl.arange(0, WH)
    offs_x = (stride_x_b * b +
            stride_x_wh * offs_wh[:, None] +
            stride_x_h * h +
            stride_x_d * offs_d[None, :])
    x = tl.load(x_ptr + offs_x)

    #load x_rotated
    offs_wh = tl.arange(0, WH)
    offs_d_rotated = tl.where(offs_d < D_MODEL // 4, offs_d + D_MODEL // 4, offs_d)
    offs_d_rotated = tl.where((offs_d >= D_MODEL // 4) & (offs_d < D_MODEL // 2), offs_d_rotated - D_MODEL // 4, offs_d_rotated)
    offs_d_rotated = tl.where((offs_d >= D_MODEL // 2) & (offs_d < 3*D_MODEL // 4), offs_d_rotated + D_MODEL // 4, offs_d_rotated)
    offs_d_rotated = tl.where((offs_d >= 3*D_MODEL // 4) & (offs_d < D_MODEL), offs_d_rotated - D_MODEL // 4, offs_d_rotated)
    offs_x_rotated = (stride_x_b * b +
            stride_x_wh * offs_wh[:, None] +
            stride_x_h * h +
            stride_x_d * offs_d_rotated[None, :])
    x_rotated = tl.load(x_ptr + offs_x_rotated)
    neg_x_rotated = tl.where((offs_d >= D_MODEL // 4) & (offs_d < D_MODEL // 2), 1, 0)
    neg_x_rotated = tl.where((offs_d >= 3*D_MODEL // 4) & (offs_d < D_MODEL), 1, neg_x_rotated)
    x_rotated = tl.where(neg_x_rotated, x_rotated, -x_rotated)

    #compute x1
    x1 = x * cos_h + x_rotated * sin_h

    #compute x2
    x2 = x * cos_w + x_rotated * sin_w

    #compute output
    out = x1 + x2

    #store output
    tl.store(out_ptr + offs_x, out)

#TODO: For now D_MODEL is assumed to be power of 2. Expand to handle other value of D.
def _rope_fwd(
    x: torch.Tensor,
    out: torch.Tensor,
    freqs: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part : bool,
    nope_first : bool,
    transpose_output: bool = False
) -> torch.Tensor :
    s, b, h, d = x.shape

    if freqs.shape[-1] == d // 2:
        if reuse_freqs_front_part:
            have_nope = False
        else:
            have_nope = True
    elif freqs.shape[-1] == d // 4:
        have_nope = True
    else:
        have_nope = False

    grid = (b,h,s)
    if rotate_style == RotateStyle.NEOX: 
        if have_nope:
            _rope_fwd_kernel_neox_nope[grid](x, freqs, out, 
                                        *x.stride(), *freqs.stride(), *out.stride(), 
                                        rotate_style, reuse_freqs_front_part, nope_first, 
                                        s, d, d // 2)
        else:
            _rope_fwd_kernel_neox[grid](x, freqs, out, 
                                    *x.stride(), *freqs.stride(), *out.stride(), 
                                    rotate_style, reuse_freqs_front_part,
                                    s, d, d // 2)
    else:
        if have_nope:
            _rope_fwd_kernel_gptj_nope[grid](x, freqs, out, 
                                    *x.stride(), *freqs.stride(), *out.stride(), 
                                    rotate_style, reuse_freqs_front_part, nope_first, 
                                    s, d, d // 2)
        else:
            _rope_fwd_kernel_gptj[grid](x, freqs, out, 
                                    *x.stride(), *freqs.stride(), *out.stride(), 
                                    rotate_style, reuse_freqs_front_part, 
                                    s, d, d // 2)

    return out

def rope_fwd(
    x: torch.Tensor,
    freqs: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part : bool,
    nope_first : bool,
    transpose_output: bool = False
) -> torch.Tensor :
    s, b, h, d = x.shape
    out = torch.empty((s,b,h,d), dtype=x.dtype, device=x.device, requires_grad=False)

    _rope_fwd(x, out, freqs, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

    return out

def rope_fwd_inplace(
    x: torch.Tensor,
    freqs: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part : bool,
    nope_first : bool,
    transpose_output: bool = False
) -> torch.Tensor :
    out = x
    _rope_fwd(x, out, freqs, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

    return out


def _rope_fwd_thd(
    x: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens: torch.Tensor,
    freqs: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part : bool,
    nope_first : bool,
    transpose_output: bool = False
) -> torch.Tensor :
    b = torch.numel(cu_seqlens)-1
    t, h, d = x.shape

    if freqs.shape[-1] == d // 2:
        if reuse_freqs_front_part:
            have_nope = False
        else:
            have_nope = True
    elif freqs.shape[-1] == d // 4:
        have_nope = True
    else:
        have_nope = False

    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int)
    max_seq_len_pow2 = triton.next_power_of_2(torch.max(seqlens).item())
    grid = (b,h,1)

    if rotate_style == RotateStyle.NEOX:
        if have_nope:
            _rope_fwd_kernel_neox_nope_thd[grid](x, cu_seqlens, freqs, out,
                                    *x.stride(), *freqs.stride(), *out.stride(),
                                    rotate_style, reuse_freqs_front_part, nope_first,
                                    max_seq_len_pow2, d, d // 2)
        else:
            _rope_fwd_kernel_neox_thd[grid](x, cu_seqlens, freqs, out,
                                    *x.stride(), *freqs.stride(), *out.stride(),
                                    rotate_style, reuse_freqs_front_part,
                                    max_seq_len_pow2, d, d // 2)
    else:
        if have_nope:
            _rope_fwd_kernel_gptj_nope_thd[grid](x, cu_seqlens, freqs, out, 
                                    *x.stride(), *freqs.stride(), *out.stride(), 
                                    rotate_style, reuse_freqs_front_part, nope_first, 
                                    max_seq_len_pow2, d, d // 2)
        else:
            _rope_fwd_kernel_gptj_thd[grid](x, cu_seqlens, freqs, out, 
                                    *x.stride(), *freqs.stride(), *out.stride(), 
                                    rotate_style, reuse_freqs_front_part, 
                                    max_seq_len_pow2, d, d // 2)

    return out

def rope_fwd_thd(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    freqs: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part : bool,
    nope_first : bool,
    transpose_output: bool = False
) -> torch.Tensor :
    t, h, d = x.shape
    out = torch.empty((t,h,d), dtype=x.dtype, device=x.device, requires_grad=False)

    _rope_fwd_thd(x, out, cu_seqlens, freqs, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

    return out

def rope_fwd_thd_inplace(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    freqs: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part : bool,
    nope_first : bool,
    transpose_output: bool = False
) -> torch.Tensor :
    out = x
    _rope_fwd_thd(x, out, cu_seqlens, freqs, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

    return out

#TODO: For now D_MODEL is assumed to be power of 2. Expand to handle other value of D.
def _rope_cached_fwd(
    x: torch.Tensor,
    out: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part : bool,
    nope_first : bool,
    transpose_output: bool = False
) -> torch.Tensor :
    s, b, h, d = x.shape

    if cos.shape[-1] == d // 2:
        if reuse_freqs_front_part:
            have_nope = False
        else:
            have_nope = True
    elif cos.shape[-1] == d // 4:
        have_nope = True
    else:
        have_nope = False

    grid = (b,h,s)
    if rotate_style == RotateStyle.NEOX: 
        if have_nope:
            _rope_fwd_kernel_neox_nope_cached[grid](x, cos, sin, out, 
                                        *x.stride(), *cos.stride(), *out.stride(), 
                                        rotate_style, reuse_freqs_front_part, nope_first, 
                                        s, d, d // 2)
        else:
            _rope_fwd_kernel_neox_cached[grid](x, cos, sin, out, 
                                    *x.stride(), *cos.stride(), *out.stride(), 
                                    rotate_style, reuse_freqs_front_part,
                                    s, d, d // 2)
    else:
        if have_nope:
            _rope_fwd_kernel_gptj_nope_cached[grid](x, cos, sin, out, 
                                    *x.stride(), *cos.stride(), *out.stride(), 
                                    rotate_style, reuse_freqs_front_part, nope_first, 
                                    s, d, d // 2)
        else:
            _rope_fwd_kernel_gptj_cached[grid](x, cos, sin, out, 
                                    *x.stride(), *cos.stride(), *out.stride(), 
                                    rotate_style, reuse_freqs_front_part, 
                                    s, d, d // 2)

    return out

def rope_cached_fwd(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part : bool,
    nope_first : bool,
    transpose_output: bool = False
):
    s, b, h, d = x.shape
    out = torch.empty((s,b,h,d), dtype=x.dtype, device=x.device, requires_grad=False)

    _rope_cached_fwd(x, out, cos, sin, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

    return out

def rope_cached_fwd_inplace(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part : bool,
    nope_first : bool,
    transpose_output: bool = False
):
    out = x
    _rope_cached_fwd(x, out, cos, sin, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

    return out

def _rope_cached_positions_fwd(
    x: torch.Tensor,
    out: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part : bool,
    nope_first : bool,
    transpose_output: bool = False
) -> torch.Tensor :
    s, b, h, d = x.shape
    if cos.shape[-1] == d // 2:
        if reuse_freqs_front_part:
            have_nope = False
        else:
            have_nope = True
    elif cos.shape[-1] == d // 4:
        have_nope = True
    else:
        have_nope = False

    grid = (b,h,s)
    if rotate_style == RotateStyle.NEOX: 
        if have_nope:
            _rope_fwd_kernel_neox_nope_cached_position[grid](x, cos, sin, positions, out, 
                                        *x.stride(), *cos.stride(), *positions.stride(), *out.stride(), 
                                        reuse_freqs_front_part, nope_first, 
                                        s, d, d // 2)
        else:
            _rope_fwd_kernel_neox_cached_position[grid](x, cos, sin, positions, out, 
                                    *x.stride(), *cos.stride(),*positions.stride(), *out.stride(),
                                    reuse_freqs_front_part, s, d, d // 2)
    else:
        if have_nope:
            _rope_fwd_kernel_gptj_nope_cached_position[grid](x, cos, sin, positions, out, 
                                    *x.stride(), *cos.stride(), *positions.stride(),*out.stride(), 
                                    reuse_freqs_front_part, nope_first, 
                                    s, d, d // 2)
        else:
            _rope_fwd_kernel_gptj_cached_position[grid](x, cos, sin, positions, out, 
                                    *x.stride(), *cos.stride(),*positions.stride(), *out.stride(), 
                                    reuse_freqs_front_part, 
                                    s, d, d // 2)

    return out

def rope_cached_positions_fwd(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part : bool,
    nope_first : bool,
    transpose_output: bool = False
) -> torch.Tensor :
    s, b, h, d = x.shape
    out = torch.empty((s,b,h,d), dtype=x.dtype, device=x.device, requires_grad=False)

    _rope_cached_positions_fwd(x, out, cos, sin, positions, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

    return out

def rope_cached_positions_fwd_inplace(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part : bool,
    nope_first : bool,
    transpose_output: bool = False
) -> torch.Tensor :
    out = x

    _rope_cached_positions_fwd(out, out, cos, sin, positions, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

    return out

def _rope_cached_positions_offsets_fwd(
    x: torch.Tensor,
    out: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    offsets: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part : bool,
    nope_first : bool,
    transpose_output: bool = False
) -> torch.Tensor :
    s, b, h, d = x.shape
    if cos.shape[-1] == d // 2:
        if reuse_freqs_front_part:
            have_nope = False
        else:
            have_nope = True
    elif cos.shape[-1] == d // 4:
        have_nope = True
    else:
        have_nope = False

    grid = (b,h,s)
    if rotate_style == RotateStyle.NEOX: 
        if have_nope:
            _rope_fwd_kernel_neox_nope_cached_position_off[grid](x, cos, sin, positions, offsets, out, 
                                        *x.stride(), *cos.stride(), *positions.stride(), *out.stride(), 
                                        reuse_freqs_front_part, nope_first, 
                                        s, d, d // 2)
        else:
            _rope_fwd_kernel_neox_cached_position_off[grid](x, cos, sin, positions, offsets, out, 
                                    *x.stride(), *cos.stride(),*positions.stride(), *out.stride(),
                                    reuse_freqs_front_part, s, d, d // 2)
    else:
        if have_nope:
            _rope_fwd_kernel_gptj_nope_cached_position_off[grid](x, cos, sin, positions, offsets, out, 
                                    *x.stride(), *cos.stride(), *positions.stride(),*out.stride(), 
                                    reuse_freqs_front_part, nope_first, 
                                    s, d, d // 2)
        else:
            _rope_fwd_kernel_gptj_cached_position_off[grid](x, cos, sin, positions, offsets, out, 
                                    *x.stride(), *cos.stride(),*positions.stride(), *out.stride(), 
                                    reuse_freqs_front_part, 
                                    s, d, d // 2)

    return out

def rope_cached_positions_offsets_fwd(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    offsets: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part : bool,
    nope_first : bool,
    transpose_output: bool = False
) -> torch.Tensor :
    s, b, h, d = x.shape
    out = torch.empty((s,b,h,d), dtype=x.dtype, device=x.device, requires_grad=False)

    _rope_cached_positions_offsets_fwd(x, out, cos, sin, positions, offsets, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

    return out

def rope_cached_positions_offsets_fwd_inplace(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    offsets: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part : bool,
    nope_first : bool,
    transpose_output: bool = False
) -> torch.Tensor :
    out = x

    _rope_cached_positions_offsets_fwd(out, out, cos, sin, positions, offsets, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

    return out

def _rope_cached_thd_positions_2c_fwd(
    x: torch.Tensor,
    y: torch.Tensor,
    out_x: torch.Tensor,
    out_y: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part : bool,
    nope_first : bool,
    transpose_output: bool = False
) -> torch.Tensor :
    t, h, d = x.shape
    if cos.shape[-1] == d // 2:
        if reuse_freqs_front_part:
            have_nope = False
        else:
            have_nope = True
    elif cos.shape[-1] == d // 4:
        have_nope = True
    else:
        have_nope = False

    #TODO: create heuristics for SPLIT_SEQ_SIZE based h and d
    SPLIT_SEQ_SIZE = 32
    grid = (h, triton.cdiv(t, SPLIT_SEQ_SIZE), 1)
    num_warps = 4
    waves_per_eu = 0
    if rotate_style == RotateStyle.GPTJ:
        _rope_fwd_kernel_gptj_cached_thd_position_2c[grid](x, y,  cos, sin, positions, out_x, out_y,
                                    *x.stride(), *cos.stride(),*positions.stride(), *out_x.stride(), 
                                    reuse_freqs_front_part = reuse_freqs_front_part, 
                                    SEQ_LEN = t,
                                    D_MODEL = d,
                                    D_MODEL_HALF = d // 2,
                                    SPLIT_SEQ_SIZE = SPLIT_SEQ_SIZE,
                                    num_warps = num_warps,
                                    waves_per_eu = waves_per_eu)
    elif rotate_style == RotateStyle.NEOX:
        #TODO: add a new kernel for NOEX    
        raise NotImplementedError("NEOX ratate style has not been implemented.")
    
    return out_x, out_y

def rope_cached_thd_positions_2c_fwd(
    x: torch.Tensor,
    y: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part : bool,
    nope_first : bool,
    transpose_output: bool = False 
):
    t, h, d = x.shape
    out_x = torch.empty((t,h,d), dtype=x.dtype, device=x.device, requires_grad=False)
    out_y = torch.empty((t,h,d), dtype=x.dtype, device=x.device, requires_grad=False)

    _rope_cached_thd_positions_2c_fwd(x, y, out_x, out_y, cos, sin, positions, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

    return out_x, out_y

def rope_cached_thd_positions_2c_fwd_inplace(
    x: torch.Tensor,
    y: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part : bool,
    nope_first : bool,
    transpose_output: bool = False 
):
    out_x = x
    out_y = y

    _rope_cached_thd_positions_2c_fwd(x, y, out_x, out_y, cos, sin, positions, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

    return out_x, out_y

def _rope_fwd_2d(
    x: torch.Tensor,
    out: torch.Tensor,
    cos_h: torch.Tensor,
    sin_h: torch.Tensor,
    cos_w: torch.Tensor,
    sin_w: torch.Tensor,
    img_height: torch.Tensor,
    img_width: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part : bool,
    nope_first : bool,
    transpose_output: bool = False 
):
    b, wh, h, d = x.shape
    #out = torch.empty((b,wh,h,d), dtype=x.dtype, device=x.device, requires_grad=False)

    grid = (b,h,1)
    _rope_fwd_2d_kernel_neox[grid](x, cos_h, sin_h, cos_w, sin_w, out, *x.stride(), *cos_h.stride(), *cos_w.stride(), wh, img_height, img_width, D_MODEL=d)

    return out

def rope_fwd_2d(
    x: torch.Tensor,
    cos_h: torch.Tensor,
    sin_h: torch.Tensor,
    cos_w: torch.Tensor,
    sin_w: torch.Tensor,
    img_height: torch.Tensor,
    img_width: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part : bool,
    nope_first : bool,
    transpose_output: bool = False 
):
    b, wh, h, d = x.shape
    out = torch.empty((b,wh,h,d), dtype=x.dtype, device=x.device, requires_grad=False)

    #grid = (b,h,1)
    #_rope_fwd_2d_kernel_neox[grid](x, cos_h, sin_h, cos_w, sin_w, out, *x.stride(), *cos_h.stride(), *cos_w.stride(), wh, img_height, img_width, D_MODEL=d)

    _rope_fwd_2d(x, out, cos_h, sin_h, cos_w, sin_w, img_height, img_width, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

    return out

def rope_fwd_2d_inplace(
    x: torch.Tensor,
    cos_h: torch.Tensor,
    sin_h: torch.Tensor,
    cos_w: torch.Tensor,
    sin_w: torch.Tensor,
    img_height: torch.Tensor,
    img_width: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part : bool,
    nope_first : bool,
    transpose_output: bool = False 
):
    out = x
    _rope_fwd_2d(x, out, cos_h, sin_h, cos_w, sin_w, img_height, img_width, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

    return out

class RoPE(autograd.Function):
    @staticmethod
    def forward(ctx, x:torch.Tensor, freqs:torch.Tensor, rotate_style: int, reuse_freqs_front_part : bool, nope_first : bool, transpose_output: bool = False) ->torch.Tensor:
        ctx.rotate_style = rotate_style
        ctx.reuse_freqs_front_part = reuse_freqs_front_part
        ctx.nope_first = nope_first
        ctx.transpose_output = transpose_output
        ctx.save_for_backward(freqs)
        return rope_fwd(x, freqs, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

class RoPETHD(autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, cu_seqlens: torch.Tensor, freqs: torch.Tensor, rotate_style: int, reuse_freqs_front_part : bool, nope_first : bool):
        ctx.rotate_style = rotate_style
        ctx.reuse_freqs_front_part = reuse_freqs_front_part
        ctx.nope_first = nope_first
        ctx.save_for_backward(cu_seqlens, freqs)
        return rope_fwd_thd(x, cu_seqlens, freqs, rotate_style, reuse_freqs_front_part, nope_first)

class RoPECached(autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotate_style: int, reuse_freqs_front_part : bool, nope_first : bool, transpose_output: bool = False) -> torch.Tensor:
        ctx.rotate_style = rotate_style
        ctx.reuse_freqs_front_part = reuse_freqs_front_part
        ctx.nope_first = nope_first
        ctx.transpose_output = transpose_output
        ctx.save_for_backward(cos, sin)
        return rope_cached_fwd(x, cos, sin, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

class RoPE2D(autograd.Function):
    @staticmethod
    def forward(ctx, x:torch.Tensor, 
                cos_height: torch.Tensor, sin_height: torch.Tensor, 
                cos_width: torch.Tensor, sin_width: torch.Tensor,
                img_height: int, img_width: int,
                rotate_style: int,
                reuse_freqs_front_part : bool,
                nope_first : bool) -> torch.Tensor:
        ctx.img_height = img_height
        ctx.img_width = img_width
        ctx.rotate_style = rotate_style
        ctx.reuse_freqs_front_part = reuse_freqs_front_part
        ctx.nope_first = nope_first
        ctx.save_for_backward(cos_height, sin_height, cos_width, sin_width)
        return rope_2d_fwd(x, cos_height, sin_height, cos_width, sin_width, img_height, img_width, rotate_style, reuse_freqs_front_part, nope_first)
    
