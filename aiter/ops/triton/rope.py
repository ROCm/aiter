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

#TODO: For now D_MODEL is assumed to be power of 2. Expand to handle other value of D.
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


#TODO: For now D_MODEL is assumed to be power of 2. Expand to handle other value of D.
def rope_fwd_cached(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part : bool,
    nope_first : bool,
    transpose_output: bool = False
) -> torch.Tensor :
    s, b, h, d = x.shape
    out = torch.empty((s,b,h,d), dtype=x.dtype, device=x.device, requires_grad=False)

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

class RoPE(autograd.Function):
    @staticmethod
    def forward(ctx, x:torch.Tensor, freqs:torch.Tensor, rotate_style: int, reuse_freqs_front_part : bool, nope_first : bool, transpose_output: bool = False) ->torch.Tensor:
        ctx.rotate_style = rotate_style
        ctx.reuse_freqs_front_part = reuse_freqs_front_part
        ctx.nope_first = nope_first
        ctx.transpose_output = transpose_output
        ctx.save_for_backward(freqs)
        return rope_fwd(x, freqs, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)


class RoPECached(autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotate_style: int, reuse_freqs_front_part : bool, nope_first : bool, transpose_output: bool = False) -> torch.Tensor:
        ctx.rotate_style = rotate_style
        ctx.reuse_freqs_front_part = reuse_freqs_front_part
        ctx.nope_first = nope_first
        ctx.transpose_output = transpose_output
        ctx.save_for_backward(cos, sin)
        return rope_fwd_cached(x, cos, sin, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)