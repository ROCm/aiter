"""MoE GEMM stage1/stage2 kernel implementations (FLIR MFMA FP8/FP16).

This module intentionally contains the **kernel builder code** for:
- `moe_gemm1` (stage1)
- `moe_gemm2` (stage2)

It is extracted from `tests/kernels/test_moe_gemm.py` so that:
- `kernels/` holds the implementation
- `tests/` holds correctness/perf harnesses
"""

import os
from contextlib import contextmanager

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import torch.nn.functional as F
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr.arith import ArithValue

from flydsl.expr import range_constexpr

try:
    from flydsl.runtime.device import (
        get_rocm_arch as get_hip_arch,
        supports_bf16_global_atomics,
        bf16_global_atomics_arch_description,
    )
except ImportError:
    from flydsl.runtime.device import get_rocm_arch as get_hip_arch

    def supports_bf16_global_atomics(arch: str) -> bool:
        return str(arch).startswith(("gfx94", "gfx95", "gfx12"))

    def bf16_global_atomics_arch_description() -> str:
        return "gfx94+/gfx95+/gfx12+"


from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from flydsl._mlir import ir
from flydsl.expr.typing import T

from flydsl.expr import arith as _arith_expr
from flydsl.dialects.ext import gpu, buffer_ops, llvm, vector, rocdl, scf, memref

from .mfma_preshuffle_pipeline import (
    buffer_copy_gmem16_dwordx4,
    lds_store_16b_xor16,
    make_preshuffle_b_layout,
    load_b_pack_k32,
    tile_chunk_coord_i32,
    swizzle_xor16,
)
from .mfma_epilogues import c_shuffle_epilog, mfma_epilog
from .layout_utils import crd2idx, idx2crd, get as layout_get

import functools


class _ArithCompat:
    """Compatibility wrapper for legacy arith helper names."""

    def __getattr__(self, name):
        return getattr(_arith_expr, name)

    def _const_like(self, v, other):
        ty = getattr(other, "type", None)
        if ty is None and hasattr(other, "ir_value"):
            try:
                ty = other.ir_value().type
            except Exception:
                ty = None
        if ty is None:
            ty = T.i32
        return _arith_expr.constant(v, type=ty)

    def i32(self, v):
        return _arith_expr.constant(v, type=T.i32)

    def f32(self, v):
        return _arith_expr.constant(v, type=T.f32)

    def cmpu(self, lhs, rhs, pred: str):
        if isinstance(lhs, int):
            lhs = self._const_like(lhs, rhs)
        if isinstance(rhs, int):
            rhs = self._const_like(rhs, lhs)
        pred_map = {
            "ult": _arith_expr.CmpIPredicate.ult,
            "ule": _arith_expr.CmpIPredicate.ule,
            "ne": _arith_expr.CmpIPredicate.ne,
        }
        p = pred_map.get(pred)
        if p is None:
            raise ValueError(f"Unsupported cmpu predicate: {pred!r}")
        return _arith_expr.cmpi(p, lhs, rhs)

    def as_value(self, v):
        if isinstance(v, ArithValue):
            return v
        if isinstance(v, ir.Value):
            return ArithValue(v)
        return v

    def ext_f(self, ty, v):
        return _arith_expr.extf(ty, v)

    def minimum(self, lhs, rhs):
        lhs = self.as_value(lhs)
        return lhs.minimumf(rhs)

    def maximum(self, lhs, rhs):
        lhs = self.as_value(lhs)
        return lhs.maximumf(rhs)


arith = _ArithCompat()


@contextmanager
def _if_then(if_op):
    """Compat helper for SCF IfOp then-region across old/new Python APIs."""
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


def make_preshuffle_scale_layout(
    arith_mod,
    *,
    c_mn: ir.Value,
    c_k: ir.Value,
    mn_pack: int = 2,
    k_pack: int = 2,
    elem_bytes: int = 4,
    scale_block_size: int = 32,
):
    def _to_i32(v):
        if hasattr(v, "ir_value"):
            v = v.ir_value()
        if hasattr(v, "type") and isinstance(v.type, ir.IndexType):
            return arith_mod.index_cast(T.i32, v)
        return v

    c16 = arith_mod.constant(16, type=T.i32)
    c4 = arith_mod.constant(4, type=T.i32)
    c_mn_pack = arith_mod.constant(mn_pack, type=T.i32)
    c_k_pack = arith_mod.constant(k_pack, type=T.i32)
    c_mn = _to_i32(c_mn)
    c_k = _to_i32(c_k)
    c_scale_block = arith_mod.constant(scale_block_size, type=T.i32)
    c_k_scale = c_k // c_scale_block

    c_mn1 = c_mn // c16 // c_mn_pack
    c_k1 = c_k_scale // c4 // c_k_pack
    if elem_bytes != mn_pack * k_pack:
        raise ValueError(
            f"elem_bytes of scale must be {mn_pack} * {k_pack}, got {elem_bytes!r}"
        )

    stride_nlane = arith_mod.constant(1, type=T.i32)
    stride_klane = c16
    stride_k0 = c4 * stride_klane
    stride_n0 = c_k1 * stride_k0
    stride_b_scale = (stride_n0, stride_k0, stride_klane, stride_nlane)
    return fx.make_layout((c_mn1, c_k1, c4, c16), stride=stride_b_scale)


def _mixed_out_torch_dtype(out_dtype: str):
    return torch.bfloat16 if str(out_dtype).lower() == "bf16" else torch.float16


def _decode_routing(
    sorted_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    *,
    token_num: int,
    topk: int,
    tile_m: int,
    valid_len: int,
):
    topk_ids = torch.zeros(
        (token_num, topk), dtype=torch.int64, device=sorted_ids.device
    )
    topk_weights = torch.zeros(
        (token_num, topk), dtype=torch.float32, device=sorted_ids.device
    )
    row_count = min(int(sorted_ids.numel()), int(sorted_weights.numel()))
    rows = sorted_ids[:row_count].to(torch.int64)
    weights = sorted_weights[:row_count].to(torch.float32)
    token_ids = rows & 0xFFFFFF
    slot_ids = rows >> 24
    valid = (token_ids < token_num) & (slot_ids < topk)
    token_ids = token_ids[valid]
    slot_ids = slot_ids[valid]
    row_indices = torch.nonzero(valid, as_tuple=False).flatten()
    expert_ids = sorted_expert_ids[(row_indices // tile_m)].to(torch.int64)
    topk_ids[token_ids, slot_ids] = expert_ids
    topk_weights[token_ids, slot_ids] = weights[valid]
    return topk_ids, topk_weights, token_ids, slot_ids


def _invert_shuffle_weight_fp4(x: torch.Tensor, layout=(16, 16)) -> torch.Tensor:
    x = x.contiguous()
    x_type = x.dtype
    if hasattr(torch, "float4_e2m1fn_x2") and x_type == torch.float4_e2m1fn_x2:
        x = x.view(torch.uint8)
    in_n, i_k = layout
    block_k = i_k * 2
    lane_pack = 16 // x.element_size()
    x = x.view(
        -1,
        x.shape[-2] // in_n,
        x.shape[-1] // block_k,
        block_k // lane_pack,
        in_n,
        lane_pack,
    )
    x = x.permute(0, 1, 4, 2, 3, 5).contiguous()
    return x.view(
        x.shape[0], x.shape[1] * x.shape[2], x.shape[3] * x.shape[4] * x.shape[5]
    ).view(x_type)


def _invert_e8m0_shuffle_2d(scale: torch.Tensor) -> torch.Tensor:
    scale = scale.contiguous()
    m, n = scale.shape
    scale = scale.view(m // 32, n // 8, 4, 16, 2, 2)
    scale = scale.permute(0, 5, 3, 1, 4, 2).contiguous()
    return scale.view(m, n)


def _invert_e8m0_shuffle(scale: torch.Tensor) -> torch.Tensor:
    if scale.ndim == 2:
        return _invert_e8m0_shuffle_2d(scale)
    if scale.ndim == 3:
        return torch.stack([_invert_e8m0_shuffle_2d(s) for s in scale], dim=0)
    raise ValueError(f"Unsupported shuffled scale rank: {scale.ndim}")


def _e8m0_to_f32_local(scale: torch.Tensor) -> torch.Tensor:
    scale_u8 = scale.view(torch.uint8)
    zero_case = scale_u8 == 0
    nan_case = scale_u8 == 0xFF
    scale_i32 = scale_u8.to(torch.int32) << 23
    scale_i32[zero_case] = 0x00400000
    scale_i32[nan_case] = 0x7F800001
    return scale_i32.view(torch.float32)


def _mxfp4_to_f32_local(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch, "float4_e2m1fn_x2") and x.dtype == torch.float4_e2m1fn_x2:
        x = x.view(torch.uint8)
    x = x.to(torch.uint8)
    x = x.repeat_interleave(2, dim=-1)
    x[..., ::2] = x[..., ::2] & 0xF
    x[..., 1::2] = x[..., 1::2] >> 4
    lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=x.device,
    )
    return lut[x.long()]


def _dequant_fp4_activation_stage1(
    a_packed: torch.Tensor,
    sorted_scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    *,
    token_num: int,
    model_dim: int,
    topk: int,
    tile_m: int,
    valid_len: int,
) -> torch.Tensor:
    a_f32 = _mxfp4_to_f32_local(a_packed.view(token_num, model_dim // 2))
    padded_rows = (
        (max(int(sorted_ids.numel()), int(valid_len)) + tile_m - 1) // tile_m
    ) * tile_m
    scale_groups = int(sorted_scale.numel() // padded_rows)
    scale_group_size = model_dim // scale_groups
    scale_sorted = _e8m0_to_f32_local(sorted_scale.view(padded_rows, scale_groups))
    scale_unsorted = torch.ones(
        (token_num, scale_groups), dtype=torch.float32, device=a_f32.device
    )
    filled = torch.zeros((token_num,), dtype=torch.bool, device=a_f32.device)
    id_rows = min(int(sorted_ids.numel()), padded_rows)
    rows = sorted_ids[:id_rows].to(torch.int64)
    token_ids = rows & 0xFFFFFF
    slot_ids = rows >> 24
    valid = (token_ids < token_num) & (slot_ids < topk)
    for row_idx in torch.nonzero(valid, as_tuple=False).flatten():
        t = int(token_ids[row_idx])
        if not filled[t]:
            scale_unsorted[t] = scale_sorted[row_idx]
            filled[t] = True
    return (
        a_f32.view(token_num, scale_groups, scale_group_size)
        * scale_unsorted.unsqueeze(-1)
    ).view(token_num, model_dim)


def _dequant_fp4_activation_stage2(
    a_packed: torch.Tensor,
    sorted_scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    *,
    token_num: int,
    inter_dim: int,
    topk: int,
    tile_m: int,
    valid_len: int,
) -> torch.Tensor:
    a_f32 = _mxfp4_to_f32_local(a_packed.view(token_num, topk, inter_dim // 2))
    padded_rows = (
        (max(int(sorted_ids.numel()), int(valid_len)) + tile_m - 1) // tile_m
    ) * tile_m
    scale_groups = int(sorted_scale.numel() // padded_rows)
    scale_group_size = inter_dim // scale_groups
    scale_sorted = _e8m0_to_f32_local(sorted_scale.view(padded_rows, scale_groups))
    scale_unsorted = torch.ones(
        (token_num, topk, scale_groups), dtype=torch.float32, device=a_f32.device
    )
    id_rows = min(int(sorted_ids.numel()), padded_rows)
    rows = sorted_ids[:id_rows].to(torch.int64)
    token_ids = rows & 0xFFFFFF
    slot_ids = rows >> 24
    valid = (token_ids < token_num) & (slot_ids < topk)
    for row_idx in torch.nonzero(valid, as_tuple=False).flatten():
        t = int(token_ids[row_idx])
        s = int(slot_ids[row_idx])
        scale_unsorted[t, s] = scale_sorted[row_idx]
    return (
        a_f32.view(token_num, topk, scale_groups, scale_group_size)
        * scale_unsorted.unsqueeze(-1)
    ).view(token_num, topk, inter_dim)


def _dequant_fp8_activation_stage1(
    a_q: torch.Tensor, a_scale: torch.Tensor, *, token_num: int, model_dim: int
) -> torch.Tensor:
    return a_q.view(token_num, model_dim).to(torch.float32) * a_scale.view(
        token_num, -1
    ).to(torch.float32)


def _dequant_fp8_activation_stage2(
    a_q: torch.Tensor,
    a_scale: torch.Tensor,
    *,
    token_num: int,
    topk: int,
    inter_dim: int,
) -> torch.Tensor:
    return a_q.view(token_num, topk, inter_dim).to(torch.float32) * a_scale.view(
        token_num, topk, -1
    ).to(torch.float32)


def _dequant_fp4_weight(
    w_shuf: torch.Tensor,
    w_scale_shuf: torch.Tensor,
    *,
    experts: int,
    out_dim: int,
    k_dim: int,
) -> torch.Tensor:
    w_unshuf = _invert_shuffle_weight_fp4(w_shuf.view(experts, out_dim, k_dim // 2))
    scale_groups_stored = int(w_scale_shuf.numel() // (experts * out_dim))
    scale_unshuf = _invert_e8m0_shuffle(
        w_scale_shuf.view(experts, out_dim, scale_groups_stored)
    )
    scale_group_size = k_dim // scale_groups_stored
    w_f32 = _mxfp4_to_f32_local(w_unshuf)
    scale_f32 = _e8m0_to_f32_local(scale_unshuf)
    return (
        w_f32.view(experts, out_dim, scale_groups_stored, scale_group_size)
        * scale_f32.unsqueeze(-1)
    ).view(experts, out_dim, k_dim)


def _launch_mixed_moe_stage1_torch(
    *,
    out_flat: torch.Tensor,
    a_flat: torch.Tensor,
    w_flat: torch.Tensor,
    a_scale_flat: torch.Tensor,
    w_scale_flat: torch.Tensor,
    sorted_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    inter_total: int,
    model_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    doweight: bool,
):
    valid_len = int(num_valid_ids.view(-1)[0].item())
    topk_ids, topk_weights, _, _ = _decode_routing(
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        token_num=token_num,
        topk=topk,
        tile_m=tile_m,
        valid_len=valid_len,
    )
    if a_dtype == "fp4":
        a_deq = _dequant_fp4_activation_stage1(
            a_flat,
            a_scale_flat,
            sorted_ids,
            token_num=token_num,
            model_dim=model_dim,
            topk=topk,
            tile_m=tile_m,
            valid_len=valid_len,
        )
    elif a_dtype == "fp8":
        a_deq = _dequant_fp8_activation_stage1(
            a_flat, a_scale_flat, token_num=token_num, model_dim=model_dim
        )
    else:
        raise NotImplementedError(
            f"torch fallback only supports fp4/fp8 activations, got {a_dtype}"
        )
    if b_dtype != "fp4":
        raise NotImplementedError(
            f"torch fallback only supports fp4 weights, got {b_dtype}"
        )
    w_deq = _dequant_fp4_weight(
        w_flat, w_scale_flat, experts=experts, out_dim=inter_total, k_dim=model_dim
    )
    inter_dim = inter_total // 2
    out = out_flat.view(token_num, topk, inter_dim)
    out.zero_()
    for e in range(experts):
        t_idx, s_idx = torch.where(topk_ids == e)
        if t_idx.numel() == 0:
            continue
        y = F.linear(a_deq[t_idx], w_deq[e])
        gate, up = y[:, :inter_dim], y[:, inter_dim:]
        y_act = F.silu(gate) * up
        if doweight:
            y_act = y_act * topk_weights[t_idx, s_idx].unsqueeze(-1)
        out[t_idx, s_idx] = y_act.to(_mixed_out_torch_dtype(out_dtype))


def _launch_mixed_moe_stage2_torch(
    *,
    out: torch.Tensor,
    a_flat: torch.Tensor,
    w_flat: torch.Tensor,
    a_scale_flat: torch.Tensor,
    w_scale_flat: torch.Tensor,
    sorted_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    a_dtype: str,
    b_dtype: str,
    doweight: bool,
):
    valid_len = int(num_valid_ids.view(-1)[0].item())
    topk_ids, topk_weights, _, _ = _decode_routing(
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        token_num=token_num,
        topk=topk,
        tile_m=tile_m,
        valid_len=valid_len,
    )
    if a_dtype == "fp4":
        a_deq = _dequant_fp4_activation_stage2(
            a_flat,
            a_scale_flat,
            sorted_ids,
            token_num=token_num,
            inter_dim=inter_dim,
            topk=topk,
            tile_m=tile_m,
            valid_len=valid_len,
        )
    elif a_dtype == "fp8":
        a_deq = _dequant_fp8_activation_stage2(
            a_flat, a_scale_flat, token_num=token_num, topk=topk, inter_dim=inter_dim
        )
    else:
        raise NotImplementedError(
            f"torch fallback only supports fp4/fp8 activations, got {a_dtype}"
        )
    if b_dtype != "fp4":
        raise NotImplementedError(
            f"torch fallback only supports fp4 weights, got {b_dtype}"
        )
    w_deq = _dequant_fp4_weight(
        w_flat, w_scale_flat, experts=experts, out_dim=model_dim, k_dim=inter_dim
    )
    out.zero_()
    for e in range(experts):
        t_idx, s_idx = torch.where(topk_ids == e)
        if t_idx.numel() == 0:
            continue
        y = F.linear(a_deq[t_idx, s_idx], w_deq[e])
        if doweight:
            y = y * topk_weights[t_idx, s_idx].unsqueeze(-1)
        out.index_add_(0, t_idx, y.to(out.dtype))


@functools.lru_cache(maxsize=None)
def compile_mixed_moe_gemm1(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    # NOTE: aiter swap passes these for API symmetry; stage1 uses dynamic memrefs so they are ignored.
    doweight_stage1: bool,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "f16",
    act: str = "swiglu",
    use_cshuffle_epilog: bool | None = None,
    enable_bias: bool = False,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    persist_m: int = 1,
):
    """Compile stage1 kernel (`moe_gemm1`) and return the compiled executable.

    a_dtype:
      - "fp8": X is fp8
      - "fp16": X is fp16 (caller uses tile_k halved vs fp8 to match MFMA K halving)
      - "int8": X is int8
      - "fp4": X is fp4

    b_dtype:
      - "fp8": W is fp8
      - "fp16": W is fp16 (caller uses tile_k halved vs fp8 to match MFMA K halving)
      - "int8": W is int8
      - "int4": W4A8 path: X is int8, W is packed int4 (2 values per byte) unpacked to int8 in-kernel
      - "fp4": W is fp4
    """
    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    if a_dtype not in ("fp8", "fp16", "int8", "fp4"):
        raise ValueError(
            f"a_dtype must be one of ('fp8','fp16','int8','fp4'), got {a_dtype!r}"
        )
    if b_dtype not in ("fp8", "fp16", "int8", "int4", "fp4"):
        raise ValueError(
            f"b_dtype must be one of ('fp8','fp16','int8','int4','fp4'), got {b_dtype!r}"
        )

    use_native = os.environ.get("FLYDSL_MIXED_USE_NATIVE", "0") in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    if not use_native:

        def _launch_via_torch(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_max_token_ids,
            arg_bias,
            i32_tokens_in,
            i32_inter_in,
            i32_k_in,
            i32_size_expert_ids_in,
            stream_ptr,
        ):
            del arg_bias, i32_size_expert_ids_in, stream_ptr
            _launch_mixed_moe_stage1_torch(
                out_flat=arg_out,
                a_flat=arg_x,
                w_flat=arg_w,
                a_scale_flat=arg_scale_x,
                w_scale_flat=arg_scale_w,
                sorted_ids=arg_sorted_token_ids,
                sorted_expert_ids=arg_expert_ids,
                sorted_weights=arg_sorted_weights,
                num_valid_ids=arg_max_token_ids,
                token_num=int(i32_tokens_in),
                inter_total=int(i32_inter_in),
                model_dim=int(i32_k_in),
                experts=experts,
                topk=topk,
                tile_m=tile_m,
                a_dtype=a_dtype,
                b_dtype=b_dtype,
                out_dtype=out_dtype,
                doweight=doweight_stage1,
            )

        return _launch_via_torch

    is_f16_a = a_dtype == "fp16"
    is_f16_b = b_dtype == "fp16"
    is_f16 = is_f16_a or is_f16_b

    is_f8_a = a_dtype == "fp8"
    is_f4_a = a_dtype == "fp4"
    is_f4_b = b_dtype == "fp4"

    pack_M = 2
    pack_N = 2
    pack_K = 2

    elem_bytes = 1

    a_elem_bytes = 2 if is_f16_a else 1
    b_elem_bytes = 1
    tile_k_bytes = int(tile_k) * int(a_elem_bytes)

    a_elem_vec_pack = 2 if is_f4_a else 1
    cbsz = 0 if is_f8_a else 4
    blgp = 4

    # enable_bias = False

    # K64-byte micro-step: always 64 bytes per `ku`. For fp16, this is 32 elements (2xK16 MFMA).
    if (tile_k_bytes % 64) != 0:
        raise ValueError(
            f"tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes} "
            f"(tile_k={tile_k}, elem_bytes={a_elem_bytes})"
        )
    is_int4 = b_dtype == "int4"
    # INT4 here means W4A8: X is int8, W is packed int4 and unpacked to int8 in-kernel.
    # is_int8 = (in_dtype == "int8") or is_int4
    is_int8 = False

    mfma_i32_k32 = None
    if is_int8:
        mfma_i32_k32 = getattr(rocdl, "mfma_i32_16x16x32i8", None) or getattr(
            rocdl, "mfma_i32_16x16x32_i8", None
        )
        if mfma_i32_k32 is None:
            raise AttributeError(
                "INT8 K32 MFMA op not found: expected `rocdl.mfma_i32_16x16x32i8` "
                "(or `rocdl.mfma_i32_16x16x32_i8`)."
            )

    def _x_elem_type():
        if is_f4_b:
            return T.f8 if is_f8_a else T.i8
        return T.f16 if is_f16 else (T.i8 if is_int8 else T.f8)

    def _w_elem_type():
        if is_f4_b:
            return T.i8
        return T.f16 if is_f16 else (T.i8 if is_int8 else T.f8)

    def _scale_elem_type():
        return T.i32

    def _x_lds_elem_type():
        return T.f16 if is_f16_a else (T.i8 if is_int8 else T.f8)

    def _out_elem_type():
        return T.bf16 if out_dtype == "bf16" else T.i32

    def _out_lds_elem_type():
        return T.f32

    def _out_vec_type():
        return T.bf16x1 if out_dtype == "bf16" else T.f8x1

    # size_out = tokens * topk * inter_dim
    # size_x = tokens * model_dim
    # # W is packed int4 for W4A8: 2 values per byte.
    # size_w = (experts * (2 * inter_dim) * model_dim) // 2 if is_int4 else (experts * (2 * inter_dim) * model_dim)

    DYN = ir.ShapedType.get_dynamic_size()
    size_out = DYN
    size_x = DYN
    # W is packed int4 for W4A8: 2 values per byte.
    size_w = (
        (experts * (2 * inter_dim) * model_dim) // 2
        if is_int4
        else (experts * (2 * inter_dim) * model_dim)
    )
    size_sorted = DYN
    size_expert_ids = DYN

    total_threads = 256
    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(a_elem_bytes)
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(
            "tile_m*tile_k*elem_bytes must be divisible by "
            f"{total_threads}: tile_m={tile_m}, tile_k={tile_k}, elem_bytes={a_elem_bytes}"
        )
    bytes_per_thread_x = bytes_x_per_tile // total_threads
    # Keep MoE stage1 X gmem->LDS pipeline consistent with the optimized GEMM kernel:
    # split into <=16B pieces and use buffer-load dwordx4 for gmem prefetch.
    # (Compute the split lens inside the kernel so the code matches GEMM structure.)

    # CK-style LDS128 mode (same idea as test_preshuffle_gemm.py):
    # - LDS stride == tile_k (no extra padding) + XOR16 swizzle
    # - Use ds_{read,write}_b128 (16B) and extract 8B halves for MFMA steps
    _ck_lds128 = os.environ.get("FLIR_CK_LDS128", "1") in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    pad_k = 0 if _ck_lds128 else 8
    lds_stride = tile_k + pad_k
    if use_cshuffle_epilog is None:
        use_cshuffle_epilog = os.environ.get("FLIR_MOE_STAGE1_CSHUFFLE", "0") in (
            "1",
            "true",
            "True",
            "YES",
            "yes",
        )
    use_cshuffle_epilog = bool(use_cshuffle_epilog)

    epilog_tag = "cshuffle" if use_cshuffle_epilog else "direct"
    module_name = (
        f"mfma_moe1_a{a_dtype}_w{b_dtype}_{epilog_tag}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"_pm{persist_m}"
    ).replace("-", "_")

    # -- LDS sizing (pure Python; no MLIR Context needed) ---------------------
    # Reuse the same LDS bytes for both:
    # - ping-pong X tiles (2 * tile_m * lds_stride * elem_bytes bytes)
    # - optional CShuffle tile (stage1 uses 2xf16 vector store, sized in 4B pairs)
    _use_cshuffle_epilog = bool(use_cshuffle_epilog)
    lds_x_bytes = 2 * int(tile_m) * int(lds_stride) * int(a_elem_bytes)
    lds_out_bytes = 4 * int(tile_m) * (int(tile_n) // 2) if _use_cshuffle_epilog else 0
    lds_total_bytes = max(lds_x_bytes, lds_out_bytes)
    lds_total_elems = lds_total_bytes if a_elem_bytes == 1 else (lds_total_bytes // 2)
    lds_alloc_bytes = int(lds_total_elems) * int(a_elem_bytes)
    lds_alloc_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_alloc_offset + lds_alloc_bytes

    if True:

        @flyc.kernel
        def moe_gemm1(
            arg_out: fx.Tensor,
            arg_x: fx.Tensor,
            arg_w: fx.Tensor,
            arg_scale_x: fx.Tensor,
            arg_scale_w: fx.Tensor,
            arg_sorted_token_ids: fx.Tensor,
            arg_expert_ids: fx.Tensor,
            arg_sorted_weights: fx.Tensor,
            arg_max_token_ids: fx.Tensor,
            arg_bias: fx.Tensor,
            i32_tokens_in: fx.Int32,
            i32_inter_in: fx.Int32,
            i32_k_in: fx.Int32,
            i32_size_expert_ids_in: fx.Int32,
        ):
            # unwrap tensor handles to memrefs for ext dialect helpers
            arg_out = arg_out.value
            arg_x = arg_x.value
            arg_w = arg_w.value
            arg_scale_x = arg_scale_x.value
            arg_scale_w = arg_scale_w.value
            arg_sorted_token_ids = arg_sorted_token_ids.value
            arg_expert_ids = arg_expert_ids.value
            arg_sorted_weights = arg_sorted_weights.value
            arg_max_token_ids = arg_max_token_ids.value
            arg_bias = arg_bias.value

            tokens_in = arith.index_cast(ir.IndexType.get(), i32_tokens_in.ir_value())
            inter_in = arith.ArithValue(
                arith.index_cast(ir.IndexType.get(), i32_inter_in.ir_value())
            )
            k_in = arith.index_cast(ir.IndexType.get(), i32_k_in.ir_value())
            size_expert_ids_in = arith.index_cast(
                ir.IndexType.get(), i32_size_expert_ids_in.ir_value()
            )
            tokens_i32_v = i32_tokens_in.ir_value()
            k_i32_v = i32_k_in.ir_value()

            x_elem = T.f16 if is_f16_a else (T.i8 if is_int8 else T.f8)
            # For int4, weights are stored as packed bytes (i8) and unpacked to i8 packs.
            w_elem = T.f16 if is_f16_b else (T.i8 if is_int8 else T.f8)
            f16 = T.f16
            f32 = T.f32
            i32 = T.i32
            i64 = T.i64
            vec4_f32 = T.vec(4, f32)
            vec4_i32 = T.vec(4, i32)
            vec4_f16 = T.vec(4, f16)
            vec4_f8 = T.vec(4, T.f8)
            vec1_f16 = T.vec(1, f16)
            vec1_f32 = T.vec(1, f32)
            vec16_elems = 16 if a_elem_bytes == 1 else 8
            vec8_elems = 8 if a_elem_bytes == 1 else 4
            vec4_elems = 4 if a_elem_bytes == 1 else 2
            vec8_x = T.vec(vec8_elems, x_elem)
            vec16_x = T.vec(vec16_elems, x_elem)
            vec1_i64 = T.vec(1, i64)
            vec2_i64 = T.vec(2, i64)
            x_lds_elem = _x_lds_elem_type()

            def silu(x):
                # Align with CK's device fast path:
                #   emu = exp(-x)  ~= exp2(log2e * (-x))  -> v_exp_f32
                #   sig = rcp(1 + emu)                   -> v_rcp_f32
                #   y = x * sig
                #
                # Using llvm.amdgcn intrinsics prevents lowering to the div_scale/div_fixup
                # sequences that introduce extra compares/cndmasks.
                t = x * (-1.4426950408889634)  # -log2(e)
                emu = llvm.call_intrinsic(f32, "llvm.amdgcn.exp2.f32", [t], [], [])
                den = 1.0 + emu
                sig = llvm.call_intrinsic(f32, "llvm.amdgcn.rcp.f32", [den], [], [])
                return x * sig

            def swiglu(gate, up, alpha=1.702, limit=7.0):
                # Align with CK's device fast path
                #
                # Using llvm.amdgcn intrinsics prevents lowering to the div_scale/div_fixup
                # sequences that introduce extra compares/cndmasks.
                gate = arith.minimum(gate, limit)
                up = arith.minimum(up, limit)
                up = arith.maximum(up, -limit)

                t = gate * (alpha) * (-1.4426950408889634)  # -log2(e)
                emu = llvm.call_intrinsic(f32, "llvm.amdgcn.exp2.f32", [t], [], [])
                den = 1.0 + emu
                sig = llvm.call_intrinsic(f32, "llvm.amdgcn.rcp.f32", [den], [], [])
                return gate * sig * (up + 1)

            acc_init = (
                arith.constant_vector(0, vec4_i32)
                if is_int8
                else arith.constant_vector(0.0, vec4_f32)
            )

            # Lccouts
            layout_x = fx.make_layout((tokens_i32_v, k_i32_v), stride=(k_i32_v, 1))

            # B preshuffle layout: match GEMM test helper exactly.
            c_n_total = arith.constant(experts * (2 * inter_dim), index=True)
            kpack_bytes = 8 if is_int4 else 16
            b_layout = make_preshuffle_b_layout(
                arith,
                c_n=c_n_total,
                c_k=k_in // pack_K,
                kpack_bytes=kpack_bytes,
                elem_bytes=b_elem_bytes,
            )
            layout_b = b_layout.layout_b

            m_repeat = tile_m // 16
            k_unroll = tile_k_bytes // 128  # K64-byte micro-step

            # A&B's scale preshuffle layout
            layout_a_scale = make_preshuffle_scale_layout(
                arith, c_mn=tokens_in, c_k=k_in
            )
            layout_b_scale = make_preshuffle_scale_layout(
                arith, c_mn=c_n_total, c_k=k_in
            )

            shape_lds = fx.make_shape(tile_m, tile_k)
            stride_lds = fx.make_stride(lds_stride, 1)
            layout_lds = fx.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            # Align with Aiter launch mapping (NSwizzle==false):
            # - blockIdx.x -> N dimension (tile along inter_dim)
            # - blockIdx.y -> expert-block id / M dimension (tile along sorted M)
            by = gpu.block_id("x")  # tile along inter_dim
            bx = gpu.block_id("y")  # tile along sorted M

            # Block validity: compute as early as possible so invalid blocks skip all buffer-resource
            # setup, LDS pointer math, and gmem prefetch work.
            bx_m = bx * arith.constant(tile_m, index=True)
            by_n = by * arith.constant(tile_n, index=True)

            maxids_rsrc = buffer_ops.create_buffer_resource(
                arg_max_token_ids, max_size=False, num_records_bytes=arith.i32(4)
            )
            max_token_id_i32 = buffer_ops.buffer_load(
                maxids_rsrc, arith.constant(0, index=True), vec_width=1, dtype=i32
            )

            bias_rsrc = (
                buffer_ops.create_buffer_resource(arg_bias, max_size=False)
                if enable_bias
                else None
            )

            bx_m_i32 = arith.index_cast(i32, bx_m)
            by_n_i32 = arith.index_cast(i32, by_n)
            blk_valid = arith.cmpu(bx_m_i32, max_token_id_i32, "ult")
            # Common constants/atoms (hoisted): keep IR small like GEMM.
            # CK-style XOR16 swizzle parameter (constant, power-of-two in our configs).
            k_blocks16 = arith.constant(tile_k_bytes // 16, index=True)
            layout_tx_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
            layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))

            _if_blk = scf.IfOp(blk_valid)
            with _if_then(_if_blk):
                base_ptr = allocator.get_base()
                lds_x_ptr = SmemPtr(
                    base_ptr,
                    lds_alloc_offset,
                    x_lds_elem,
                    shape=(lds_total_elems,),
                )
                lds_x = lds_x_ptr.get()
                # Alias LDS bytes as fp16 for optional CShuffle epilogue.
                _use_cshuffle_epilog = bool(use_cshuffle_epilog)

                lds_out = (
                    SmemPtr(
                        base_ptr,
                        lds_x_ptr.byte_offset,
                        _out_lds_elem_type(),
                        shape=(tile_m * tile_n,),
                    ).get()
                    if _use_cshuffle_epilog
                    else None
                )

                # Use logical buffer sizes (descriptor num_records) so hardware OOB checking can be
                # used directly (CK-style). This allows us to avoid `select`-based masking for
                # invalid lanes and rely on the buffer instruction's built-in bounds behavior.
                x_rsrc = buffer_ops.create_buffer_resource(
                    arg_x, max_size=False, num_records_bytes=tokens_in * model_dim
                )
                w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)
                out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=False)

                # fp16 path ignores scales completely (implicit scale=1.0).
                sx_rsrc = (
                    None
                    if is_f16_a
                    else buffer_ops.create_buffer_resource(arg_scale_x, max_size=False)
                )
                sw_rsrc = (
                    None
                    if is_f16_b
                    else buffer_ops.create_buffer_resource(arg_scale_w, max_size=False)
                )
                sorted_rsrc = buffer_ops.create_buffer_resource(
                    arg_sorted_token_ids, max_size=False
                )
                sorted_w_rsrc = buffer_ops.create_buffer_resource(
                    arg_sorted_weights, max_size=False
                )

                # expert ids: [blocks] i32 -> bytes = size_expert_ids_in*4
                eid_nbytes_i32 = arith.index_cast(
                    i32, size_expert_ids_in * arith.constant(4, index=True)
                )
                expert_rsrc = buffer_ops.create_buffer_resource(
                    arg_expert_ids, max_size=False, num_records_bytes=eid_nbytes_i32
                )

                # Expert id for this M tile (keep address math in `index`)
                expert_i32 = buffer_ops.buffer_load(
                    expert_rsrc, bx, vec_width=1, dtype=i32
                )
                exp_valid = arith.cmpu(expert_i32, experts, "ult")  # todo fix
                _ifexpert_of = scf.IfOp(exp_valid)
                with _if_then(_ifexpert_of):
                    expert_idx = arith.index_cast(ir.IndexType.get(), expert_i32)
                    inter2_idx = arith.constant(2 * inter_dim, index=True)
                    expert_off_idx = expert_idx * inter2_idx  # index

                    bx_m = bx * arith.constant(tile_m, index=True)

                    # ---- X gmem->reg prefetch (match preshuffle GEMM mapping) ----
                    # Keep a fixed 16B gmem->reg schedule (dwordx4) to match preshuffle_gemm.py.
                    if bytes_per_thread_x % 16 != 0:
                        raise ValueError(
                            f"bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 16"
                        )
                    x_load_bytes = 16
                    num_x_loads = bytes_per_thread_x // x_load_bytes
                    chunk_i32 = x_load_bytes // 4  # dwords per chunk (1/2/4)

                    # Work in dword units along K: K_dwords = (K_bytes)/4.
                    c_k_div4 = (
                        k_in * arith.constant(int(elem_bytes), index=True)
                    ) // arith.index(4)
                    c_k_div4_i32 = arith.index_cast(i32, c_k_div4)
                    layout_x_div4 = fx.make_layout(
                        (tokens_i32_v, c_k_div4_i32), stride=(c_k_div4_i32, 1)
                    )
                    tile_k_dwords = (int(tile_k) * int(elem_bytes)) // 4
                    layout_x_tile_div4 = fx.make_layout(
                        (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
                    )
                    c_chunk_i32 = arith.constant(chunk_i32, index=True)
                    tx_i32_base = tx * c_chunk_i32
                    mask24 = arith.i32(0xFFFFFF)
                    # Keep i32 constants available for epilogue index math.
                    topk_i32 = arith.i32(topk)

                    tokens_i32 = arith.index_cast(i32, tokens_in)

                    def x_tile_chunk_coord_i32(i: int):
                        return tile_chunk_coord_i32(
                            arith,
                            tx_i32_base=tx_i32_base,
                            i=i,
                            total_threads=total_threads,
                            layout_tile_div4=layout_x_tile_div4,
                            chunk_i32=chunk_i32,
                        )

                    # CK-aligned: decode token once (per thread's M-slice) and build a base row offset.
                    x_row_base_div4 = []
                    x_col_local_i32 = []
                    x_row_local = []
                    for i in range_constexpr(num_x_loads):
                        row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                        x_row_local.append(row_local)
                        x_col_local_i32.append(col_local_i32)

                        sorted_row_i = bx_m + row_local
                        fused_i = buffer_ops.buffer_load(
                            sorted_rsrc, sorted_row_i, vec_width=1, dtype=i32
                        )
                        t_i32 = fused_i & mask24
                        t_idx = arith.index_cast(ir.IndexType.get(), t_i32)
                        x_row_base_div4.append(t_idx * c_k_div4)

                    vec1_i32 = T.vec(1, i32)
                    vec2_i32 = T.vec(2, i32)
                    vec4_i32 = T.vec(4, i32)
                    vec4_x = T.vec(4, x_elem)

                    def load_x(idx_i32):
                        """Load `x_load_bytes` bytes from X (gmem) into regs.

                        For 16B, keep the fast dwordx4 path. For 8B/4B, use byte offsets.
                        """
                        idx_elem = (
                            idx_i32 if elem_bytes == 1 else (idx_i32 * arith.index(2))
                        )
                        return buffer_copy_gmem16_dwordx4(
                            buffer_ops,
                            vector,
                            elem_type=x_elem,
                            idx_i32=idx_elem,
                            rsrc=x_rsrc,
                            vec_elems=vec16_elems,
                        )

                    def load_x_tile(base_k):
                        """Prefetch the per-thread X tile portion (gmem -> regs) for a given K base (in elements)."""
                        base_k_div4 = (
                            base_k * arith.constant(int(elem_bytes), index=True)
                        ) // arith.index(4)
                        parts = []
                        for i in range_constexpr(num_x_loads):
                            idx_i32 = (
                                x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                            )
                            x_vec = load_x(idx_i32)
                            parts.append(vector.bitcast(vec4_i32, x_vec))
                        return parts

                    # tx -> wave/lane (GEMM-style decomposition).
                    coord_wl = idx2crd(tx, layout_tx_wave_lane)
                    wave_id = layout_get(coord_wl, 0)
                    lane_id = layout_get(coord_wl, 1)
                    coord_l16 = idx2crd(lane_id, layout_lane16)
                    lane_div_16 = layout_get(coord_l16, 0)
                    lane_mod_16 = layout_get(coord_l16, 1)

                    # Match GEMM naming/pattern: row in LDS is lane_mod_16, and col base is lane_div_16*16B (KPackBytes=16).
                    row_a_lds = lane_mod_16
                    # col_offset_base = lane_div_16 * arith.constant(32, index=True)
                    col_offset_base = lane_div_16 * arith.constant(16, index=True)

                    # Dynamic N tiling within block (same as existing kernels)
                    num_waves = 4
                    n_per_wave = tile_n // num_waves
                    num_acc_n = n_per_wave // 16
                    c_n_per_wave = arith.constant(n_per_wave, index=True)
                    wave_mod_4 = wave_id % arith.index(4)
                    n_tile_base = wave_mod_4 * c_n_per_wave

                    # fp4 pack
                    k_unroll_packed = k_unroll // pack_K
                    m_repeat_packed = m_repeat // pack_M
                    num_acc_n_packed = num_acc_n // pack_N

                    # Precompute n_blk/n_intra for gate and up rows (GEMM-style: idx2crd/get)
                    col_g_list = []
                    valid_col_list = []
                    inter_idx = arith.constant(inter_dim, index=True)
                    # layout for (row -> (blk,intra)) where intra is 0..15
                    c_n0 = c_n_total // arith.index(16)
                    c_n0_i32 = arith.index_cast(i32, c_n0)
                    layout_n_blk_intra = fx.make_layout((c_n0_i32, 16), stride=(16, 1))
                    n_intra_list = []
                    n_blk_list = []
                    for i in range_constexpr(num_acc_n):
                        offset = i * 16

                        col_g = by_n + n_tile_base
                        col_g = col_g // 2 + offset
                        col_g = col_g + lane_mod_16
                        col_g_list.append(col_g)

                        c_offset = arith.constant(offset, index=True)
                        global_n = by_n + n_tile_base + c_offset + lane_mod_16
                        row_w = expert_off_idx + global_n
                        coord_n = idx2crd(row_w, layout_n_blk_intra)
                        n_blk_list.append(layout_get(coord_n, 0))
                        n_intra_list.append(layout_get(coord_n, 1))

                    # --- B Load Logic (K64) - shared layout with preshuffle GEMM ---
                    def load_b_packs_k64(base_k, ku: int, ni: int):
                        # K64 micro-step = 2x K32 MFMA steps. Reuse the shared helper.
                        b0 = load_b_pack_k32(
                            buffer_ops,
                            arith,
                            vector,
                            arg_b=arg_w,
                            b_rsrc=w_rsrc,
                            layout_b=layout_b,
                            base_k=base_k,
                            ki_step=ku * 2,
                            n_blk=n_blk_list[ni],
                            n_intra=n_intra_list[ni],
                            lane_div_16=lane_div_16,
                            elem_type=_w_elem_type(),
                            kpack_bytes=kpack_bytes,
                            elem_bytes=b_elem_bytes,
                            unpack_int4=bool(is_int4),
                        )
                        b1 = load_b_pack_k32(
                            buffer_ops,
                            arith,
                            vector,
                            arg_b=arg_w,
                            b_rsrc=w_rsrc,
                            layout_b=layout_b,
                            base_k=base_k,
                            ki_step=ku * 2 + 1,
                            n_blk=n_blk_list[ni],
                            n_intra=n_intra_list[ni],
                            lane_div_16=lane_div_16,
                            elem_type=_w_elem_type(),
                            kpack_bytes=kpack_bytes,
                            elem_bytes=b_elem_bytes,
                            unpack_int4=bool(is_int4),
                        )
                        return b0, b1

                    def load_b_tile(base_k):
                        b_tile = []
                        for ku in range_constexpr(k_unroll):
                            packs0 = []
                            packs1 = []
                            for ni in range_constexpr(num_acc_n):
                                b0, b1 = load_b_packs_k64(base_k, ku, ni)
                                packs0.append(b0)
                                packs1.append(b1)
                            b_tile.append((packs0, packs1))
                        return b_tile

                    def load_scale(arg_scale, rsrc, layout, ku, mni):
                        k_lane = lane_div_16
                        n_lane = lane_mod_16
                        idx_pack = crd2idx((mni, ku, k_lane, n_lane), layout)
                        return buffer_ops.buffer_load(
                            rsrc, idx_pack, vec_width=1, dtype=T.i32
                        )

                    def load_b_scale_tile(base_k):
                        b_scale_tile = []
                        for ku in range_constexpr(k_unroll_packed):
                            for ni in range_constexpr(num_acc_n_packed):
                                scale = load_scale(
                                    arg_scale_w,
                                    sw_rsrc,
                                    layout_b_scale,
                                    ku + base_k,
                                    ni
                                    + (expert_off_idx + by_n + n_tile_base)
                                    // pack_N
                                    // 16,
                                )
                                b_scale_tile.append(scale)
                        return b_scale_tile

                    def load_a_scale_tile(base_k):
                        a_scale_tile = []
                        for ku in range_constexpr(k_unroll_packed):
                            for mi in range_constexpr(m_repeat_packed):
                                scale = load_scale(
                                    arg_scale_x,
                                    sx_rsrc,
                                    layout_a_scale,
                                    ku + base_k,
                                    mi + bx_m // pack_M // 16,
                                )
                                a_scale_tile.append(scale)
                        return a_scale_tile

                    def prefetch_ab_scale_tile(base_k):
                        return [None, load_b_scale_tile(base_k)]

                    acc_gate = [acc_init] * (num_acc_n // 2 * m_repeat)
                    acc_up = [acc_init] * (num_acc_n // 2 * m_repeat)

                    # ---- Pipeline helpers: store X tile to LDS with ping-pong base ----
                    def store_x_tile_to_lds(vec_x_in_parts, lds_base):
                        for i in range_constexpr(num_x_loads):
                            row_local = x_row_local[i]
                            col_local_i32 = x_col_local_i32[i]
                            if x_load_bytes == 16:
                                lds_store_16b_xor16(
                                    arith,
                                    vector,
                                    lds_memref=lds_x,
                                    vec16_ty=vec16_x,
                                    layout_lds=layout_lds,
                                    row_local=row_local,
                                    col_local_i32=col_local_i32,
                                    tx_c4=arith.index(4),
                                    k_blocks16=k_blocks16,
                                    lds_base=lds_base,
                                    vec_part_i32x4=vec_x_in_parts[i],
                                    elem_bytes=elem_bytes,
                                )

                    # --- A LDS load helper for K64 (load 16B once, extract 2x i64 halves) ---
                    def lds_load_packs_k64(curr_row_a_lds, col_base, lds_base):
                        # Swizzle in bytes, then convert to element offset for memref indexing.
                        col_base_swz_bytes = swizzle_xor16(
                            curr_row_a_lds, col_base, k_blocks16
                        )
                        col_base_swz = (
                            col_base_swz_bytes
                            if elem_bytes == 1
                            else (col_base_swz_bytes // arith.index(2))
                        )
                        idx_a16 = crd2idx(
                            (
                                arith.as_value(curr_row_a_lds),
                                arith.as_value(col_base_swz),
                            ),
                            layout_lds,
                        )
                        idx_a16 = idx_a16 + lds_base
                        loaded_a16 = vector.load_op(vec16_x, lds_x, [idx_a16])
                        a_i64x2 = vector.bitcast(vec2_i64, loaded_a16)
                        a0 = vector.extract(
                            a_i64x2, static_position=[0], dynamic_position=[]
                        )
                        a1 = vector.extract(
                            a_i64x2, static_position=[1], dynamic_position=[]
                        )
                        return a0, a1

                    def compute_f8f6f4_tile(
                        acc_gate_in,
                        acc_up_in,
                        b_tile_in,
                        lds_base,
                        *,
                        a0_prefetch=None,
                        a_scale=None,
                        b_scale=None,
                        prefetch_epilogue: bool = False,
                    ):
                        gate_list = list(acc_gate_in)
                        up_list = list(acc_up_in)

                        epilogue_pf = None
                        if enable_bias and prefetch_epilogue:
                            expert_off_idx + by_n + n_tile_base

                            gate_bias = []
                            up_bias = []
                            for ni in range_constexpr(num_acc_n_packed):
                                global_n = (
                                    (by_n + n_tile_base) // 2 + ni * 16 + lane_mod_16
                                )
                                gate_offset = expert_off_idx + global_n
                                up_offset = expert_off_idx + global_n + inter_dim
                                gate_bias.append(
                                    buffer_ops.buffer_load(
                                        bias_rsrc, gate_offset, vec_width=1, dtype=f32
                                    )
                                )
                                up_bias.append(
                                    buffer_ops.buffer_load(
                                        bias_rsrc, up_offset, vec_width=1, dtype=f32
                                    )
                                )
                            epilogue_pf = (gate_bias, up_bias)

                        # ---------------- gfx95 fast path (K128 MFMA scale) ----------------
                        # This is the key optimization from `zhimding/develop_0107` for FP8:
                        # use mfma.scale 16x16x128 to reduce instruction count in the hot loop.
                        #
                        # Notes:
                        # - Only valid for fp8 path (not int8/int4) and gfx95+
                        # - Requires tile_k divisible by 128
                        # - mfma.scale takes 9 operands: 3 vectors + 6 i32 flags/scales.
                        if (int(tile_k) % 128) != 0:
                            raise ValueError(
                                f"tile_k must be divisible by 128 for mfma_scale_x128, got tile_k={tile_k}"
                            )

                        mfma_res_ty = T.f32x4
                        vec4_i64 = T.vec(4, T.i64)
                        vec8_i32 = T.vec(8, T.i32)
                        c0_i64 = arith.constant(0, type=T.i64)

                        def pack_i64x4_to_i32x8(x0, x1, x2, x3):
                            v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
                            return vector.bitcast(vec8_i32, v4)

                        for ku128 in range_constexpr(k_unroll_packed):
                            for mi in range_constexpr(m_repeat_packed):
                                # a_scale_i32 = a_scale[ku128 * m_repeat_packed + mi]
                                # a_scale_val = vector.extract(a_scale_i32, static_position=[0], dynamic_position=[])
                                for ni in range_constexpr(num_acc_n_packed):
                                    b_scale_val = b_scale[ku128 * num_acc_n_packed + ni]
                                    for ikxdl in range_constexpr(pack_K):
                                        k_idx = ku128 * pack_K + ikxdl
                                        b_packs0, b_packs1 = b_tile_in[k_idx]
                                        col_base = (
                                            col_offset_base
                                            + (k_idx * 128) // a_elem_vec_pack
                                        )
                                        for imxdl in range_constexpr(pack_M):
                                            col_base0 = col_base
                                            mi_idx = mi * pack_M + imxdl
                                            mi_val = arith.constant(
                                                mi_idx * 16, index=True
                                            )
                                            curr_row_a_lds = row_a_lds + mi_val

                                            if (
                                                (a0_prefetch is not None)
                                                and (k_idx == 0)
                                                and (mi_idx == 0)
                                            ):
                                                a0, a1 = a0_prefetch
                                            else:
                                                a0, a1 = lds_load_packs_k64(
                                                    curr_row_a_lds, col_base0, lds_base
                                                )

                                            if is_f8_a:
                                                col_base1 = col_base + 64
                                                a2, a3 = lds_load_packs_k64(
                                                    curr_row_a_lds, col_base1, lds_base
                                                )
                                                a128 = pack_i64x4_to_i32x8(
                                                    a0, a1, a2, a3
                                                )
                                            else:
                                                a128 = pack_i64x4_to_i32x8(
                                                    a0, a1, c0_i64, c0_i64
                                                )

                                            for inxdl in range_constexpr(pack_N):
                                                if inxdl % 2 == 0:
                                                    current_accs_list = gate_list
                                                else:
                                                    current_accs_list = up_list
                                                ni_idx = ni * pack_N + inxdl

                                                b0 = b_packs0[ni_idx]
                                                b1 = b_packs1[ni_idx]
                                                b128 = pack_i64x4_to_i32x8(
                                                    b0, b1, c0_i64, c0_i64
                                                )

                                                acc_idx = mi_idx * num_acc_n_packed + ni
                                                rocdl.sched_barrier(0)
                                                current_accs_list[acc_idx] = (
                                                    rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                                        mfma_res_ty,
                                                        [
                                                            a128,
                                                            b128,
                                                            current_accs_list[acc_idx],
                                                            cbsz,
                                                            blgp,
                                                            # use per tensor quant a1 for now,
                                                            0,
                                                            0x3F800000,
                                                            ikxdl * pack_N + inxdl,
                                                            b_scale_val,
                                                        ],
                                                    )
                                                )

                        return gate_list, up_list, epilogue_pf

                    # ---------------- 2-stage pipeline (ping-pong LDS + B tile prefetch) ----------------
                    lds_tile_elems = arith.constant(tile_m * lds_stride, index=True)
                    lds_base_cur = arith.index(0)
                    lds_base_nxt = lds_tile_elems

                    # Optional scheduler hints (copied from tuned GEMM); can be disabled via env.
                    rocdl.sched_barrier(0)

                    def hot_loop_scheduler():
                        mfma_group = num_acc_n * 2
                        # K64 micro-step: 2x K32 MFMA per gemm.
                        mfma_total = (k_unroll * 2) * m_repeat * mfma_group
                        mfma_per_iter = 2 * mfma_group
                        sche_iters = (
                            0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
                        )

                        # DS-read preload (CK default is 2); clamp to non-negative.
                        rocdl.sched_dsrd(2)
                        rocdl.sched_mfma(2)
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(1)
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(1)

                        # DS-write hints near the end: match total X LDS-store micro-ops per thread.
                        dswr_tail = num_x_loads
                        if dswr_tail > sche_iters:
                            dswr_tail = sche_iters
                        dswr_start = sche_iters - dswr_tail
                        for sche_i in range_constexpr(sche_iters):
                            rocdl.sched_vmem(1)
                            rocdl.sched_mfma(mfma_group)
                            rocdl.sched_dsrd(1)
                            rocdl.sched_mfma(mfma_group)
                            if sche_i >= dswr_start - 1:
                                rocdl.sched_dswr(1)
                        rocdl.sched_barrier(0)

                    # Prologue: prefetch tile0, store to LDS(cur), sync.
                    k0 = arith.index(0)
                    x_regs0 = load_x_tile(k0)
                    w_regs0 = load_b_tile(k0)

                    a_scale_pong = None
                    a_scale_ping = None
                    # a_scale_pong, b_scale_pong = prefetch_ab_scale_tile(k0 // 2)
                    _, b_scale_pong = prefetch_ab_scale_tile(k0 // 2)
                    store_x_tile_to_lds(x_regs0, lds_base_cur)
                    gpu.barrier()

                    # Loop-carried ping/pong state.
                    lds_base_pong = lds_base_cur  # current/compute
                    lds_base_ping = lds_base_nxt  # next/load+store
                    w_regs_pong = w_regs0

                    # Cross-tile A0 LDS prefetch (default-on): prefetch the first A-pack (K64) for the
                    # tile we are about to compute from LDS, to overlap with upcoming VMEM.
                    a0_prefetch_pong = lds_load_packs_k64(
                        row_a_lds, col_offset_base, lds_base_pong
                    )

                    # Unrolled ping-pong main loop (2 tiles per iteration), leaving 2 tail tiles.
                    c2_tile_k = arith.constant(tile_k * 2, index=True)
                    c_k_main2 = k_in - c2_tile_k

                    for k_iv in range(arith.index(0), c_k_main2, c2_tile_k):
                        # ---- stage 0: prefetch+store ping, compute pong ----
                        next_k1 = k_iv + tile_k
                        x_regs_ping = load_x_tile(next_k1)
                        w_regs_ping = load_b_tile(next_k1 // 2)
                        _, b_scale_ping = prefetch_ab_scale_tile(
                            next_k1 // pack_K // 128
                        )

                        acc_gate, acc_up, _ = compute_f8f6f4_tile(
                            acc_gate,
                            acc_up,
                            w_regs_pong,
                            lds_base_pong,
                            a0_prefetch=a0_prefetch_pong,
                            a_scale=a_scale_pong,
                            b_scale=b_scale_pong,
                        )
                        a0_prefetch_pong = None
                        store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                        # hot_loop_scheduler()
                        gpu.barrier()

                        # Cross-tile prefetch for the ping tile we are about to compute.
                        a0_prefetch_ping = lds_load_packs_k64(
                            row_a_lds, col_offset_base, lds_base_ping
                        )

                        # ---- stage 1: prefetch+store pong, compute ping ----
                        next_k2 = k_iv + c2_tile_k
                        x_regs_pong = load_x_tile(next_k2)
                        w_regs_pong = load_b_tile(next_k2 // 2)
                        _, b_scale_pong = prefetch_ab_scale_tile(
                            next_k2 // pack_K // 128
                        )

                        acc_gate, acc_up, _ = compute_f8f6f4_tile(
                            acc_gate,
                            acc_up,
                            w_regs_ping,
                            lds_base_ping,
                            a0_prefetch=a0_prefetch_ping,
                            a_scale=a_scale_ping,
                            b_scale=b_scale_ping,
                        )
                        a0_prefetch_ping = None
                        store_x_tile_to_lds(x_regs_pong, lds_base_pong)
                        # hot_loop_scheduler()
                        gpu.barrier()

                        # Cross-tile prefetch for the next pong tile.
                        a0_prefetch_pong = lds_load_packs_k64(
                            row_a_lds, col_offset_base, lds_base_pong
                        )

                    # Tail: 2 remaining tiles at (k_in - 2*tile_k) and (k_in - tile_k).
                    k_tail1 = k_in - tile_k
                    x_regs_ping = load_x_tile(k_tail1)
                    w_regs_ping = load_b_tile(k_tail1 // 2)
                    _, b_scale_ping = prefetch_ab_scale_tile(k_tail1 // pack_K // 128)

                    acc_gate, acc_up, _ = compute_f8f6f4_tile(
                        acc_gate,
                        acc_up,
                        w_regs_pong,
                        lds_base_pong,
                        a0_prefetch=a0_prefetch_pong,
                        a_scale=a_scale_pong,
                        b_scale=b_scale_pong,
                    )
                    a0_prefetch_pong = None
                    store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                    # hot_loop_scheduler()
                    gpu.barrier()

                    # Cross-tile prefetch for the final ping tile.
                    a0_prefetch_ping = lds_load_packs_k64(
                        row_a_lds, col_offset_base, lds_base_ping
                    )

                    # Epilogue: compute last tile with epilogue scale prefetch to overlap loads with MFMA.
                    acc_gate, acc_up, epilogue_pf = compute_f8f6f4_tile(
                        acc_gate,
                        acc_up,
                        w_regs_ping,
                        lds_base_ping,
                        a0_prefetch=a0_prefetch_ping,
                        a_scale=a_scale_ping,
                        b_scale=b_scale_ping,
                        prefetch_epilogue=True,
                    )

                    # Store epilogue to out[t, slot, inter]
                    expert_off = expert_off_idx
                    bx_m0 = bx_m
                    topk_i32_v = topk_i32
                    inter_i32_v = arith.i32(inter_dim)
                    mask24_i32 = arith.i32(0xFFFFFF)

                    # Epilogue hoists to keep IR + Python build time small:
                    col_i32_list = []
                    for ni in range_constexpr(num_acc_n):
                        col_i32_list.append(arith.index_cast(i32, col_g_list[ni]))

                    lane_div_16_mul4 = lane_div_16 * arith.index(4)
                    inter_i32_local = inter_i32_v

                    # Optional: CK-style CShuffle epilogue for better global store coalescing.
                    # Uses EVec=4 (buffer store "x4" of fp16 elements).
                    _use_cshuffle_epilog = (out_dtype == "fp8") or bool(
                        use_cshuffle_epilog
                    )

                    mask_even_i32 = arith.i32(0xFFFFFFFE)

                    if _use_cshuffle_epilog:
                        if lds_out is None:
                            raise RuntimeError(
                                "CShuffle epilogue enabled but lds_out is not allocated/aliased."
                            )

                        def write_row_to_lds(
                            *,
                            mi: int,
                            ii: int,
                            row_in_tile,
                            row,
                            row_base_lds,
                            col_base_local,
                            num_acc_n: int,
                            lds_out,
                        ):
                            # `row` is the sorted-row index (bx_m + row_in_tile).
                            fused2 = buffer_ops.buffer_load(
                                sorted_rsrc, row, vec_width=1, dtype=i32
                            )
                            t2 = fused2 & mask24_i32

                            # Sorted weight aligned with `row` (matches aiter moe_sorting output).
                            if doweight_stage1:
                                tw = buffer_ops.buffer_load(
                                    sorted_w_rsrc, row, vec_width=1, dtype=f32
                                )

                            for ni in range_constexpr(num_acc_n_packed):
                                col_local = col_base_local + (ni * 16)

                                acc_idx = mi * num_acc_n + ni
                                vg = vector.extract(
                                    acc_gate[acc_idx],
                                    static_position=[ii],
                                    dynamic_position=[],
                                )
                                vu = vector.extract(
                                    acc_up[acc_idx],
                                    static_position=[ii],
                                    dynamic_position=[],
                                )

                                if enable_bias:
                                    gate_bias_list, up_bias_list = epilogue_pf
                                    vg = vg + gate_bias_list[ni]
                                    vu = vu + up_bias_list[ni]

                                if act == "swiglu_fast":
                                    y = swiglu(vg, vu)
                                else:
                                    y = silu(vg) * vu

                                if doweight_stage1:
                                    y = y * tw

                                lds_idx = row_base_lds + col_local
                                v1 = vector.from_elements(vec1_f32, [y])
                                vector.store(v1, lds_out, [lds_idx], alignment=1)

                        def precompute_row(*, row_local, row):
                            fused2 = buffer_ops.buffer_load(
                                sorted_rsrc, row, vec_width=1, dtype=i32
                            )
                            t2 = fused2 & mask24_i32
                            s2 = fused2 >> 24
                            return (t2 * topk_i32_v + s2) * inter_i32_local

                        def store_pair(
                            *, row_local, row, row_ctx, col_pair0, col_g0, frag
                        ):
                            # Guard against sentinel token ids (t == tokens) produced by aiter moe_sorting padding.
                            # OOB buffer stores are not guaranteed to be safe on all paths, so predicate explicitly.
                            fused2 = buffer_ops.buffer_load(
                                sorted_rsrc, row, vec_width=1, dtype=i32
                            )
                            t2 = fused2 & mask24_i32
                            t_valid = arith.cmpu(t2, tokens_i32, "ult")
                            _if_valid = scf.IfOp(t_valid)
                            with _if_then(_if_valid):
                                frag = vector.bitcast(vec4_f32, frag)
                                frag0 = vector.extract(
                                    frag, static_position=[0], dynamic_position=[]
                                )
                                frag1 = vector.extract(
                                    frag, static_position=[1], dynamic_position=[]
                                )
                                frag2 = vector.extract(
                                    frag, static_position=[2], dynamic_position=[]
                                )
                                frag3 = vector.extract(
                                    frag, static_position=[3], dynamic_position=[]
                                )

                                out_fp8 = arith.i32(0)
                                out_fp8 = rocdl.cvt_pk_fp8_f32(
                                    src_a=arith._unwrap_value(frag0),
                                    src_b=arith._unwrap_value(frag1),
                                    old=arith._unwrap_value(out_fp8),
                                    word_sel=0,
                                    res=T.i32,
                                )
                                out_fp8 = rocdl.cvt_pk_fp8_f32(
                                    src_a=arith._unwrap_value(frag2),
                                    src_b=arith._unwrap_value(frag3),
                                    old=arith._unwrap_value(out_fp8),
                                    word_sel=1,
                                    res=T.i32,
                                )

                                idx0 = row_ctx
                                col_i32 = arith.index_cast(i32, col_g0)
                                idx_out = idx0 + col_i32
                                buffer_ops.buffer_store(out_fp8, out_rsrc, idx_out // 4)

                        mfma_epilog(
                            use_cshuffle=True,
                            arith=arith,
                            vector=vector,
                            gpu=gpu,
                            range_constexpr=range_constexpr,
                            tile_m=tile_m,
                            tile_n=tile_n // 2,
                            e_vec=4,
                            m_repeat=m_repeat,
                            num_acc_n=num_acc_n_packed,
                            tx=tx,
                            lane_div_16=lane_div_16,
                            lane_mod_16=lane_mod_16,
                            bx_m=bx_m,
                            by_n=by_n // 2,
                            n_tile_base=n_tile_base // 2,
                            lds_out=lds_out,
                            frag_elem_type=T.f32,
                            write_row_to_lds=write_row_to_lds,
                            precompute_row=precompute_row,
                            store_pair=store_pair,
                        )
                        return

                    def _stage1_store_row(*, mi: int, ii: int, row_in_tile, row):
                        # `row` is the sorted-row index (bx_m + row_in_tile).
                        fused2 = buffer_ops.buffer_load(
                            sorted_rsrc, row, vec_width=1, dtype=i32
                        )
                        t2 = fused2 & mask24_i32
                        s2 = fused2 >> 24

                        t_valid = arith.cmpu(t2, tokens_i32, "ult")
                        # No explicit mask: rely on buffer descriptor OOB to zero-fill when t2 is the
                        # sentinel (t2 == tokens) or otherwise out-of-range.

                        # out linear index base = ((t*topk + s)*inter_dim) (invariant across ni)
                        idx0 = (t2 * topk_i32_v + s2) * inter_i32_local

                        # Sorted weight aligned with `row` (matches aiter moe_sorting output).
                        if doweight_stage1:
                            tw = buffer_ops.buffer_load(
                                sorted_w_rsrc, row, vec_width=1, dtype=f32
                            )

                        _if_valid = scf.IfOp(t_valid)
                        with _if_then(_if_valid):
                            for ni in range_constexpr(num_acc_n_packed):
                                col_i32 = col_i32_list[ni]
                                acc_idx = mi * num_acc_n_packed + ni
                                vg = vector.extract(
                                    acc_gate[acc_idx],
                                    static_position=[ii],
                                    dynamic_position=[],
                                )
                                vu = vector.extract(
                                    acc_up[acc_idx],
                                    static_position=[ii],
                                    dynamic_position=[],
                                )
                                if enable_bias:
                                    gate_bias_list, up_bias_list = epilogue_pf
                                    vg = vg + gate_bias_list[ni]
                                    vu = vu + up_bias_list[ni]

                                if act == "swiglu_fast":
                                    y = swiglu(vg, vu)
                                else:
                                    y = silu(vg) * vu

                                if doweight_stage1:
                                    y = y * tw

                                y = arith.trunc_f(_out_elem_type(), y)
                                idx_out = idx0 + col_i32
                                buffer_ops.buffer_store(y, out_rsrc, idx_out)

                    mfma_epilog(
                        use_cshuffle=False,
                        arith=arith,
                        range_constexpr=range_constexpr,
                        m_repeat=m_repeat,
                        lane_div_16=lane_div_16,
                        bx_m=bx_m,
                        body_row=_stage1_store_row,
                    )

    # -- Host launcher (flyc.jit + .launch) --------------------------------
    _cache_tag = (
        module_name,
        a_dtype,
        b_dtype,
        out_dtype,
        tile_m,
        tile_n,
        tile_k,
        doweight_stage1,
        act,
        enable_bias,
        model_dim_pad,
        inter_dim_pad,
        use_cshuffle_epilog,
        persist_m,
    )

    @flyc.jit
    def launch_mixed_moe_gemm1(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_max_token_ids: fx.Tensor,
        arg_bias: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream_ptr: int,
    ):
        _ = _cache_tag
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        inter_in = arith.ArithValue(
            arith.index_cast(ir.IndexType.get(), i32_inter_in.ir_value())
        )
        gx = (arith.constant(2, index=True) * inter_in) // arith.constant(
            tile_n, index=True
        )
        gy = arith.index_cast(ir.IndexType.get(), i32_size_expert_ids_in.ir_value())
        moe_gemm1(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_max_token_ids,
            arg_bias,
            i32_tokens_in,
            i32_inter_in,
            i32_k_in,
            i32_size_expert_ids_in,
        ).launch(
            grid=(gx, gy, 1),
            block=(256, 1, 1),
            stream=stream_ptr,
        )

    return launch_mixed_moe_gemm1


@functools.lru_cache(maxsize=None)
def compile_mixed_moe_gemm2(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "f16",
    use_cshuffle_epilog: bool | None = None,
    # Optional experiment: write per-(token,slot) output (no atomics) into an output shaped
    # [tokens*topk, model_dim] (or [tokens, topk, model_dim] flattened), then reduce over topk outside.
    # This can reduce atomic contention for small tokens at the cost of extra bandwidth / reduction.
    accumulate: bool = True,
    enable_bias: bool = False,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    persist_m: int = 1,
):
    """Compile stage2 kernel (`moe_gemm2`) and return the compiled executable.

    a_dtype:
      - "fp8": A2 is fp8
      - "fp16": A2 is fp16 (caller uses tile_k halved vs fp8 to match MFMA K halving)
      - "int8": A2 is int8
      - "fp4": A2 is fp4

    b_dtype:
      - "fp8": W is fp8
      - "fp16": W is fp16 (caller uses tile_k halved vs fp8 to match MFMA K halving)
      - "int8": W is int8
      - "int4": W4A8 path: A2 is int8, W is packed int4 (2 values per byte) unpacked to int8 in-kernel
      - "fp4": W is fp4

    Stage2 output supports:
      - out_dtype="f16": fp16 half2 atomics (fast, can overflow to +/-inf for bf16 workloads)
      - out_dtype="f32": fp32 scalar atomics (slower, but avoids fp16 atomic overflow)

    `use_cshuffle_epilog` controls whether we use the LDS CShuffle epilogue before
    global atomics (recommended for performance).
    """
    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    if a_dtype not in ("fp8", "fp16", "int8", "fp4"):
        raise ValueError(
            f"a_dtype must be one of ('fp8','fp16','int8','fp4'), got {a_dtype!r}"
        )
    if b_dtype not in ("fp8", "fp16", "int8", "int4", "fp4"):
        raise ValueError(
            f"b_dtype must be one of ('fp8','fp16','int8','int4','fp4'), got {b_dtype!r}"
        )

    use_native = os.environ.get("FLYDSL_MIXED_USE_NATIVE", "0") in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    if not use_native:

        def _launch_via_torch(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_num_valid_ids,
            arg_bias,
            i32_tokens_in,
            i32_n_in,
            i32_k_in,
            i32_size_expert_ids_in,
            stream_ptr,
        ):
            del arg_bias, i32_size_expert_ids_in, stream_ptr
            _launch_mixed_moe_stage2_torch(
                out=arg_out,
                a_flat=arg_x,
                w_flat=arg_w,
                a_scale_flat=arg_scale_x,
                w_scale_flat=arg_scale_w,
                sorted_ids=arg_sorted_token_ids,
                sorted_expert_ids=arg_expert_ids,
                sorted_weights=arg_sorted_weights,
                num_valid_ids=arg_num_valid_ids,
                token_num=int(i32_tokens_in),
                model_dim=int(i32_n_in),
                inter_dim=int(i32_k_in),
                experts=experts,
                topk=topk,
                tile_m=tile_m,
                a_dtype=a_dtype,
                b_dtype=b_dtype,
                doweight=doweight_stage2,
            )

        return _launch_via_torch

    is_f16_a = a_dtype == "fp16"
    is_f16_b = b_dtype == "fp16"
    is_f16 = is_f16_a or is_f16_b

    is_f8_a = a_dtype == "fp8"
    is_f4_a = a_dtype == "fp4"
    is_f4_b = b_dtype == "fp4"

    pack_M = 2
    pack_N = 2
    pack_K = 2

    elem_bytes = 1

    a_elem_bytes = 2 if is_f16_a else 1
    b_elem_bytes = 1
    tile_k_bytes = int(tile_k) * int(a_elem_bytes)

    a_elem_vec_pack = 2 if is_f4_a else 1
    cbsz = 0 if is_f8_a else 4
    blgp = 4

    # K64-byte micro-step: always 64 bytes per `ku`. For fp16, this is 32 elements (2xK16 MFMA).
    if (tile_k_bytes % 64) != 0:
        raise ValueError(
            f"tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes} "
            f"(tile_k={tile_k}, elem_bytes={a_elem_bytes})"
        )

    out_s = str(out_dtype).strip().lower()
    if out_s not in ("f16", "fp16", "half", "bf16", "bfloat16", "f32", "fp32", "float"):
        raise ValueError(
            f"out_dtype must be 'f16', 'bf16', or 'f32', got {out_dtype!r}"
        )
    out_is_f32 = out_s in ("f32", "fp32", "float")
    out_is_bf16 = out_s in ("bf16", "bfloat16")
    if (not bool(accumulate)) and out_is_f32:
        raise ValueError(
            "compile_moe_gemm2(accumulate=False) only supports out_dtype in {'f16','bf16'}"
        )
    is_int4 = b_dtype == "int4"
    # INT4 here means W4A8: A2 is int8, W is packed int4 and unpacked to int8 in-kernel.
    is_int8 = False

    mfma_i32_k32 = None
    if is_int8:
        mfma_i32_k32 = getattr(rocdl, "mfma_i32_16x16x32i8", None) or getattr(
            rocdl, "mfma_i32_16x16x32_i8", None
        )
        if mfma_i32_k32 is None:
            raise AttributeError(
                "INT8 K32 MFMA op not found: expected `rocdl.mfma_i32_16x16x32i8` "
                "(or `rocdl.mfma_i32_16x16x32_i8`)."
            )

    def _x_elem_type():
        if is_f4_b:
            return T.f8 if is_f8_a else T.i8
        return T.f16 if is_f16_a else (T.i8 if is_int8 else T.f8)

    def _w_elem_type():
        if is_f4_b:
            return T.i8
        return T.f16 if is_f16_b else (T.i8 if is_int8 else T.f8)

    def _scale_elem_type():
        return T.i32

    def _x_lds_elem_type():
        return T.f16 if is_f16_a else (T.i8 if is_int8 else T.f8)

    DYN = ir.ShapedType.get_dynamic_size()
    size_out = DYN
    size_x = DYN
    size_sorted = DYN
    size_expert_ids_shape = DYN
    size_scale_x = DYN
    # W is packed int4 for W4A8: 2 values per byte.
    size_w = (
        (experts * model_dim * inter_dim) // 2
        if is_int4
        else (experts * model_dim * inter_dim)
    )

    total_threads = 256
    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(a_elem_bytes)
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(
            "tile_m*tile_k*elem_bytes must be divisible by "
            f"{total_threads}: tile_m={tile_m}, tile_k={tile_k}, elem_bytes={a_elem_bytes}"
        )
    bytes_per_thread_x = bytes_x_per_tile // total_threads

    _ck_lds128 = os.environ.get("FLIR_CK_LDS128", "1") in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    pad_k = 0 if _ck_lds128 else 8
    lds_stride = tile_k + pad_k
    if out_is_bf16:
        if not supports_bf16_global_atomics(gpu_arch):
            raise ValueError(
                f"out_dtype='bf16' requires bf16 global atomics ({bf16_global_atomics_arch_description()}), got arch={gpu_arch!r}"
            )

    if out_is_f32:
        # Match origin/dev_a16w4: f32 output uses scalar atomics and does NOT use the CShuffle epilogue.
        _use_cshuffle_epilog = (
            False if use_cshuffle_epilog is None else bool(use_cshuffle_epilog)
        )
        if _use_cshuffle_epilog:
            raise ValueError(
                "out_dtype='f32' does not support CShuffle epilogue (set use_cshuffle_epilog=False)."
            )
    else:
        if use_cshuffle_epilog is None:
            _use_cshuffle_epilog = os.environ.get("FLIR_MOE_STAGE2_CSHUFFLE", "1") in (
                "1",
                "true",
                "True",
                "YES",
                "yes",
            )
        else:
            _use_cshuffle_epilog = bool(use_cshuffle_epilog)
        if not _use_cshuffle_epilog:
            raise ValueError(
                "stage2 f16 output currently requires CShuffle epilogue (FLIR_MOE_STAGE2_CSHUFFLE=1)."
            )

    # NOTE: Keep this as a callable so we don't require an MLIR Context at Python-time.
    out_elem = T.f32 if out_is_f32 else (T.bf16 if out_is_bf16 else T.f16)
    epilog_tag = "cshuffle"
    # IMPORTANT: include tiling in the module name to avoid accidentally reusing a compiled
    # binary for a different (tile_m, tile_n, tile_k) configuration.
    # See stage1 note: include ABI tag to prevent binary reuse across signature changes.
    # IMPORTANT: module name participates in FlyDSL's compile cache key.
    # Dynamic-shape variant: safe to reuse across (tokens/sorted_size/size_expert_ids) at runtime.
    # Keep a distinct ABI tag so the compile cache never mixes with historical signatures.
    module_name = (
        f"mfma_moe2_a{a_dtype}_w{b_dtype}_{out_s}_{epilog_tag}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"_pm{persist_m}"
    ).replace("-", "_")

    # -- LDS sizing (pure Python; no MLIR Context needed) ---------------------
    # Reuse a single allocation for both:
    # - ping-pong A2 tiles (2 * tile_m * lds_stride * elem_bytes bytes)
    # - epilogue CShuffle tile (tile_m * tile_n f16 -> 2 * tile_m * tile_n bytes)
    lds_x_bytes = 2 * int(tile_m) * int(lds_stride) * int(a_elem_bytes)
    lds_out_bytes = (
        2 * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
    )  # f16 bytes
    lds_total_bytes = max(lds_x_bytes, lds_out_bytes)
    lds_total_elems = lds_total_bytes if a_elem_bytes == 1 else (lds_total_bytes // 2)
    lds_alloc_bytes = int(lds_total_elems) * int(a_elem_bytes)
    lds_alloc_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_alloc_offset + lds_alloc_bytes

    if True:

        @flyc.kernel
        def moe_gemm2(
            arg_out: fx.Tensor,
            arg_x: fx.Tensor,
            arg_w: fx.Tensor,
            arg_scale_x: fx.Tensor,
            arg_scale_w: fx.Tensor,
            arg_sorted_token_ids: fx.Tensor,
            arg_expert_ids: fx.Tensor,
            arg_sorted_weights: fx.Tensor,
            arg_num_valid_ids: fx.Tensor,
            arg_bias: fx.Tensor,
            i32_tokens_in: fx.Int32,
            i32_n_in: fx.Int32,
            i32_k_in: fx.Int32,
            i32_size_expert_ids_in: fx.Int32,
        ):
            arg_out = arg_out.value
            arg_x = arg_x.value
            arg_w = arg_w.value
            arg_scale_x = arg_scale_x.value
            arg_scale_w = arg_scale_w.value
            arg_sorted_token_ids = arg_sorted_token_ids.value
            arg_expert_ids = arg_expert_ids.value
            arg_sorted_weights = arg_sorted_weights.value
            arg_num_valid_ids = arg_num_valid_ids.value
            arg_bias = arg_bias.value

            tokens_in = arith.index_cast(ir.IndexType.get(), i32_tokens_in.ir_value())
            n_in = arith.ArithValue(
                arith.index_cast(ir.IndexType.get(), i32_n_in.ir_value())
            )
            k_in = arith.index_cast(ir.IndexType.get(), i32_k_in.ir_value())
            size_expert_ids_in = arith.index_cast(
                ir.IndexType.get(), i32_size_expert_ids_in.ir_value()
            )
            tokens_i32_v = i32_tokens_in.ir_value()
            n_i32_v = i32_n_in.ir_value()
            k_i32_v = i32_k_in.ir_value()
            x_elem = T.f16 if is_f16_a else (T.i8 if is_int8 else T.f8)
            # For int4, weights are stored as packed bytes (i8) and unpacked to i8 packs.
            w_elem = T.f16 if is_f16_b else (T.i8 if is_int8 else T.f8)
            f16 = T.f16
            f32 = T.f32
            i32 = T.i32
            i64 = T.i64
            vec4_f32 = T.vec(4, f32)
            vec4_i32 = T.vec(4, i32)
            vec1_f16 = T.vec(1, f16)
            vec2_f16 = T.vec(2, f16)
            vec4_f16 = T.vec(4, f16)
            vec16_elems = 16 if a_elem_bytes == 1 else 8
            vec8_elems = 8 if a_elem_bytes == 1 else 4
            vec4_elems = 4 if a_elem_bytes == 1 else 2
            vec8_x = T.vec(vec8_elems, x_elem)
            vec16_x = T.vec(vec16_elems, x_elem)
            vec1_i64 = T.vec(1, i64)
            vec2_i64 = T.vec(2, i64)
            x_lds_elem = _x_lds_elem_type()

            acc_init = (
                arith.constant_vector(0, vec4_i32)
                if is_int8
                else arith.constant_vector(0.0, vec4_f32)
            )

            # A2 layout (flatten token-slot -> M).
            topk_idx = arith.constant(topk, index=True)
            m_in = tokens_in * topk_idx
            m_i32_v = arith.as_value(tokens_i32_v) * arith.i32(topk)
            layout_x = fx.make_layout((m_i32_v, k_i32_v), stride=(k_i32_v, 1))

            # B preshuffle layout: [experts*model_dim, inter_dim]
            c_n_total = arith.constant(experts * model_dim, index=True)
            kpack_bytes = 8 if is_int4 else 16
            b_layout = make_preshuffle_b_layout(
                arith,
                c_n=c_n_total,
                c_k=k_in // pack_K,
                kpack_bytes=kpack_bytes,
                elem_bytes=b_elem_bytes,
            )
            layout_b = b_layout.layout_b
            c_k0 = (
                k_in * arith.constant(int(a_elem_bytes), index=True)
            ) // arith.index(64)

            def check_c_n_valid_gate(base_n):
                return arith.cmpu(base_n, model_dim - model_dim_pad, "ult")

            def check_c_k_valid_gate(base_k):
                return arith.cmpu(base_k, inter_dim - inter_dim_pad, "ult")

            # A&B's scale preshuffle layout
            layout_a_scale = make_preshuffle_scale_layout(arith, c_mn=m_in, c_k=k_in)
            layout_b_scale = make_preshuffle_scale_layout(
                arith, c_mn=c_n_total, c_k=k_in
            )

            shape_lds = fx.make_shape(tile_m, tile_k)
            stride_lds = fx.make_stride(lds_stride, 1)
            layout_lds = fx.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            # Align with Aiter launch mapping:
            # - blockIdx.x -> N dimension (tile along model_dim)
            # - blockIdx.y -> expert-block id / M dimension (tile along sorted M)
            by = gpu.block_id("x")  # tile along model_dim
            bx = gpu.block_id("y")  # tile along sorted M

            # XOR16 swizzle parameter (in bytes; constant, power-of-two in our configs).
            k_blocks16 = arith.constant(tile_k_bytes // 16, index=True)
            layout_tx_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
            layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))
            layout_lin_rowcol = fx.make_layout((tile_m, tile_k), stride=(tile_k, 1))

            base_ptr = allocator.get_base()
            lds_x_ptr = SmemPtr(
                base_ptr,
                lds_alloc_offset,
                x_lds_elem,
                shape=(lds_total_elems,),
            )
            lds_x = lds_x_ptr.get()
            # Alias the same underlying LDS bytes as f16/bf16 for epilogue shuffle.
            lds_out = (
                SmemPtr(
                    base_ptr,
                    lds_x_ptr.byte_offset,
                    (T.bf16 if out_is_bf16 else T.f16),
                    shape=(tile_m * tile_n,),
                ).get()
                if _use_cshuffle_epilog
                else None
            )

            # Buffer resources.
            # For dynamic memrefs, `max_size=False` cannot infer the logical size from the memref *type*,
            # so we should pass `num_records_bytes` explicitly for stable hardware OOB behavior.
            c_topk = arith.constant(topk, index=True)

            # X(A2): [tokens*topk, inter_dim] bytes = tokens*topk*k*elem_bytes
            x_nbytes_idx = (
                (tokens_in * c_topk)
                * inter_dim
                * arith.constant(int(a_elem_bytes), index=True)
            )
            x_nbytes_i32 = arith.index_cast(i32, x_nbytes_idx)
            x_rsrc = buffer_ops.create_buffer_resource(
                arg_x, max_size=False, num_records_bytes=x_nbytes_i32
            )

            w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)

            # OUT: [tokens, model_dim] -> clamp to descriptor max (i32 bytes) to avoid overflow on huge tokens.
            out_elem_bytes = 4 if out_is_f32 else 2
            out_nbytes_idx = (
                tokens_in * n_in * arith.constant(out_elem_bytes, index=True)
            )
            if not bool(accumulate):
                out_nbytes_idx = (
                    tokens_in
                    * arith.index(topk)
                    * n_in
                    * arith.constant(out_elem_bytes, index=True)
                )
            out_nbytes_i32 = arith.index_cast(i32, out_nbytes_idx)
            out_rsrc = buffer_ops.create_buffer_resource(
                arg_out, max_size=False, num_records_bytes=out_nbytes_i32
            )
            # fp16 path ignores scales completely (implicit scale=1.0).
            if is_f16_a:
                sx_rsrc = None
            else:
                # scale_x (A2 scale): [tokens*topk] f32 -> bytes = tokens*topk*4
                sx_nbytes_idx = (tokens_in * c_topk) * arith.constant(4, index=True)
                sx_nbytes_i32 = arith.index_cast(i32, sx_nbytes_idx)
                sx_rsrc = buffer_ops.create_buffer_resource(
                    arg_scale_x, max_size=False, num_records_bytes=sx_nbytes_i32
                )

            if is_f16_b:
                sw_rsrc = None
            else:
                # scale_w: [experts*model_dim] f32 (static shape in practice)
                sw_rsrc = buffer_ops.create_buffer_resource(arg_scale_w, max_size=False)

            # sorted_token_ids / sorted_weights: [blocks*tile_m] (CK-style padded length)
            sorted_nbytes_idx = (
                size_expert_ids_in
                * arith.constant(tile_m, index=True)
                * arith.constant(4, index=True)
            )
            sorted_nbytes_i32 = arith.index_cast(i32, sorted_nbytes_idx)
            sorted_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_token_ids,
                max_size=False,
                num_records_bytes=sorted_nbytes_i32,
            )
            sorted_w_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_weights, max_size=False, num_records_bytes=sorted_nbytes_i32
            )

            # expert ids: [blocks] i32 -> bytes = size_expert_ids_in*4
            eid_nbytes_idx = size_expert_ids_in * arith.constant(4, index=True)
            eid_nbytes_i32 = arith.index_cast(i32, eid_nbytes_idx)
            expert_rsrc = buffer_ops.create_buffer_resource(
                arg_expert_ids, max_size=False, num_records_bytes=eid_nbytes_i32
            )
            bx_m = bx * arith.constant(tile_m, index=True)

            # Early-exit guard (as in 2ce65fb): some routing paths can produce extra/garbage
            # expert blocks beyond `num_valid_ids`. Skip those blocks entirely to avoid OOB.
            numids_rsrc = buffer_ops.create_buffer_resource(
                arg_num_valid_ids, max_size=False, num_records_bytes=arith.i32(4)
            )
            num_valid_i32 = buffer_ops.buffer_load(
                numids_rsrc, arith.constant(0, index=True), vec_width=1, dtype=i32
            )
            bx_m_i32 = arith.index_cast(i32, bx_m)
            blk_valid = arith.cmpu(bx_m_i32, num_valid_i32, "ult")

            bias_rsrc = (
                buffer_ops.create_buffer_resource(arg_bias, max_size=False)
                if enable_bias
                else None
            )

            expert_i32 = buffer_ops.buffer_load(expert_rsrc, bx, vec_width=1, dtype=i32)
            expert_idx = arith.index_cast(ir.IndexType.get(), expert_i32)
            exp_valid = arith.cmpu(expert_i32, experts, "ult")  # todo fix

            def _moe_gemm2_then_body():
                # Expert id for this M tile.
                n_idx = arith.constant(model_dim, index=True)
                expert_off_idx = expert_idx * n_idx  # index

                # ---- X gmem->reg prefetch (match preshuffle GEMM mapping) ----
                # Keep a fixed 16B gmem->reg schedule (dwordx4) to match preshuffle_gemm.py.
                if bytes_per_thread_x % 16 != 0:
                    raise ValueError(
                        f"bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 16"
                    )
                x_load_bytes = 16
                num_x_loads = bytes_per_thread_x // x_load_bytes
                chunk_i32 = x_load_bytes // 4  # dwords per chunk (1/2/4)
                vec4_i32 = T.vec(4, i32)

                c_k_div4 = (
                    k_in * arith.constant(int(a_elem_bytes), index=True)
                ) // arith.index(4)
                c_k_div4_i32 = arith.index_cast(i32, c_k_div4)
                layout_x_div4 = fx.make_layout(
                    (m_i32_v, c_k_div4_i32), stride=(c_k_div4_i32, 1)
                )
                tile_k_dwords = (int(tile_k) * int(a_elem_bytes)) // 4
                layout_x_tile_div4 = fx.make_layout(
                    (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
                )
                c_chunk_i32 = arith.constant(chunk_i32, index=True)
                tx_i32_base = tx * c_chunk_i32

                topk_i32 = arith.i32(topk)
                mask24 = arith.i32(0xFFFFFF)
                # Sentinel clamp uses `tokens` as the upper bound: t_valid = (t < tokens).
                tokens_i32 = arith.index_cast(i32, tokens_in)

                def x_tile_chunk_coord_i32(i: int):
                    return tile_chunk_coord_i32(
                        arith,
                        tx_i32_base=tx_i32_base,
                        i=i,
                        total_threads=total_threads,
                        layout_tile_div4=layout_x_tile_div4,
                        chunk_i32=chunk_i32,
                    )

                vec1_i32 = T.vec(1, i32)
                vec2_i32 = T.vec(2, i32)
                vec4_x = T.vec(4, x_elem)

                def load_x(idx_i32):
                    idx_elem = (
                        idx_i32 if a_elem_bytes == 1 else (idx_i32 * arith.index(2))
                    )
                    return buffer_copy_gmem16_dwordx4(
                        buffer_ops,
                        vector,
                        elem_type=x_elem,
                        idx_i32=idx_elem,
                        rsrc=x_rsrc,
                        vec_elems=vec16_elems,
                    )

                # decode routed token once (per thread's M-slice) and build a base offset.
                x_row_base_div4 = []
                x_col_local_i32 = []
                x_row_local = []
                for i in range_constexpr(num_x_loads):
                    row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                    x_row_local.append(row_local)
                    x_col_local_i32.append(col_local_i32)

                    sorted_row_i = bx_m + row_local
                    fused_i = buffer_ops.buffer_load(
                        sorted_rsrc, sorted_row_i, vec_width=1, dtype=i32
                    )
                    t_i32 = fused_i & mask24
                    s_i32 = arith.shrui(fused_i, arith.i32(24))
                    # Keep `blk_valid` only; remove per-row token validity checks.

                    t_valid = arith.cmpu(t_i32, tokens_i32, "ult")
                    s_valid = arith.cmpu(s_i32, topk_i32, "ult")
                    ts_valid = t_valid & s_valid
                    t_safe = arith.select(ts_valid, t_i32, arith.i32(0))
                    s_safe = arith.select(ts_valid, s_i32, arith.i32(0))
                    row_ts_i32 = t_safe * topk_i32 + s_safe
                    row_ts_idx = arith.index_cast(ir.IndexType.get(), row_ts_i32)

                    # Base row offset in dword units: row_ts_idx * (k_in/4)
                    x_row_base_div4.append(row_ts_idx * c_k_div4)

                def load_x_tile(base_k):
                    base_k_div4 = (
                        base_k * arith.constant(int(a_elem_bytes), index=True)
                    ) // arith.index(4)
                    parts = []
                    for i in range_constexpr(num_x_loads):
                        idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                        x_vec = load_x(idx_i32)

                        if x_load_bytes == 16:
                            parts.append(vector.bitcast(vec4_i32, x_vec))
                        elif x_load_bytes == 8:
                            parts.append(vector.bitcast(vec2_i32, x_vec))
                        else:
                            parts.append(vector.bitcast(vec1_i32, x_vec))
                    return parts

                # tx -> wave/lane (GEMM-style decomposition).
                coord_wl = idx2crd(tx, layout_tx_wave_lane)
                wave_id = layout_get(coord_wl, 0)
                lane_id = layout_get(coord_wl, 1)
                coord_l16 = idx2crd(lane_id, layout_lane16)
                lane_div_16 = layout_get(coord_l16, 0)
                lane_mod_16 = layout_get(coord_l16, 1)

                row_a_lds = lane_mod_16

                col_offset_base = lane_div_16 * arith.constant(16, index=True)

                # Dynamic N tiling within block.
                by_n = by * arith.constant(tile_n, index=True)
                num_waves = 4
                n_per_wave = tile_n // num_waves
                num_acc_n = n_per_wave // 16
                c_n_per_wave = arith.constant(n_per_wave, index=True)
                wave_mod_4 = wave_id % arith.constant(4, index=True)
                n_tile_base = wave_mod_4 * c_n_per_wave

                # Precompute (n_blk, n_intra) for B, and col indices for output.
                n_intra_list = []
                n_blk_list = []
                col_g_list = []
                c_n0 = c_n_total // arith.index(16)
                c_n0_i32 = arith.index_cast(i32, c_n0)
                layout_n_blk_intra = fx.make_layout((c_n0_i32, 16), stride=(16, 1))

                for i in range_constexpr(num_acc_n):
                    offset = i * 16

                    col_g = by_n + n_tile_base
                    col_g = col_g // 2 + offset
                    col_g = col_g + lane_mod_16
                    col_g_list.append(col_g)

                    c_offset = arith.constant(offset, index=True)
                    global_n = by_n + n_tile_base + c_offset + lane_mod_16
                    row_w = expert_off_idx + global_n
                    coord_n = idx2crd(row_w, layout_n_blk_intra)
                    n_blk_list.append(layout_get(coord_n, 0))
                    n_intra_list.append(layout_get(coord_n, 1))

                m_repeat = tile_m // 16
                k_unroll = tile_k_bytes // 128  # K64-byte micro-step (2x MFMA)

                # fp4 pack
                k_unroll_packed = k_unroll // pack_K
                m_repeat_packed = m_repeat // pack_M
                num_acc_n_packed = num_acc_n // pack_N

                # --- B Load Logic (K64) - shared layout with preshuffle GEMM ---
                def load_b_packs_k64(base_k, ku: int, ni: int):
                    b0 = load_b_pack_k32(
                        buffer_ops,
                        arith,
                        vector,
                        arg_b=arg_w,
                        b_rsrc=w_rsrc,
                        layout_b=layout_b,
                        base_k=base_k,
                        ki_step=ku * 2,
                        n_blk=n_blk_list[ni],
                        n_intra=n_intra_list[ni],
                        lane_div_16=lane_div_16,
                        elem_type=_w_elem_type(),
                        kpack_bytes=kpack_bytes,
                        elem_bytes=b_elem_bytes,
                        unpack_int4=bool(is_int4),
                    )
                    b1 = load_b_pack_k32(
                        buffer_ops,
                        arith,
                        vector,
                        arg_b=arg_w,
                        b_rsrc=w_rsrc,
                        layout_b=layout_b,
                        base_k=base_k,
                        ki_step=ku * 2 + 1,
                        n_blk=n_blk_list[ni],
                        n_intra=n_intra_list[ni],
                        lane_div_16=lane_div_16,
                        elem_type=_w_elem_type(),
                        kpack_bytes=kpack_bytes,
                        elem_bytes=b_elem_bytes,
                        unpack_int4=bool(is_int4),
                    )
                    return b0, b1

                def load_b_tile(base_k):
                    b_tile = []
                    for ku in range_constexpr(k_unroll):
                        packs0 = []
                        packs1 = []
                        for ni in range_constexpr(num_acc_n):
                            b0, b1 = load_b_packs_k64(base_k, ku, ni)
                            packs0.append(b0)
                            packs1.append(b1)
                        b_tile.append((packs0, packs1))
                    return b_tile

                def load_scale(arg_scale, rsrc, layout, ku, mni):
                    k_lane = lane_div_16
                    n_lane = lane_mod_16
                    idx_pack = crd2idx((mni, ku, k_lane, n_lane), layout)
                    return buffer_ops.buffer_load(
                        rsrc, idx_pack, vec_width=1, dtype=T.i32
                    )

                def load_b_scale_tile(base_k):
                    b_scale_tile = []
                    for ku in range_constexpr(k_unroll_packed):
                        for ni in range_constexpr(num_acc_n_packed):
                            scale = load_scale(
                                arg_scale_w,
                                sw_rsrc,
                                layout_b_scale,
                                ku + base_k,
                                ni
                                + (expert_off_idx + by_n + n_tile_base) // pack_N // 16,
                            )
                            b_scale_tile.append(scale)
                    return b_scale_tile

                def load_a_scale_tile(base_k):
                    a_scale_tile = []
                    for ku in range_constexpr(k_unroll_packed):
                        for mi in range_constexpr(m_repeat_packed):
                            scale = load_scale(
                                arg_scale_x,
                                sx_rsrc,
                                layout_a_scale,
                                ku + base_k,
                                mi + bx_m // pack_M // 16,
                            )
                            a_scale_tile.append(scale)
                    return a_scale_tile

                def prefetch_ab_scale_tile(base_k):
                    # return [load_a_scale_tile(base_k), load_b_scale_tile(base_k)]
                    return [None, load_b_scale_tile(base_k)]

                # ---- Pipeline helpers: store X tile to LDS with ping-pong base ----
                def store_x_tile_to_lds(vec_x_in_parts, lds_base):
                    for i in range_constexpr(num_x_loads):
                        row_local = x_row_local[i]
                        col_local_i32 = x_col_local_i32[i]
                        if x_load_bytes == 16:
                            lds_store_16b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec16_ty=vec16_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=arith.index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x4=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )

                # --- A LDS load helper for K64 (load 16B once, extract 2x i64 halves) ---
                def lds_load_packs_k64(curr_row_a_lds, col_base, lds_base):
                    # Swizzle in bytes, then convert to element offset for memref indexing.
                    col_base_swz_bytes = swizzle_xor16(
                        curr_row_a_lds, col_base, k_blocks16
                    )
                    col_base_swz = (
                        col_base_swz_bytes
                        if elem_bytes == 1
                        else (col_base_swz_bytes // arith.index(2))
                    )
                    idx_a16 = crd2idx(
                        (arith.as_value(curr_row_a_lds), arith.as_value(col_base_swz)),
                        layout_lds,
                    )
                    idx_a16 = idx_a16 + lds_base
                    loaded_a16 = vector.load_op(vec16_x, lds_x, [idx_a16])
                    a_i64x2 = vector.bitcast(vec2_i64, loaded_a16)
                    a0 = vector.extract(
                        a_i64x2, static_position=[0], dynamic_position=[]
                    )
                    a1 = vector.extract(
                        a_i64x2, static_position=[1], dynamic_position=[]
                    )
                    return a0, a1

                def compute_tile(
                    acc_in,
                    b_tile_in,
                    lds_base,
                    a_scale=None,
                    b_scale=None,
                    *,
                    prefetch_epilogue: bool = False,
                    a0_prefetch=None,
                ):
                    acc_list = list(acc_in)
                    mfma_res_ty = vec4_i32 if is_int8 else vec4_f32

                    epilogue_pf = None
                    bias = None
                    if prefetch_epilogue:
                        if enable_bias:
                            bias = []
                            for ni in range_constexpr(num_acc_n):
                                global_n = by_n + n_tile_base + ni * 16 + lane_mod_16
                                bias_offset = expert_off_idx + global_n
                                bias.append(
                                    buffer_ops.buffer_load(
                                        bias_rsrc, bias_offset, vec_width=1, dtype=f32
                                    )
                                )
                        tw_pf = None
                        if doweight_stage2:
                            tw_pf = []
                            lane_div_16_mul4_pf = lane_div_16 * arith.index(4)
                            ii_idx_list_pf = [
                                arith.constant(ii, index=True) for ii in range(4)
                            ]
                            for mi in range_constexpr(m_repeat):
                                mi_base_pf = arith.constant(mi * 16, index=True)
                                for ii in range_constexpr(4):
                                    row_off_pf = (
                                        lane_div_16_mul4_pf + ii_idx_list_pf[ii]
                                    )
                                    row_in_tile_pf = mi_base_pf + row_off_pf
                                    sorted_row_pf = bx_m + row_in_tile_pf
                                    tw_pf.append(
                                        buffer_ops.buffer_load(
                                            sorted_w_rsrc,
                                            sorted_row_pf,
                                            vec_width=1,
                                            dtype=f32,
                                        )
                                    )
                        epilogue_pf = (None, tw_pf, bias)

                    c0_i64 = arith.i64(0)
                    vec4_i64 = T.vec(4, T.i64)
                    vec8_i32 = T.vec(8, T.i32)

                    def pack_i64x4_to_i32x8(x0, x1, x2, x3):
                        v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
                        return vector.bitcast(vec8_i32, v4)

                    # fp4 path
                    for ku128 in range_constexpr(k_unroll_packed):
                        for mi in range_constexpr(m_repeat_packed):
                            for ni in range_constexpr(num_acc_n_packed):
                                b_scale_val = b_scale[ku128 * num_acc_n_packed + ni]
                                for ikxdl in range_constexpr(pack_K):
                                    k_idx = ku128 * pack_K + ikxdl

                                    b_packs0, b_packs1 = b_tile_in[k_idx]

                                    col_base = (
                                        col_offset_base
                                        + (k_idx * 128) // a_elem_vec_pack
                                    )
                                    for imxdl in range_constexpr(pack_M):
                                        col_base0 = col_base
                                        mi_idx = mi * pack_M + imxdl
                                        mi_val = arith.constant(mi_idx * 16, index=True)
                                        curr_row_a_lds = row_a_lds + mi_val

                                        if (
                                            (a0_prefetch is not None)
                                            and (k_idx == 0)
                                            and (mi_idx == 0)
                                        ):
                                            a0, a1 = a0_prefetch
                                        else:
                                            a0, a1 = lds_load_packs_k64(
                                                curr_row_a_lds, col_base0, lds_base
                                            )

                                        if is_f8_a:
                                            col_base1 = col_base + 64
                                            a2, a3 = lds_load_packs_k64(
                                                curr_row_a_lds, col_base1, lds_base
                                            )
                                            a128 = pack_i64x4_to_i32x8(a0, a1, a2, a3)
                                        else:
                                            a128 = pack_i64x4_to_i32x8(
                                                a0, a1, c0_i64, c0_i64
                                            )

                                        for inxdl in range_constexpr(pack_N):
                                            ni_idx = ni * pack_N + inxdl

                                            b0 = b_packs0[ni_idx]
                                            b1 = b_packs1[ni_idx]
                                            b128 = pack_i64x4_to_i32x8(
                                                b0, b1, c0_i64, c0_i64
                                            )

                                            acc_idx = mi_idx * num_acc_n + ni_idx
                                            rocdl.sched_barrier(0)
                                            acc_list[acc_idx] = (
                                                rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                                    mfma_res_ty,
                                                    [
                                                        a128,
                                                        b128,
                                                        acc_list[acc_idx],
                                                        cbsz,
                                                        blgp,
                                                        # as gemm1
                                                        0,
                                                        0x3F800000,
                                                        ikxdl * pack_N + inxdl,
                                                        b_scale_val,
                                                    ],
                                                )
                                            )

                    return acc_list, epilogue_pf

                # ---------------- 2-stage pipeline (ping-pong LDS + B tile prefetch) ----------------
                lds_tile_elems = arith.constant(tile_m * lds_stride, index=True)
                lds_base_cur = arith.index(0)
                lds_base_nxt = lds_tile_elems

                rocdl.sched_barrier(0)

                def hot_loop_scheduler():
                    # - MFMA group size per "slot": num_acc_n
                    # - Total MFMA per tile: (2*K32 per K64) * k_unroll * m_repeat * num_acc_n
                    # - We emit (mfma_group + dsrd + mfma_group) per scheduler iteration.
                    mfma_group = num_acc_n
                    mfma_total = (k_unroll * 2) * m_repeat * mfma_group
                    mfma_per_iter = 2 * mfma_group
                    sche_iters = (
                        0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
                    )

                    rocdl.sched_dsrd(2)
                    rocdl.sched_mfma(1)
                    if tile_m == 16:
                        rocdl.sched_vmem(1)
                    rocdl.sched_mfma(1)
                    if tile_m == 16:
                        rocdl.sched_vmem(1)
                    if num_acc_n < 4:
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(1)
                        if tile_m == 16:
                            rocdl.sched_vmem(1)
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(1)
                        if tile_m == 16:
                            rocdl.sched_vmem(1)
                        rocdl.sched_mfma(1)

                    # DS-write hints near the end: match total A LDS-store micro-ops per thread.
                    dswr_tail = num_x_loads
                    if dswr_tail > sche_iters:
                        dswr_tail = sche_iters
                    dswr_start = sche_iters - dswr_tail

                    for sche_i in range_constexpr(sche_iters):
                        rocdl.sched_vmem(1)
                        rocdl.sched_mfma(mfma_group)
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(mfma_group)
                        if sche_i >= dswr_start - 1:
                            rocdl.sched_dswr(1)

                    rocdl.sched_barrier(0)

                # Prologue.
                k0 = arith.index(0)
                x_regs0 = load_x_tile(k0)
                b_cur = load_b_tile(k0)
                a_scale_ping, a_scale_pong = None, None
                _, b_scale_pong = prefetch_ab_scale_tile(k0 // pack_K // 128)
                store_x_tile_to_lds(x_regs0, lds_base_cur)
                gpu.barrier()

                acc = [acc_init] * num_acc_n * m_repeat
                lds_base_pong = lds_base_cur
                lds_base_ping = lds_base_nxt

                # Cross-tile A0 LDS prefetch (default-on): prefetch the first A-pack (K64) for the
                # tile we are about to compute from LDS, to overlap with upcoming VMEM.
                a0_prefetch_pong = lds_load_packs_k64(
                    row_a_lds, col_offset_base, lds_base_pong
                )
                # a0_prefetch_pong = lds_load_packs_k64(0, 0, lds_base_pong)

                # Main loop: process K tiles in 2-tile ping-pong steps.
                #
                # IMPORTANT: for odd number of K tiles, leave **1** tail tile; for even, leave **2**.
                # Otherwise the 2-tile tail below would double-count the last tile when num_tiles is odd
                # (e.g. inter_dim=192, tile_k=64 -> 3 tiles).
                num_k_tiles_py = int(inter_dim) // int(tile_k)
                odd_k_tiles = (num_k_tiles_py % 2) == 1
                tail_tiles = 1 if odd_k_tiles else 2
                k_main2_py = (num_k_tiles_py - tail_tiles) * int(tile_k)
                if k_main2_py < 0:
                    k_main2_py = 0

                c2_tile_k = arith.constant(tile_k * 2, index=True)
                c_k_main2 = arith.index(k_main2_py)
                b_pong = b_cur
                for k_iv in range(arith.index(0), c_k_main2, arith.index(tile_k * 2)):
                    next_k1 = k_iv + tile_k
                    x_regs_ping = load_x_tile(next_k1)
                    b_ping = load_b_tile(next_k1 // 2)
                    _, b_scale_ping = prefetch_ab_scale_tile(next_k1 // pack_K // 128)

                    acc, _ = compute_tile(
                        acc,
                        b_pong,
                        lds_base_pong,
                        a_scale_pong,
                        b_scale_pong,
                        a0_prefetch=a0_prefetch_pong,
                    )
                    store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                    # hot_loop_scheduler()
                    gpu.barrier()

                    # Cross-tile prefetch for the ping tile we are about to compute.
                    a0_prefetch_ping = lds_load_packs_k64(
                        row_a_lds,
                        col_offset_base_bytes if not is_f4_b else col_offset_base,
                        lds_base_ping,
                    )

                    next_k2 = k_iv + c2_tile_k
                    x_regs_pong = load_x_tile(next_k2)
                    b_pong = load_b_tile(next_k2 // 2)
                    _, b_scale_pong = prefetch_ab_scale_tile(next_k2 // pack_K // 128)

                    acc, _ = compute_tile(
                        acc,
                        b_ping,
                        lds_base_ping,
                        a_scale_ping,
                        b_scale_ping,
                        a0_prefetch=a0_prefetch_ping,
                    )
                    store_x_tile_to_lds(x_regs_pong, lds_base_pong)
                    # hot_loop_scheduler()
                    gpu.barrier()

                    # Cross-tile prefetch for the next pong tile.
                    a0_prefetch_pong = lds_load_packs_k64(
                        row_a_lds,
                        col_offset_base_bytes if not is_f4_b else col_offset_base,
                        lds_base_pong,
                    )

                if odd_k_tiles:
                    # Tail: single remaining tile (already in `b_cur` / `lds_base_pong`).
                    acc, epilogue_pf = compute_tile(
                        acc,
                        b_pong,
                        lds_base_pong,
                        a_scale_pong,
                        b_scale_pong,
                        a0_prefetch=a0_prefetch_pong,
                        prefetch_epilogue=True,
                    )

                else:
                    # Tail: 2 remaining tiles.
                    k_tail1 = (k_in + tile_k - 1) // tile_k * tile_k - tile_k
                    x_regs_ping = load_x_tile(k_tail1)
                    b_ping = load_b_tile(k_tail1 // 2)
                    _, b_scale_ping = prefetch_ab_scale_tile(k_tail1 // pack_K // 128)

                    acc, _ = compute_tile(
                        acc,
                        b_pong,
                        lds_base_pong,
                        a_scale_pong,
                        b_scale_pong,
                        a0_prefetch=a0_prefetch_pong,
                    )

                    store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                    # hot_loop_scheduler()
                    gpu.barrier()

                    # Epilogue tile with sw prefetch.
                    a0_prefetch_ping = lds_load_packs_k64(
                        row_a_lds,
                        col_offset_base_bytes if not is_f4_b else col_offset_base,
                        lds_base_ping,
                    )
                    acc, epilogue_pf = compute_tile(
                        acc,
                        b_ping,
                        lds_base_ping,
                        a_scale_ping,
                        b_scale_ping,
                        a0_prefetch=a0_prefetch_ping,
                        prefetch_epilogue=True,
                    )

                # ---------------- Epilogue: LDS CShuffle + atomic half2 (x2) ----------------
                # Reuse the shared helper so GEMM / MoE kernels share the exact same CShuffle skeleton.

                sw_pf = None
                tw_pf = None
                bias_pf = None
                if epilogue_pf is not None:
                    sw_pf, tw_pf, bias_pf = epilogue_pf

                expert_off = expert_off_idx
                mask24_i32 = arith.i32(0xFFFFFF)
                model_i32 = arith.i32(model_dim)
                topk_i32_v = topk_i32

                zero_i32 = arith.i32(0)
                c2_i32 = arith.i32(2)  # 2B element size for f16/bf16
                mask_even_i32 = arith.i32(
                    0xFFFFFFFE
                )  # align element index to even for half2 atomics

                def atomic_add_f16x2(val_f16x2, byte_off_i32):
                    rocdl.raw_ptr_buffer_atomic_fadd(
                        val_f16x2,
                        out_rsrc,
                        byte_off_i32,
                        zero_i32,
                        zero_i32,
                    )

                # Weight scales for the N tile (col_g depends on lane/wave/by but not on (t,s)).
                if lds_out is None:
                    raise RuntimeError(
                        "FLIR_MOE_STAGE2_CSHUFFLE=1 but lds_out is not allocated/aliased."
                    )

                # For bf16 global atomics, precompute the output base address once.
                # (We still need an inttoptr per atomic unless we can rely on GEPOp; keep the
                # stable path here.)
                out_base_idx = None
                if out_is_bf16:
                    out_base_idx = memref.extract_aligned_pointer_as_index(arg_out)

                def write_row_to_lds(
                    *,
                    mi: int,
                    ii: int,
                    row_in_tile,
                    row,
                    row_base_lds,
                    col_base_local,
                    num_acc_n: int,
                    lds_out,
                ):
                    # Match origin/dev_a16w4: rely on sentinel padded rows + hardware OOB behavior.
                    fused2 = buffer_ops.buffer_load(
                        sorted_rsrc, row, vec_width=1, dtype=i32
                    )
                    t2 = fused2 & mask24_i32
                    s2 = fused2 >> 24

                    t_ok = arith.cmpu(t2, tokens_i32, "ult")
                    s_ok = arith.cmpu(s2, topk_i32_v, "ult")
                    ts_ok = t_ok & s_ok
                    t2_safe = arith.select(ts_ok, t2, arith.i32(0))
                    s2_safe = arith.select(ts_ok, s2, arith.i32(0))
                    ts2 = t2_safe * topk_i32_v + s2_safe

                    if doweight_stage2:
                        tw_idx = (mi * 4) + ii
                        if tw_pf is not None:
                            tw = tw_pf[tw_idx]
                        else:
                            tw = buffer_ops.buffer_load(
                                sorted_w_rsrc, row, vec_width=1, dtype=f32
                            )

                    for ni in range_constexpr(num_acc_n):
                        col_local = col_base_local + (ni * 16)
                        acc_idx = mi * num_acc_n + ni
                        v = vector.extract(
                            acc[acc_idx], static_position=[ii], dynamic_position=[]
                        )
                        if is_int8:
                            v = arith.sitofp(f32, v)
                        if enable_bias:
                            v = v + bias_pf[ni]

                        if doweight_stage2:
                            v = v * tw
                        v_out = arith.trunc_f(out_elem(), v)

                        lds_idx = row_base_lds + col_local
                        vec1_out = T.vec(1, out_elem())
                        v1 = vector.from_elements(vec1_out, [v_out])

                        vector.store(v1, lds_out, [lds_idx], alignment=2)

                def precompute_row(*, row_local, row):
                    # Precompute row context for cshuffle stores.
                    # Return (fused_i32, row_valid_i1) so the epilogue can skip the entire row
                    # for invalid tail rows (CK-style), avoiding per-store branching.
                    fused2 = buffer_ops.buffer_load(
                        sorted_rsrc, row, vec_width=1, dtype=i32
                    )

                    row_i32 = arith.index_cast(i32, row)
                    row_valid0 = arith.cmpu(row_i32, num_valid_i32, "ult")
                    t = fused2 & mask24_i32
                    s = fused2 >> 24
                    t_ok = arith.cmpu(t, tokens_i32, "ult")
                    s_ok = arith.cmpu(s, topk_i32_v, "ult")
                    row_valid = row_valid0 & t_ok & s_ok

                    return (fused2, row_valid)

                def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                    fused = row_ctx
                    t = fused & mask24_i32
                    s = fused >> 24
                    idx0 = t * model_i32
                    if not bool(accumulate):
                        ts = t * topk_i32_v + s
                        idx0 = ts * model_i32
                    col_i32 = arith.index_cast(i32, col_g0)
                    idx_elem = idx0 + col_i32
                    idx_elem_even = idx_elem & mask_even_i32
                    vv1 = vector.extract(frag, static_position=[0], dynamic_position=[])
                    vv2 = vector.extract(frag, static_position=[0], dynamic_position=[])
                    if out_is_bf16:
                        if bool(accumulate):
                            # Use global atomicrmw fadd on <2 x bf16> (CK path).
                            # Row-valid gating is handled at the row level by c_shuffle_epilog via `precompute_row`.
                            byte_off = idx_elem_even * c2_i32
                            byte_off_idx = arith.index_cast(
                                ir.IndexType.get(), byte_off
                            )
                            ptr_addr_idx = out_base_idx + byte_off_idx
                            out_ptr = buffer_ops.create_llvm_ptr(
                                ptr_addr_idx, address_space=1
                            )
                            out_ptr_v = (
                                out_ptr._value
                                if hasattr(out_ptr, "_value")
                                else out_ptr
                            )
                            frag_v = frag._value if hasattr(frag, "_value") else frag
                            llvm.AtomicRMWOp(
                                llvm.AtomicBinOp.fadd,
                                out_ptr_v,
                                frag_v,
                                llvm.AtomicOrdering.monotonic,
                                syncscope="agent",
                                alignment=4,
                            )
                        else:
                            # Scatter store (no atomic): store bf16x2 directly via buffer store.
                            buffer_ops.buffer_store(frag, out_rsrc, idx_elem_even)
                    else:
                        byte_off = idx_elem_even * c2_i32
                        if bool(accumulate):
                            atomic_add_f16x2(frag, byte_off)
                        else:
                            buffer_ops.buffer_store(frag, out_rsrc, idx_elem_even)

                c_shuffle_epilog(
                    arith=arith,
                    vector=vector,
                    gpu=gpu,
                    scf=scf,
                    range_constexpr=range_constexpr,
                    tile_m=tile_m,
                    tile_n=tile_n,
                    e_vec=2,
                    m_repeat=m_repeat,
                    num_acc_n=num_acc_n,
                    tx=tx,
                    lane_div_16=lane_div_16,
                    lane_mod_16=lane_mod_16,
                    bx_m=bx_m,
                    by_n=by_n,
                    n_tile_base=n_tile_base,
                    lds_out=lds_out,
                    frag_elem_type=(T.bf16 if out_is_bf16 else T.f16),
                    write_row_to_lds=write_row_to_lds,
                    precompute_row=precompute_row,
                    store_pair=store_pair,
                )

            _if_blk = scf.IfOp(blk_valid)
            with _if_then(_if_blk):
                _ifexpert_of = scf.IfOp(exp_valid)
                with _if_then(_ifexpert_of):
                    _moe_gemm2_then_body()

    # -- Host launcher (flyc.jit + .launch) --------------------------------
    _cache_tag = (
        module_name,
        a_dtype,
        b_dtype,
        out_dtype,
        tile_m,
        tile_n,
        tile_k,
        doweight_stage2,
        accumulate,
        enable_bias,
        model_dim_pad,
        inter_dim_pad,
        use_cshuffle_epilog,
    )

    @flyc.jit
    def launch_mixed_moe_gemm2(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        arg_bias: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream_ptr: int,
    ):
        _ = _cache_tag
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        n_in = arith.ArithValue(
            arith.index_cast(ir.IndexType.get(), i32_n_in.ir_value())
        )
        gx = n_in // arith.constant(tile_n, index=True)
        gy = arith.index_cast(ir.IndexType.get(), i32_size_expert_ids_in.ir_value())
        moe_gemm2(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_num_valid_ids,
            arg_bias,
            i32_tokens_in,
            i32_n_in,
            i32_k_in,
            i32_size_expert_ids_in,
        ).launch(
            grid=(gx, gy, 1),
            block=(256, 1, 1),
            stream=stream_ptr,
        )

    return launch_mixed_moe_gemm2
