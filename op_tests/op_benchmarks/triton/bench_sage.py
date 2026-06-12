from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import argparse
import glob
import json
import logging
import os
import re
import sys
import tempfile
import time

import torch
import triton

import aiter
from aiter.ops.mha import flash_attn_func, flash_attn_fp8_pertensor_func

from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_3
from aiter.ops.triton.attention.mha_v3 import _quantize_bshd
from aiter.ops.triton.attention.fav3_sage import (
    build_attention_lut,
    fav3_sage_func,
    fav3_sage_wrapper_func,
    fav3_sage_vfa_wrapper_func,
    get_sage_fwd_configs,
)
from aiter.ops.triton.attention.fav3_sage_attention_mxfp4_wrapper import (
    fav3_sage_mxfp4_func,
    fav3_sage_mxfp4_wrapper,
    get_sage_fwd_configs_mxfp4,
)
from aiter.ops.triton.attention.block_sparse import (
    block_attn_mask_to_ragged_lut,
    compute_m_proxy_topn,
)
from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
    create_hadamard_matrix,
    sage_quant,
    sage_quant_mxfp4,
)
from aiter.test_mha_common import attention_ref, attention_ref_block_sparse

from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
)
from op_tests.triton_tests.attention.test_fav3_sage import (
    check_attention_outputs,
    compare_accuracy,
)

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


KernelName = Literal[
    "sage_fp8",
    "sage_fp8_vfa",
    "sage_mxfp4",
    "fav3_fp8",
    "aiter_fp8",
    "aiter_bf16",
]

ALL_KERNELS: List[str] = [
    "sage_fp8",
    "sage_fp8_vfa",
    "sage_mxfp4",
    "fav3_fp8",
    "aiter_fp8",
    "aiter_bf16",
]

# Kernels that consume fp8/mxfp4 inputs (1-byte Q/K) rather than the raw dtype.
QUANTIZED_KERNELS = ("sage_fp8", "sage_fp8_vfa", "sage_mxfp4", "fav3_fp8", "aiter_fp8")
# Kernels whose V is also 1 byte.
FP8_V_KERNELS = ("fav3_fp8", "aiter_fp8")
SAGE_FP8_KERNELS = ("sage_fp8", "sage_fp8_vfa")


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass
class ShapeSpec:
    batch: int
    hq: int
    hk: int
    n_ctx_q: int
    n_ctx_k: int
    d_head: int
    d_head_v: int


def dense_flops(shape: ShapeSpec) -> float:
    """Total dense attention FLOPs (QK^T + PV) for the given shape."""
    return (
        2.0
        * shape.batch
        * shape.hq
        * shape.n_ctx_q
        * shape.n_ctx_k
        * (shape.d_head + shape.d_head_v)
    )


@dataclass
class LoadedMask:
    mask: torch.Tensor
    batch: int
    num_q_blocks: int
    num_kv_blocks: int


def layout_preprocess(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: Literal["bshd", "bhsd"],
    target_layout: Literal["bshd", "bhsd"] = "bshd",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if layout != target_layout:
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
    return q, k, v


def primary_output(result: Any) -> Any:
    if isinstance(result, torch.Tensor):
        return result
    if isinstance(result, (tuple, list)) and len(result) > 0:
        return result[0]
    return result


def infer_shape_spec(
    q: torch.Tensor,
    v: torch.Tensor,
    layout: Literal["bshd", "bhsd"],
) -> ShapeSpec:
    if layout == "bshd":
        batch, n_ctx_q, hq, d_head = q.shape
        _, n_ctx_k, hk, d_head_v = v.shape
    else:
        batch, hq, n_ctx_q, d_head = q.shape
        _, hk, n_ctx_k, d_head_v = v.shape
    return ShapeSpec(
        batch=batch,
        hq=hq,
        hk=hk,
        n_ctx_q=n_ctx_q,
        n_ctx_k=n_ctx_k,
        d_head=d_head,
        d_head_v=d_head_v,
    )


def _array_ndim(arr: Any) -> int:
    if not isinstance(arr, list):
        return 0
    if not arr:
        return 1
    return 1 + _array_ndim(arr[0])


def _mask_array_to_tensor(
    mask_arr: List[Any],
    device: torch.device,
) -> LoadedMask:
    if not mask_arr:
        raise ValueError("mask array is empty")

    depth = _array_ndim(mask_arr)
    if depth == 2:
        mask = torch.tensor(mask_arr, dtype=torch.bool, device=device)
        num_q_blocks, num_kv_blocks = mask.shape
        mask = mask.unsqueeze(0)
        return LoadedMask(mask, 1, num_q_blocks, num_kv_blocks)

    if depth == 3:
        mask = torch.tensor(mask_arr, dtype=torch.bool, device=device)
        batch, num_q_blocks, num_kv_blocks = mask.shape
        return LoadedMask(mask, batch, num_q_blocks, num_kv_blocks)

    raise ValueError(f"mask must be 2D or 3D, got {depth}D")


def _mask_from_json_item(item: Dict[str, Any], device: torch.device) -> LoadedMask:
    """Build a LoadedMask from a JSON object, validating any declared block counts."""
    if "mask" not in item:
        raise ValueError("Each mask entry must include key 'mask'")
    m = _mask_array_to_tensor(item["mask"], device)
    for key, inferred in (
        ("num_q_blocks", m.num_q_blocks),
        ("num_kv_blocks", m.num_kv_blocks),
    ):
        if key in item and item[key] != inferred:
            raise ValueError(f"{key} mismatch: inferred {inferred}, got {item[key]}")
    return m


def load_block_mask_from_json(
    path: Optional[str],
    device: torch.device,
) -> Optional[Union[LoadedMask, List[LoadedMask]]]:
    if not path or not path.strip():
        return None

    path = path.strip()
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Block mask file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    if not data:
        return None
    if "masks" in data:
        return [_mask_from_json_item(item, device) for item in data["masks"]]
    if "mask" in data:
        return _mask_from_json_item(data, device)
    return None


def kernel_block_sizes(kernel: KernelName) -> Tuple[int, int]:
    if kernel == "sage_mxfp4":
        cfg = get_sage_fwd_configs_mxfp4()
    else:
        cfg = get_sage_fwd_configs()
    return cfg["BLOCK_M"], cfg["BLOCK_N"]


def use_vfa(args: argparse.Namespace) -> bool:
    """Whether this run exercises the VFA running-max freeze (the sage_fp8_vfa kernel)."""
    return args.kernel == "sage_fp8_vfa"


def builds_qk_lut(args: argparse.Namespace) -> bool:
    """Whether this run builds a Q/K-derived block LUT via ``build_attention_lut``.

    True iff ``--sparge`` is set: plain SpargeAttn block selection, or (for
    ``sage_fp8_vfa``) SpargeAttn selection plus the VFA running-max freeze. The
    standalone dense-VFA build was removed, so VFA always rides on a sparge LUT
    (or a static mask, handled separately in ``run_benchmark``).
    """
    return bool(args.sparge)


def _ragged_lut_to_block_mask(
    block_lut: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch: int,
    hq: int,
    num_q_blocks: int,
    num_kv_blocks: int,
    device: torch.device,
) -> torch.Tensor:
    """Reconstruct the dense (B, HQ, num_q_blocks, num_kv_blocks) bool mask a LUT encodes."""
    kv_block_indices, _, lut_count = block_lut
    num_seg = batch * hq * num_q_blocks
    total = int(lut_count.sum().item())
    mask = torch.zeros(num_seg, num_kv_blocks, dtype=torch.bool, device=device)
    if total > 0:
        seg = torch.repeat_interleave(
            torch.arange(num_seg, device=device), lut_count.to(torch.long)
        )
        mask[seg, kv_block_indices[:total].to(torch.long)] = True
    return mask.view(batch, hq, num_q_blocks, num_kv_blocks)


def build_attn_mode_lut(
    args: argparse.Namespace,
    q_bhsd: torch.Tensor,
    k_bhsd: torch.Tensor,
    device: torch.device,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], int, torch.Tensor]:
    """Build a Sparge (+ optional VFA) ragged LUT (+ freeze count) from bhsd Q/K.

    Plain ``--sparge`` builds a SpargeAttn-selected LUT; ``sage_fp8_vfa``
    additionally front-loads the top-``n_sample`` blocks for the VFA running-max
    freeze (``use_vfa``). Returns ``(block_lut, freeze_softmax_max_count,
    block_attn_mask)``; the mask is reconstructed from the LUT so the
    ``--compare-to-ref`` block-sparse reference path keeps working.
    """
    batch, hq, n_ctx_q, _ = q_bhsd.shape
    n_ctx_k = k_bhsd.shape[2]
    block_m, block_n = kernel_block_sizes(args.kernel)

    simthreshd1 = torch.full(
        (hq,), args.simthreshd1, device=device, dtype=torch.float32
    )
    cdfthreshd = torch.full((hq,), args.cdfthreshd, device=device, dtype=torch.float32)

    block_lut, freeze = build_attention_lut(
        q_bhsd,
        k_bhsd,
        simthreshd1=simthreshd1,
        cdfthreshd=cdfthreshd,
        use_vfa=use_vfa(args),
        n_sample=args.n_sample,
        is_causal=False,
        block_m=block_m,
        block_n=block_n,
    )

    num_q_blocks = (n_ctx_q + block_m - 1) // block_m
    num_kv_blocks = (n_ctx_k + block_n - 1) // block_n
    block_attn_mask = _ragged_lut_to_block_mask(
        block_lut, batch, hq, num_q_blocks, num_kv_blocks, device
    )
    return block_lut, freeze, block_attn_mask


def maybe_expand_mask(
    mask: LoadedMask,
    batch: int,
    hq: int,
) -> torch.Tensor:
    out = mask.mask
    if mask.batch != batch:
        if mask.batch == 1:
            out = out.expand(batch, -1, -1).clone()
        else:
            raise ValueError(
                f"Mask batch ({mask.batch}) does not match benchmark batch ({batch})"
            )

    if out.dim() == 3:
        out = out.unsqueeze(1).expand(batch, hq, mask.num_q_blocks, mask.num_kv_blocks)
    return out.clone()


def build_block_mask(
    args: argparse.Namespace,
    shape: ShapeSpec,
    loaded_single_mask: Optional[LoadedMask],
) -> Optional[torch.Tensor]:
    """Return the block mask loaded from ``--block-mask-file`` (or ``None``).

    Validates that the file mask's block grid matches the benchmark shape and
    broadcasts it to ``(batch, hq, ...)``.
    """
    if loaded_single_mask is not None:
        block_m, block_n = kernel_block_sizes(args.kernel)
        expected_q_blocks = (shape.n_ctx_q + block_m - 1) // block_m
        expected_kv_blocks = (shape.n_ctx_k + block_n - 1) // block_n

        if loaded_single_mask.num_q_blocks != expected_q_blocks:
            raise ValueError(
                f"Mask q blocks mismatch: expected {expected_q_blocks}, got {loaded_single_mask.num_q_blocks}"
            )
        if loaded_single_mask.num_kv_blocks != expected_kv_blocks:
            raise ValueError(
                f"Mask kv blocks mismatch: expected {expected_kv_blocks}, got {loaded_single_mask.num_kv_blocks}"
            )

        return maybe_expand_mask(loaded_single_mask, shape.batch, shape.hq)

    return None


def sparse_flops_from_lut(
    kernel: KernelName,
    block_lut: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    shape: ShapeSpec,
) -> Tuple[float, float]:
    _, _, lut_count = block_lut
    num_sparse_pairs = lut_count.sum().item()

    block_m, block_n = kernel_block_sizes(kernel)
    num_q_blocks = ceil_div(shape.n_ctx_q, block_m)
    num_kv_blocks = ceil_div(shape.n_ctx_k, block_n)
    num_dense_pairs = shape.batch * shape.hq * num_q_blocks * num_kv_blocks

    total_dense_flops = dense_flops(shape)
    if num_dense_pairs == 0:
        return 0.0, total_dense_flops
    return total_dense_flops * (num_sparse_pairs / num_dense_pairs), total_dense_flops


def fp8_quantize(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    quant_dtype = aiter.dtypes.fp8
    dtype_max = torch.finfo(quant_dtype).max

    def _quant(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return aiter.per_tensor_quant(
            x,
            scale=torch.abs(x).max() if scale is None else scale,
            quant_dtype=quant_dtype,
            dtypeMax=dtype_max,
        )

    q_quant, q_descale = _quant(q)
    k_quant, k_descale = _quant(k)
    v_quant, v_descale = _quant(v)
    return q_quant, k_quant, v_quant, q_descale, k_descale, v_descale


def _unpack_block_lut(
    block_lut: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[
    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], bool
]:
    """Unpack block LUT into (kv_block_indices, lut_start, lut_count, use_block_sparse)."""
    if block_lut is not None:
        kv_block_indices, lut_start, lut_count = block_lut
        return kv_block_indices, lut_start, lut_count, True
    return None, None, None, False


def _call_flash_attn_3(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    v_fp8: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> Any:
    """Thin wrapper around flash_attn_3.fwd with default args for unused features."""
    return flash_attn_3.fwd(
        q_fp8,
        k_fp8,
        v_fp8,
        None,
        None,
        None,
        None,
        None,
        None,
        None,  # out, alibi_slopes, etc.
        None,
        None,
        None,
        None,
        None,
        None,
        None,  # unused optional tensors
        None,
        None,
        None,  # rng states, padding
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
        -1,
        -1,  # window_size
        0,
        0.0,
        False,  # attention_chunk, softcap, deterministic
        None,
        1,
        None,  # descale_out, sm_margin, seqused_k
        0,  # num_splits
    )


def make_fav3_fp8_runner(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float],
    causal: bool,
    e2e: bool = False,
) -> Any:
    batch, _, num_q_heads, head_dim = q.shape
    _, _, num_kv_heads, _ = k.shape

    fp8_dtype = aiter.dtypes.fp8
    group_size = num_q_heads // num_kv_heads if num_q_heads != num_kv_heads else None

    if softmax_scale is None:
        softmax_scale = head_dim**-0.5

    def _quantize():
        q_fp8, q_ds = _quantize_bshd(q, fp8_dtype, group_size=group_size)
        k_fp8, k_ds = _quantize_bshd(k, fp8_dtype)
        v_fp8, v_ds = _quantize_bshd(v, fp8_dtype)
        return q_fp8, k_fp8, v_fp8, q_ds, k_ds, v_ds

    if e2e:
        return lambda: _call_flash_attn_3(*_quantize(), softmax_scale, causal)

    q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale = _quantize()

    assert q_descale.shape == (batch, num_kv_heads)
    assert k_descale.shape == (batch, num_kv_heads)
    assert v_descale.shape == (batch, num_kv_heads)

    return lambda: _call_flash_attn_3(
        q_fp8,
        k_fp8,
        v_fp8,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
    )


def make_torch_ref_runner(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
) -> Any:
    return lambda: attention_ref(
        q, k, v, dropout_p=0.0, dropout_mask=None, causal=causal
    )


def make_kernel_runner(
    args: argparse.Namespace,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_lut: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    freeze_softmax_max_count: int = -1,
) -> Any:
    q_bshd, k_bshd, v_bshd = layout_preprocess(
        q, k, v, layout=args.layout, target_layout="bshd"
    )
    head_dim = q_bshd.shape[-1]
    softmax_scale = head_dim**-0.5

    # sage_fp8_vfa without a block LUT (no --sparge / mask): dense VFA. In --e2e
    # mode the full wrapper (quant + m_init + kernel) is timed; the non-e2e
    # kernel-only path is handled inside the SAGE_FP8_KERNELS branch below
    # (quant + m_init precomputed outside the timed loop, like plain sage_fp8).
    dense_vfa = args.kernel == "sage_fp8_vfa" and block_lut is None
    if dense_vfa and args.e2e:
        return lambda: fav3_sage_vfa_wrapper_func(
            q,
            k,
            v,
            softmax_scale,
            return_lse=False,
            layout=args.layout,
            n_sample_blocks=args.n_sample,
            block_lut=None,
        )

    # sage_fp8 (and sage_fp8_vfa with a LUT) run the unified kernel driven by a
    # build_attention_lut LUT plus the matching freeze_softmax_max_count
    # (see build_attn_mode_lut / use_vfa).
    if args.kernel in SAGE_FP8_KERNELS:
        if args.e2e:
            return lambda: fav3_sage_wrapper_func(
                q,
                k,
                v,
                softmax_scale,
                causal=args.causal,
                return_lse=False,
                layout=args.layout,
                block_lut=block_lut,
                freeze_softmax_max_count=freeze_softmax_max_count,
            )

        cfg = get_sage_fwd_configs()
        fp8_type = aiter.dtypes.fp8
        fp8_max = torch.finfo(fp8_type).max

        q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale = sage_quant(
            q,
            k,
            v,
            fp8_type,
            fp8_max,
            BLKQ=cfg["BLOCK_M"],
            BLKK=cfg["BLOCK_N"],
            sm_scale=softmax_scale,
            layout=args.layout,
        )

        # Dense VFA, non-e2e: the m_init running-max estimate is preparation
        # (like quantization), so precompute it outside the timed loop so the
        # measurement isolates the VFA kernel speedup, matching plain sage_fp8.
        m_init = None
        if dense_vfa:
            m_init = compute_m_proxy_topn(
                q, k, q_int8, k_int8, q_scale, k_scale,
                BLKQ=cfg["BLOCK_M"], BLKK=cfg["BLOCK_N"],
                layout=args.layout, n_blocks=args.n_sample,
            )

        kv_idx, lut_s, lut_c, sparse = _unpack_block_lut(block_lut)
        sparge_load_balancing = sparse and getattr(args, "sparge_load_balancing", False)
        return lambda: fav3_sage_func(
            q_int8,
            k_int8,
            v_fp8,
            q_scale,
            k_scale,
            v_scale,
            softmax_scale=softmax_scale,
            causal=args.causal,
            return_lse=False,
            layout=args.layout,
            config=cfg,
            kv_block_indices=kv_idx,
            lut_start=lut_s,
            lut_count=lut_c,
            use_block_sparse=sparse,
            freeze_softmax_max_count=freeze_softmax_max_count,
            sparge_load_balancing=sparge_load_balancing,
            m_init=m_init,
        )

    if args.kernel == "sage_mxfp4":
        block_r = args.block_r
        if block_r > q.shape[-1]:
            raise ValueError(f"block_r ({block_r}) must be <= head dim ({q.shape[-1]})")

        r = create_hadamard_matrix(block_r, device=q.device, dtype=q.dtype) / (
            block_r**0.5
        )

        if args.e2e:
            return lambda: fav3_sage_mxfp4_wrapper(
                q,
                k,
                v,
                causal=args.causal,
                layout=args.layout,
                q_smooth=args.qsmooth,
                hadamard_rotation=args.hadamard_rotate,
                R=r,
                block_lut=block_lut,
            )

        cfg = get_sage_fwd_configs_mxfp4()
        fp8_type = aiter.dtypes.fp8
        fp8_max = torch.finfo(fp8_type).max

        (
            q_quant,
            q_descale,
            k_quant,
            k_descale,
            v_quant,
            v_descale,
            delta_s,
        ) = sage_quant_mxfp4(
            q,
            k,
            v,
            fp8_type,
            fp8_max,
            BLKQ=cfg["BLOCK_M"],
            BLKK=64,
            layout=args.layout,
            R=r,
            BLOCK_R=block_r,
            q_smoothing=args.qsmooth,
        )

        kv_idx, lut_s, lut_c, sparse = _unpack_block_lut(block_lut)
        return lambda: fav3_sage_mxfp4_func(
            q=q_quant,
            k=k_quant,
            v=v_quant,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            bias=delta_s,
            causal=args.causal,
            layout=args.layout,
            config=cfg,
            kv_block_indices=kv_idx,
            lut_start=lut_s,
            lut_count=lut_c,
            use_block_sparse=sparse,
        )

    if args.kernel == "aiter_bf16":
        return lambda: flash_attn_func(
            q_bshd,
            k_bshd,
            v_bshd,
            dropout_p=0.0,
            causal=args.causal,
            return_attn_probs=False,
        )

    if args.kernel == "aiter_fp8":

        def _run_aiter_fp8():
            q_fp8, k_fp8, v_fp8, q_ds, k_ds, v_ds = fp8_quantize(q_bshd, k_bshd, v_bshd)
            return flash_attn_fp8_pertensor_func(
                q_fp8,
                k_fp8,
                v_fp8,
                q_descale=q_ds,
                k_descale=k_ds,
                v_descale=v_ds,
            )

        if args.e2e:
            return _run_aiter_fp8

        q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale = fp8_quantize(
            q_bshd, k_bshd, v_bshd
        )
        return lambda: flash_attn_fp8_pertensor_func(
            q_fp8,
            k_fp8,
            v_fp8,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    if args.kernel == "fav3_fp8":
        return make_fav3_fp8_runner(
            q_bshd,
            k_bshd,
            v_bshd,
            softmax_scale=softmax_scale,
            causal=args.causal,
            e2e=args.e2e,
        )

    raise ValueError(f"Unsupported kernel: {args.kernel}")


def to_bshd_output_if_needed(
    out: torch.Tensor,
    layout: Literal["bshd", "bhsd"],
) -> torch.Tensor:
    if layout == "bhsd":
        return out.permute(0, 2, 1, 3).contiguous()
    return out


def make_reference_output(
    args: argparse.Namespace,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_attn_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    q_bshd, k_bshd, v_bshd = layout_preprocess(
        q, k, v, layout=args.layout, target_layout="bshd"
    )
    ref = args.ref or "torch"

    if ref == "torch":
        block_m, block_n = kernel_block_sizes(args.kernel)
        ref_out = attention_ref_block_sparse(
            q_bshd,
            k_bshd,
            v_bshd,
            block_attn_mask,
            block_m,
            block_n,
            dropout_p=0.0,
            dropout_mask=None,
            upcast=True,
        )
        return primary_output(ref_out)

    if ref == "aiter_bf16":
        return primary_output(
            flash_attn_func(
                q_bshd,
                k_bshd,
                v_bshd,
                dropout_p=0.0,
                causal=args.causal,
                return_attn_probs=False,
            )
        )

    return primary_output(make_torch_ref_runner(q_bshd, k_bshd, v_bshd, args.causal)())


def compute_memory_bytes(
    shape: ShapeSpec,
    q_element_size: int,
    k_element_size: int,
    v_element_size: int,
) -> float:
    total_num_tokens_q = shape.batch * shape.n_ctx_q
    total_num_tokens_k = shape.batch * shape.n_ctx_k

    q_size = total_num_tokens_q * shape.hq * shape.d_head * q_element_size
    k_size = total_num_tokens_k * shape.hk * shape.d_head * k_element_size
    v_size = total_num_tokens_k * shape.hk * shape.d_head_v * v_element_size
    o_size = total_num_tokens_q * shape.hq * shape.d_head_v * q_element_size
    return q_size + k_size + v_size + o_size


def benchmark_single_case(
    args: argparse.Namespace,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    provider: str,
    loaded_single_mask: Optional[LoadedMask],
    explicit_block_attn_mask: Optional[torch.Tensor] = None,
) -> float:
    shape = infer_shape_spec(q, v, args.layout)
    freeze_softmax_max_count = -1

    # Block-mask precedence (highest first):
    #   1. an explicit list-mode mask, 2. a --block-mask-file mask,
    #   3. a random --block-sparsity mask, 4. the Q/K-derived --sparge LUT.
    # A static mask (1-3) is used directly and wins over the --sparge build.
    file_mask = (
        explicit_block_attn_mask
        if explicit_block_attn_mask is not None
        else build_block_mask(args, shape, loaded_single_mask)
    )
    if file_mask is None and args.block_sparsity is not None:
        device = q.device
        block_m, block_n = kernel_block_sizes(args.kernel)
        num_q_blocks = (shape.n_ctx_q + block_m - 1) // block_m
        num_kv_blocks = (shape.n_ctx_k + block_n - 1) // block_n
        warmup_mask = (
            torch.rand(
                shape.batch, shape.hq, num_q_blocks, num_kv_blocks, device=device
            )
            > args.block_sparsity
        ).to(torch.bool)
        file_mask = warmup_mask

    if file_mask is not None:
        block_attn_mask = file_mask
        block_lut = block_attn_mask_to_ragged_lut(
            block_attn_mask, return_none_if_dense=True
        )
        # sage_fp8_vfa on a static (random) mask: don't reorder the LUT to
        # front-load the top-n blocks (no build_attention_lut). Keep the LUT in
        # ascending block order and just run the kernel with the n_sample freeze
        # count so the VFA running-max-freeze path is still exercised.
        if use_vfa(args) and block_lut is not None:
            freeze_softmax_max_count = args.n_sample
    elif builds_qk_lut(args):
        # Sparge (+ optional VFA): build the ragged LUT (+ freeze count) from the
        # bhsd Q/K via build_attention_lut, replacing the mask-driven LUT path.
        q_bhsd, k_bhsd, _ = layout_preprocess(
            q, k, v, layout=args.layout, target_layout="bhsd"
        )
        block_lut, freeze_softmax_max_count, block_attn_mask = build_attn_mode_lut(
            args, q_bhsd, k_bhsd, q.device
        )
    else:
        block_attn_mask = None
        block_lut = None

    fn = make_kernel_runner(
        args,
        q,
        k,
        v,
        block_lut=block_lut,
        freeze_softmax_max_count=freeze_softmax_max_count,
    )
    ms = triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep)

    if args.compare_to_ref:
        current_primary = primary_output(fn())
        current_primary = to_bshd_output_if_needed(current_primary, args.layout)
        ref_primary = make_reference_output(args, q, k, v, block_attn_mask)
        compare_accuracy(current_primary, ref_primary)
        if use_vfa(args):
            # VFA freezes the online-softmax running max after the first few
            # K-block iterations, so a handful of weights drift from the exact
            # reference.  Allow a higher pass-rate so the check passes by default.
            check_attention_outputs(
                current_primary,
                ref_primary,
                fp8=True,
                atol=3.0e-1,
                rtol=2.0e-1,
                max_diff_percentage=2.0,
            )
        elif args.kernel in ("sage_mxfp4", "sage_fp8"):
            # FP8/MXFP4 paths are intrinsically noisier than BF16 reference;
            # use FP8-style tolerances.
            check_attention_outputs(
                current_primary, ref_primary, fp8=True, atol=3.0e-1, rtol=2.0e-1
            )
        else:
            check_attention_outputs(current_primary, ref_primary, fp8=False)

    total_flops = dense_flops(shape)

    qk_elem_size = 1 if args.kernel in QUANTIZED_KERNELS else q.element_size()
    v_elem_size = 1 if args.kernel in FP8_V_KERNELS else v.element_size()
    mem = compute_memory_bytes(shape, qk_elem_size, qk_elem_size, v_elem_size)

    sparse_flops = None
    if block_lut is not None:
        sparse_flops, _ = sparse_flops_from_lut(args.kernel, block_lut, shape)

    if "time(ms)" in provider:
        return ms
    if "sparse_throughput(TFLOPS)" in provider:
        flops = sparse_flops if sparse_flops is not None else total_flops
        return flops / ms * 1e-9
    if "throughput(TFLOPS)" in provider:
        return total_flops / ms * 1e-9
    if "bandwidth(GB/s)" in provider:
        return mem / ms * 1e-6
    if "arithmetic_intensity(FLOP/byte)" in provider:
        return total_flops / mem
    return ms


def metric_lines(args: argparse.Namespace, include_sparse_metric: bool) -> List[str]:
    metric_map = {
        "time": "time(ms)",
        "throughput": "throughput(TFLOPS)",
        "bandwidth": "bandwidth(GB/s)",
        "arithint": "arithmetic_intensity(FLOP/byte)",
        "sparseput": "sparse_throughput(TFLOPS)",
    }

    if args.compare_to_ref:
        return ["time(ms)"]

    if args.metric == "all":
        # By default (when --metric not specified), show only throughput (matching bench_fav3_sage.py)
        result = [metric_map["throughput"]]
        if include_sparse_metric:
            result.append(metric_map["sparseput"])
        return result

    if args.metric == "sparseput" and not include_sparse_metric:
        raise ValueError("sparse_throughput requires --sparge or --block-mask-file")

    if args.metric not in metric_map:
        raise ValueError(f"Unknown metric: {args.metric}")

    return [metric_map[args.metric]]


def make_styles(num_lines: int) -> List[Tuple[str, str]]:
    palette = ["red", "green", "yellow", "blue", "cyan", "magenta"]
    return [(palette[i % len(palette)], "-") for i in range(num_lines)]


def has_sparse_metric(args: argparse.Namespace) -> bool:
    """Whether this run produces a block-sparse LUT (and thus a sparse-throughput metric)."""
    return (
        args.block_mask_file is not None
        or args.block_sparsity is not None
        or builds_qk_lut(args)
    )


def create_single_shape_config(args: argparse.Namespace) -> List[Any]:
    hk = args.hk if args.hk else args.hq
    sk = args.sk if args.sk else args.sq
    d_head = args.d if args.d else 128
    d_head_v = args.dv if args.dv else d_head

    lines = metric_lines(args, has_sparse_metric(args))

    return [
        triton.testing.Benchmark(
            x_names=["BATCH", "HQ", "HK", "N_CTX_Q", "N_CTX_K"],
            x_vals=[(args.b, args.hq, hk, args.sq, sk)],
            line_arg="provider",
            line_vals=lines,
            line_names=lines,
            styles=make_styles(len(lines)),
            ylabel="",
            plot_name=get_caller_name_no_ext(),
            args={
                "D_HEAD": d_head,
                "D_HEAD_V": d_head_v,
                "dtype": arg_to_torch_dtype[args.dtype],
                "layout": args.layout,
                "causal": args.causal,
            },
        )
    ]


def create_captured_config(
    args: argparse.Namespace,
    inputs: List[Dict[str, Any]],
) -> List[Any]:
    lines = metric_lines(args, has_sparse_metric(args))

    return [
        triton.testing.Benchmark(
            x_names=["INPUT_IDX"],
            x_vals=[(i,) for i in range(len(inputs))],
            line_arg="provider",
            line_vals=lines,
            line_names=lines,
            styles=make_styles(len(lines)),
            ylabel="",
            plot_name="bench_sage_captured",
            args={"inputs": inputs},
        )
    ]


def create_mask_list_config(
    args: argparse.Namespace,
    masks: List[LoadedMask],
) -> List[Any]:
    lines = metric_lines(args, include_sparse_metric=True)
    hk = args.hk if args.hk else args.hq

    return [
        triton.testing.Benchmark(
            x_names=["MASK_IDX"],
            x_vals=[(i,) for i in range(len(masks))],
            line_arg="provider",
            line_vals=lines,
            line_names=lines,
            styles=make_styles(len(lines)),
            ylabel="",
            plot_name=get_caller_name_no_ext() + "_masks",
            args={
                "masks": masks,
                "D_HEAD": args.d,
                "D_HEAD_V": args.dv,
                "dtype": arg_to_torch_dtype[args.dtype],
                "layout": args.layout,
                "causal": args.causal,
                "args": args,
                "HQ": args.hq,
                "HK": hk,
            },
        )
    ]


def load_captured_inputs(input_dir: str) -> List[Dict[str, Any]]:
    input_files = sorted(glob.glob(os.path.join(input_dir, "*_input_*.pt")))
    if not input_files:
        raise FileNotFoundError(f"No captured input files found in {input_dir}")

    inputs = []
    for file_path in input_files:
        inputs.append(torch.load(file_path, weights_only=False))

    logger.info("Loaded %d captured inputs", len(inputs))
    return inputs


def validate_args(args: argparse.Namespace) -> None:
    if not args.load_captured:
        required = [args.b, args.hq, args.sq, args.d]
        if any(v <= 0 for v in required):
            raise ValueError("For generated inputs provide positive --b --hq --sq --d")

    if args.dv <= 0:
        args.dv = args.d
    if args.hk <= 0:
        args.hk = args.hq
    if args.sk <= 0:
        args.sk = args.sq

    if args.compare_to_ref and args.ref not in ("torch", "aiter_bf16"):
        raise ValueError("--ref must be one of: torch, aiter_bf16")

    if args.block_sparsity is not None:
        if not 0.0 <= args.block_sparsity <= 1.0:
            raise ValueError("--block-sparsity must be in [0, 1]")
        if args.block_mask_file:
            raise ValueError(
                "--block-sparsity cannot be combined with --block-mask-file"
            )
        if args.kernel == "all":
            raise ValueError("--kernel=all does not support --block-sparsity")
        if args.sparge:
            logger.warning(
                "--block-sparsity overrides --sparge: using a random block-sparse "
                "mask instead of the Q/K-derived sparge selection"
            )
    elif args.sparge and not args.load_captured:
        # Sparge block selection is data-driven; on randomly generated Q/K it keeps
        # essentially every block (a near-dense LUT), so the sparse metrics match
        # the dense ones. Steer users to real data or a random mask instead.
        logger.warning(
            "--sparge on randomly generated input produces a near-dense block "
            "selection (random Q/K keeps almost all blocks), so sparse_throughput "
            "will match throughput. Use --load-captured with real data, or "
            "--block-sparsity R to benchmark a random block-sparse mask."
        )

    # sage_fp8_vfa with a block LUT (--sparge or a static mask) rides the VFA
    # running-max freeze on the LUT; without one it runs the dense VFA path
    # (fav3_sage_vfa_wrapper_func, m_init estimate from compute_m_proxy_topn).

    if builds_qk_lut(args):
        if args.causal:
            raise NotImplementedError(
                "--sparge / sage_fp8_vfa does not support causal attention "
                "(the block-sparse LUT path is non-causal)"
            )
        if args.block_mask_file:
            # The file mask is used directly; only plain SpargeAttn accepts an
            # external mask (VFA builds a front-loaded LUT from Q/K).
            if use_vfa(args):
                raise ValueError(
                    "--block-mask-file can only be combined with "
                    "--kernel=sage_fp8 --sparge, not the sage_fp8_vfa kernel"
                )
        elif args.kernel not in ("sage_fp8", "sage_fp8_vfa", "all"):
            # Building the LUT from Q/K (no external mask) needs the sage path.
            raise ValueError(
                f"--sparge requires --kernel=sage_fp8 or sage_fp8_vfa, "
                f"got {args.kernel}"
            )

    if args.kernel == "all":
        if args.block_mask_file:
            raise ValueError("--kernel=all does not support --block-mask-file")
        if args.compare_to_ref:
            raise ValueError("--kernel=all does not support --compare-to-ref")
        if args.load_captured:
            raise ValueError("--kernel=all does not support --load-captured")

    if args.e2e and args.kernel not in QUANTIZED_KERNELS and args.kernel != "all":
        logger.warning("--e2e has no effect for kernel %s", args.kernel)

    if args.kernel not in ("sage_mxfp4", "all") and (
        args.qsmooth or args.hadamard_rotate is False
    ):
        logger.warning("MXFP4-specific flags are ignored unless --kernel=sage_mxfp4")


def run_benchmark_generated(
    args: argparse.Namespace,
    loaded_single_mask: Optional[LoadedMask],
) -> None:
    @triton.testing.perf_report(create_single_shape_config(args))
    def bench_mha(
        BATCH,
        HQ,
        HK,
        N_CTX_Q,
        N_CTX_K,
        D_HEAD,
        D_HEAD_V,
        dtype,
        layout,
        causal,
        provider,
        device="cuda",
    ):
        q = torch.randn((BATCH, HQ, N_CTX_Q, D_HEAD), device=device, dtype=dtype)
        k = torch.randn((BATCH, HK, N_CTX_K, D_HEAD), device=device, dtype=dtype)
        v = torch.randn((BATCH, HK, N_CTX_K, D_HEAD_V), device=device, dtype=dtype)

        q, k, v = layout_preprocess(q, k, v, layout="bhsd", target_layout=layout)

        return benchmark_single_case(
            args,
            q,
            k,
            v,
            provider,
            loaded_single_mask=loaded_single_mask,
        )

    bench_mha.run(save_path="." if args.o else None, print_data=True)


def run_benchmark_captured(
    args: argparse.Namespace,
    loaded_single_mask: Optional[LoadedMask],
) -> None:
    inputs = load_captured_inputs(args.captured_dir)

    @triton.testing.perf_report(create_captured_config(args, inputs))
    def bench_mha_captured(INPUT_IDX, inputs, provider, device="cuda"):
        inp = inputs[INPUT_IDX]
        q = inp["q"].to(device)
        k = inp["k"].to(device)
        v = inp["v"].to(device)
        return benchmark_single_case(
            args,
            q,
            k,
            v,
            provider,
            loaded_single_mask=loaded_single_mask,
        )

    bench_mha_captured.run(save_path="." if args.o else None, print_data=True)


def run_benchmark_mask_list(args: argparse.Namespace, masks: List[LoadedMask]) -> None:
    block_m, block_n = kernel_block_sizes(args.kernel)

    @triton.testing.perf_report(create_mask_list_config(args, masks))
    def bench_mha_masks(
        MASK_IDX,
        masks,
        D_HEAD,
        D_HEAD_V,
        dtype,
        layout,
        causal,
        args,
        HQ,
        HK,
        provider,
        device="cuda",
    ):
        loaded = masks[MASK_IDX]
        mask = maybe_expand_mask(loaded, loaded.batch, HQ)

        n_ctx_q = loaded.num_q_blocks * block_m
        n_ctx_k = loaded.num_kv_blocks * block_n

        q = torch.randn((loaded.batch, HQ, n_ctx_q, D_HEAD), device=device, dtype=dtype)
        k = torch.randn((loaded.batch, HK, n_ctx_k, D_HEAD), device=device, dtype=dtype)
        v = torch.randn(
            (loaded.batch, HK, n_ctx_k, D_HEAD_V), device=device, dtype=dtype
        )
        q, k, v = layout_preprocess(q, k, v, layout="bhsd", target_layout=layout)
        return benchmark_single_case(
            args,
            q,
            k,
            v,
            provider,
            loaded_single_mask=None,
            explicit_block_attn_mask=mask,
        )

    bench_mha_masks.run(save_path="." if args.o else None, print_data=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified SAGE attention benchmark (FAv3, MXFP4, AITER, FP8)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--kernel",
        type=str,
        default="sage_fp8",
        choices=[*ALL_KERNELS, "all"],
        help="Kernel implementation to benchmark. Use 'all' to compare all backends.",
    )

    parser.add_argument("--b", type=int, default=0, help="Batch size")
    parser.add_argument("--hq", type=int, default=0, help="Number of Q heads")
    parser.add_argument("--hk", type=int, default=0, help="Number of KV heads")
    parser.add_argument("--sq", type=int, default=0, help="Query sequence length")
    parser.add_argument("--sk", type=int, default=0, help="KV sequence length")
    parser.add_argument("--d", type=int, default=0, help="Q/K head dimension")
    parser.add_argument("--dv", type=int, default=0, help="V head dimension")

    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"]
    )
    parser.add_argument("--layout", type=str, default="bshd", choices=["bshd", "bhsd"])
    parser.add_argument("--causal", action="store_true", help="Enable causal attention")

    parser.add_argument(
        "--metric",
        type=str,
        default="all",
        choices=[
            "all",
            "time",
            "throughput",
            "bandwidth",
            "arithint",
            "sparseput",
        ],
        help="Metric(s) to report (default: time+throughput only; 'all' does not include bandwidth/arithint)",
    )

    parser.add_argument("-o", action="store_true", help="Write Triton output CSV")
    parser.add_argument(
        "--print-vgpr", action="store_true", help="Print kernel VGPR usage"
    )

    parser.add_argument(
        "--compare-to-ref", action="store_true", help="Compare against reference"
    )
    parser.add_argument(
        "--ref",
        type=str,
        default="torch",
        choices=["torch", "aiter_bf16"],
        help="Reference kernel for --compare-to-ref",
    )

    parser.add_argument(
        "--load-captured",
        action="store_true",
        help="Use captured tensors from disk instead of random generation",
    )
    parser.add_argument(
        "--captured-dir",
        type=str,
        default="./captured_inputs",
        help="Directory containing *_input_*.pt files",
    )

    parser.add_argument(
        "--sparge",
        action="store_true",
        help="SpargeAttn block sparsity: build the block LUT from Q/K via "
        "build_attention_lut using --simthreshd1/--cdfthreshd. With "
        "--kernel=sage_fp8 this runs 'sparge' (block selection only); with "
        "--kernel=sage_fp8_vfa it runs 'both' (sparge selection + VFA "
        "running-max freeze). If --block-mask-file is also given, the mask is "
        "taken from that file instead.",
    )
    parser.add_argument(
        "--block-mask-file",
        type=str,
        default=None,
        help="JSON file with block masks. Used directly as the block mask "
        "(takes precedence over the Q/K-derived --sparge mask).",
    )
    parser.add_argument(
        "--block-sparsity",
        type=float,
        default=None,
        help="Random block sparsity ratio in [0,1] (fraction of (q_block, "
        "kv_block) pairs masked out). Generates a random block-sparse mask, "
        "useful for benchmarking sparsity on randomly generated inputs where "
        "--sparge selection is meaningless. Takes precedence over --sparge.",
    )

    parser.add_argument(
        "--sparge-load-balancing",
        action="store_true",
        help="Load-balance the block-sparse launch: sort the (batch, head, "
        "q_block) tiles by descending KV-block count (longest-processing-time "
        "first) and run them through a persistent kernel whose resident "
        "programs share a global atomic work queue, so heavy tiles never strand "
        "idle CUs in the tail. Only affects the block-sparse "
        "sage_fp8/sage_fp8_vfa path.",
    )
    parser.add_argument(
        "--n-sample",
        type=int,
        default=16,
        help="Number of top pooled-score K blocks front-loaded per q-block for "
        "the sage_fp8_vfa kernel ('both' mode) / used as the static-mask freeze count",
    )
    parser.add_argument(
        "--simthreshd1",
        type=float,
        default=0.1,
        help="Per-head intra-block similarity threshold for --sparge meansim "
        "selection",
    )
    parser.add_argument(
        "--cdfthreshd",
        type=float,
        default=0.9,
        help="Per-head cumulative-probability threshold for --sparge " "selection",
    )

    parser.add_argument(
        "--e2e",
        action="store_true",
        help="Include quantization overhead in benchmark timing",
    )
    parser.add_argument(
        "--hadamard-rotate",
        type=lambda v: bool(int(v)),
        default=True,
        help="(MXFP4 only) Apply Hadamard rotation: 1/0",
    )
    parser.add_argument(
        "--block-r",
        type=int,
        default=128,
        help="(MXFP4 only) Hadamard block size, must be <= head dim",
    )
    parser.add_argument(
        "--qsmooth",
        action="store_true",
        help="(MXFP4 only) Enable Q smoothing",
    )

    parser.add_argument(
        "--rep",
        type=int,
        default=100,
        help="do_bench rep time in ms",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="do_bench warmup time in ms",
    )

    return parser.parse_args()


def print_vgpr_from_bench(runner: Any) -> None:
    """Run benchmark with Triton dumps enabled and print kernel VGPR metadata.

    This avoids relying on benchmark_utils table parsing, which can fail when
    Triton does not emit the expected result table format.
    """
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        output_file = temp_file.name

    old_stdout, old_stderr = sys.stdout, sys.stderr
    env_keys = [
        "AMDGCN_ENABLE_DUMP",
        "TRITON_ALWAYS_COMPILE",
        "TRITON_PRINT_AUTOTUNING",
    ]
    old_env = {k: os.environ.get(k) for k in env_keys}

    try:
        with open(output_file, "w+") as temp_file:
            sys.stdout = temp_file
            sys.stderr = temp_file

            os.environ["AMDGCN_ENABLE_DUMP"] = "1"
            os.environ["TRITON_ALWAYS_COMPILE"] = "1"
            os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
            runner()

            sys.stdout.flush()
            sys.stderr.flush()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        for k in env_keys:
            if old_env[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old_env[k]

    time.sleep(0.2)

    try:
        with open(output_file, "r") as f:
            lines = f.readlines()
    finally:
        os.unlink(output_file)

    vgpr_info: List[str] = []
    for line in lines:
        if re.search(r"Autotuning kernel", line):
            vgpr_info.append(line.strip())
        if re.search(r"Triton autotuning for function", line):
            vgpr_info.append(line.strip())
        if re.search(r"\.name:", line):
            vgpr_info.append(line.strip())
        if re.search(r"\.vgpr_count:", line) or re.search(r"\.vgpr_spill_count:", line):
            vgpr_info.append(line.strip())

    if vgpr_info:
        print("\n".join(vgpr_info))
    else:
        print("No VGPR metadata found in Triton dump output.")


def run_all_kernels(args: argparse.Namespace) -> None:
    """Run all backends on the same QKV inputs and print a comparison table."""
    dtype = arg_to_torch_dtype[args.dtype]
    device = "cuda"
    hk = args.hk if args.hk else args.hq
    sk = args.sk if args.sk else args.sq
    d_head = args.d if args.d else 128
    d_head_v = args.dv if args.dv else d_head

    q = torch.randn((args.b, args.hq, args.sq, d_head), device=device, dtype=dtype)
    k = torch.randn((args.b, hk, sk, d_head), device=device, dtype=dtype)
    v = torch.randn((args.b, hk, sk, d_head_v), device=device, dtype=dtype)
    q, k, v = layout_preprocess(q, k, v, layout="bhsd", target_layout=args.layout)

    shape = infer_shape_spec(q, v, args.layout)
    total_flops = dense_flops(shape)

    saved_kernel = args.kernel
    rows: List[Tuple[str, float, float]] = []

    q_bhsd, k_bhsd, _ = layout_preprocess(
        q, k, v, layout=args.layout, target_layout="bhsd"
    )

    for kernel_name in ALL_KERNELS:
        args.kernel = kernel_name
        try:
            block_lut, freeze = None, -1
            if builds_qk_lut(args):
                block_lut, freeze, _ = build_attn_mode_lut(args, q_bhsd, k_bhsd, device)
            fn = make_kernel_runner(
                args, q, k, v, block_lut=block_lut, freeze_softmax_max_count=freeze
            )
            ms = triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep)
            tflops = total_flops / ms * 1e-9
            rows.append((kernel_name, ms, tflops))
        except Exception as e:
            logger.warning("Skipping %s: %s", kernel_name, e)
            rows.append((kernel_name, float("nan"), float("nan")))

    args.kernel = saved_kernel

    print(
        f"\nbench_sage --kernel=all  (b={args.b} hq={args.hq} sq={args.sq} sk={sk} d={d_head}):"
    )
    print(f"{'kernel':<16} {'time(ms)':>10} {'TFLOPS':>10}")
    print("-" * 38)
    for name, ms, tflops in rows:
        if ms != ms:  # nan
            print(f"{name:<16} {'SKIP':>10} {'SKIP':>10}")
        else:
            print(f"{name:<16} {ms:>10.4f} {tflops:>10.2f}")


def run_with_optional_vgpr(args: argparse.Namespace, runner: Any) -> int:
    if args.print_vgpr:
        print_vgpr_from_bench(runner)
    else:
        runner()
    return 0


def main() -> int:
    args = parse_args()
    validate_args(args)

    loaded_masks = load_block_mask_from_json(args.block_mask_file, torch.device("cuda"))
    loaded_single_mask: Optional[LoadedMask] = None

    if isinstance(loaded_masks, list):
        if args.load_captured:
            raise ValueError("List mask mode and --load-captured cannot be combined")
        if args.hq <= 0 or args.d <= 0:
            raise ValueError("For list mask mode, provide positive --hq and --d")
        if args.dv <= 0:
            args.dv = args.d
        if args.hk <= 0:
            args.hk = args.hq
        return run_with_optional_vgpr(
            args,
            lambda: run_benchmark_mask_list(args, loaded_masks),
        )

    if isinstance(loaded_masks, LoadedMask):
        loaded_single_mask = loaded_masks

    if args.kernel == "all":
        return run_with_optional_vgpr(args, lambda: run_all_kernels(args))

    if args.load_captured:

        def default_runner():
            run_benchmark_captured(args, loaded_single_mask)

    else:

        def default_runner():
            run_benchmark_generated(args, loaded_single_mask)

    return run_with_optional_vgpr(args, default_runner)


if __name__ == "__main__":
    sys.exit(main())
