from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import argparse
import csv
import glob
import json
import logging
import os
import re
import sys
import tempfile
import time

import numpy as np
import torch
import triton

import aiter
from aiter.ops.mha import (
    flash_attn_func,
    flash_attn_fp8_pertensor_func,
    flash_attn_i8fp8_pertensor_func,
    flash_attn_mxfp4_func,
)

from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_3
from aiter.ops.triton.attention.mha_v3 import _quantize_bshd
from aiter.ops.triton.attention.fav3_sage import (
    fav3_sage_func,
    fav3_sage_wrapper_func,
    get_sage_fwd_configs,
)
from aiter.ops.triton.attention.fav3_sage_attention_mxfp4_wrapper import (
    fav3_sage_mxfp4_func,
    fav3_sage_mxfp4_wrapper,
    get_sage_fwd_configs_mxfp4,
)
from aiter.ops.triton.attention.utils import block_attn_mask_to_ragged_lut
from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
    create_hadamard_matrix,
    sage_quant,
    sage_quant_mxfp4,
    rotation_smooth_qk,
)
from aiter.ops.triton._triton_kernels.quant.sage_attention_quant import (
    sage_quant_v_kernel,
)
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
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
    "sage_mxfp4",
    "fav3_fp8",
    "aiter_fp8",
    "aiter_i8fp8",
    "aiter_mxfp4",
    "aiter_mxfp6",
    "aiter_bf16",
]

ALL_KERNELS: List[str] = [
    "sage_fp8",
    "sage_mxfp4",
    "aiter_fp8",
    "aiter_i8fp8",
    "aiter_mxfp4",
    "aiter_mxfp6",
    "aiter_bf16",
]

QUANT_KERNELS = {
    "sage_fp8",
    "sage_mxfp4",
    "fav3_fp8",
    "aiter_fp8",
    "aiter_i8fp8",
    "aiter_mxfp4",
    "aiter_mxfp6",
}


@dataclass
class ShapeSpec:
    batch: int
    hq: int
    hk: int
    n_ctx_q: int
    n_ctx_k: int
    d_head: int
    d_head_v: int


@dataclass
class LoadedMask:
    mask: torch.Tensor
    batch: int
    num_q_blocks: int
    num_kv_blocks: int


@dataclass
class AccuracyMetrics:
    mae: float
    maxe: float
    cosine: float


@dataclass
class AllKernelRow:
    kernel: str
    ms: float
    tflops: float
    accuracy: Optional[AccuracyMetrics] = None


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


def _generate_transformer_qkv(
    batch: int,
    hq: int,
    hk: int,
    sq: int,
    sk: int,
    d_head: int,
    d_head_v: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Realistic LLM activations: RMS-norm + per-channel log-normal scales + shared low-rank Q/K component + V outlier dims/tokens. Returns fp32 q/k/v.
    q = torch.randn((batch, hq, sq, d_head), device=device, dtype=torch.float32)
    k = torch.randn((batch, hk, sk, d_head), device=device, dtype=torch.float32)
    v = torch.randn((batch, hk, sk, d_head_v), device=device, dtype=torch.float32)

    q = q / q.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
    k = k / k.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
    v = v / v.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()

    q_channel_scale = torch.exp(
        0.35 * torch.randn((1, hq, 1, d_head), device=device)
    ).clamp(0.35, 2.5)
    k_channel_scale = torch.exp(
        0.35 * torch.randn((1, hk, 1, d_head), device=device)
    ).clamp(0.35, 2.5)
    v_channel_scale = torch.exp(
        0.45 * torch.randn((1, hk, 1, d_head_v), device=device)
    ).clamp(0.25, 3.5)
    q = q * q_channel_scale
    k = k * k_channel_scale
    v = v * v_channel_scale

    shared_heads = min(hq, hk)
    shared_seq = min(sq, sk)
    shared_d = min(d_head, d_head_v)
    if shared_heads > 0 and shared_seq > 0:
        shared = torch.randn(
            (batch, shared_heads, shared_seq, shared_d),
            device=device,
            dtype=torch.float32,
        )
        q[:, :shared_heads, :shared_seq, :shared_d] += 0.35 * shared
        k[:, :shared_heads, :shared_seq, :shared_d] += 0.35 * shared

    num_v_outlier_dims = max(1, d_head_v // 16)
    v_outlier_dims = torch.randperm(d_head_v, device=device)[:num_v_outlier_dims]
    v[..., v_outlier_dims] *= 4.0
    num_v_outlier_tokens = max(1, sk // 128)
    v_outlier_tokens = torch.randperm(sk, device=device)[:num_v_outlier_tokens]
    v[:, :, v_outlier_tokens, :] *= 2.5

    return q, k, v


def generate_test_tensors(
    batch: int,
    hq: int,
    hk: int,
    sq: int,
    sk: int,
    d_head: int,
    d_head_v: int,
    dtype: torch.dtype,
    device: str,
    distribution: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # "normal": plain iid Gaussian Q/K/V -- the simplest smoke-test inputs.
    if distribution == "normal":
        q = torch.randn((batch, hq, sq, d_head), device=device, dtype=dtype)
        k = torch.randn((batch, hk, sk, d_head), device=device, dtype=dtype)
        v = torch.randn((batch, hk, sk, d_head_v), device=device, dtype=dtype)
        return q, k, v

    # "sink": realistic StreamingLLM-style pattern where a few leading "sink" tokens attract most attention mass -- peaked yet in-distribution for long context.
    if distribution == "sink":
        q, k, v = _generate_transformer_qkv(
            batch, hq, hk, sq, sk, d_head, d_head_v, device
        )
        g = torch.nn.functional.normalize(
            torch.randn((batch, 1, 1, d_head), device=device, dtype=torch.float32),
            dim=-1,
        )
        num_sinks = min(sk, 4)
        k[:, :, :num_sinks, :] += 12.0 * g
        q = q + 3.0 * g
        return q.to(dtype), k.to(dtype), v.to(dtype)

    if distribution == "underflow":
        # Reproduces the fp8 underflow tile-skip regression on the microbench.
        #
        # A strong "hotspot" in the first KV tile (keys [0:128]) establishes a
        # high frozen softmax max. Every later KV tile then sits far below the
        # e4m3 round-to-zero floor (~2^-11 of the max ≈ 7.62 nats), so the
        # kernel's underflow tile-skip path fires on those tiles. A per-query-row
        # jitter on the hotspot strength makes the all-underflow condition
        # row-dependent, so the two anti-phase co-resident wave groups (which own
        # different query-row blocks) disagree on which tiles to skip. The
        # shared-VALU lockstep barrier then eats the saving while the extra
        # underflow compare is still paid on every no-mask tile -> net slowdown,
        # matching the real-Wan result.
        #
        # Tunables (env):
        #   AITER_UNDERFLOW_GAP    max hotspot logit in nats (default 16.0)
        #   AITER_UNDERFLOW_JITTER per-row hotspot factor ~ U[jitter, 1]
        #                          (default 0.4 -> asymmetric/realistic regression;
        #                           set 1.0 for the symmetric best-case where every
        #                           later tile underflows for both partner waves)
        gap = float(os.environ.get("AITER_UNDERFLOW_GAP", "16.0"))
        jitter = float(os.environ.get("AITER_UNDERFLOW_JITTER", "0.4"))
        jitter = min(max(jitter, 0.0), 1.0)
        scale = float(d_head) ** -0.5  # kernel softmax scale (1/sqrt(d_head))
        hot_keys = min(128, sk)        # one KV tile

        # Single shared hotspot direction (unit vector), broadcast over heads.
        u = torch.randn((1, 1, 1, d_head), device=device, dtype=torch.float32)
        u = u / u.pow(2).sum(dim=-1, keepdim=True).add(1e-12).sqrt()

        # Amplitude so a fully-aligned Q/K pair (row_factor=1) yields a hotspot
        # logit == gap after the 1/sqrt(d_head) softmax scaling.
        amp = (gap / scale) ** 0.5

        # Per-query-row hotspot factor in [jitter, 1]; the hotspot logit for a
        # row is gap * row_factor, so rows with row_factor < ~7.62/gap will NOT
        # fully underflow the later tiles -> partner-wave disagreement.
        row_factor = jitter + (1.0 - jitter) * torch.rand(
            (batch, hq, sq, 1), device=device, dtype=torch.float32
        )
        q = amp * row_factor * u + 0.35 * torch.randn(
            (batch, hq, sq, d_head), device=device, dtype=torch.float32
        )

        # Remaining keys: small, near-orthogonal -> low logits. Hotspot keys
        # (first tile) aligned with u at amplitude `amp`.
        k = 0.30 * torch.randn(
            (batch, hk, sk, d_head), device=device, dtype=torch.float32
        )
        k[:, :, :hot_keys, :] = amp * u + 0.10 * torch.randn(
            (batch, hk, hot_keys, d_head), device=device, dtype=torch.float32
        )

        v = 0.5 * torch.randn(
            (batch, hk, sk, d_head_v), device=device, dtype=torch.float32
        )
        return q.to(dtype), k.to(dtype), v.to(dtype)

    if distribution == "latesink":
        # ADVERSARIAL TRIPWIRE for the frozen-max rollback (added 2026-06-14 after the black-video
        # regression). Mirrors `underflow` but places the high-norm "attention sink" hotspot in the
        # LAST KV tile instead of the first. With a frozen-max rollback that seeds from tile 0, the
        # seed is LOW and the late hotspot's logit blows far past it -> the Schraudolph u32-cvt
        # saturates to 0xFFFFFFFF (NaN bits) -> corrupt P -> NaN/black. The exact (proper running
        # max) path is immune (S - m_new <= 0 always). Random transformer/normal/underflow never
        # produce a late-tile outlier, so this is the structured input cosine-on-random missed.
        #   AITER_LATESINK_GAP : late-hotspot logit in nats (default 40.0 -> well past the cvt
        #                        saturation at scale_log2e*(S-seed) > 128 for 1/sqrt(d) scaling)
        gap = float(os.environ.get("AITER_LATESINK_GAP", "40.0"))
        scale = float(d_head) ** -0.5
        hot_keys = min(128, sk)              # one KV tile
        u = torch.randn((1, 1, 1, d_head), device=device, dtype=torch.float32)
        u = u / u.pow(2).sum(dim=-1, keepdim=True).add(1e-12).sqrt()
        amp = (gap / scale) ** 0.5
        # Q fully aligned with the sink direction so the late tile dominates.
        q = amp * u + 0.35 * torch.randn(
            (batch, hq, sq, d_head), device=device, dtype=torch.float32
        )
        # All keys small/near-orthogonal EXCEPT the LAST tile, which holds the sink.
        k = 0.30 * torch.randn(
            (batch, hk, sk, d_head), device=device, dtype=torch.float32
        )
        k[:, :, sk - hot_keys:, :] = amp * u + 0.10 * torch.randn(
            (batch, hk, hot_keys, d_head), device=device, dtype=torch.float32
        )
        v = 0.5 * torch.randn(
            (batch, hk, sk, d_head_v), device=device, dtype=torch.float32
        )
        return q.to(dtype), k.to(dtype), v.to(dtype)

    if distribution != "transformer":
        raise ValueError(f"Unsupported input distribution: {distribution}")

    # "transformer": realistic LLM activation statistics (see _generate_transformer_qkv).
    q, k, v = _generate_transformer_qkv(batch, hq, hk, sq, sk, d_head, d_head_v, device)
    return q.to(dtype), k.to(dtype), v.to(dtype)


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
        loaded = []
        for item in data["masks"]:
            if "mask" not in item:
                raise ValueError("Each element in 'masks' must include key 'mask'")
            m = _mask_array_to_tensor(item["mask"], device)
            if "num_q_blocks" in item and item["num_q_blocks"] != m.num_q_blocks:
                raise ValueError(
                    f"num_q_blocks mismatch: inferred {m.num_q_blocks}, got {item['num_q_blocks']}"
                )
            if "num_kv_blocks" in item and item["num_kv_blocks"] != m.num_kv_blocks:
                raise ValueError(
                    f"num_kv_blocks mismatch: inferred {m.num_kv_blocks}, got {item['num_kv_blocks']}"
                )
            loaded.append(m)
        return loaded

    if "mask" in data:
        m = _mask_array_to_tensor(data["mask"], device)
        if "num_q_blocks" in data and data["num_q_blocks"] != m.num_q_blocks:
            raise ValueError(
                f"num_q_blocks mismatch: inferred {m.num_q_blocks}, got {data['num_q_blocks']}"
            )
        if "num_kv_blocks" in data and data["num_kv_blocks"] != m.num_kv_blocks:
            raise ValueError(
                f"num_kv_blocks mismatch: inferred {m.num_kv_blocks}, got {data['num_kv_blocks']}"
            )
        return m

    return None


def kernel_block_sizes(kernel: KernelName) -> Tuple[int, int]:
    if kernel == "sage_mxfp4":
        cfg = get_sage_fwd_configs_mxfp4()
    else:
        cfg = get_sage_fwd_configs()
    return cfg["BLOCK_M"], cfg["BLOCK_N"]


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
    device: torch.device,
    loaded_single_mask: Optional[LoadedMask],
) -> Optional[torch.Tensor]:
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

    if args.block_sparsity is None:
        return None

    block_m, block_n = kernel_block_sizes(args.kernel)
    num_q_blocks = (shape.n_ctx_q + block_m - 1) // block_m
    num_kv_blocks = (shape.n_ctx_k + block_n - 1) // block_n

    return (
        torch.rand(
            shape.batch,
            shape.hq,
            num_q_blocks,
            num_kv_blocks,
            device=device,
        )
        > args.block_sparsity
    ).to(torch.bool)


def sparse_flops_from_lut(
    kernel: KernelName,
    block_lut: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    shape: ShapeSpec,
) -> Tuple[float, float]:
    _, _, lut_count = block_lut
    num_sparse_pairs = lut_count.sum().item()

    block_m, block_n = kernel_block_sizes(kernel)
    num_q_blocks = (shape.n_ctx_q + block_m - 1) // block_m
    num_kv_blocks = (shape.n_ctx_k + block_n - 1) // block_n
    num_dense_pairs = shape.batch * shape.hq * num_q_blocks * num_kv_blocks

    total_dense_flops = (
        2.0
        * shape.batch
        * shape.hq
        * shape.n_ctx_q
        * shape.n_ctx_k
        * (shape.d_head + shape.d_head_v)
    )

    if num_dense_pairs == 0:
        return 0.0, total_dense_flops

    sparse_flops = total_dense_flops * (num_sparse_pairs / num_dense_pairs)
    return sparse_flops, total_dense_flops


def fp8_quantize(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    quant_dtype = aiter.dtypes.fp8
    q_quant, q_descale = aiter.per_tensor_quant(
        q,
        scale=torch.abs(q).max() if scale is None else scale,
        quant_dtype=quant_dtype,
        dtypeMax=torch.finfo(quant_dtype).max,
    )
    k_quant, k_descale = aiter.per_tensor_quant(
        k,
        scale=torch.abs(k).max() if scale is None else scale,
        quant_dtype=quant_dtype,
        dtypeMax=torch.finfo(quant_dtype).max,
    )
    v_quant, v_descale = aiter.per_tensor_quant(
        v,
        scale=torch.abs(v).max() if scale is None else scale,
        quant_dtype=quant_dtype,
        dtypeMax=torch.finfo(quant_dtype).max,
    )
    return q_quant, k_quant, v_quant, q_descale, k_descale, v_descale


def i8fp8_quantize(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_clip: float = 1.0,
    k_clip: float = 1.0,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Quantize Q/K to int8, V to fp8 (Sage-style)."""
    # Q -> int8
    q_amax = torch.abs(q).max() * q_clip
    q_scale = q_amax / 127.0
    q_int8 = torch.clamp(torch.round(q / q_scale), -128, 127).to(torch.int8)
    q_descale = q_scale.reshape(1).to(torch.float32)
    # K -> int8
    k_amax = torch.abs(k).max() * k_clip
    k_scale = k_amax / 127.0
    k_int8 = torch.clamp(torch.round(k / k_scale), -128, 127).to(torch.int8)
    k_descale = k_scale.reshape(1).to(torch.float32)
    # V -> fp8
    quant_dtype = aiter.dtypes.fp8
    v_quant, v_descale = aiter.per_tensor_quant(
        v,
        scale=torch.abs(v).max(),
        quant_dtype=quant_dtype,
        dtypeMax=torch.finfo(quant_dtype).max,
    )
    return q_int8, k_int8, v_quant, q_descale, k_descale, v_descale


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


# f6f6 kernel (double-FP6: fp6 Q,K + fp6 V). Built offline by mi350_fmha_hd128_f6f6.py
# with its OWN symbol (_ZN5aiter28fmha_fwd_hd128_mxfp6_gfx950E) and OWN config row
# (dtype "mxfp6bf16"), so it has a dedicated .co slot (fwd_hd128_mxfp6.co) and
# coexists with aiter_mxfp4 -- no overlay. The C++ dispatch routes 96-wide
# (fp6-packed) q/k to this slot automatically. The .co is loaded straight from the
# AITER hsa dir like every other kernel; to swap kernels, copy your build over
# hsa/gfx950/fmha_v3_fwd/fwd_hd128_mxfp6.co.


def _deployed_mxfp4_co_path() -> str:
    from aiter.jit.core import get_asm_dir

    return os.path.join(get_asm_dir(), "fmha_v3_fwd", "fwd_hd128_mxfp4.co")


# By default the fp6 FMHA encoding lives IN-TREE (aiter.ops.triton.quant.
# mxfp6_fmha_pack). Set AITER_MXFP6_PACK=/path/to/packer.py to override with an
# external packer module by path.
_MXFP6_PACK_PATH = os.environ.get("AITER_MXFP6_PACK")
_host_fp6_pack_mod = None


def _load_host_fp6_pack():
    """Return the MXFP6-E2M3 host packer module (cached).

    Uses the in-tree aiter packer by default; honors AITER_MXFP6_PACK to load an
    external packer module by path."""
    global _host_fp6_pack_mod
    if _host_fp6_pack_mod is None:
        if _MXFP6_PACK_PATH:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "host_fp6_pack", _MXFP6_PACK_PATH
            )
            if spec is None or spec.loader is None:
                raise FileNotFoundError(
                    f"host fp6 packer not found at {_MXFP6_PACK_PATH}; "
                    "unset AITER_MXFP6_PACK to use the in-tree packer."
                )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        else:
            from aiter.ops.triton.quant import mxfp6_fmha_pack as mod
        _host_fp6_pack_mod = mod
    return _host_fp6_pack_mod


def _sage_quant_mxfp6(
    q,
    k,
    v,
    FP8_TYPE,
    FP8_MAX,
    BLKQ,
    BLKK,
    sm_scale=None,
    q_smoothing=False,
    layout="bshd",
    R=None,
    BLOCK_R=32,
    qk_packer=None,
    k_colmajor_packer=None,
):
    """MXFP6-E2M3 QK quantize for the aiter_mxfp6 bench path.

    Local copy of sage_quant_mxfp4's body so the shared production wrapper
    (sage_attention_quant_wrappers.sage_quant_mxfp4) stays untouched. When
    qk_packer is supplied, Q/K are packed to MXFP6-E2M3 (both QK MFMA operands in
    fp6) instead of the fp4 (e2m1) downcast; V quant (sage_quant_v_kernel) and
    delta_s are identical to the mxfp4 path."""
    v_fp8 = torch.empty_like(v, dtype=FP8_TYPE, device=v.device)
    assert layout == "bshd", f"aiter_mxfp6 bench path expects bshd, got {layout}"
    b, qo_len, h_qo, head_dim = q.shape
    _, kv_len, h_kv, _ = v.shape
    stride_bz_v, stride_h_v, stride_seq_v, stride_d_v = (
        v.stride(0),
        v.stride(2),
        v.stride(1),
        v.stride(3),
    )
    K_NUM_BLKS = (kv_len + BLKK - 1) // BLKK

    v_scale = v.abs().amax(dim=1).to(torch.float32) / FP8_MAX
    grid = (b * h_kv * K_NUM_BLKS,)

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    q, k, delta_s = rotation_smooth_qk(
        q,
        k,
        BLKQ,
        R=R,
        BLOCK_R=BLOCK_R,
        q_smoothing=q_smoothing,
        layout=layout,
        sm_scale=(sm_scale * 1.4426950408889634),
    )

    sage_quant_v_kernel[grid](
        v,
        v_fp8,
        v_scale,
        stride_bz_v,
        stride_h_v,
        stride_seq_v,
        stride_d_v,
        v_scale.stride(0),
        v_scale.stride(1),
        b,
        h_kv,
        K_NUM_BLKS,
        kv_len,
        D=head_dim,
        BLK_K=BLKK,
        num_stages=3,
        num_warps=8,
    )

    if qk_packer is not None:
        q_fp4, q_scale = qk_packer(q)
        if k_colmajor_packer is not None:
            # Alternate K-operand layout (the coalesced LDS-order packer); the K
            # scale stays on the existing separate gather.
            k_fp4, k_scale = k_colmajor_packer(k)
        else:
            k_fp4, k_scale = qk_packer(k)
    else:
        q_fp4, q_scale = downcast_to_mxfp(q, torch.uint8, axis=-1)
        k_fp4, k_scale = downcast_to_mxfp(k, torch.uint8, axis=-1)

    return q_fp4, q_scale, k_fp4, k_scale, v_fp8, v_scale, delta_s


def _build_fp6_qk_packer(device):
    """Return a packer: rotated/smoothed float Q|K [...,128] -> (uint8 [...,96]
    interleaved MXFP6-E2M3 data, uint8 [...,4] E8M0 scale) on `device`.

    GPU (Triton) pack by default: byte-identical to the host numpy packer for
    bf16/fp16/fp32 Q/K (v/2^E is an exponent shift, exact in fp32) and runs every
    do_bench iteration cheaply on the rotated tensors. Set AITER_MXFP6_QK_TRITON=0
    to fall back to the (memoized) CPU numpy round-trip."""
    hp = _load_host_fp6_pack()
    _use_triton = (
        os.environ.get("AITER_MXFP6_QK_TRITON", "1") != "0"
        and getattr(hp, "_HAVE_TRITON", False)
        and hasattr(hp, "quantize_fp6_lastdim_triton")
    )
    if _use_triton:

        def _packer(t: torch.Tensor):
            return hp.quantize_fp6_lastdim_triton(t)

        return _packer

    _cache: dict = {}

    def _packer(t: torch.Tensor):
        key = (t.data_ptr(), tuple(t.shape), t.dtype)
        hit = _cache.get(key)
        if hit is not None:
            return hit
        arr = t.detach().to(torch.float32).cpu().numpy()
        packed, scale = hp.quantize_fp6_lastdim(arr)
        out = (
            torch.from_numpy(packed).to(device),
            torch.from_numpy(scale).to(device),
        )
        _cache[key] = out
        return out

    return _packer


def _build_hostv_v_packer(device):
    """Return a packer: fp8 V [b, sk, h_kv, d] -> strided uint8 V tensor in the
    kernel's native-fp6 d-major tile-flat HBM layout.

    The packed buffer is contiguous [b, h_kv, n_tiles*12800] uint8; it is exposed
    to flash_attn_mxfp4_func as a [b, sk, h_kv, d] tensor via as_strided with the
    kernel's byte strides (v_Seqs=100, v_Hs=n_tiles*12800, v_Bs=h_kv*v_Hs). The
    last dim keeps size d (=128) so the C++ derives the correct output extent and
    the contiguous-last-dim check passes; the kernel does its own tile-flat
    addressing and ignores the per-token/per-d element strides.

    Packs the RAW fp8 V magnitudes (NO per-channel descale); the per-channel
    v_descale is applied in the kernel epilogue, exactly as in the in-kernel
    ("both") path -- so this is numerically a layout change only.

    Memoized on the input tensor identity (do_bench reuses fixed tensors)."""
    hp = _load_host_fp6_pack()
    _cache: dict = {}
    # GPU (Triton) V pack by default: byte-identical to the host packer but ~1000x
    # faster (no CPU round-trip). Set AITER_MXFP6_V_TRITON=0 to force the host pack.
    _use_triton = os.environ.get("AITER_MXFP6_V_TRITON", "1") != "0"

    def _packer(v_fp8: torch.Tensor):
        key = (v_fp8.data_ptr(), tuple(v_fp8.shape), v_fp8.dtype)
        hit = _cache.get(key)
        if hit is not None:
            return hit
        b, sk, h_kv, d = v_fp8.shape
        assert sk % 128 == 0, f"hostv requires seqlen_k % 128 == 0, got {sk}"
        n_tiles = sk // 128
        if _use_triton and getattr(hp, "_HAVE_TRITON", False):
            # Triton GPU packer: returns torch uint8 [b, h_kv, n_tiles*12800] on
            # the V device; no CPU transpose / round-trip.
            packed_t = hp.quantize_fp6_v_clean_triton(v_fp8, tile=128)
            packed_flat = packed_t.reshape(-1)  # device uint8, contiguous
        else:
            # Host (numpy) fallback: byte-identical to the Triton packer.
            v_f = v_fp8.detach().to(torch.float32).cpu().numpy()  # [b, sk, h_kv, d]
            v_dmajor = np.transpose(v_f, (0, 2, 3, 1))  # [b, h_kv, d, sk]
            packed = hp.quantize_fp6_v_clean(v_dmajor, tile=128)
            # packed: uint8 [b, h_kv, n_tiles*12800]
            packed_flat = torch.from_numpy(np.ascontiguousarray(packed).reshape(-1))
        tile_bytes = d * 96 + d * 4  # 12800 for d=128
        v_hs = n_tiles * tile_bytes
        v_bs = h_kv * v_hs
        # as_strided can read up to (sk-1)*100 + (h_kv-1)*v_hs + (d-1) which
        # slightly exceeds b*v_bs; pad the storage tail so the view is in-bounds.
        buf = torch.empty(b * v_bs + 256, dtype=torch.uint8, device=packed_flat.device)
        buf[: packed_flat.numel()] = packed_flat
        buf = buf.to(device)
        v_view = buf.as_strided((b, sk, h_kv, d), (v_bs, 100, v_hs, 1))
        _cache[key] = v_view
        return v_view

    return _packer


# ---------------------------------------------------------------------------
# Coalesced K-load packer (LDS-order): pre-arranges the fp6 K so the cooperative
# load is a CONTIGUOUS (coalesced) copy that lands byte-identically in the kernel's
# chunk-major LDS image (lds_read_K_data + MFMA unchanged). Drives the kernel's
# coalesced K load; fixes the vL1D address-gen serialization (Stalled-on-Address
# 18.5%->0.7%, L1-L2 txns 332M->241M, +1.3%). Permutation lives in the canonical
# packer (mxfp6_fmha_pack.quantize_fp6_k_lds_order).
# ---------------------------------------------------------------------------
def _build_fp6_k_coalesced_packer(device):
    """float K [b,sk,h,128] -> (LDS-order uint8 view [b,sk,h,96] over a [b,h,n_tiles*16384] buffer,
    E8M0 scale [b,sk,h,4]). seq stride = 128 (16384/128) so the kernel's _s_k_Seqs=128 -> tile base =
    token*128.

    GPU PERMUTATION of the (Triton) base fp6 pack: byte-identical to the numpy
    quantize_fp6_k_lds_order (LAYOUT-only gather) but runs entirely on-device, so the
    full pack is seconds instead of the minutes the per-tile numpy CPU round-trip took
    for long seqlens. Memoized per fixed do_bench tensor; the [n_tiles*16384] gather
    index is built once on-device per (n_tiles, device)."""
    hp = _load_host_fp6_pack()
    base = _build_fp6_qk_packer(device)  # GPU (b,sk,h,96) packed + (b,sk,h,4) scale, byte-identical
    # Per-tile [16384] token-major byte index (canonical), lifted to the device once.
    idx16k = torch.as_tensor(
        hp._k_lds_order_gather_index(), dtype=torch.long, device=device
    )
    _cache: dict = {}
    _gidx: dict = {}

    def _gather_index(nt: int, total: int):
        # Full [nt*16384] gather index = tile_base(t*12288) + idx16k, OOB (>=total, the dup/overflow
        # tail) clamped to 0 and masked to 0 -- exactly the numpy np.where(valid, ...) behaviour.
        g = _gidx.get(nt)
        if g is None:
            full = (
                torch.arange(nt, device=device, dtype=torch.long) * 12288
            ).unsqueeze(1) + idx16k.unsqueeze(0)
            full = full.reshape(-1)
            valid = full < total
            gc = torch.where(valid, full, torch.zeros_like(full))
            g = (gc, valid)
            _gidx[nt] = g
        return g

    def _packer(k_thd: torch.Tensor):
        key = (k_thd.data_ptr(), tuple(k_thd.shape), k_thd.dtype)
        hit = _cache.get(key)
        if hit is not None:
            return hit
        b, sk, h, d = k_thd.shape
        assert d == 128 and sk % 128 == 0, (d, sk)
        nt = sk // 128
        packed, scale = base(k_thd)  # GPU uint8 [b,sk,h,96], [b,sk,h,4]
        total = sk * 96
        # token-major flat per (b,h): [b, h, sk*96]
        km = packed.permute(0, 2, 1, 3).reshape(b, h, total).contiguous()
        gc, valid = _gather_index(nt, total)
        out = km[:, :, gc]  # [b, h, nt*16384] on-device gather
        out = torch.where(valid.view(1, 1, -1), out, torch.zeros_like(out))
        k_hs = nt * 16384
        k_bs = h * k_hs
        data = out.reshape(-1)  # contiguous [b*h*nt*16384] = b*k_bs
        buf = torch.empty(b * k_bs + 256, dtype=torch.uint8, device=device)
        buf[: data.numel()] = data
        # seq stride 128 -> kernel _s_k_Seqs=128 -> tile base = token*128 (=tile*16384).
        k_view = buf.as_strided((b, sk, h, 96), (k_bs, 128, k_hs, 1))
        out_t = (k_view, scale)
        _cache[key] = out_t
        return out_t

    return _packer


def make_kernel_runner(
    args: argparse.Namespace,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_lut: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Any:
    q_bshd, k_bshd, v_bshd = layout_preprocess(
        q, k, v, layout=args.layout, target_layout="bshd"
    )
    head_dim = q_bshd.shape[-1]
    softmax_scale = head_dim**-0.5

    if args.kernel == "sage_fp8":
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

        kv_idx, lut_s, lut_c, sparse = _unpack_block_lut(block_lut)
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
            q_fp8, k_fp8, v_fp8, q_ds, k_ds, v_ds = fp8_quantize(
                q_bshd,
                k_bshd,
                v_bshd,
            )
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
            q_bshd,
            k_bshd,
            v_bshd,
        )
        return lambda: flash_attn_fp8_pertensor_func(
            q_fp8,
            k_fp8,
            v_fp8,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    if args.kernel == "aiter_i8fp8":
        q_clip = args.q_clip if args.q_clip is not None else args.qk_clip
        k_clip = args.k_clip if args.k_clip is not None else args.qk_clip

        def _run_aiter_i8fp8():
            q_i8, k_i8, v_fp8, q_ds, k_ds, v_ds = i8fp8_quantize(
                q_bshd,
                k_bshd,
                v_bshd,
                q_clip=q_clip,
                k_clip=k_clip,
            )
            return flash_attn_i8fp8_pertensor_func(
                q_i8,
                k_i8,
                v_fp8,
                q_descale=q_ds,
                k_descale=k_ds,
                v_descale=v_ds,
            )

        if args.e2e:
            return _run_aiter_i8fp8

        q_i8, k_i8, v_fp8, q_descale, k_descale, v_descale = i8fp8_quantize(
            q_bshd,
            k_bshd,
            v_bshd,
            q_clip=q_clip,
            k_clip=k_clip,
        )
        return lambda: flash_attn_i8fp8_pertensor_func(
            q_i8,
            k_i8,
            v_fp8,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    if args.kernel == "aiter_mxfp4":
        # Drive flash_attn_mxfp4_func with mxfp4 inputs from sage_quant_mxfp4 (the
        # same quantizer fav3_sage_mxfp4 uses): q/k are uint8-packed fp4 (head dim
        # 128 -> 64) with per-block E8M0 descales (head_dim/32), v is fp8 with
        # per-channel fp32 descales (as in the i8fp8 path).
        logger.info(
            "aiter_mxfp4: using sage_quant_mxfp4 (real fp4 q/k, per-block "
            "E8M0 descales, fp8 v with per-channel descales)."
        )
        cfg = get_sage_fwd_configs_mxfp4()
        fp8_type = aiter.dtypes.fp8
        fp8_max = torch.finfo(fp8_type).max

        block_r = args.block_r
        if block_r > q_bshd.shape[-1]:
            raise ValueError(
                f"block_r ({block_r}) must be <= head dim ({q_bshd.shape[-1]})"
            )
        r = create_hadamard_matrix(
            block_r, device=q_bshd.device, dtype=q_bshd.dtype
        ) / (block_r**0.5)

        # sage_quant_mxfp4 folds sm_scale into Q before fp4 quant, so the kernel
        # consumes a pre-scaled Q and must NOT re-apply the scale (doing so
        # double-scales the softmax). Pin the fold scale to the same softmax_scale
        # used by the reference and pass it through explicitly.
        def _quantize_mxfp4():
            return sage_quant_mxfp4(
                q_bshd,
                k_bshd,
                v_bshd,
                fp8_type,
                fp8_max,
                BLKQ=cfg["BLOCK_M"],
                BLKK=64,
                layout="bshd",
                R=r,
                BLOCK_R=block_r,
                sm_scale=softmax_scale,
                q_smoothing=args.qsmooth,
            )

        def _run_aiter_mxfp4():
            q_fp4, q_descale, k_fp4, k_descale, v_fp8, v_descale, _delta_s = (
                _quantize_mxfp4()
            )
            return flash_attn_mxfp4_func(
                q_fp4,
                k_fp4,
                v_fp8,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                softmax_scale=softmax_scale,
            )

        if args.e2e:
            return _run_aiter_mxfp4

        q_fp4, q_descale, k_fp4, k_descale, v_fp8, v_descale, _delta_s = (
            _quantize_mxfp4()
        )
        return lambda: flash_attn_mxfp4_func(
            q_fp4,
            k_fp4,
            v_fp8,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            softmax_scale=softmax_scale,
        )

    if args.kernel == "aiter_mxfp6":
        # mxfp6 path: Q,K are packed fp6 [.,.,.,96] and V is packed to the
        # kernel's native-fp6 d-major tile-flat layout. The C++ dispatch routes
        # the fp6 q/k to the DEDICATED fwd_hd128_mxfp6.co slot (own symbol/config),
        # loaded straight from the AITER hsa dir, so aiter_mxfp4 and aiter_mxfp6
        # coexist with no overlay.
        cfg = get_sage_fwd_configs_mxfp4()
        fp8_type = aiter.dtypes.fp8
        fp8_max = torch.finfo(fp8_type).max

        block_r = args.block_r
        if block_r > q_bshd.shape[-1]:
            raise ValueError(
                f"block_r ({block_r}) must be <= head dim ({q_bshd.shape[-1]})"
            )
        r = create_hadamard_matrix(
            block_r, device=q_bshd.device, dtype=q_bshd.dtype
        ) / (block_r**0.5)

        # Build the fp6 QK and V packers ONCE (they memoize per input tensor);
        # rebuilding them inside _quantize_mxfp6 would discard the cache every
        # do_bench iteration.
        _mxfp6_qk_packer = _build_fp6_qk_packer(q_bshd.device)
        _mxfp6_v_packer = _build_hostv_v_packer(q_bshd.device)
        # fp6fp8 validation mode: QK stays fp6 but V is passed as raw FP8 (E4M3)
        # with a per-channel fp32 descale (the mxfp4/fp8 PV back-end), bypassing
        # the native-fp6 V packer. Used to validate the fp6fp8 kernel deployed
        # into the mxfp6 .co slot (the mxfp6bf16 C++ route has in_bpe=1 and reads
        # the V stride from the tensor, so an fp8 V flows through unchanged).
        _fp6fp8_mode = os.environ.get("AITER_FP6FP8", "0") != "0"
        # AITER_KCOAL (default 1) packs K in LDS-order for the kernel's coalesced contiguous K load
        # (the shipped path). Set AITER_KCOAL=0 ONLY to feed token-major K to an OLD token-strided
        # .co for A/B. Routes through the k_colmajor_packer slot (the K-only alt-packer slot).
        if os.environ.get("AITER_KCOAL", "1") != "0":
            _k_colmajor = _build_fp6_k_coalesced_packer(q_bshd.device)
        else:
            _k_colmajor = None

        def _quantize_mxfp6():
            return _sage_quant_mxfp6(
                q_bshd,
                k_bshd,
                v_bshd,
                fp8_type,
                fp8_max,
                BLKQ=cfg["BLOCK_M"],
                BLKK=64,
                layout="bshd",
                R=r,
                BLOCK_R=block_r,
                sm_scale=softmax_scale,
                q_smoothing=args.qsmooth,
                qk_packer=_mxfp6_qk_packer,
                k_colmajor_packer=_k_colmajor,
            )

        def _run_aiter_mxfp6():
            q_fp4, q_descale, k_fp4, k_descale, v_fp8, v_descale, _delta_s = (
                _quantize_mxfp6()
            )
            v_arg = v_fp8 if _fp6fp8_mode else _mxfp6_v_packer(v_fp8)
            return flash_attn_mxfp4_func(
                q_fp4,
                k_fp4,
                v_arg,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                softmax_scale=softmax_scale,
            )

        if args.e2e:
            return _run_aiter_mxfp6

        q_fp4, q_descale, k_fp4, k_descale, v_fp8, v_descale, _delta_s = (
            _quantize_mxfp6()
        )
        v_arg = v_fp8 if _fp6fp8_mode else _mxfp6_v_packer(v_fp8)
        return lambda: flash_attn_mxfp4_func(
            q_fp4,
            k_fp4,
            v_arg,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            softmax_scale=softmax_scale,
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


def compute_accuracy_metrics(
    current: torch.Tensor,
    reference: torch.Tensor,
) -> AccuracyMetrics:
    current_f = current.float()
    reference_f = reference.float()
    abs_diff = (current_f - reference_f).abs()
    cosine = torch.nn.functional.cosine_similarity(
        current_f.flatten(), reference_f.flatten(), dim=0
    ).item()
    return AccuracyMetrics(
        mae=abs_diff.mean().item(),
        maxe=abs_diff.max().item(),
        cosine=cosine,
    )


def fp8_max_diff_percentage(args: argparse.Namespace) -> float:
    if args.input_distribution in ("transformer", "sink"):
        return 2.0
    return 0.5


def check_output_against_reference(
    args: argparse.Namespace,
    current: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    # Guard against NaN/Inf in the kernel output before any accuracy stats are
    # computed (a non-finite output silently wrecks cosine/MAE and is the usual
    # symptom of softmax tail overflow -- see the "latesink" input distribution).
    n_nan = int(torch.isnan(current).sum().item())
    n_inf = int(torch.isinf(current).sum().item())
    if n_nan or n_inf:
        print(f"[NAN-CHECK] FAIL kernel={args.kernel} nan={n_nan} inf={n_inf}")
    else:
        print(f"[NAN-CHECK] PASS kernel={args.kernel} (output finite)")
    compare_accuracy(current, reference)
    if args.kernel in QUANT_KERNELS:
        check_attention_outputs(
            current,
            reference,
            fp8=True,
            max_diff_percentage=fp8_max_diff_percentage(args),
        )
    else:
        check_attention_outputs(current, reference, fp8=False)


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
    ref = args.ref

    if block_attn_mask is not None:
        if ref != "torch":
            raise ValueError(
                "Block sparse comparison currently supports --ref=torch only"
            )
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
    block_attn_mask = (
        explicit_block_attn_mask
        if explicit_block_attn_mask is not None
        else build_block_mask(args, shape, q.device, loaded_single_mask)
    )
    block_lut = (
        block_attn_mask_to_ragged_lut(block_attn_mask, return_none_if_dense=True)
        if block_attn_mask is not None
        else None
    )

    fn = make_kernel_runner(args, q, k, v, block_lut=block_lut)
    ms = triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep)

    if args.compare_to_ref:
        current_primary = primary_output(fn())
        current_primary = to_bshd_output_if_needed(current_primary, args.layout)
        ref_primary = make_reference_output(args, q, k, v, block_attn_mask)
        check_output_against_reference(args, current_primary, ref_primary)

    total_flops = (
        2.0
        * shape.batch
        * shape.hq
        * shape.n_ctx_q
        * shape.n_ctx_k
        * (shape.d_head + shape.d_head_v)
    )

    if args.kernel in (
        "fav3_fp8",
        "aiter_fp8",
        "aiter_i8fp8",
        "aiter_mxfp4",
        "aiter_mxfp6",
        "sage_fp8",
        "sage_mxfp4",
    ):
        q_elem_size = 1
        k_elem_size = 1
    else:
        q_elem_size = q.element_size()
        k_elem_size = k.element_size()

    v_elem_size = (
        1
        if args.kernel
        in ("fav3_fp8", "aiter_fp8", "aiter_i8fp8", "aiter_mxfp4", "aiter_mxfp6")
        else v.element_size()
    )
    mem = compute_memory_bytes(shape, q_elem_size, k_elem_size, v_elem_size)

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
        raise ValueError(
            "sparse_throughput requires --block-sparsity or --block-mask-file"
        )

    if args.metric not in metric_map:
        raise ValueError(f"Unknown metric: {args.metric}")

    return [metric_map[args.metric]]


def make_styles(num_lines: int) -> List[Tuple[str, str]]:
    palette = ["red", "green", "yellow", "blue", "cyan", "magenta"]
    return [(palette[i % len(palette)], "-") for i in range(num_lines)]


def create_single_shape_config(args: argparse.Namespace) -> List[Any]:
    hk = args.hk if args.hk else args.hq
    sk = args.sk if args.sk else args.sq
    d_head = args.d if args.d else 128
    d_head_v = args.dv if args.dv else d_head

    include_sparse_metric = (
        args.block_sparsity is not None or args.block_mask_file is not None
    )
    lines = metric_lines(args, include_sparse_metric)

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
    include_sparse_metric = (
        args.block_sparsity is not None or args.block_mask_file is not None
    )
    lines = metric_lines(args, include_sparse_metric)

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

    if args.block_sparsity is not None and not (0.0 <= args.block_sparsity <= 1.0):
        raise ValueError(
            f"--block-sparsity must be in [0,1], got {args.block_sparsity}"
        )

    if args.block_sparsity is not None and args.block_mask_file:
        logger.info("Using --block-mask-file; ignoring --block-sparsity")

    if args.ref not in ("torch", "aiter_bf16"):
        raise ValueError("--ref must be one of: torch, aiter_bf16")

    if args.kernel == "all":
        if args.block_sparsity is not None or args.block_mask_file:
            raise ValueError("--kernel=all does not support block-sparse mode")
        if args.load_captured:
            raise ValueError("--kernel=all does not support --load-captured")

    _quantized_kernels = (
        "sage_fp8",
        "sage_mxfp4",
        "fav3_fp8",
        "aiter_fp8",
        "aiter_i8fp8",
        "aiter_mxfp4",
        "aiter_mxfp6",
    )

    if args.e2e and args.kernel not in _quantized_kernels and args.kernel != "all":
        logger.warning("--e2e has no effect for kernel %s", args.kernel)

    if args.kernel not in ("sage_mxfp4", "aiter_mxfp4", "aiter_mxfp6", "all") and (
        args.qsmooth or args.hadamard_rotate is False
    ):
        logger.warning(
            "MXFP4/6-specific flags are ignored unless --kernel=sage/aiter_mxfp4/6"
        )


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
        q, k, v = generate_test_tensors(
            BATCH,
            HQ,
            HK,
            N_CTX_Q,
            N_CTX_K,
            D_HEAD,
            D_HEAD_V,
            dtype,
            device,
            args.input_distribution,
        )

        q.requires_grad = False
        k.requires_grad = False
        v.requires_grad = False

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

        q, k, v = generate_test_tensors(
            loaded.batch,
            HQ,
            HK,
            n_ctx_q,
            n_ctx_k,
            D_HEAD,
            D_HEAD_V,
            dtype,
            device,
            args.input_distribution,
        )
        q.requires_grad = False
        k.requires_grad = False
        v.requires_grad = False

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


def run_block_sparse_repetitions(
    args: argparse.Namespace,
    loaded_single_mask: Optional[LoadedMask],
) -> None:
    if loaded_single_mask is not None:
        raise ValueError(
            "--n-repetitions is only supported with random --block-sparsity"
        )

    if args.load_captured:
        raise ValueError(
            "--n-repetitions is supported only with generated random inputs"
        )

    dtype = arg_to_torch_dtype[args.dtype]
    device = "cuda"

    q, k, v = generate_test_tensors(
        args.b,
        args.hq,
        args.hk,
        args.sq,
        args.sk,
        args.d,
        args.dv,
        dtype,
        device,
        args.input_distribution,
    )
    q.requires_grad = False
    k.requires_grad = False
    v.requires_grad = False
    q, k, v = layout_preprocess(q, k, v, layout="bhsd", target_layout=args.layout)

    shape = infer_shape_spec(q, v, args.layout)
    block_m, block_n = kernel_block_sizes(args.kernel)
    num_q_blocks = (shape.n_ctx_q + block_m - 1) // block_m
    num_kv_blocks = (shape.n_ctx_k + block_n - 1) // block_n

    warmup_mask = (
        torch.rand(shape.batch, shape.hq, num_q_blocks, num_kv_blocks, device=device)
        > args.block_sparsity
    ).to(torch.bool)
    warmup_lut = block_attn_mask_to_ragged_lut(warmup_mask, return_none_if_dense=True)
    fn_warmup = make_kernel_runner(args, q, k, v, block_lut=warmup_lut)
    triton.testing.do_bench(fn_warmup, warmup=args.warmup, rep=args.rep)

    total_flops = (
        2.0
        * shape.batch
        * shape.hq
        * shape.n_ctx_q
        * shape.n_ctx_k
        * (shape.d_head + shape.d_head_v)
    )

    latencies_ms: List[float] = []
    tflops_dense: List[float] = []
    tflops_effective: List[float] = []

    for _ in range(args.n_repetitions):
        mask = (
            torch.rand(
                shape.batch, shape.hq, num_q_blocks, num_kv_blocks, device=device
            )
            > args.block_sparsity
        ).to(torch.bool)
        lut = block_attn_mask_to_ragged_lut(mask, return_none_if_dense=True)

        fn = make_kernel_runner(args, q, k, v, block_lut=lut)
        ms = triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep)
        latencies_ms.append(ms)

        dense_tflops = (total_flops / (ms * 1e-3)) / 1e12
        tflops_dense.append(dense_tflops)

        sparse_flops, _ = sparse_flops_from_lut(args.kernel, lut, shape)
        effective_tflops = (sparse_flops / (ms * 1e-3)) / 1e12
        tflops_effective.append(effective_tflops)

    def stats(x: List[float]) -> Dict[str, float]:
        t = torch.tensor(x)
        return {
            "median": torch.quantile(t, 0.5).item(),
            "q1": torch.quantile(t, 0.25).item(),
            "q3": torch.quantile(t, 0.75).item(),
            "p10": torch.quantile(t, 0.1).item(),
            "p90": torch.quantile(t, 0.9).item(),
        }

    st_dense = stats(tflops_dense)
    st_lat = stats(latencies_ms)
    st_eff = stats(tflops_effective)

    summary = (
        f"kernel={args.kernel}, block_sparsity={args.block_sparsity}, n_repetitions={args.n_repetitions}: "
        f"median_TFLOPS={st_dense['median']:.4f}, Q1={st_dense['q1']:.4f}, Q3={st_dense['q3']:.4f}, "
        f"p10={st_dense['p10']:.4f}, p90={st_dense['p90']:.4f} | "
        f"median_latency_ms={st_lat['median']:.4f}, Q1={st_lat['q1']:.4f}, Q3={st_lat['q3']:.4f}, "
        f"p10={st_lat['p10']:.4f}, p90={st_lat['p90']:.4f} | "
        f"median_effective_TFLOPS={st_eff['median']:.4f}, Q1={st_eff['q1']:.4f}, "
        f"Q3={st_eff['q3']:.4f}, p10={st_eff['p10']:.4f}, p90={st_eff['p90']:.4f}"
    )
    logger.info(summary)
    print(summary)

    if args.o:
        csv_path = "bench_sage_block_sparse_repetitions.csv"
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "kernel",
                        "BATCH",
                        "HQ",
                        "N_CTX_Q",
                        "N_CTX_K",
                        "D_HEAD",
                        "D_HEAD_V",
                        "block_sparsity",
                        "n_repetitions",
                        "median_TFLOPS",
                        "q1_TFLOPS",
                        "q3_TFLOPS",
                        "p10_TFLOPS",
                        "p90_TFLOPS",
                        "median_latency_ms",
                        "q1_latency_ms",
                        "q3_latency_ms",
                        "p10_latency_ms",
                        "p90_latency_ms",
                        "median_effective_TFLOPS",
                        "q1_effective_TFLOPS",
                        "q3_effective_TFLOPS",
                        "p10_effective_TFLOPS",
                        "p90_effective_TFLOPS",
                    ]
                )
            writer.writerow(
                [
                    args.kernel,
                    shape.batch,
                    shape.hq,
                    shape.n_ctx_q,
                    shape.n_ctx_k,
                    shape.d_head,
                    shape.d_head_v,
                    args.block_sparsity,
                    args.n_repetitions,
                    st_dense["median"],
                    st_dense["q1"],
                    st_dense["q3"],
                    st_dense["p10"],
                    st_dense["p90"],
                    st_lat["median"],
                    st_lat["q1"],
                    st_lat["q3"],
                    st_lat["p10"],
                    st_lat["p90"],
                    st_eff["median"],
                    st_eff["q1"],
                    st_eff["q3"],
                    st_eff["p10"],
                    st_eff["p90"],
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified SAGE attention benchmark (FAv3, MXFP4, AITER, FP8)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--kernel",
        type=str,
        default="sage_fp8",
        choices=[
            "sage_fp8",
            "sage_mxfp4",
            "fav3_fp8",
            "aiter_fp8",
            "aiter_i8fp8",
            "aiter_mxfp4",
            "aiter_mxfp6",
            "aiter_bf16",
            "all",
        ],
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
        "--input-distribution",
        type=str,
        default="transformer",
        choices=["normal", "transformer", "sink", "underflow", "latesink"],
        help=(
            "Distribution used for generated Q/K/V tensors. 'sink' is a realistic "
            "StreamingLLM attention sink pattern; 'underflow'/'latesink' are "
            "adversarial fp8 tile-skip / frozen-max rollback regression tripwires."
        ),
    )
    parser.add_argument(
        "--qk-clip",
        type=float,
        default=1.0,
        help="Clip factor applied to Q and K absmax before int8 quantization for aiter_i8fp8",
    )
    parser.add_argument(
        "--q-clip",
        type=float,
        default=None,
        help="Optional Q-only absmax clip factor for aiter_i8fp8; overrides --qk-clip for Q",
    )
    parser.add_argument(
        "--k-clip",
        type=float,
        default=None,
        help="Optional K-only absmax clip factor for aiter_i8fp8; overrides --qk-clip for K",
    )
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
        "--ref",
        type=str,
        default="aiter_bf16",
        choices=["torch", "aiter_bf16"],
        help="Reference kernel for accuracy metrics/checks. --kernel=all reports MAE/MaxE/Cosine against this reference.",
    )
    parser.add_argument(
        "--compare-to-ref",
        action="store_true",
        help="Run correctness checks against the selected --ref",
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
        "--block-sparsity",
        type=float,
        default=None,
        help="Random block sparsity ratio in [0,1]",
    )
    parser.add_argument(
        "--block-mask-file",
        type=str,
        default=None,
        help="JSON file with block masks; takes precedence over --block-sparsity",
    )
    parser.add_argument(
        "--n-repetitions",
        type=int,
        default=None,
        help="With random block sparsity: run repeated masks and report quantiles",
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

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed torch RNG before generating Q/K/V so runs are reproducible "
        "(use the same --seed across kernels to compare on identical inputs)",
    )

    args = parser.parse_args()
    for name in (
        "qk_clip",
        "q_clip",
        "k_clip",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be > 0")
    return args


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


def benchmark_all_kernel_row(
    args: argparse.Namespace,
    kernel_name: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    total_flops: float,
    ref_primary: Optional[torch.Tensor],
) -> AllKernelRow:
    saved_kernel = args.kernel
    args.kernel = kernel_name
    try:
        fn = make_kernel_runner(args, q, k, v, block_lut=None)
        ms = triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep)
        tflops = total_flops / ms * 1e-9
        accuracy = None
        if ref_primary is not None:
            current_primary = primary_output(fn())
            current_primary = to_bshd_output_if_needed(current_primary, args.layout)
            accuracy = compute_accuracy_metrics(current_primary, ref_primary)
        return AllKernelRow(kernel_name, ms, tflops, accuracy)
    finally:
        args.kernel = saved_kernel


def skipped_all_kernel_row(kernel_name: str) -> AllKernelRow:
    return AllKernelRow(kernel_name, float("nan"), float("nan"), None)


def print_all_kernel_table(
    rows: List[AllKernelRow],
    include_accuracy: bool,
) -> None:
    if not include_accuracy:
        print(f"{'kernel':<16} {'time(ms)':>10} {'TFLOPS':>10}")
        print("-" * 38)
        for row in rows:
            if row.ms != row.ms:  # nan
                print(f"{row.kernel:<16} {'SKIP':>10} {'SKIP':>10}")
            else:
                print(f"{row.kernel:<16} {row.ms:>10.4f} {row.tflops:>10.2f}")
        return

    print(
        f"{'kernel':<16} {'time(ms)':>10} {'TFLOPS':>10} {'MAE':>12} {'MaxE':>12} {'Cosine':>12}"
    )
    print("-" * 78)
    for row in rows:
        if row.ms != row.ms or row.accuracy is None:  # nan or failed accuracy run
            print(
                f"{row.kernel:<16} {'SKIP':>10} {'SKIP':>10} {'SKIP':>12} {'SKIP':>12} {'SKIP':>12}"
            )
        else:
            print(
                f"{row.kernel:<16} {row.ms:>10.4f} {row.tflops:>10.2f} "
                f"{row.accuracy.mae:>12.3e} {row.accuracy.maxe:>12.3e} "
                f"{row.accuracy.cosine:>12.6f}"
            )


def run_all_kernels(args: argparse.Namespace) -> None:
    """Run all backends on the same QKV inputs and print a comparison table."""
    dtype = arg_to_torch_dtype[args.dtype]
    device = "cuda"
    hk = args.hk if args.hk else args.hq
    sk = args.sk if args.sk else args.sq
    d_head = args.d if args.d else 128
    d_head_v = args.dv if args.dv else d_head

    q, k, v = generate_test_tensors(
        args.b,
        args.hq,
        hk,
        args.sq,
        sk,
        d_head,
        d_head_v,
        dtype,
        device,
        args.input_distribution,
    )
    q.requires_grad = False
    k.requires_grad = False
    v.requires_grad = False
    q, k, v = layout_preprocess(q, k, v, layout="bhsd", target_layout=args.layout)

    shape = infer_shape_spec(q, v, args.layout)
    ref_primary = make_reference_output(args, q, k, v, block_attn_mask=None).float()
    total_flops = (
        2.0
        * shape.batch
        * shape.hq
        * shape.n_ctx_q
        * shape.n_ctx_k
        * (shape.d_head + shape.d_head_v)
    )

    rows: List[AllKernelRow] = []

    for kernel_name in ALL_KERNELS:
        try:
            rows.append(
                benchmark_all_kernel_row(
                    args,
                    kernel_name,
                    q,
                    k,
                    v,
                    total_flops,
                    ref_primary,
                )
            )
        except Exception as e:
            logger.warning("Skipping %s: %s", kernel_name, e)
            rows.append(skipped_all_kernel_row(kernel_name))

    print(
        f"\nbench_sage --kernel=all  (b={args.b} hq={args.hq} sq={args.sq} sk={sk} d={d_head} input={args.input_distribution}):"
    )
    print_all_kernel_table(rows, include_accuracy=True)


def run_with_optional_vgpr(args: argparse.Namespace, runner: Any) -> int:
    if args.print_vgpr:
        print_vgpr_from_bench(runner)
    else:
        runner()
    return 0


def main() -> int:
    args = parse_args()
    validate_args(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

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

    if (
        args.block_sparsity is not None
        and args.n_repetitions is not None
        and args.block_mask_file is None
    ):
        return run_with_optional_vgpr(
            args,
            lambda: run_block_sparse_repetitions(args, loaded_single_mask),
        )

    if args.load_captured:

        def default_runner():
            run_benchmark_captured(args, loaded_single_mask)

    else:

        def default_runner():
            run_benchmark_generated(args, loaded_single_mask)

    return run_with_optional_vgpr(args, default_runner)


if __name__ == "__main__":
    sys.exit(main())
