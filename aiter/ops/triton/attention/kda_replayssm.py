# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Checkpoint replay and fused KDA verification on gfx1250."""

import functools

import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx1250.attention.kda_replayssm import (
    _replayssm_kda_fwd_cta_kernel,
)
from aiter.ops.triton._triton_kernels.kimi_delta_attn.fast_launch import fast_launch
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir

_launch = fast_launch(_replayssm_kda_fwd_cta_kernel)


@functools.lru_cache(maxsize=8)
def _config(arch, cu_num):
    # Include the CU count to keep device-specific config trees separate.
    path = resolve_config_dir("attention", "kda_replayssm", backend="gluon")
    config = load_config_json(f"{path}/DEFAULT.json")
    if config["TPW_K"] != 16 or config["num_warps"] != 8:
        raise ValueError("ReplaySSM requires 16 K lanes and 8 warps")
    return config


def replayssm_sigmoid_gating_delta_rule(
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    ckpt,
    buf_k,
    buf_u,
    buf_g,
    write_pos,
    slot_idx,
    cu_seqlens,
    max_query_len,
    o=None,
    scale=None,
    use_qk_l2norm_in_kernel=False,
    lower_bound=None,
):
    """Replay committed state and process a variable-length KDA token batch.

    q/k have shape [1, tokens, H, 128]. v/a have shape
    [1, tokens, HV, 128], b has shape [1, tokens, HV], and HV is a
    multiple of H. A_log and dt_bias contain HV and HV*128 FP32 values.
    Activations may use BF16, FP16 or FP32. K and V must both be 128.

    ckpt is FP32 [slots, HV, 128, 128]. buf_k, buf_u and buf_g are
    [slots, HV, capacity, 128] records. k/u records may use BF16, FP16
    or FP32; g records use FP32. The last record dimension is contiguous.
    write_pos is int32 [slots], slot_idx is int32 [requests], and
    cu_seqlens is int32 [requests+1]. All tensors use the current GPU.

    The caller supplies valid sequence bounds, distinct live slots and
    cursors in [-1, capacity-max_query_len]. Each sequence has at most
    max_query_len tokens. capacity must be at least 2*max_query_len,
    and the padded history tile must fit 64 rows. A negative slot or
    empty sequence writes nothing. Slot zero is valid.

    This call updates ckpt only on flush and appends k/u/g records.
    It does not commit tokens or advance write_pos. Return output with
    shape [1, tokens, HV, 128]. o may have that shape or omit dimension
    zero. Skipped output rows keep their supplied values; without o,
    those rows are undefined. This entry requires gfx1250.
    """
    arch = get_arch()
    if arch not in ("gfx1250", "gfx1250-strict"):
        raise NotImplementedError("ReplaySSM KDA requires gfx1250")
    if q.ndim != 4 or q.shape[0] != 1 or not q.is_cuda:
        raise ValueError("q must be a GPU tensor with shape [1, tokens, H, K]")
    _, tokens, H, K = q.shape
    if v.ndim != 4 or v.shape[:2] != (1, tokens):
        raise ValueError("v must have shape [1, tokens, HV, V]")
    HV, V = v.shape[2:]
    if K != 128 or V != 128 or H <= 0 or HV <= 0 or HV % H:
        raise NotImplementedError("ReplaySSM supports K=V=128 and HV divisible by H")
    if not isinstance(max_query_len, int) or max_query_len < 1:
        raise ValueError("max_query_len must be a positive integer")
    if (
        k.shape != q.shape
        or a.shape != (1, tokens, HV, K)
        or b.shape != (1, tokens, HV)
    ):
        raise ValueError("k, a and b shapes must match the token and head counts")
    floating = (torch.bfloat16, torch.float16, torch.float32)
    if any(x.dtype not in floating for x in (q, k, v, a, b, buf_k, buf_u)):
        raise ValueError("activations and k/u records must use BF16, FP16 or FP32")
    if any(x.dtype != torch.float32 for x in (A_log, dt_bias, ckpt, buf_g)):
        raise ValueError("gates, checkpoint and g records must use FP32")
    if ckpt.ndim != 4 or ckpt.shape[1:] != (HV, V, K):
        raise ValueError("ckpt must have shape [slots, HV, V, K]")
    slots = ckpt.shape[0]
    if slots == 0 or ckpt.stride()[1:] != (V * K, K, 1):
        raise ValueError("ckpt needs at least one slot and contiguous head matrices")
    if buf_k.ndim != 4:
        raise ValueError("records must have shape [slots, HV, capacity, 128]")
    cap = buf_k.shape[2]
    if cap < 2 * max_query_len:
        raise ValueError("record capacity must be at least 2*max_query_len")
    BH = max(16, triton.next_power_of_2(cap - max_query_len))
    if BH > 64:
        raise NotImplementedError("ReplaySSM supports history tiles up to 64 rows")
    if any(
        x.shape != (slots, HV, cap, K) or x.stride(-1) != 1
        for x in (buf_k, buf_u, buf_g)
    ):
        raise ValueError("record shapes or inner strides do not match the checkpoint")
    if A_log.numel() != HV or dt_bias.numel() != HV * K:
        raise ValueError("A_log and dt_bias must contain HV and HV*K values")
    if not A_log.is_contiguous() or not dt_bias.is_contiguous():
        raise ValueError("A_log and dt_bias must be contiguous")
    if (
        cu_seqlens.ndim != 1
        or slot_idx.ndim != 1
        or cu_seqlens.numel() != slot_idx.numel() + 1
    ):
        raise ValueError("sequence bounds and slot indices must be one-dimensional")
    if write_pos.shape != (slots,):
        raise ValueError("write_pos must contain one cursor per slot")
    if any(
        x.dtype != torch.int32 or not x.is_contiguous()
        for x in (write_pos, slot_idx, cu_seqlens)
    ):
        raise ValueError("metadata must use contiguous int32 tensors")
    tensors = (
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        ckpt,
        buf_k,
        buf_u,
        buf_g,
        write_pos,
        slot_idx,
        cu_seqlens,
    )
    if q.device.index != torch.cuda.current_device() or any(
        x.device != q.device for x in tensors
    ):
        raise ValueError("all tensors must use the current GPU")
    out = q.new_empty((1, tokens, HV, V)) if o is None else o
    if out.shape == (tokens, HV, V):
        out = out.unsqueeze(0)
    if (
        out.shape != (1, tokens, HV, V)
        or out.device != q.device
        or out.dtype not in floating
        or not out.is_contiguous()
    ):
        raise ValueError(
            "o must be a contiguous floating-point output on the input GPU"
        )
    if slot_idx.numel() == 0:
        return out
    q, k, v, a = (x.contiguous() for x in (q, k, v, a))
    properties = torch.cuda.get_device_properties(q.device)
    cfg = _config(arch, properties.multi_processor_count)
    BT = min(triton.next_power_of_2(max_query_len), cfg["max_token_tile"])
    _launch[(slot_idx.numel() * HV,)](
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        ckpt=ckpt,
        buf_k=buf_k,
        buf_u=buf_u,
        buf_g=buf_g,
        write_pos=write_pos,
        slot_idx=slot_idx,
        cu_seqlens=cu_seqlens,
        scale=K**-0.5 if scale is None else scale,
        LOWER_BOUND=0.0 if lower_bound is None else lower_bound,
        T_MAX=max_query_len,
        CAP=cap,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BH=BH,
        BT=BT,
        stride_ckpt_slot=ckpt.stride(0),
        stride_bufk_slot=buf_k.stride(0),
        stride_bufk_pos=buf_k.stride(2),
        stride_bufk_hv=buf_k.stride(1),
        stride_bufu_slot=buf_u.stride(0),
        stride_bufu_pos=buf_u.stride(2),
        stride_bufu_hv=buf_u.stride(1),
        stride_bufg_slot=buf_g.stride(0),
        stride_bufg_pos=buf_g.stride(2),
        stride_bufg_hv=buf_g.stride(1),
        stride_a_token=a.stride(1),
        stride_b_token=b.stride(1),
        stride_b_head=b.stride(2),
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        USE_LOWER_BOUND=lower_bound is not None,
        TPW_K=cfg["TPW_K"],
        WARPS=cfg["num_warps"],
        FUSED=cfg["fused_loads"],
        num_warps=cfg["num_warps"],
        waves_per_eu=cfg["waves_per_eu"],
    )
    return out
