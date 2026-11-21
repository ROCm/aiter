# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused MLA qkv_a projection + q/kv latent RMSNorm for FP8 block-scaled weights.

One call replaces the decode chain ``per-1x128 FP8 quant -> gemm_a8w8_blockscale_bpreshuffle
-> fused_qk_rmsnorm`` of an MLA attention layer with a fused ``q_a_proj``/``kv_a_proj_with_mqa``
weight (GLM-5.x, DeepSeek-V3 style). The output is the BF16 ``q_lora`` after ``q_a_layernorm``,
the BF16 ``kv_lora`` after ``kv_a_layernorm`` and the BF16 ``k_pe`` (rope) slice.

The kernels are Gluon split-K schedules for small decode batches (M = 1, 2, 4, 8, 16, 32)
on gfx950, tuned for the GLM-5.2 FP8 shape. Each M has one schedule in
``configs/gfx950/gluon/fusions/fused_mla_qkv_a_proj_rmsnorm/DEFAULT.json``. The split-K
partials are FP32.

Support is narrow and is reported by :func:`fused_mla_qkv_a_proj_rmsnorm_supported`.
A caller probes it once and keeps the three-kernel chain as the fallback. Two hard gates:

* gfx950 only: the kernels use CDNA4 buffer addressing, ``mfma_scaled`` and explicit
  register layouts.
* Triton >= 3.8: the kernels are written in the Gluon dialect. On an older Triton the gate
  returns ``False`` and this module stays importable (the Gluon kernels are imported lazily).
"""

import functools
import re

import torch
import triton

from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

__all__ = [
    "fused_mla_qkv_a_proj_rmsnorm",
    "fused_mla_qkv_a_proj_rmsnorm_supported",
]

_CONFIG_NAME = "FUSED_MLA_QKV_A_PROJ_RMSNORM"
_GROUP = 128  # FP8 quant group of the activation and block size of the weight scales
_WEIGHT_DTYPE = torch.float8_e4m3fn


def _arch_supported() -> tuple[bool, str]:
    """gfx950 probe; a detection failure is reported as unsupported."""
    try:
        from aiter.ops.triton.utils._triton.arch_info import get_arch

        arch = get_arch()
    except Exception as exc:  # noqa: BLE001
        return False, f"architecture detection failed ({exc})"
    if arch != "gfx950":
        return False, f"gfx950 only, got {arch}"
    return True, ""


@functools.lru_cache(maxsize=1)
def _gluon_supported() -> tuple[bool, str]:
    """Cached probe: Triton >= 3.8 with the Gluon dialect (mirrors fused_gdn_prefill_qkvz)."""
    try:
        import triton
    except ImportError as exc:
        return False, f"triton not importable ({exc})"
    version = triton.__version__
    matched = re.match(r"(\d+)\.(\d+)", version or "")
    if matched is None or (int(matched.group(1)), int(matched.group(2))) < (3, 8):
        return False, f"Triton >= 3.8 required for this Gluon dialect, got {version}"
    try:
        import triton.experimental.gluon
    except ImportError:
        return False, f"triton.experimental.gluon unavailable (Triton {version})"
    return True, ""


@functools.cache
def _schedule(m: int) -> dict | None:
    """Schedule of batch size ``m`` from the config file, or ``None`` when M has no row."""
    cfg_dir = resolve_config_dir("fusions", _CONFIG_NAME, backend="gluon")
    return load_config_json(f"{cfg_dir}/DEFAULT.json").get(f"M_{m}")


def _k_multiple(schedule: dict) -> int:
    """K must be a multiple of this value: the K groups one split-K step consumes."""
    if schedule["producer"] == "chains":
        return _GROUP * schedule["chains"] * schedule["splits"]
    return _GROUP * schedule["groups_per_wave"] * schedule["waves"]


def fused_mla_qkv_a_proj_rmsnorm_supported(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    rope_dim: int = 64,
) -> tuple[bool, str]:
    """Report whether this call is covered, and if not, why.

    Returns ``(True, "")`` or ``(False, reason)``. Every consumed tensor is checked (shape,
    dtype, device, layout): the kernels use fixed-stride addressing, so a permissive gate
    would let them read out of bounds. The reason is meant to be logged once by the caller
    on its fallback path.
    """
    arch_ok, arch_reason = _arch_supported()
    if not arch_ok:
        return False, arch_reason
    gluon_ok, gluon_reason = _gluon_supported()
    if not gluon_ok:
        return False, gluon_reason

    named = {
        "hidden": hidden,
        "weight": weight,
        "weight_scale": weight_scale,
        "q_norm_weight": q_norm_weight,
        "kv_norm_weight": kv_norm_weight,
    }
    for name, t in named.items():
        if not isinstance(t, torch.Tensor) or not t.is_cuda:
            return False, f"{name} must be a CUDA tensor"
    if any(t.device != hidden.device for t in named.values()):
        return False, "all inputs must be on the same device"
    if hidden.ndim != 2 or hidden.dtype is not torch.bfloat16 or hidden.stride(1) != 1:
        return False, "hidden must be a bf16 [M, K] tensor with stride(1) == 1"
    if (
        weight.ndim != 2
        or weight.dtype is not _WEIGHT_DTYPE
        or not weight.is_contiguous()
    ):
        return False, f"weight must be a contiguous {_WEIGHT_DTYPE} [N, K] tensor"
    if q_norm_weight.ndim != 1 or kv_norm_weight.ndim != 1:
        return False, "q_norm_weight and kv_norm_weight must be 1-D"
    if (
        q_norm_weight.dtype is not torch.bfloat16
        or kv_norm_weight.dtype is not torch.bfloat16
    ):
        return False, "q_norm_weight and kv_norm_weight must be bf16"
    if not (q_norm_weight.is_contiguous() and kv_norm_weight.is_contiguous()):
        return False, "q_norm_weight and kv_norm_weight must be contiguous"

    m, k = hidden.shape
    n = weight.shape[0]
    q, kv = q_norm_weight.numel(), kv_norm_weight.numel()
    if weight.shape[1] != k:
        return False, f"weight K {weight.shape[1]} != hidden K {k}"
    if not (0 < rope_dim <= 128) or q + kv + rope_dim != n:
        return False, (
            f"weight N {n} != q {q} + kv {kv} + rope_dim {rope_dim} (0 < rope_dim <= 128)"
        )
    if k % _GROUP != 0 or q % _GROUP != 0 or kv % _GROUP != 0:
        return False, f"K, q and kv widths must be multiples of {_GROUP}"
    scale_shape = (triton.cdiv(n, _GROUP), k // _GROUP)
    if (
        weight_scale.dtype is not torch.float32
        or tuple(weight_scale.shape) != scale_shape
        or not weight_scale.is_contiguous()
    ):
        return False, f"weight_scale must be a contiguous fp32 {scale_shape} tensor"

    schedule = _schedule(m)
    if schedule is None:
        return False, f"no schedule for M={m} (covered: 1, 2, 4, 8, 16, 32)"
    if k % _k_multiple(schedule) != 0 or n % schedule["tile_n"] != 0:
        return False, (
            f"K {k} must be a multiple of {_k_multiple(schedule)} and N {n} a multiple of "
            f"{schedule['tile_n']} for M={m}"
        )
    return True, ""


def _run_small(
    hidden, weight, weight_scale, q_norm_weight, kv_norm_weight, rope_dim, eps, cfg
):
    from aiter.ops.triton._gluon_kernels.gfx950.fusions.fused_mla_qkv_a_proj_rmsnorm import (
        _finish,
        _project_native,
        _project_stream,
        _project_two,
    )

    producers = {
        "native": _project_native,
        "two": _project_two,
        "stream": _project_stream,
    }
    m, k = hidden.shape
    q, kv = q_norm_weight.numel(), kv_norm_weight.numel()
    n = q + kv + rope_dim
    parts = k // (_GROUP * cfg["groups_per_wave"] * cfg["waves"])
    pitch = triton.cdiv(n, cfg["scratch_alignment"]) * cfg["scratch_alignment"]
    device = hidden.device
    partial = torch.empty((parts, m, pitch), dtype=torch.float32, device=device)
    outputs = torch.empty((m * n,), dtype=torch.bfloat16, device=device)
    q_out = outputs[: m * q].view(m, q)
    kv_out = outputs[m * q : m * (q + kv)].view(m, kv)
    rope_out = outputs[m * (q + kv) :].view(m, rope_dim)

    options = {}
    if cfg["waves_per_eu"]:
        options["waves_per_eu"] = cfg["waves_per_eu"]
    if cfg["fragment_n"]:
        options["FN"] = cfg["fragment_n"]
    producers[cfg["producer"]][n // cfg["tile_n"], parts](
        hidden,
        weight,
        weight_scale,
        partial,
        m,
        pitch,
        k,
        hidden.stride(0),
        cfg["tile_n"],
        cfg["groups_per_wave"],
        cfg["waves"],
        cfg["activation_vec"],
        num_warps=cfg["waves"],
        enable_fp_fusion=cfg["waves"] != 1,
        **options,
    )
    _finish[m, cfg["q_shards"] + cfg["kv_shards"]](
        partial,
        q_norm_weight,
        kv_norm_weight,
        q_out,
        kv_out,
        rope_out,
        m,
        pitch,
        q,
        kv,
        rope_dim,
        eps,
        parts,
        cfg["q_shards"],
        cfg["kv_shards"],
        triton.next_power_of_2(q),
        triton.next_power_of_2(kv if cfg["narrow_rope"] else kv + rope_dim),
        triton.next_power_of_2(rope_dim),
        cfg["narrow_rope"],
        cfg["buffer_q_gamma"],
        cfg["buffer_q_output"],
        cfg["kv_vec"],
        num_warps=4,
        enable_fp_fusion=False,
    )
    return q_out, kv_out, rope_out


def _run_chains(
    hidden, weight, weight_scale, q_norm_weight, kv_norm_weight, rope_dim, eps, cfg
):
    from aiter.ops.triton._gluon_kernels.gfx950.fusions.fused_mla_qkv_a_proj_rmsnorm import (
        _m32_finish,
        _m32_project,
    )

    m, k = hidden.shape
    q, kv = q_norm_weight.numel(), kv_norm_weight.numel()
    n = q + kv + rope_dim
    device = hidden.device
    partial = torch.empty((m, cfg["splits"], n), dtype=torch.float32, device=device)
    outputs = torch.empty((m * n,), dtype=torch.bfloat16, device=device)
    q_out = outputs[: m * q].view(m, q)
    kv_out = outputs[m * q : m * (q + kv)].view(m, kv)
    rope_out = outputs[m * (q + kv) :].view(m, rope_dim)
    _m32_project[cfg["splits"], n // cfg["tile_n"]](
        hidden,
        weight,
        weight_scale,
        partial,
        m,
        n,
        k,
        hidden.stride(0),
        cfg["tile_n"],
        cfg["splits"],
        cfg["chains"],
        cfg["fold_vec"],
        cfg["fold_row_lanes"],
        tuple(cfg["fold_order"]),
        num_warps=cfg["num_warps"],
        waves_per_eu=cfg["waves_per_eu"],
        enable_fp_fusion=True,
    )
    _m32_finish[m, 2](
        partial,
        q_norm_weight,
        kv_norm_weight,
        q_out,
        kv_out,
        rope_out,
        n,
        q,
        kv,
        rope_dim,
        eps,
        cfg["splits"],
        cfg["finish_q_vec"],
        cfg["finish_kv_vec"],
        triton.next_power_of_2(q),
        triton.next_power_of_2(kv + rope_dim),
        num_warps=cfg["finish_num_warps"],
        enable_fp_fusion=False,
    )
    return q_out, kv_out, rope_out


def fused_mla_qkv_a_proj_rmsnorm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    *,
    rope_dim: int = 64,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused qkv_a projection + q/kv RMSNorm of an MLA layer, FP8 block-scaled weight.

    Computes ``y = quant_1x128(hidden) @ weight^T`` with the 128x128 weight scales, rounds
    ``y`` to BF16, and returns ``(rmsnorm(y[:, :q]) * q_norm_weight, rmsnorm(y[:, q:q+kv])
    * kv_norm_weight, y[:, q+kv:])``. The activation quant is dynamic per (row, 128 group)
    FP8 e4m3 with ``scale = amax / 448``, as ``get_hip_quant(QuantType.per_1x128)``.

    Args:
        hidden: bf16 ``[M, K]`` activation (``stride(1) == 1``).
        weight: fp8 e4m3fn ``[N, K]`` fused ``q_a | kv_a | k_pe`` weight, permuted by
            ``shuffle_weight(w, layout=(16, 16))``.
        weight_scale: fp32 ``[ceil(N / 128), K / 128]`` block scales (row-major).
        q_norm_weight: bf16 ``[q_lora_rank]`` RMSNorm weight.
        kv_norm_weight: bf16 ``[kv_lora_rank]`` RMSNorm weight.
        rope_dim: width of the k_pe slice (keyword-only).
        eps: RMSNorm epsilon of both norms (keyword-only).

    Returns:
        ``(q_lora, kv_lora, k_pe)``: bf16 ``[M, q_lora_rank]``, ``[M, kv_lora_rank]`` and
        ``[M, rope_dim]``.

    Limits: gfx950, Triton >= 3.8, M in {1, 2, 4, 8, 16, 32}; see
    :func:`fused_mla_qkv_a_proj_rmsnorm_supported`.

    Raises:
        ValueError: if the call is not supported.
    """
    ok, reason = fused_mla_qkv_a_proj_rmsnorm_supported(
        hidden, weight, weight_scale, q_norm_weight, kv_norm_weight, rope_dim
    )
    if not ok:
        raise ValueError(
            f"fused_mla_qkv_a_proj_rmsnorm does not support this call: {reason}"
        )
    _LOGGER.info(
        "FUSED_MLA_QKV_A_PROJ_RMSNORM: hidden=%s weight=%s rope_dim=%d",
        tuple(hidden.shape),
        tuple(weight.shape),
        rope_dim,
    )
    cfg = _schedule(hidden.shape[0])
    run = _run_chains if cfg["producer"] == "chains" else _run_small
    return run(
        hidden, weight, weight_scale, q_norm_weight, kv_norm_weight, rope_dim, eps, cfg
    )
