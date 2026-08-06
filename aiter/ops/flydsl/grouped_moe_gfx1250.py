# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 grouped MoE GEMM (a8w4 / a4w4) via FlyDSL.

Supports optional DeepGEMM-style contiguous-M scheduler
(AITER_GROUPED_DEEPGEMM_CONTIGUOUS=1 or CSV grouped_contiguous_m=1).
"""

import csv
import functools
import os

import torch

from aiter import ActivationType, QuantType, dtypes, logger
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

# Opt-in switch for the gfx1250 FlyDSL grouped-GEMM path.
_TRUTHY_ENV = ("1", "true", "True", "yes", "YES")
_GROUPED_CONFIG_CACHE = {}
_WARNED_NAIVE_EPILOGUE = False
# Cache the contiguous uint8 view of static MoE weights so a non-contiguous
# weight is materialized at most once (not re-copied on every fused_moe call).
_GROUPED_WEIGHT_CACHE = {}

# Opt-in kernel-bench hook: a caller sets a list here to collect
# (name, callable) per-kernel launches; None in production.
kernel_bench_callable = None


def _grouped_weight_uint8(w: torch.Tensor) -> torch.Tensor:
    """Contiguous uint8 view of a static MoE weight, cached by data_ptr."""
    key = (w.data_ptr(), tuple(w.shape), tuple(w.stride()), str(w.dtype))
    cached = _GROUPED_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    out = (w if w.dtype == torch.uint8 else w.view(torch.uint8)).contiguous()
    if len(_GROUPED_WEIGHT_CACHE) > 64:
        _GROUPED_WEIGHT_CACHE.clear()
    _GROUPED_WEIGHT_CACHE[key] = out
    return out


def _as_bool(value, default: bool) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip() in _TRUTHY_ENV


def _as_int(value, default: int | None) -> int | None:
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def _dtype_name(dtype) -> str:
    if dtype is torch.bfloat16 or dtype == dtypes.bf16:
        return "torch.bfloat16"
    if dtype is torch.float16 or dtype == dtypes.fp16:
        return "torch.float16"
    return str(dtype)


def _enum_name(value) -> str:
    if hasattr(value, "name"):
        return f"{type(value).__name__}.{value.name}"
    return str(value)


def _load_grouped_config_rows():
    cfg_path = os.environ.get("AITER_CONFIG_GROUPED_FMOE")
    if not cfg_path:
        try:
            from aiter.jit.core import AITER_CONFIGS

            cfg_path = AITER_CONFIGS.AITER_CONFIG_GROUPED_FMOE_FILE
        except Exception:  # noqa: BLE001
            cfg_path = ""
    cached = _GROUPED_CONFIG_CACHE.get(cfg_path)
    if cached is not None:
        return cached
    rows = []
    for path in str(cfg_path).split(os.pathsep):
        if not path or not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    _GROUPED_CONFIG_CACHE[cfg_path] = rows
    return rows


def _next_pow2(n):
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


_PADDED_M_TIERS = [32768, 131072]


def _get_padded_m(m):
    if m < _PADDED_M_TIERS[0]:
        return _next_pow2(m)
    for tier in reversed(_PADDED_M_TIERS):
        if m >= tier:
            return tier
    return _PADDED_M_TIERS[0]


@functools.lru_cache(maxsize=1024)
def _find_grouped_config(
    *,
    token_num: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    activation,
    dtype,
    q_dtype_a,
    q_dtype_w,
    quant_type,
):
    from aiter.jit.utils.chip_info import get_cu_num

    keys = {
        "gfx": str(get_gfx()),
        "cu_num": str(get_cu_num()),
        "token": str(int(token_num)),
        "model_dim": str(int(model_dim)),
        "inter_dim": str(int(inter_dim)),
        "expert": str(int(experts)),
        "topk": str(int(topk)),
        "act_type": _enum_name(activation),
        "dtype": _dtype_name(dtype),
        "q_dtype_a": str(q_dtype_a),
        "q_dtype_w": str(q_dtype_w),
        "q_type": _enum_name(quant_type),
    }
    rows = _load_grouped_config_rows()

    # Hardware is locked by (gfx, cu_num): gfx (architecture) is always a hard
    # constraint, while cu_num can be relaxed as a fallback. Columns missing from
    # the CSV (e.g. older configs without a 'gfx' column) are skipped, so this
    # stays backward compatible with pre-gfx tuned files.
    def _matches(row, *, require_cu_num: bool):
        for k, v in keys.items():
            if k == "cu_num" and not require_cu_num:
                continue
            if row.get(k) and str(row.get(k)).strip() != v:
                return False
        return True

    matches = [row for row in rows if _matches(row, require_cu_num=True)]
    if not matches:
        matches = [row for row in rows if _matches(row, require_cu_num=False)]
    if not matches:
        if os.environ.get("AITER_GROUPED_DEBUG", "0") not in (
            "",
            "0",
            "false",
            "False",
        ):
            print(
                f"[grouped-gemm-debug] no grouped CSV config match for {keys}; "
                f"loaded_rows={len(rows)}",
                flush=True,
            )
        return None
    matches.sort(key=lambda r: float(r.get("us") or 0.0))
    return matches[0]


def _use_grouped_gemm_enabled() -> bool:
    env_enabled = os.environ.get("AITER_USE_GROUPED_GEMM", "0") in _TRUTHY_ENV
    is_gfx1250 = get_gfx() == "gfx1250"
    return env_enabled or is_gfx1250


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError(f"alignment must be > 0, got {alignment}")
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _make_contiguous_psum_layout(
    *,
    masked_m: torch.Tensor,
    rows_to_tokens: torch.Tensor,
    topids_to_rows: torch.Tensor,
    experts: int,
    max_m: int,
    tile_m: int,
    token_num: int,
    topk: int,
):
    """Build DeepGEMM psum layout.

    ``contiguous_m`` is a static upper bound, which keeps this CUDAGraph-safe.
    """
    device = masked_m.device

    starts_t, psum_t, _ = contiguous_psum(masked_m, int(experts), int(tile_m))
    ub = int(token_num) * int(topk) + int(experts) * int(tile_m) - int(topk)
    contiguous_m = max(int(tile_m), _align_up(ub, int(tile_m)))

    old_flat = topids_to_rows.reshape(-1)
    expert = torch.div(old_flat, int(max_m), rounding_mode="floor")
    slot = old_flat - expert * int(max_m)
    new_flat = starts_t[expert.to(torch.long)] + slot
    remapped_topids = new_flat.to(torch.int32).view_as(topids_to_rows)

    # Inverse map (contiguous row -> source token) via one scatter.
    remapped_rows = torch.full(
        (int(contiguous_m),), -1, device=device, dtype=torch.int32
    )
    src_tokens = rows_to_tokens[old_flat.to(torch.long)]
    remapped_rows[new_flat.to(torch.long)] = src_tokens

    return remapped_topids, remapped_rows, psum_t, int(contiguous_m)


def _grouped_a8w4_preshuffle_e8m0_scale(
    scale: torch.Tensor,
    warp_tile: int,
    scale_k_per_tile: int = 4,
) -> torch.Tensor:
    # Preshuffle row/k-scale axes; experts stay as the leading batch dim.
    scale = scale.view(torch.uint8).contiguous()
    E, _, k_scale = scale.shape
    wmma_rep = int(warp_tile) // 16
    k_groups = k_scale // scale_k_per_tile
    k_wmma_steps = scale_k_per_tile // 4
    g = scale.view(E, -1, wmma_rep, 16, k_groups, k_wmma_steps, 4)
    g = g.permute(0, 1, 3, 4, 5, 2, 6).contiguous()
    return g.reshape(E, -1, k_groups * k_wmma_steps * wmma_rep * 4)


def _grouped_a8w4_prepare_scale_batch(
    scale: torch.Tensor,
    *,
    experts: int,
    rows: int,
    k_dim: int,
    warp_tile: int,
    tile_k: int,
    device: torch.device,
) -> torch.Tensor:
    scale_u8 = scale.view(torch.uint8).contiguous()
    raw_shape = (experts, rows, k_dim // 32)
    wmma_rep = int(warp_tile) // 16
    preshuffled_shape = (experts, rows // wmma_rep, (k_dim // 32) * wmma_rep)
    if tuple(scale_u8.shape) == preshuffled_shape:
        return scale_u8
    if tuple(scale_u8.shape) == (experts * rows, k_dim // 32):
        scale_u8 = scale_u8.view(raw_shape)
    elif tuple(scale_u8.shape) != raw_shape:
        raise ValueError(
            f"scale shape must be raw {raw_shape}, "
            f"flat raw {(experts * rows, k_dim // 32)} "
            f"or preshuffled {preshuffled_shape}, got {tuple(scale_u8.shape)}"
        )
    scale_k_per_tile = int(tile_k) // 32
    return _grouped_a8w4_preshuffle_e8m0_scale(
        scale_u8, warp_tile=warp_tile, scale_k_per_tile=scale_k_per_tile
    ).to(device=device)


@functools.cache
def _get_compiled_g2l_lut():
    """Compile and cache the single-block FlyDSL g2l-LUT builder."""
    from aiter.ops.flydsl.kernels.moe_g2l_lut import build_moe_g2l_lut_module

    return build_moe_g2l_lut_module()


# Single-workgroup scan ceiling (matches moe_g2l_lut.MAX_G2L_EXPERTS); larger
# masks fall back to the torch chain.
_G2L_MAX_N = 512


def _build_g2l_lut(
    expert_mask: torch.Tensor,
    E: int,
    device,
    nvt: torch.Tensor | None = None,
    topk: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Build the EP global->local expert LUT (and zero the route counter).

    Returns ``(g2l_lut, counter, nvr)`` where ``counter`` is the zero-inited
    ``(E,)`` per-bucket route counter produced by the fused kernel (or ``None``
    on the torch fallback, so callers allocate/zero it themselves) and ``nvr`` is
    the ``(1,)`` int32 ``num_valid_routes = nvt * topk`` scalar the same kernel
    computes on-device (or ``None`` when ``nvt``/``topk`` are not supplied or the
    torch fallback is taken, so callers compute it themselves). Folding the
    ``nvt * topk`` into this pre-route kernel removes the standalone torch
    elementwise ``* topk`` launch on the EP decode hot path.

    ``g2l_lut[global_id]`` gives the local bucket in [0, E) for enabled experts
    or the sentinel ``E`` for dropped (non-local) routes. Result is int32 on
    ``device``.

    Fast path: a single FlyDSL kernel (``moe_g2l_lut``) does ``ne + cumsum + sub
    + where`` in one pass -- one launch instead of ~6 elementwise/scan kernels,
    and on the same compiler/runtime as the rest of the gfx1250 grouped path
    (no Triton in the decode hot path). This depends only on ``expert_mask``
    (static per rank) but cannot be memoised here: ``fused_moe`` is dispatched
    through ``torch.ops.aiter.*``, so the op layer hands this function a fresh
    copy of ``expert_mask`` (new object *and* storage) on essentially every call,
    so neither object- nor data_ptr-keyed caching hits. Collapsing the chain into
    one kernel is the portable win; fully removing it would require precomputing
    the LUT at ``expert_mask`` creation and threading it through the op schema.
    """
    n = expert_mask.numel()
    # nvr is only folded in when both the dynamic-token scalar and topk are known.
    _want_nvr = nvt is not None and topk is not None
    if os.environ.get("AITER_G2L_TORCH", "0") not in _TRUTHY_ENV and n <= _G2L_MAX_N:
        try:
            mask = (
                expert_mask.to(device=device, dtype=torch.int32)
                .reshape(-1)
                .contiguous()
            )
            lut = torch.empty(n, dtype=torch.int32, device=device)
            # The kernel also zero-inits this per-bucket route counter (folds the
            # separate host torch.zeros(E) that moe_route_g2l increments).
            counter = torch.empty(E, dtype=torch.int32, device=device)
            # (1,) int32 num_valid_routes = nvt * topk, computed on-device by the
            # same single-block kernel (folds the standalone torch ``* topk``). A
            # valid nvt pointer is always passed so the kernel store is uniform;
            # the result is only surfaced to the caller when nvr was requested.
            nvt_i32 = (
                nvt.reshape(-1)[:1].to(device=device, dtype=torch.int32).contiguous()
                if _want_nvr
                else torch.zeros(1, dtype=torch.int32, device=device)
            )
            nvr = torch.empty(1, dtype=torch.int32, device=device)
            _get_compiled_g2l_lut()(
                ptr_arg(mask),
                ptr_arg(lut),
                ptr_arg(counter),
                ptr_arg(nvt_i32),
                ptr_arg(nvr),
                int(n),
                int(E),
                int(topk) if _want_nvr else 0,
                stream=torch.cuda.current_stream(),
            )
            return lut, counter, (nvr if _want_nvr else None)
        except Exception as exc:  # noqa: BLE001  # pragma: no cover
            logger.debug(
                "[grouped_a8w4] flydsl g2l build unavailable (%s); "
                "falling back to torch",
                exc,
            )
    mask_bool = expert_mask.to(device=device).reshape(-1) != 0
    lut = torch.cumsum(mask_bool.to(torch.int32), 0) - 1
    lut = (
        torch.where(mask_bool, lut, torch.full_like(lut, E))
        .to(torch.int32)
        .contiguous()
    )
    return lut, None, None


def _tdm_align_up(x: int, a: int) -> int:
    return ((int(x) + a - 1) // a) * a


def _grouped_a8w4_tdm_moe(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    *,
    E,
    model_dim,
    inter_dim,
    dtype,
    activation,
    w1_scale,
    w2_scale,
    bias1,
    bias2,
    swiglu_limit,
    doweight_stage1,
    tile_m=64,
    tile_n=256,
    tile_k=256,
    num_buffers=3,
    tile_m2=None,
    tile_n2=None,
    tile_k2=None,
    num_buffers2=None,
    data_format="a8w4",
    expert_mask=None,
    num_local_tokens=None,
    situ_beta=1.0,
    situ_linear_beta=1.0,
):
    import functools

    import torch

    from aiter.ops.flydsl.batched_gemm_mxfp4 import flydsl_grouped_gemm_a8w4_masked
    from aiter.ops.flydsl.moe_kernels import (
        flydsl_moe_fused_quant_preshuffle,
        flydsl_moe_topids_to_rows,
    )

    device = hidden_states.device
    token_num, topk = topk_ids.shape
    if tile_m2 is None:
        tile_m2 = tile_m
    if tile_n2 is None:
        tile_n2 = tile_n
    if tile_k2 is None:
        tile_k2 = tile_k
    if num_buffers2 is None:
        num_buffers2 = num_buffers
    wmma_rep = tile_m // 16
    wmma_rep2 = tile_m2 // 16
    _align_m = max(tile_m, tile_m2)
    contiguous_m = max(
        _align_m, _tdm_align_up(token_num * topk + E * _align_m - topk, _align_m)
    )
    max_m = max(_align_m, _tdm_align_up(token_num * topk, _align_m))

    # Expert-Parallel (EP) wiring. ``topk_ids`` then carry GLOBAL expert ids; the
    # route kernel remaps them to local buckets via ``g2l_lut`` (sentinel E =
    # dropped/non-local route), casts the f32 route weights into ``_gather_w_buf``
    # (kept -> weight_dtype, dropped -> 0), and skips the EP dead-tail (routes >=
    # num_valid_routes / tokens >= num_valid_tokens). Non-local routes claim no
    # per-expert slot, so ``_masked_m`` / ``psum`` -- and the rows the GEMMs
    # actually compute -- cover only the routes this rank owns (~1/ep of what it
    # received). ``contiguous_m`` below stays the static worst case to keep the
    # launch grid CUDAGraph-safe; the surplus tiles fall past ``psum`` and exit
    # early. The felix TDM batched GEMMs themselves are EP-agnostic: they operate
    # on the already-routed contiguous layout.
    _is_ep = expert_mask is not None
    _g2l_lut = None
    _g2l_counter = None
    _gather_w_buf = None
    _ep_nvr = None
    _ep_nvt = None
    if _is_ep:
        if num_local_tokens is not None:
            _ep_nvt = (
                num_local_tokens.reshape(-1)[:1]
                .to(device=device, dtype=torch.int32)
                .contiguous()
            )
        else:
            # torch.full builds directly on-device (fill kernel, no H2D copy), so
            # this stays legal under CUDA graph capture. torch.tensor([...],
            # device=cuda) would allocate a CPU tensor and cudaMemcpy it, which
            # capture rejects unless pinned.
            _ep_nvt = torch.full((1,), int(token_num), dtype=torch.int32, device=device)
        _g2l_lut, _g2l_counter, _g2l_nvr = _build_g2l_lut(
            expert_mask, E, device, nvt=_ep_nvt, topk=int(topk)
        )
        _ep_nvr = (
            _g2l_nvr if _g2l_nvr is not None else (_ep_nvt * int(topk)).contiguous()
        )
        # Route kernel writes every entry (kept -> weight_dtype cast, dropped -> 0),
        # so the buffer is left uninitialised (fully kernel-written).
        _gather_w_buf = torch.empty((token_num, topk), dtype=dtype, device=device)
        _masked_m, topids_to_rows = flydsl_moe_topids_to_rows(
            topk_ids,
            E,
            max_m,
            g2l_lut=_g2l_lut,
            gather_w=_gather_w_buf,
            weight_in=topk_weight,
            counter=_g2l_counter,
            num_local_tokens=num_local_tokens,
            num_valid_routes=_ep_nvr,
        )
    else:
        _masked_m, topids_to_rows = flydsl_moe_topids_to_rows(topk_ids, E, max_m)
    _starts, psum, _ = contiguous_psum_remap(
        _masked_m, topids_to_rows, E, max_m, tile_m, num_valid_routes=_ep_nvr
    )
    psum = psum.to(torch.int32).contiguous()

    out_is_f16 = 1 if (dtype == torch.float16 or dtype == dtypes.fp16) else 0
    two_inter = 2 * inter_dim
    # Stage1 epilogue code: 1 silu, 2 swiglu, 3 SiTUv2. The caller has already
    # rejected anything else.
    if activation == ActivationType.Swiglu:
        stage1_act = 2
    elif activation == ActivationType.Situv2:
        stage1_act = 3
    else:
        stage1_act = 1
    # SiTUv2 is bounded by construction and takes no clamp, so the limit only
    # ever applies to swiglu.
    sl = (
        float(swiglu_limit)
        if swiglu_limit
        else (7.0 if activation == ActivationType.Swiglu else float("inf"))
    )
    _situ_kw = {"situ_beta": situ_beta, "situ_linear_beta": situ_linear_beta}
    _b1 = (
        bias1.to(dtype).contiguous()
        if (bias1 is not None and bias1.numel() > 0)
        else None
    )
    _b2 = (
        bias2.to(dtype).contiguous()
        if (bias2 is not None and bias2.numel() > 0)
        else None
    )
    _is_fp4 = data_format == "fp4"
    _quant_mode = "fp4" if _is_fp4 else "fp8"
    _a_is_fp4 = 1 if _is_fp4 else 0

    a1_payload, a1_scale = flydsl_moe_fused_quant_preshuffle(
        hidden_states.reshape(1, token_num, model_dim),
        1,
        contiguous_m,
        wmma_rep=wmma_rep,
        quant_mode=_quant_mode,
        masked_m=None,
        topids_to_rows=topids_to_rows,
        source_topk=topk,
        num_valid_routes=_ep_nvr,
    )

    # Fuse gemm1 silu/swiglu + fp8 quantization + scale preshuffle into the
    # kernel epilogue (a8w4 only), eliminating the standalone
    # flydsl_moe_fused_quant_preshuffle call between gemm1 and gemm2.
    _fuse_quant = (not _is_fp4) and (_b1 is None)
    w1_u8 = _grouped_weight_uint8(w1)
    w1s_i32 = w1_scale.reshape(-1).view(torch.int32)

    if _fuse_quant:
        # Pre-allocate fp8 payload + preshuffled e8m0 scale for gemm1 output.
        # These are written directly by the kernel's fused quant epilogue.
        payload_bytes = inter_dim  # fp8: 1 byte per element
        scale_bytes = inter_dim // 32  # one e8m0 byte per 32-element MX block
        a2_payload = torch.empty(
            (1, contiguous_m, payload_bytes), dtype=torch.uint8, device=device
        )
        a2_scale = torch.empty(
            (1, contiguous_m // wmma_rep2, scale_bytes * wmma_rep2),
            dtype=torch.uint8,
            device=device,
        )
        # The gemm1 kernel writes fp8 payload to `a2_payload` (passed as
        # `out` / arg_c) and preshuffled e8m0 scale to `a2_scale` (passed via
        # quant_scale / arg_quant_scale).
        flydsl_grouped_gemm_a8w4_masked(
            a2_payload.view(torch.uint8),
            a1_payload,
            w1_u8,
            a1_scale,
            w1s_i32,
            psum,
            n_experts=E,
            contiguous_m=contiguous_m,
            N=two_inter,
            K=model_dim,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            out_is_f16=out_is_f16,
            a_is_fp4=_a_is_fp4,
            stage1_act=stage1_act,
            bias=_b1,
            swiglu_limit=sl,
            num_buffers=num_buffers,
            stage1_quant_out=1,
            quant_scale=a2_scale,
            quant_wmma_rep=wmma_rep2,
            **_situ_kw,
        )
    else:
        # Original path: bf16 intermediate + separate quant kernel.
        y = torch.empty((1, contiguous_m, inter_dim), dtype=dtype, device=device)
        flydsl_grouped_gemm_a8w4_masked(
            y,
            a1_payload,
            w1_u8,
            a1_scale,
            w1s_i32,
            psum,
            n_experts=E,
            contiguous_m=contiguous_m,
            N=two_inter,
            K=model_dim,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            out_is_f16=out_is_f16,
            a_is_fp4=_a_is_fp4,
            stage1_act=stage1_act,
            bias=_b1,
            swiglu_limit=sl,
            num_buffers=num_buffers,
            **_situ_kw,
        )
        a2_payload, a2_scale = flydsl_moe_fused_quant_preshuffle(
            y,
            1,
            contiguous_m,
            wmma_rep=wmma_rep2,
            quant_mode=_quant_mode,
            masked_m=None,
            topids_to_rows=None,
        )

    grouped_out = torch.empty((1, contiguous_m, model_dim), dtype=dtype, device=device)
    w2_u8 = _grouped_weight_uint8(w2)
    w2s_i32 = w2_scale.reshape(-1).view(torch.int32)
    flydsl_grouped_gemm_a8w4_masked(
        grouped_out,
        a2_payload,
        w2_u8,
        a2_scale,
        w2s_i32,
        psum,
        n_experts=E,
        contiguous_m=contiguous_m,
        N=model_dim,
        K=inter_dim,
        tile_m=tile_m2,
        tile_n=tile_n2,
        tile_k=tile_k2,
        out_is_f16=out_is_f16,
        a_is_fp4=_a_is_fp4,
        stage1_act=0,
        bias=_b2,
        num_buffers=num_buffers2,
    )

    if kernel_bench_callable is not None:
        if _fuse_quant:
            kernel_bench_callable.append(
                (
                    "gemm1",
                    functools.partial(
                        flydsl_grouped_gemm_a8w4_masked,
                        a2_payload.view(torch.uint8),
                        a1_payload,
                        w1_u8,
                        a1_scale,
                        w1s_i32,
                        psum,
                        n_experts=E,
                        contiguous_m=contiguous_m,
                        N=two_inter,
                        K=model_dim,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        tile_k=tile_k,
                        out_is_f16=out_is_f16,
                        a_is_fp4=_a_is_fp4,
                        stage1_act=stage1_act,
                        bias=_b1,
                        swiglu_limit=sl,
                        num_buffers=num_buffers,
                        stage1_quant_out=1,
                        quant_scale=a2_scale,
                        quant_wmma_rep=wmma_rep2,
                        **_situ_kw,
                    ),
                )
            )
        else:
            kernel_bench_callable.append(
                (
                    "gemm1",
                    functools.partial(
                        flydsl_grouped_gemm_a8w4_masked,
                        y,
                        a1_payload,
                        w1_u8,
                        a1_scale,
                        w1s_i32,
                        psum,
                        n_experts=E,
                        contiguous_m=contiguous_m,
                        N=two_inter,
                        K=model_dim,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        tile_k=tile_k,
                        out_is_f16=out_is_f16,
                        a_is_fp4=_a_is_fp4,
                        stage1_act=stage1_act,
                        bias=_b1,
                        swiglu_limit=sl,
                        num_buffers=num_buffers,
                        **_situ_kw,
                    ),
                )
            )
        kernel_bench_callable.append(
            (
                "gemm2",
                functools.partial(
                    flydsl_grouped_gemm_a8w4_masked,
                    grouped_out,
                    a2_payload,
                    w2_u8,
                    a2_scale,
                    w2s_i32,
                    psum,
                    n_experts=E,
                    contiguous_m=contiguous_m,
                    N=model_dim,
                    K=inter_dim,
                    tile_m=tile_m2,
                    tile_n=tile_n2,
                    tile_k=tile_k2,
                    out_is_f16=out_is_f16,
                    a_is_fp4=_a_is_fp4,
                    stage1_act=0,
                    bias=_b2,
                    num_buffers=num_buffers2,
                ),
            )
        )

    moe_out = torch.empty((token_num, model_dim), dtype=dtype, device=device)
    if _is_ep:
        # Route kernel already produced gather weights (dropped routes zeroed);
        # the dead-tail (tokens >= total_recv) is skipped via num_valid_tokens, and
        # non-local routes read through a zero-sized descriptor (DROPPED_ROUTE_ROW),
        # so a token whose every route is remote reduces to zeros.
        gather_w = _gather_w_buf
    else:
        gather_w = (
            torch.ones((token_num, topk), dtype=topk_weight.dtype, device=device)
            if doweight_stage1
            else topk_weight.contiguous()
        )
    flydsl_moe_gather_reduce(
        grouped_out,
        topids_to_rows,
        gather_w,
        out=moe_out,
        num_valid_tokens=(_ep_nvt if _is_ep else None),
    )
    os.environ["AITER_LAST_FUSED_MOE_IMPL"] = "grouped_a8w4_tdm"
    return moe_out


def grouped_gemm_gfx1250_a8w4(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    E: int,
    model_dim: int,
    inter_dim: int,
    dtype: torch.dtype,
    activation: ActivationType,
    quant_type: QuantType,
    q_dtype_a,
    q_dtype_w,
    isG1U1: bool,
    doweight_stage1: bool,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    expert_mask: torch.Tensor | None,
    hidden_pad: int,
    intermediate_pad: int,
    bias1: torch.Tensor | None,
    bias2: torch.Tensor | None,
    swiglu_limit: float | None = None,
    num_local_tokens: torch.Tensor | None = None,
    situ_beta: float = 1.0,
    situ_linear_beta: float = 1.0,
):
    """Grouped a8w4/a4w4 MoE on the felix TDM batched GEMM (gfx1250).

    ``w1`` MUST be GUGU: gate/up row-interleaved ([g0,u0,g1,u1,...]), i.e. what
    ``moe_shuffle_weight(..., is_guinterleave=True, gate_up=True)`` produces.
    The TDM kernel reads that layout unconditionally -- passing GGUU (separated
    gate/up) weights is not detected and yields wrong results.

    Returns ``None`` when the shape/dtype/arch is not served here, so the caller
    can fall back to the generic MoE.
    """

    def _grouped_dbg(msg: str, stacklevel: int = 1):
        if os.environ.get("AITER_GROUPED_DEBUG", "0") not in (
            "",
            "0",
            "false",
            "False",
        ):
            import inspect

            frame = inspect.stack()[stacklevel]
            print(
                f"[grouped-gemm-debug] {frame.filename}:{frame.lineno} {msg}",
                flush=True,
            )

    def _fmt(v):
        if isinstance(v, torch.Tensor):
            return f"Tensor(shape={tuple(v.shape)}, dtype={v.dtype})"
        return repr(v)

    _grouped_dbg(
        "inputs: "
        + ", ".join(
            f"{k}={_fmt(v)}"
            for k, v in [
                ("hidden_states", hidden_states),
                ("w1", w1),
                ("w2", w2),
                ("topk_weight", topk_weight),
                ("topk_ids", topk_ids),
                ("E", E),
                ("model_dim", model_dim),
                ("inter_dim", inter_dim),
                ("dtype", dtype),
                ("activation", activation),
                ("quant_type", quant_type),
                ("q_dtype_a", q_dtype_a),
                ("q_dtype_w", q_dtype_w),
                ("isG1U1", isG1U1),
                ("doweight_stage1", doweight_stage1),
                ("w1_scale", w1_scale),
                ("w2_scale", w2_scale),
                ("expert_mask", expert_mask),
                ("hidden_pad", hidden_pad),
                ("intermediate_pad", intermediate_pad),
                ("bias1", bias1),
                ("bias2", bias2),
            ]
        )
    )
    _grouped_dbg("enter grouped helper")
    # Main opt-in plus legacy kill switch.
    if not _use_grouped_gemm_enabled():
        _grouped_dbg("AITER_USE_GROUPED_GEMM not enabled; skip grouped mode")
        return None
    if os.environ.get("AITER_DISABLE_GROUPED_A8W4", "0") == "1":
        _grouped_dbg("AITER_DISABLE_GROUPED_A8W4 enabled; skip grouped mode")
        return None
    os.environ["AITER_LAST_FUSED_MOE_IMPL"] = "default"
    _is_ep = expert_mask is not None
    if _is_ep:
        _grouped_dbg(f"EP enabled: expert_mask numel={expert_mask.numel()}, E={E}")
    if hidden_pad != 0 or intermediate_pad != 0:
        hidden_pad = 0
        intermediate_pad = 0
        _grouped_dbg("haspad")
        # return None
    if not isG1U1 or quant_type != QuantType.per_1x32:
        _grouped_dbg("not g1u1 or not 1x32")
        return None
    if activation not in (
        ActivationType.Silu,
        ActivationType.Swiglu,
        ActivationType.Situv2,
    ):
        _grouped_dbg("unsupported activation")
        return None
    is_grouped_a4w4 = q_dtype_a == dtypes.fp4x2 and q_dtype_w == dtypes.fp4x2
    is_grouped_a8w4 = q_dtype_a == dtypes.fp8 and (
        q_dtype_w == dtypes.fp4x2 or w1.dtype == torch.uint8
    )
    if not (is_grouped_a4w4 or is_grouped_a8w4):
        return None
    data_format = "fp4" if is_grouped_a4w4 else "a8w4"
    # Normalize uint8-viewed fp4 weights back to fp4x2 for CSV key matching.
    q_dtype_w_key = (
        dtypes.fp4x2
        if (q_dtype_w == dtypes.fp4x2 or w1.dtype == torch.uint8)
        else q_dtype_w
    )
    _grouped_dbg(f"eligible data_format={data_format}")
    if w1_scale is None or w2_scale is None:
        return None
    _gfx_env = ";".join(
        str(os.environ.get(k, "")).lower()
        for k in ("GPU_ARCHS", "TARGET_ARCH", "AITER_GPU_ARCHS")
    )
    _force_gfx1250 = os.environ.get("AITER_FORCE_GFX1250", "0") in _TRUTHY_ENV
    if get_gfx() != "gfx1250" and "gfx1250" not in _gfx_env and not _force_gfx1250:
        return None

    device = hidden_states.device
    token_num, topk = topk_ids.shape
    if token_num == 0:
        # No tokens to compute (common in EP when a rank receives 0 dispatched
        # tokens). The grouped route/GEMM kernels would launch with a zero-sized
        # grid -> hipErrorInvalidValue, so short-circuit to an empty output.
        return torch.zeros((0, model_dim), dtype=dtype, device=device)
    # Defaults for the tile knobs the TDM path reads; the CSV row overrides them.
    tile_m = 64
    n_warp = 4
    num_buffers = 2
    cfg_row = _find_grouped_config(
        token_num=_get_padded_m(token_num),
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        # In EP each rank only holds a subset of experts, so the caller-visible
        # topk is not a stable tuning key; use -1 to match EP-agnostic CSV rows.
        topk=(-1 if _is_ep else topk),
        activation=activation,
        dtype=dtype,
        q_dtype_a=q_dtype_a,
        q_dtype_w=q_dtype_w_key,
        quant_type=quant_type,
    )
    if cfg_row is not None:
        tile_m = _as_int(cfg_row.get("tile_m"), tile_m)
        n_warp = _as_int(cfg_row.get("n_warp"), n_warp)
        num_buffers = _as_int(cfg_row.get("num_buffers"), num_buffers)
        _grouped_dbg(f"using grouped CSV config: {cfg_row}")
    else:
        logger.info(
            "no grouped CSV config matched (token=%d model_dim=%d inter_dim=%d "
            "experts=%d topk=%d act=%s dtype=%s q_dtype_a=%s q_dtype_w=%s "
            "quant_type=%s); using defaults tile_m=%d n_warp=%d "
            "num_buffers=%d",
            _get_padded_m(token_num),
            model_dim,
            inter_dim,
            E,
            topk,
            activation,
            dtype,
            q_dtype_a,
            q_dtype_w_key,
            quant_type,
            tile_m,
            n_warp,
            num_buffers,
        )
    # TDM batched kernel dispatch. EP (expert_mask) goes through the same TDM
    # GEMM; its route/quant/gather wiring lives in _grouped_a8w4_tdm_moe.
    # doweight_stage1 is not supported with EP here (the route kernel owns the
    # gather weights), so those calls return None for the generic fallback.
    if expert_mask is None or not doweight_stage1:
        _tdm_kw = {}
        if cfg_row is not None:
            _tdm_kw["tile_m"] = _as_int(cfg_row.get("tile_m"), tile_m)
            _tdm_kw["tile_n"] = _as_int(cfg_row.get("tile_n"), int(n_warp) * 64)
            _tdm_kw["tile_k"] = _as_int(cfg_row.get("tile_k"), 256)
            _tdm_kw["num_buffers"] = _as_int(cfg_row.get("num_buffers"), num_buffers)
            _tdm_kw["tile_m2"] = _as_int(cfg_row.get("tile_m2"), _tdm_kw["tile_m"])
            _tdm_kw["tile_n2"] = _as_int(cfg_row.get("tile_n2"), _tdm_kw["tile_n"])
            _tdm_kw["tile_k2"] = _as_int(cfg_row.get("tile_k2"), _tdm_kw["tile_k"])
            _tdm_kw["num_buffers2"] = _as_int(
                cfg_row.get("num_buffer_stage2"), _tdm_kw["num_buffers"]
            )

        # Env overrides for tuning (present-check so any set value wins over CSV /
        # defaults). Stage2 (*2) falls back to the stage1 value when unset. Set
        # AITER_TDM_TILE_M / _TILE_N / _TILE_K / _NUM_BUFFERS (+ *_M2/_N2/_K2/
        # _NUM_BUFFERS2) to sweep the felix TDM batched GEMM tiles.
        def _tdm_env(name):
            v = os.environ.get(name)
            return int(v) if (v is not None and v != "") else None

        _ov_m = _tdm_env("AITER_TDM_TILE_M")
        _ov_n = _tdm_env("AITER_TDM_TILE_N")
        _ov_k = _tdm_env("AITER_TDM_TILE_K")
        _ov_nb = _tdm_env("AITER_TDM_NUM_BUFFERS")
        _ov_m2 = _tdm_env("AITER_TDM_TILE_M2")
        _ov_n2 = _tdm_env("AITER_TDM_TILE_N2")
        _ov_k2 = _tdm_env("AITER_TDM_TILE_K2")
        _ov_nb2 = _tdm_env("AITER_TDM_NUM_BUFFERS2")
        if any(
            v is not None
            for v in (_ov_m, _ov_n, _ov_k, _ov_nb, _ov_m2, _ov_n2, _ov_k2, _ov_nb2)
        ):
            _base_m = _ov_m if _ov_m is not None else _tdm_kw.get("tile_m", tile_m)
            _base_n = (
                _ov_n if _ov_n is not None else _tdm_kw.get("tile_n", int(n_warp) * 64)
            )
            _base_k = _ov_k if _ov_k is not None else _tdm_kw.get("tile_k", 256)
            _base_nb = (
                _ov_nb
                if _ov_nb is not None
                else _tdm_kw.get("num_buffers", num_buffers)
            )
            _tdm_kw["tile_m"] = _base_m
            _tdm_kw["tile_n"] = _base_n
            _tdm_kw["tile_k"] = _base_k
            _tdm_kw["num_buffers"] = _base_nb
            # Stage2 ties to the (possibly overridden) stage1 base unless its own
            # *_M2/_N2/_K2/_NUM_BUFFERS2 override is set. This intentionally
            # discards any CSV stage2 value so an env sweep stays self-consistent.
            _tdm_kw["tile_m2"] = _ov_m2 if _ov_m2 is not None else _base_m
            _tdm_kw["tile_n2"] = _ov_n2 if _ov_n2 is not None else _base_n
            _tdm_kw["tile_k2"] = _ov_k2 if _ov_k2 is not None else _base_k
            _tdm_kw["num_buffers2"] = _ov_nb2 if _ov_nb2 is not None else _base_nb
        return _grouped_a8w4_tdm_moe(
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
            E=E,
            model_dim=model_dim,
            inter_dim=inter_dim,
            dtype=dtype,
            activation=activation,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            bias1=bias1,
            bias2=bias2,
            swiglu_limit=swiglu_limit,
            doweight_stage1=doweight_stage1,
            data_format=data_format,
            expert_mask=expert_mask,
            num_local_tokens=num_local_tokens,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            **_tdm_kw,
        )

    # Only the felix TDM grouped path is kept; the previous non-TDM grouped
    # GEMM (gemm_mxscale_gfx1250 / moe_grouped_gemm_mxscale_gfx1250) was
    # removed. Anything the TDM path cannot serve falls back to the caller's
    # generic MoE via None.
    # TODO(aot): AOT has no coverage for the TDM batched GEMM
    # (batched_gemm_mxfp4); grouped kernels are JIT-compiled at first use
    # until that is added back.
    return None


# --- Functions moved from moe_kernels.py for grouped gemm ---
@functools.cache
def _get_compiled_gather_reduce(
    model_dim: int,
    topk: int,
    out_dtype: str,
    split_k: int = 1,
    vec_dwords: int = 2,
    w_dtype: str = "f32",
):
    """Compile and cache the one-pass MoE gather-reduce kernel."""
    from aiter.ops.flydsl.kernels.moe_gather_reduce import (
        build_moe_gather_reduce_module,
    )

    return build_moe_gather_reduce_module(
        model_dim, topk, out_dtype, split_k, vec_dwords, w_dtype
    )


def _choose_gather_reduce_vec(token_num: int, model_dim: int) -> int:
    """Prefer CTA parallelism first; use wider vec only once CTA count is ample."""
    out_dwords = int(model_dim) // 2
    n_iters_v4 = (out_dwords + 256 * 4 - 1) // (256 * 4)
    return 4 if int(token_num) * n_iters_v4 >= 256 else 2


@functools.cache
def _get_compiled_route_maps():
    """Compile and cache the atomic route -> grouped-row map kernel."""
    from aiter.ops.flydsl.kernels.moe_route_maps import build_moe_route_maps_module

    return build_moe_route_maps_module()


@functools.cache
def _get_compiled_contiguous_psum():
    """Compile and cache the contiguous M-tile prefix-sum kernel."""
    from aiter.ops.flydsl.kernels.moe_contiguous_psum import (
        build_moe_contiguous_psum_module,
    )

    return build_moe_contiguous_psum_module()


@functools.cache
def _get_compiled_contiguous_psum_remap():
    """Compile and cache the contiguous prefix-sum + row-remap kernel."""
    from aiter.ops.flydsl.kernels.moe_contiguous_psum import (
        build_moe_contiguous_psum_remap_module,
    )

    return build_moe_contiguous_psum_remap_module()


@functools.cache
def _get_compiled_route_psum_fused():
    """Compile and cache the single-TG fused route+atomic+psum+remap kernel."""
    from aiter.ops.flydsl.kernels.moe_contiguous_psum import (
        build_moe_route_psum_fused_module,
    )

    return build_moe_route_psum_fused_module()


# One workgroup handles every route. NUMEL is advisory -- the route sweep is
# grid-stride, so a larger count is correct but stops being worth fusing.
# EXPERTS is a hard limit, enforced below: the scan and the LDS route counter
# are both one slot per lane.
_FUSED_ROUTE_PSUM_MAX_NUMEL = 4096
_FUSED_ROUTE_PSUM_MAX_EXPERTS = 512


def fused_route_psum_remap(
    topk_ids: torch.Tensor,
    experts: int,
    max_m: int,
    tile_m: int,
):
    """Single-launch route+atomic+psum+remap for small token counts.

    Equivalent to ``flydsl_moe_topids_to_rows`` followed by
    ``contiguous_psum_remap``, but fused into one workgroup. Returns
    (masked_m, topids_to_rows[token_num, topk], psum).
    """
    device = topk_ids.device
    token_num, topk = topk_ids.shape
    numel = token_num * topk
    experts = int(experts)
    # Unlike contiguous_psum/_remap, this kernel's scan is still single-pass:
    # its LDS route counter is one slot per expert, so widening E needs a bigger
    # allocation, not just a carry. Fail loudly rather than silently drop the
    # experts past the block, which is the bug the chunked scan fixed there.
    if experts > _FUSED_ROUTE_PSUM_MAX_EXPERTS:
        raise ValueError(
            f"fused_route_psum_remap supports at most "
            f"{_FUSED_ROUTE_PSUM_MAX_EXPERTS} experts, got {experts}; "
            f"use flydsl_moe_topids_to_rows + contiguous_psum_remap instead"
        )
    topids_to_rows = torch.empty(numel, dtype=torch.int32, device=device)
    masked_m = torch.empty(experts, dtype=torch.int32, device=device)
    starts = torch.empty(experts, dtype=torch.int32, device=device)
    psum = torch.empty(experts, dtype=torch.int32, device=device)
    launch = _get_compiled_route_psum_fused()
    launch(
        ptr_arg(topk_ids.to(torch.int32).reshape(-1)),
        ptr_arg(topids_to_rows),
        ptr_arg(masked_m),
        ptr_arg(starts),
        ptr_arg(psum),
        int(numel),
        experts,
        int(max_m),
        int(tile_m),
        stream=torch.cuda.current_stream(),
    )
    return masked_m, topids_to_rows.view(token_num, topk), psum


def contiguous_psum(masked_m: torch.Tensor, experts: int, tile_m: int):
    """Tile-aligned exclusive prefix sum over per-expert counts."""
    device = masked_m.device
    experts = int(experts)
    masked_m_i32 = masked_m[:experts].to(torch.int32)
    starts = torch.empty(experts, dtype=torch.int32, device=device)
    psum = torch.empty(experts, dtype=torch.int32, device=device)
    contiguous_m_t = torch.empty(1, dtype=torch.int32, device=device)
    launch = _get_compiled_contiguous_psum()
    launch(
        ptr_arg(masked_m_i32),
        ptr_arg(starts),
        ptr_arg(psum),
        ptr_arg(contiguous_m_t),
        experts,
        int(tile_m),
        stream=torch.cuda.current_stream(),
    )
    return starts, psum, contiguous_m_t


def contiguous_psum_remap(
    masked_m: torch.Tensor,
    topids_to_rows: torch.Tensor,
    experts: int,
    route_max_m: int,
    tile_m: int,
    num_valid_routes: torch.Tensor | None = None,
):
    """Tile-aligned psum and in-place masked-row -> contiguous-row remap."""
    device = masked_m.device
    experts = int(experts)
    masked_m_i32 = masked_m[:experts].to(torch.int32)
    starts = torch.empty(experts, dtype=torch.int32, device=device)
    psum = torch.empty(experts, dtype=torch.int32, device=device)
    contiguous_m_t = torch.empty(1, dtype=torch.int32, device=device)
    topids_flat = topids_to_rows.reshape(-1)
    # Only remap the valid routes; dead-tail rows are unwritten and must not be
    # used as a row index. Default (no truncation) covers every route.
    if num_valid_routes is None:
        # A 0-element tensor has data_ptr() == 0; the kernel reads that null
        # pointer as "no truncation".
        num_valid_routes_i32 = torch.empty(0, dtype=torch.int32, device=device)
    else:
        num_valid_routes_i32 = (
            num_valid_routes.reshape(-1)[:1]
            .to(device=device, dtype=torch.int32)
            .contiguous()
        )
    launch = _get_compiled_contiguous_psum_remap()
    launch(
        ptr_arg(masked_m_i32),
        ptr_arg(topids_flat),
        ptr_arg(starts),
        ptr_arg(psum),
        ptr_arg(contiguous_m_t),
        int(topids_flat.numel()),
        experts,
        int(route_max_m),
        int(tile_m),
        ptr_arg(num_valid_routes_i32),
        stream=torch.cuda.current_stream(),
    )
    return starts, psum, contiguous_m_t


def build_route_maps(topk_ids: torch.Tensor, E: int, max_m: int):
    """Atomic-scatter route maps. Returns (topids_to_rows, rows_to_tokens, masked_m)."""
    device = topk_ids.device
    token_num, topk = topk_ids.shape
    numel = token_num * topk
    topk_ids_i32 = topk_ids.reshape(-1).to(torch.int32).contiguous()
    atomic_buffer = torch.zeros(E, dtype=torch.int32, device=device)
    topids_to_rows = torch.empty(numel, dtype=torch.int32, device=device)
    rows_to_tokens = torch.full((E * max_m,), -1, dtype=torch.int32, device=device)
    grid_blocks = (numel + 255) // 256
    launch = _get_compiled_route_maps()
    launch(
        topk_ids_i32,
        atomic_buffer,
        topids_to_rows,
        rows_to_tokens,
        numel,
        topk,
        max_m,
        grid_blocks,
        stream=torch.cuda.current_stream(),
    )
    masked_m = atomic_buffer
    return topids_to_rows.view(token_num, topk), rows_to_tokens, masked_m


def flydsl_moe_gather_reduce(
    grouped_out: torch.Tensor,  # (E,max_m,D) or (split_k,E,max_m,D) bf16/f16
    topids_to_rows: torch.Tensor,  # (token_num, topk) int32 grouped flat rows
    gather_w: torch.Tensor,  # (token_num, topk) route weight, f32/bf16/f16
    out: torch.Tensor | None = None,
    num_valid_tokens: (
        torch.Tensor | None
    ) = None,  # (1,) int32; skip output tokens >= this (EP dead-tail)
) -> torch.Tensor:
    """One-pass gather-reduce: out[t] = sum_k w[t,k] * grouped[topids_to_rows[t,k]].

    ``gather_w`` may be f32 (native route weights, no host-side cast) or match
    ``grouped_out``'s bf16/f16; the kernel accumulates in f32 either way. Slots
    holding ``moe_route_maps.DROPPED_ROUTE_ROW`` (EP routes with no grouped row)
    contribute 0 without touching ``grouped_out``.
    """
    if grouped_out.dim() == 4:
        split_k, E, max_m, model_dim = grouped_out.shape
    else:
        split_k = 1
        E, max_m, model_dim = grouped_out.shape
    token_num, topk = topids_to_rows.shape
    device = grouped_out.device
    if grouped_out.dtype == torch.bfloat16:
        out_dtype = "bf16"
    elif grouped_out.dtype == torch.float16:
        out_dtype = "f16"
    else:
        raise ValueError(f"unsupported dtype {grouped_out.dtype}; need bf16/f16")
    if gather_w.dtype == torch.float32:
        w_dtype = "f32"
    elif gather_w.dtype == torch.bfloat16:
        w_dtype = "bf16"
    elif gather_w.dtype == torch.float16:
        w_dtype = "f16"
    else:
        raise ValueError(
            f"unsupported gather_w dtype {gather_w.dtype}; need f32/bf16/f16"
        )

    grouped_out_flat = grouped_out.contiguous().view(split_k * E * max_m, model_dim)
    if out is None:
        out = torch.empty(
            (token_num, model_dim), dtype=grouped_out.dtype, device=device
        )

    gather_vec = _choose_gather_reduce_vec(token_num, model_dim)
    launch = _get_compiled_gather_reduce(
        model_dim, topk, out_dtype, split_k, gather_vec, w_dtype
    )
    slice_stride_dw = E * max_m * (model_dim // 2)
    # Skip dead-tail output tokens whose route map is unwritten. Default (no
    # truncation) processes every token.
    if num_valid_tokens is None:
        # A 0-element tensor has data_ptr() == 0; the kernel reads that null
        # pointer as "no truncation".
        num_valid_tokens_i32 = torch.empty(0, dtype=torch.int32, device=device)
    else:
        num_valid_tokens_i32 = (
            num_valid_tokens.reshape(-1)[:1]
            .to(device=device, dtype=torch.int32)
            .contiguous()
        )
    launch(
        ptr_arg(grouped_out_flat),
        ptr_arg(topids_to_rows),
        ptr_arg(gather_w),
        ptr_arg(out),
        token_num,
        slice_stride_dw,
        ptr_arg(num_valid_tokens_i32),
        stream=torch.cuda.current_stream(),
    )
    return out


# MoE route-gather (scatter-copy) input layout helpers


@functools.cache
def _get_compiled_scatter_copy(row_bytes: int):
    """Compile and cache the one-pass row scatter-copy kernel (per row width)."""
    from aiter.ops.flydsl.kernels.moe_scatter_copy_token import (
        build_moe_scatter_copy_token_module,
    )

    return build_moe_scatter_copy_token_module(row_bytes)


def flydsl_moe_scatter_copy_token(
    a1_payload: torch.Tensor,  # (token_num, payload_w) uint8
    a1_scale_token_u8: torch.Tensor | None,  # (token_num, scale_w) uint8 or None
    rows_to_tokens: torch.Tensor,  # (E*max_m,) int32 grouped row -> token (-1 pad)
    E: int,
    max_m: int,
    grouped_a1: torch.Tensor | None = None,  # (E, max_m, payload_w) uint8 out
    a1_scale_raw: torch.Tensor | None = None,  # (E, max_m, scale_w) uint8 out
):
    """Copy token payload/scale into grouped layout via rows_to_tokens map.

    Returns (grouped_a1, a1_scale_raw)."""
    device = a1_payload.device
    payload_w = a1_payload.shape[1]
    num_dst = E * max_m

    if grouped_a1 is None:
        grouped_a1 = torch.zeros(
            (E, max_m, payload_w), dtype=torch.uint8, device=device
        )
    launch_p = _get_compiled_scatter_copy(payload_w)
    launch_p(
        a1_payload.contiguous().view(-1, payload_w),
        grouped_a1.view(num_dst, payload_w),
        rows_to_tokens,
        num_dst,
        stream=torch.cuda.current_stream(),
    )

    if a1_scale_token_u8 is not None:
        scale_w = a1_scale_token_u8.shape[1]
        if a1_scale_raw is None:
            a1_scale_raw = torch.zeros(
                (E, max_m, scale_w), dtype=torch.uint8, device=device
            )
        launch_s = _get_compiled_scatter_copy(scale_w)
        launch_s(
            a1_scale_token_u8.contiguous().view(-1, scale_w),
            a1_scale_raw.view(num_dst, scale_w),
            rows_to_tokens,
            num_dst,
            stream=torch.cuda.current_stream(),
        )

    return grouped_a1, a1_scale_raw


@functools.cache
def _get_compiled_scatter_preshuffle_scale(
    row_bytes: int, wmma_rep: int, scale_k_per_tile: int, gather: bool = True
):
    """Compile and cache the WMMA-preshuffle scale kernel (with/without gather)."""
    from aiter.ops.flydsl.kernels.moe_scatter_copy_preshuffle_scale import (
        build_moe_scatter_copy_preshuffle_scale_module,
    )

    return build_moe_scatter_copy_preshuffle_scale_module(
        row_bytes, wmma_rep, scale_k_per_tile, gather=gather
    )


def flydsl_moe_scatter_preshuffle_scale(
    a1_scale_token_u8: torch.Tensor,  # (token_num, scale_w) uint8
    rows_to_tokens: torch.Tensor,  # (E*max_m,) int32 grouped row -> token (-1 pad)
    E: int,
    max_m: int,
    *,
    wmma_rep: int,
    scale_k_per_tile: int,
    grouped_a1_scale: (
        torch.Tensor | None
    ) = None,  # (E, max_m//wmma_rep, scale_w*wmma_rep)
):
    """Fused route-gather + WMMA preshuffle for e8m0 scale rows.

    Returns ``grouped_a1_scale``.
    """
    device = a1_scale_token_u8.device
    scale_w = a1_scale_token_u8.shape[1]
    rows_per_tile = wmma_rep * 16
    assert (
        max_m % rows_per_tile == 0
    ), f"max_m ({max_m}) must be a multiple of wmma_rep*16 ({rows_per_tile})"
    tiles_per_expert = max_m // rows_per_tile

    if grouped_a1_scale is None:
        grouped_a1_scale = torch.empty(
            (E, max_m // wmma_rep, scale_w * wmma_rep), dtype=torch.uint8, device=device
        )

    launch = _get_compiled_scatter_preshuffle_scale(
        scale_w, wmma_rep, scale_k_per_tile, True
    )
    launch(
        a1_scale_token_u8.contiguous().view(-1, scale_w),
        grouped_a1_scale.view(E * (max_m // wmma_rep), scale_w * wmma_rep),
        rows_to_tokens,
        max_m,
        E,
        tiles_per_expert,
        stream=torch.cuda.current_stream(),
    )
    return grouped_a1_scale


def flydsl_moe_preshuffle_scale(
    scale_grouped_u8: torch.Tensor,  # (E, max_m, scale_w) or (E*max_m, scale_w) uint8
    E: int,
    max_m: int,
    *,
    wmma_rep: int,
    scale_k_per_tile: int,
    out: torch.Tensor | None = None,  # (E, max_m//wmma_rep, scale_w*wmma_rep)
):
    """Preshuffle grouped row-major e8m0 scale into WMMA layout. Returns out."""
    device = scale_grouped_u8.device
    scale_w = scale_grouped_u8.shape[-1]
    rows_per_tile = wmma_rep * 16
    assert (
        max_m % rows_per_tile == 0
    ), f"max_m ({max_m}) must be a multiple of wmma_rep*16 ({rows_per_tile})"
    tiles_per_expert = max_m // rows_per_tile

    if out is None:
        out = torch.empty(
            (E, max_m // wmma_rep, scale_w * wmma_rep), dtype=torch.uint8, device=device
        )

    launch = _get_compiled_scatter_preshuffle_scale(
        scale_w, wmma_rep, scale_k_per_tile, False
    )
    launch(
        scale_grouped_u8.contiguous().view(E * max_m, scale_w),
        out.view(E * (max_m // wmma_rep), scale_w * wmma_rep),
        max_m,
        E,
        tiles_per_expert,
        stream=torch.cuda.current_stream(),
    )
    return out
