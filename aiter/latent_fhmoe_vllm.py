# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Decode-only vLLM adapter for the Kimi-K3 latent FHMoE prototype.

This module intentionally avoids importing vLLM.  It accepts a duck-typed
``ROCmLatentMoERunner`` so aiter can be overlaid into a vLLM image without
adding a package dependency.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from aiter.latent_fhmoe import latent_fhmoe
from aiter.ops.shuffle import shuffle_weight

_K3_DECODE_M = 8
_K3_TP_SIZE = 8
_K3_EXPERTS = 896
_K3_TOPK = 16
_LOGGED_LIVE_PATH = False


def _raw_tensor(value: Any) -> torch.Tensor | None:
    """Unwrap triton-kernels precision/storage wrappers used by vLLM."""
    if value is None or isinstance(value, torch.Tensor):
        return value
    storage = getattr(value, "storage", None)
    data = getattr(storage, "data", None)
    return data if isinstance(data, torch.Tensor) else None


def _precision_scale(precision: Any) -> torch.Tensor | None:
    return _raw_tensor(getattr(precision, "weight_scale", None))


def _weight_scale(quant: Any, index: int) -> torch.Tensor | None:
    """Read either vLLM's raw MX scale or its precision wrapper.

    ``FusedMoEQuantConfig.w{1,2}_precision`` asserts that the descriptor scale
    is a ``PrecisionConfig``.  K3's live MXFP4 config stores a raw tensor there,
    so inspect the descriptor first instead of triggering that property.
    """
    desc = getattr(quant, f"_w{index}", None)
    scale = getattr(desc, "scale", None)
    tensor = _raw_tensor(scale)
    if tensor is not None:
        return tensor
    try:
        precision = getattr(quant, f"w{index}_precision", None)
    except AssertionError:
        return None
    return _precision_scale(precision)


def _shared_weights(runner: Any) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return graph-stable, FlyDSL-shuffled shared-expert TP shards."""
    cached = getattr(runner, "_k3_latent_shared_weights", None)
    if cached is not None:
        return cached

    shared = getattr(runner, "_shared_experts", None)
    layer = getattr(shared, "_layer", None)
    gate_up = getattr(getattr(layer, "gate_up_proj", None), "weight", None)
    down = getattr(getattr(layer, "down_proj", None), "weight", None)
    gate_up = _raw_tensor(gate_up)
    down = _raw_tensor(down)
    if gate_up is None or down is None:
        return None

    # TP8 K3: two shared experts give a local intermediate width of 768.
    if tuple(gate_up.shape) != (1536, 7168) or tuple(down.shape) != (7168, 768):
        return None
    shuffled = (
        shuffle_weight(gate_up.unsqueeze(0), layout=(16, 16)),
        shuffle_weight(down.unsqueeze(0), layout=(16, 16)),
    )
    runner._k3_latent_shared_weights = shuffled
    return shuffled


def _routed_weights(
    runner: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    routed = getattr(runner, "routed_experts", None)
    quant_method = getattr(routed, "quant_method", None)
    quant = getattr(quant_method, "moe_quant_config", None)
    if quant is None:
        quant = getattr(quant_method, "get_fused_moe_quant_config", lambda _: None)(
            routed
        )

    w1 = _raw_tensor(getattr(routed, "w13_weight", None))
    w2 = _raw_tensor(getattr(routed, "w2_weight", None))
    s1 = _weight_scale(quant, 1)
    s2 = _weight_scale(quant, 2)
    if any(value is None for value in (w1, w2, s1, s2)):
        return None
    assert w1 is not None and w2 is not None and s1 is not None and s2 is not None
    # AITER's SiTU conversion returns shuffled scales flattened over E*N.
    # The latent kernel uses the same storage but validates its logical 3-D
    # expert shape, so restore that view without copying.
    s1_shape = (_K3_EXPERTS, 768, 112)
    s2_shape = (_K3_EXPERTS, 3584, 16)
    if s1.numel() == 896 * 768 * 112:
        s1 = s1.reshape(s1_shape)
    if s2.numel() == 896 * 3584 * 16:
        s2 = s2.reshape(s2_shape)
    if (
        tuple(w1.shape) != (_K3_EXPERTS, 768, 1792)
        or tuple(w2.shape) != (_K3_EXPERTS, 3584, 192)
        or tuple(s1.shape) != s1_shape
        or tuple(s2.shape) != s2_shape
    ):
        return None
    return w1, w2, s1, s2


def maybe_run_vllm_k3_latent_fhmoe(
    runner: Any,
    routed_input: torch.Tensor,
    router_logits: torch.Tensor,
    shared_input: torch.Tensor | None,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return ``(shared_partial, routed_partial)`` for the exact IX decode shape.

    Every non-matching case returns ``None`` so the caller can use vLLM's
    existing path.  Restricting the prototype to M=8 also prevents the known
    FlyDSL runtime-integer cache alias between different token counts.
    """
    if os.environ.get("VLLM_ROCM_USE_K3_LATENT_FHMOE", "0") != "1":
        return None
    if os.environ.get("VLLM_ROCM_K3_LATENT_SEPARATED_LAYOUT", "0") == "1":
        if os.environ.get("VLLM_ROCM_K3_LATENT_STRICT", "0") == "1":
            raise RuntimeError(
                "K3 latent FHMoE requires production interleaved A8W4 W1 layout"
            )
        return None
    config = getattr(runner, "moe_config", None)
    if (
        shared_input is None
        or tuple(routed_input.shape) != (_K3_DECODE_M, 3584)
        or tuple(shared_input.shape) != (_K3_DECODE_M, 7168)
        or tuple(router_logits.shape) != (_K3_DECODE_M, _K3_EXPERTS)
        or getattr(config, "tp_size", None) != _K3_TP_SIZE
        or getattr(config, "ep_size", None) != 1
        or bool(getattr(config, "is_sequence_parallel", False))
        or getattr(runner, "expert_map", None) is not None
    ):
        return None

    capturing = torch.cuda.is_current_stream_capturing()
    if not capturing and getattr(runner, "_k3_latent_fhmoe_prepared", False):
        return None

    routed = getattr(runner, "routed_experts", None)
    ensure_init = getattr(routed, "_ensure_moe_quant_config_init", None)
    if ensure_init is not None:
        ensure_init()
    weights = _routed_weights(runner)
    shared_weights = _shared_weights(runner)
    if weights is None or shared_weights is None:
        if os.environ.get("VLLM_ROCM_K3_LATENT_STRICT", "0") == "1":
            routed = getattr(runner, "routed_experts", None)
            shared = getattr(getattr(runner, "_shared_experts", None), "_layer", None)

            def shape(value: Any) -> tuple[int, ...] | None:
                tensor = _raw_tensor(value)
                return None if tensor is None else tuple(tensor.shape)

            quant = getattr(
                getattr(routed, "quant_method", None), "moe_quant_config", None
            )
            w1_scale = getattr(getattr(quant, "_w1", None), "scale", None)
            w2_scale = getattr(getattr(quant, "_w2", None), "scale", None)
            raise RuntimeError(
                "K3 latent FHMoE live weight contract mismatch: "
                f"routed_w1={shape(getattr(routed, 'w13_weight', None))}, "
                f"routed_w2={shape(getattr(routed, 'w2_weight', None))}, "
                f"routed_s1={shape(w1_scale)} ({type(w1_scale).__name__}), "
                f"routed_s2={shape(w2_scale)} ({type(w2_scale).__name__}), "
                "shared_w1="
                f"{shape(getattr(getattr(shared, 'gate_up_proj', None), 'weight', None))}, "
                "shared_w2="
                f"{shape(getattr(getattr(shared, 'down_proj', None), 'weight', None))}"
            )
        return None

    topk_weight, topk_ids = runner.router.select_experts(
        hidden_states=routed_input,
        router_logits=router_logits,
        topk_indices_dtype=torch.int32,
        input_ids=input_ids,
    )
    if topk_ids.shape != (_K3_DECODE_M, _K3_TOPK):
        if os.environ.get("VLLM_ROCM_K3_LATENT_STRICT", "0") == "1":
            raise RuntimeError(f"Unexpected K3 top-k shape: {tuple(topk_ids.shape)}")
        return None

    global _LOGGED_LIVE_PATH
    if not _LOGGED_LIVE_PATH:
        _LOGGED_LIVE_PATH = True
        print(
            "K3 latent FHMoE: enabled TP8/EP1/M=8 interleaved A8W4 path",
            flush=True,
        )

    routed_w1, routed_w2, routed_s1, routed_s2 = weights
    shared_w1, shared_w2 = shared_weights
    routed_output, shared_output = latent_fhmoe(
        routed_input,
        routed_w1,
        routed_w2,
        routed_s1,
        routed_s2,
        topk_weight,
        topk_ids,
        shared_input,
        shared_w1,
        shared_w2,
    )
    if not capturing:
        # vLLM executes each capture size once before graph capture. Use that
        # warmup to compile FlyDSL and allocate the per-layer workspace from
        # the ordinary caching pool, but preserve the normal vLLM result.
        #
        # M=8 alone cannot select the latent result here: a chunked-prefill
        # tail can have eight local tokens on only some DCP ranks, which would
        # desynchronize the following TP collective. During capture all ranks
        # take this path, and replay executes it without re-entering Python.
        runner._k3_latent_fhmoe_prepared = True
        return None
    return shared_output, routed_output
