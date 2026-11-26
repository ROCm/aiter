# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter as a Megatron-Core ``kda_backend``.

Megatron's KDA layer (``megatron/core/ssm/gated_delta_net/kda.py``) resolves
its recurrence once with ``select_kda_recurrence(config.kda_backend)`` and
calls it with FLA's ``chunk_kda`` contract::

    o, _ = recurrence(q=, k=, v=, g=raw_g, beta=, A_log=, dt_bias=,
                      initial_state=None, output_final_state=False,
                      use_qk_l2norm_in_kernel=, use_gate_in_kernel=,
                      safe_gate=, lower_bound=, cu_seqlens=, cp_context=)

``chunk_kda`` below is that callable. ``patch_megatron_kda_backend`` registers
it as ``kda_backend="aiter"``; TP/SP head sharding, recompute, offload, Muon
and MTP stay in Megatron's layer around it.
"""

from __future__ import annotations

import functools

import torch

from aiter.ops.kda.recurrence import KdaKernels, kda_recurrence

__all__ = ["chunk_kda", "make_chunk_kda", "patch_megatron_kda_backend"]

# FLA keywords that pick a schedule rather than a result.
_IGNORED_KWARGS = ("disable_recompute", "cu_seqlens_cpu", "chunk_indices")


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    cp_context=None,
    *,
    backend: str | KdaKernels | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """FLA ``chunk_kda`` / Megatron ``kda_backend`` contract on aiter kernels.

    Extra keywords: ``A_log``, ``dt_bias``, ``use_beta_sigmoid_in_kernel``,
    ``chunk_size``; ``disable_recompute``, ``cu_seqlens_cpu`` and
    ``chunk_indices`` are accepted and ignored; ``return_intermediate_states``
    must be false. As in Megatron's cuDNN backend, ``lower_bound`` applies only
    with ``safe_gate`` and ``use_gate_in_kernel``, and under ``cp_context``
    the local ``cu_seqlens`` come from the context.
    """
    if kwargs.pop("return_intermediate_states", False):
        raise NotImplementedError("aiter KDA does not return intermediate states.")
    for name in _IGNORED_KWARGS:
        kwargs.pop(name, None)
    A_log = kwargs.pop("A_log", None)
    dt_bias = kwargs.pop("dt_bias", None)
    use_beta_sigmoid_in_kernel = kwargs.pop("use_beta_sigmoid_in_kernel", False)
    chunk_size = kwargs.pop("chunk_size", None)
    if kwargs:
        raise TypeError(
            f"aiter KDA does not support keyword arguments {sorted(kwargs)}."
        )
    bounded = safe_gate and use_gate_in_kernel
    return kda_recurrence(
        q,
        k,
        v,
        g,
        beta,
        A_log=A_log if use_gate_in_kernel else None,
        dt_bias=dt_bias if use_gate_in_kernel else None,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        safe_gate=bounded,
        lower_bound=lower_bound if bounded else None,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        cp_context=cp_context,
        backend=backend,
    )


def make_chunk_kda(backend: str | KdaKernels | None = None):
    """``chunk_kda`` pinned to one backend (``None`` reads ``AITER_KDA_BACKEND``)."""
    return functools.partial(chunk_kda, backend=backend)


def patch_megatron_kda_backend(
    name: str = "aiter", backend: str | KdaKernels | None = None
):
    """Register ``kda_backend=name`` in an imported Megatron-Core.

    Wraps ``select_kda_recurrence`` (in ``megatron.core.ops.ssm.kda.backends``
    and the name ``megatron.core.ssm.gated_delta_net.kda`` imported) and
    ``KDALayerConfig.validate_kda``, which only admits ``"fla"`` and
    ``"cudnn"``; the config is validated as ``"fla"``. Call it before the
    model is built. Idempotent.
    """
    from megatron.core.ops.ssm.kda import backends
    from megatron.core.ssm import kda_layer_config
    from megatron.core.ssm.gated_delta_net import kda as kda_layer

    select = backends.select_kda_recurrence
    if getattr(select, "_aiter_names", None) is None:
        orig_select = select

        @functools.wraps(orig_select)
        def select(backend_name):
            if backend_name in select._aiter_names:
                return select._aiter_names[backend_name]
            return orig_select(backend_name)

        select._aiter_names = {}
        backends.select_kda_recurrence = select
        kda_layer.select_kda_recurrence = select
    select._aiter_names[name] = make_chunk_kda(backend)

    config_cls = kda_layer_config.KDALayerConfig
    validate = config_cls.validate_kda
    if getattr(validate, "_aiter_names", None) is None:
        orig_validate = validate

        @functools.wraps(orig_validate)
        def validate(self):
            backend_name = self.kda_backend
            if backend_name not in validate._aiter_names:
                return orig_validate(self)
            object.__setattr__(self, "kda_backend", "fla")
            try:
                return orig_validate(self)
            finally:
                object.__setattr__(self, "kda_backend", backend_name)

        validate._aiter_names = set()
        config_cls.validate_kda = validate
    validate._aiter_names.add(name)
    return select._aiter_names[name]
