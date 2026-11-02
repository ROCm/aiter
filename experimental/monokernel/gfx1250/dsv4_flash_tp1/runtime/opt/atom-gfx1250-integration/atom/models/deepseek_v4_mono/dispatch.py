# SPDX-License-Identifier: MIT
"""Post-load routing, following V4.1/K3 mono's installer and LazyRunner ABI."""

import logging

from torch import nn

from atom.models.deepseek_v4_mono.config import (
    config_refusal,
    select_profile,
    step_supported,
)
from atom.mono.runtime.consensus import MonoUnsupported
from atom.mono.runtime.route import LazyRunner
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


class MonoDecodeModel(nn.Module):
    def __init__(self, model, profile):
        super().__init__()
        self.model = model
        self.profile = profile

        def create():
            try:
                from atom.models.deepseek_v4_mono.runner import HybridDecodeRunner
            except ImportError as error:
                raise MonoUnsupported(
                    f"gfx1250 hybrid kernels unavailable: {error}"
                ) from error
            return HybridDecodeRunner(model, profile)

        self._mono = LazyRunner(create, f"DSV4 gfx1250 {profile.name} hybrid")

    def forward(self, input_ids, positions):
        context = get_forward_context()
        state = context.context
        eligible = state is not None and step_supported(
            input_ids.numel(),
            self.profile,
            prefill=state.is_prefill,
            ubatches=getattr(context, "ubatch_slices", None) is not None,
        )
        # Inputs are TP-uniform. Do not gate on rank-local is_dummy_run.
        # Before paged-cache binding the native warmup is the only valid
        # path. This lifecycle state is uniform across the TP group. Views
        # bind lazily afterwards so they see the final cache tensor addresses.
        cache_bound = (
            getattr(self.model.model.layers[0].attn, "unified_kv", None) is not None
        )
        if eligible and cache_bound and self._mono.ready(input_ids.numel()):
            state.input_ids = input_ids.flatten()  # hash-routing contract
            return self._mono.runner.forward(input_ids, positions)
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        return self.model.compute_logits(hidden_states)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def install_mono_decode(model, atom_config, drafter, metadata_builder):
    # Disabled/native deployments must not import a DSL or query a device.
    if not envs.ATOM_MONO_ENABLE or not envs.ATOM_DSV4_MONO:
        return model
    from aiter.jit.utils.chip_info import get_gfx_runtime

    why = config_refusal(atom_config, arch=get_gfx_runtime())
    if why is not None:
        logger.info("DSV4 gfx1250 hybrid off: %s", why)
        return model
    if drafter is not None:
        logger.info("DSV4 gfx1250 hybrid off: drafter")
        return model
    profile = select_profile(atom_config.hf_config, atom_config.tensor_parallel_size)
    return MonoDecodeModel(model, profile)
