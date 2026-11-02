# SPDX-License-Identifier: MIT
"""The native V4 layer walk with regional gfx1250 replacements.

A hybrid owns shallow module views, never a second GPU weight copy. Native
modules and their compiled forwards remain available for prefill/fallback.
The attention and MoE side-stream ordering and TP reductions are unchanged.
"""

import copy
import logging

import torch
from aiter.dist.parallel_state import get_tp_group

from atom.models.deepseek_v4 import HCState
from atom.models.deepseek_v4_mono.kernels.ops import configs, mhc_quant, projection
from atom.models.deepseek_v4_mono.linear import HybridLinear, linear_refusal
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed, tp_agree

logger = logging.getLogger("atom")


def module_view(module):
    result = copy.copy(module)
    result._modules = module._modules.copy()
    return result


class HybridDecodeRunner:
    def __init__(self, causal_lm, profile):
        self.native = causal_lm.model
        self.profile = profile
        self.group = get_tp_group().cpu_group if profile.tp > 1 else None
        self.layers = []
        self.linears = []
        self.prepared = set()
        self.refused = set()
        bind_agreed(self._bind, self.group)

    def _linear(self, native, role):
        why = linear_refusal(native)
        if why is not None:
            # An optional projection can keep its native path. QKV is the
            # consumer of the fused quant output and must match its ABI.
            if role == "wqkv_a":
                raise MonoUnsupported(f"QKV layout: {why}")
            return native
        result = HybridLinear(native, self.profile.name, role)
        self.linears.append(result)
        return result

    def _bind(self):
        # Dependency refusal happens on every rank before a hybrid launch.
        from aiter.ops.triton.fusions.mhc import mhc_post_pre_quant_gfx1250
        from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import gemm_afp8wfp8_preshuffle

        assert callable(mhc_post_pre_quant_gfx1250) and callable(
            gemm_afp8wfp8_preshuffle
        )
        if not self.native.layers:
            raise MonoUnsupported("no decoder layers")
        for original in self.native.layers:
            if getattr(original, "_moe_merge_enabled", False):
                raise MonoUnsupported("MoE PCP merge")
            layer = module_view(original)
            attn = module_view(original.attn)
            layer.attn = attn
            for name in ("wqkv_a", "wq_b", "wo_b"):
                setattr(attn, name, self._linear(getattr(attn, name), name))
            if attn.indexer is not None:
                attn.indexer = module_view(attn.indexer)
                attn.indexer.wq_b = self._linear(attn.indexer.wq_b, "indexer.wq_b")
            ffn = module_view(original.ffn)
            layer.ffn = ffn
            if ffn.shared_experts is not None and self.profile.name == "flash_tp1":
                ffn.shared_experts = module_view(ffn.shared_experts)
                for name in ("gate_up_proj", "w2"):
                    setattr(
                        ffn.shared_experts,
                        name,
                        self._linear(
                            getattr(ffn.shared_experts, name), f"shared.{name}"
                        ),
                    )
            self.layers.append(layer)
        self.res_preshuffle = self.native.enable_res_preshuffle and (
            self.layers[0]._mhc_fused_post_pre is not None
        )
        if not all(layer._mhc_fused_post_pre is not None for layer in self.layers):
            raise MonoUnsupported("mHC fused post/pre unavailable")
        self.configs = configs(self.profile.name)

    def prepare(self, rows):
        if rows in self.prepared:
            return True
        if rows in self.refused or not 0 < rows <= self.profile.max_rows:
            return False
        if torch.cuda.is_current_stream_capturing():
            return False
        # Compile with real weight layouts before capture, agree any failure
        # across ranks, and refuse only this width. Kernels here have no
        # cross-CTA persistent waits or cross-GPU mailboxes.
        ok = True
        try:
            seen = set()
            for linear in self.linears:
                w = linear.native.weight
                key = f"{rows},{w.shape[0]},{w.shape[1]}"
                if key in seen or key not in self.configs:
                    continue
                seen.add(key)
                from aiter.ops.quant import per_group_quant_hip

                x, scales = per_group_quant_hip(
                    torch.zeros(
                        (rows, w.shape[1]), dtype=torch.bfloat16, device=w.device
                    ),
                    quant_dtype=torch.float8_e4m3fn,
                    group_size=128,
                    transpose_scale=True,
                    scale_type=torch.float8_e8m0fnu,
                )
                projection(
                    x,
                    w,
                    scales,
                    linear.native.weight_scale,
                    self.profile.name,
                    linear.role,
                )
            layer = self.layers[min(1, len(self.layers) - 1)]
            x = torch.zeros(
                (rows, self.profile.hidden),
                device=layer.hc_attn_fn.device,
                dtype=torch.bfloat16,
            )
            residual = self._repeat(x)
            post = torch.zeros((rows, 4), device=x.device, dtype=torch.float32)
            comb = torch.zeros((rows, 4, 4), device=x.device, dtype=torch.float32)
            self._mhc(layer, HCState(residual, post, comb, x, self.res_preshuffle))
        except Exception:
            logger.exception("DSV4 gfx1250 hybrid failed to prepare %d rows", rows)
            ok = False
        if not tp_agree(ok, self.group):
            self.refused.add(rows)
            return False
        self.prepared.add(rows)
        logger.info("DSV4 gfx1250 %s hybrid prepared %d rows", self.profile.name, rows)
        return True

    def _repeat(self, x):
        import aiter

        if hasattr(aiter, "mhc_res_repeat"):
            return aiter.mhc_res_repeat(x, 4, self.res_preshuffle)
        if self.res_preshuffle:
            raise MonoUnsupported("shuffled residual repeat unavailable")
        return x.unsqueeze(-2).repeat(1, 4, 1)

    @staticmethod
    def _mhc(layer, state):
        return mhc_quant(
            state.x_prev,
            state.residual,
            state.post_mix,
            state.comb_mix,
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            layer.attn_norm.weight,
            float(layer.norm_eps),
            float(layer.hc_eps),
            int(layer.hc_sinkhorn_iters),
            state.res_preshuffle,
        )

    def forward(self, input_ids, positions):
        h = self.native.embed(input_ids)
        state = HCState(residual=self._repeat(h), res_preshuffle=self.res_preshuffle)
        for layer in self.layers:
            q = qs = None
            if state.x_prev is not None:
                post, comb, x, res, q, qs = self._mhc(layer, state)
                state = HCState(res, post.squeeze(-1), comb, x, state.res_preshuffle)
            else:
                state = layer.fuse_hc(
                    state,
                    layer.hc_attn_fn,
                    layer.hc_attn_scale,
                    layer.hc_attn_base,
                    layer.attn_norm.weight,
                    layer.norm_eps,
                )
            # Calling the bound view's implementation retains native attention
            # work/stream ordering; the prefix-based opaque op targets the
            # original module and would bypass the hybrid linear views.
            state.x_prev = layer.attn.forward_impl(state.x_prev, positions, q, qs)
            state = layer.fuse_hc(
                state,
                layer.hc_ffn_fn,
                layer.hc_ffn_scale,
                layer.hc_ffn_base,
                layer.ffn_norm.weight,
                layer.norm_eps,
            )
            ffn = layer.ffn
            if ffn._use_dual_stream or getattr(ffn, "_use_comm_fused_dispatch", False):
                state.x_prev = ffn.dual_stream_moe_forward(state.x_prev)
            else:
                state.x_prev = ffn.single_stream_moe_forward(state.x_prev)
        last = self.layers[-1]
        return last.hc_post(
            state.x_prev,
            state.residual,
            state.post_mix,
            state.comb_mix,
            res_preshuffle=state.res_preshuffle,
        )
