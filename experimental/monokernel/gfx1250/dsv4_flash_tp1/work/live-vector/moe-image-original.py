# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

import torch
from aiter import ActivationType, QuantType, dtypes, get_hip_quant, topk_gating
from aiter.dist.parallel_state import get_dp_group, get_tp_group
from aiter.fused_moe import fused_moe
from aiter.jit.utils.chip_info import get_gfx
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import (
    interleave_gate_up_rows,
    moe_shuffle_scale,
    moe_shuffle_weight,
)
from torch import nn
from transformers import PretrainedConfig

from atom.config import (
    Config,
    QuantizationConfig,
    get_current_atom_config,
)
from atom.model_loader.weight_utils import set_weight_attrs
from atom.model_ops.base_config import QuantizeMethodBase
from atom.model_ops.eplb import eplb_map_and_record_fused
from atom.model_ops.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEQuantConfig,
    fp8_w8a8_moe_quant_config,
    moe_kernel_token_capacity,
    mxfp4_w4a8_moe_quant_config,
    mxfp4_w4a16_moe_quant_config,
)
from atom.model_ops.fused_moe.expert_layout import (
    MoEExpertLayout,
    SharedExpertMode,
    count_local_base_experts,
    determine_expert_map,
    expert_region,
    expert_shard_dim,
    expert_shard_view,
    physical_expert_id,
)
from atom.model_ops.fused_moe.modular_kernel import (
    FusedMoEModularKernel,
    FusedMoEPrepareAndFinalize,
)
from atom.model_ops.fused_moe.mori_prepare_finalize import (
    MoriPrepareAndFinalize,
    resolve_mori_dispatch,
)
from atom.model_ops.fused_moe.shared_expert_dispatch import remap_topk_to_dispatch
from atom.model_ops.topK import (
    init_aiter_topK_meta_data,
    is_rocm_aiter_fuse_routed_scaling_factor,
    is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config,
)
from atom.model_ops.topK import rocm_aiter_grouped_topk as grouped_topk
from atom.model_ops.topK import rocm_aiter_topk_softmax as fused_topk
from atom.model_ops.utils import (
    _has_module,
    atom_parameter,
    normalize_e4m3fn_to_e4m3fnuz,
    per_tensor_dequantize,
    shuffle_weights,
)
from atom.plugin.vllm.moe import FusedMoEDecoratorForPluginMode
from atom.quant_spec import (
    NVFP4_DTYPE,
    NVFP4_GROUP_SIZE,
    LayerQuantConfig,
    should_skip_online_quant,
    should_stream_online_quant,
    validate_nvfp4_global_scales,
    validate_nvfp4_online_target,
)
from atom.quantization.quark.utils import (
    dequant_moe_weight_online,
    dequant_weight_online,
    dequantize_nvfp4,
    quant_weight_online,
)
from atom.utils import envs
from atom.utils.custom_register import direct_register_custom_op
from atom.utils.decorators import mark_trace
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")

_NUM_TBO_UBATCHES = 2


class MoEActivationQuant(Enum):
    BF16 = "bf16"
    FP8 = "fp8"
    FP4 = "fp4"

    @staticmethod
    def from_model_config(a_quant_dtype: str | None) -> "MoEActivationQuant":
        if a_quant_dtype is None or a_quant_dtype == "":
            return MoEActivationQuant.BF16
        prefix = a_quant_dtype.split("_")[0]
        if prefix == "fp8":
            return MoEActivationQuant.FP8
        if prefix in ("fp4", "uint8"):
            return MoEActivationQuant.FP4
        return MoEActivationQuant.BF16


class FusedMoeWeightScaleSupported(Enum):
    """Supported quantization strategies for MoE weight scales."""

    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"


# Saturating magnitude for the synthetic logits. The picked/unpicked gap after
# sigmoid (or softmax / sqrtsoftplus) then swamps a router correction bias, which
# is added to the score before top-k. sigmoid caps that gap at 1.0, so a bias
# spread wider than 1.0 can still reorder the selection.
_FAKE_EPLB_LOGIT = 10.0


@lru_cache(maxsize=1)
def init_balance_router_logits(
    n_routed_experts: int,
    topk: int,
    ep_size: int = 1,
    dp_size: int = 1,
    dp_rank: int = 0,
    n_group: int = 1,
    topk_group: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    max_num_tokens: int = 32768,
):
    # Synthetic router logits whose top-k is balanced across both experts and EP
    # ranks. ATOM shards experts contiguously (rank r owns [r*L, (r+1)*L), with
    # L = E // ep_size), so we walk one flat ring: position p = t * topk + j takes
    # rank p % ep_size and slot (p // ep_size) % L. Consecutive positions step the
    # rank by one, so a token's experts land on distinct ranks and rank load stays
    # even at any batch size -- down to single-token decode, unlike a per-row
    # rotation.
    #
    # t = dp_rank + i * dp_size rather than plain i: the table is substituted per
    # device before the MoE gather, so the axis to interleave on is the token-shard
    # axis, which is DP. TP ranks inside a DP group see the same tokens and must
    # share one table; pure TP-attn + EP collapses to dp_size == 1.
    device = "cuda"
    E = n_routed_experts
    L = max(1, E // ep_size)  # experts per rank; a remainder is left unused
    t = dp_rank + torch.arange(max_num_tokens, device=device) * dp_size  # (T,)

    if n_group > 1 and 0 < topk_group < n_group and topk % topk_group == 0:
        # Group-limited routing (DeepSeek): every group but topk_group of n_group
        # is masked out before top-k, so a ring laid across all groups loses the
        # choices that fall in the masked ones. Run it inside a rotating set of
        # groups instead, and inside each group across the EP ranks that group
        # spans. A token can only reach topk_group groups by construction, so rank
        # load evens out over ceil(n_group / topk_group) tokens, not at T == 1.
        gs = E // n_group  # experts per group
        rg = max(1, gs // L)  # EP ranks spanned by one group
        sub = gs // rg  # experts per (group, rank) pair
        c = topk // topk_group  # choices placed in each selected group
        m = torch.arange(topk_group, device=device).view(1, -1, 1)
        k = torch.arange(c, device=device).view(1, 1, -1)
        q = t.view(-1, 1, 1) * topk_group + m  # position on the group ring
        p = (q // n_group) * c + k  # visits to that group so far
        slot = (p % rg) * sub + (p // rg) % sub
        expert_ids = ((q % n_group) * gs + slot).reshape(max_num_tokens, topk)
    else:
        j = torch.arange(topk, device=device).unsqueeze(0)
        p = t.unsqueeze(1) * topk + j
        expert_ids = (p % ep_size) * L + (p // ep_size) % L  # (T, topk)

    router_logits = torch.full(
        (max_num_tokens, E), -_FAKE_EPLB_LOGIT, dtype=dtype, device=device
    )
    router_logits.scatter_(1, expert_ids, _FAKE_EPLB_LOGIT)
    return router_logits


@dataclass
class FusedMoEParallelConfig:
    tp_size: int
    dp_size: int
    ep_size: int
    tp_rank: int
    dp_rank: int
    ep_rank: int

    use_ep: bool  # whether to use EP or not
    local_ep_size: int

    # Config.dp_logical_size / dp_size: >1 when DP-attention simulates a
    # deployment wider than the box, in which case experts shard that many
    # times finer and each rank repeats the gathered tokens that many times.
    dp_logical_ratio: int = 1
    requested_all2all_backend: str = "auto"
    # The other half of --all2all-backend: which MoRI kernel family to ask for.
    low_latency: bool = False

    @property
    def selected_all2all_backend(self) -> str | None:
        if self.dp_size <= 1 or not self.use_ep or self.dp_logical_ratio != 1:
            return None
        if self.requested_all2all_backend == "none":
            return None
        if self.requested_all2all_backend == "rccl":
            return "rccl"
        if not envs.ATOM_DISABLE_MORI_EP and _has_module("mori"):
            return "mori"
        return None

    @property
    def use_all2all_kernels(self):
        # Routed all-to-all requires real peers. Simulated DP has a finer
        # expert map than the launched process group and stays on the fallback.
        return self.selected_all2all_backend is not None

    @property
    def use_mori_kernels(self):
        return self.selected_all2all_backend == "mori"

    @property
    def use_rccl_kernels(self):
        return self.selected_all2all_backend == "rccl"

    @staticmethod
    def make(
        tp_size_: int, dp_size_: int, parallel_config: Config
    ) -> "FusedMoEParallelConfig":
        # Width the expert sharding is cut for; wider than the real DP size only
        # while simulating. The rank stays real, so this device keeps the slice
        # it would own in the full deployment.
        dp_logical = getattr(parallel_config, "dp_logical_size", 0) or dp_size_

        def flatten_tp_across_dp(dp_rank: int):
            tp_rank = 0 if tp_size_ == 1 else get_tp_group().rank_in_group
            # There are actually dp_logical * tp_size_ devices. Update tp_size
            # and tp_rank so we shard across all devices.
            tp_size = dp_logical * tp_size_
            tp_rank = dp_rank * tp_size_ + tp_rank
            return tp_size, tp_rank

        # Only flatten DP into TP/EP when enable_dp_attention is True.
        # Otherwise, use pure DP for MoE.
        enable_dp_attention = parallel_config.enable_dp_attention
        requested_all2all_backend = getattr(
            parallel_config, "moe_all2all_backend", "auto"
        )
        low_latency = getattr(parallel_config, "enable_low_latency", False)

        # When EP shards across the flattened DP * TP space (vLLM plugin under
        # EP), the ep rank must be computed in that flattened group space.
        flatten_tp_across_dp_for_moe = (
            enable_dp_attention or parallel_config.moe_ep_flatten_tp_across_dp
        )

        # dp_logical, not dp_size_: with DP-attention simulated down to a single
        # rank the real product is 1, but the deployment being reproduced still
        # shards experts, so EP must stay on.
        use_ep = dp_logical * tp_size_ > 1 and parallel_config.enable_expert_parallel

        dp_size = dp_size_
        dp_rank = get_dp_group().rank_in_group if dp_size > 1 else 0

        if flatten_tp_across_dp_for_moe:
            tp_size, tp_rank = flatten_tp_across_dp(dp_rank)
        else:
            tp_size = tp_size_
            tp_rank = 0 if tp_size_ == 1 else get_tp_group().rank_in_group

        # PCP moe_pcp_merge: fold the prefill-context-parallel dim into the MoE
        # tensor/expert sharding so the W=pcp_size redundant copies become real
        # shards (intermediate//W*tp or expert//W*tp). The flattened rank MUST be
        # `pcp_rank * tp_size + tp_rank`: this makes each TP group [0..tp-1] own a
        # contiguous half and its PCP partner own the other half, so the existing
        # tp all_reduce + the new pcp reduce_scatter (two orthogonal groups) sum
        # all W*tp shards. The reversed mapping would make PCP partners overlap
        # and double-count. ep_size/ep_rank below inherit tp_size/tp_rank, so EP
        # is covered by the same flatten.
        from aiter.dist.parallel_state import (
            get_prefill_context_model_parallel_rank,
            get_prefill_context_model_parallel_world_size,
        )

        pcp_merge = (
            envs.ATOM_PCP_MOE_MERGE
            and get_prefill_context_model_parallel_world_size() > 1
        )
        if pcp_merge:
            pcp_size = get_prefill_context_model_parallel_world_size()
            pcp_rank = get_prefill_context_model_parallel_rank()
            tp_rank = pcp_rank * tp_size + tp_rank
            tp_size = pcp_size * tp_size

        atom_config = get_current_atom_config()

        if not use_ep:
            return FusedMoEParallelConfig(
                tp_size=tp_size,
                tp_rank=tp_rank,
                dp_size=dp_size,
                dp_rank=dp_rank,
                ep_size=1,
                ep_rank=0,
                use_ep=False,
                local_ep_size=1,
                requested_all2all_backend=requested_all2all_backend,
                low_latency=low_latency,
            )
        # DP + EP / TP + EP / DP + TP + EP
        assert use_ep
        # In EP, each device owns a set of experts fully. There is no tensor
        # parallel update tp_size, tp_rank, ep_size and ep_rank to reflect that.
        ep_size = tp_size
        ep_rank = tp_rank
        return FusedMoEParallelConfig(
            tp_size=1,
            tp_rank=0,
            dp_size=dp_size,
            dp_rank=dp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            use_ep=True,
            # CoreManager guarantees the division is exact.
            dp_logical_ratio=(
                dp_logical // dp_size if flatten_tp_across_dp_for_moe else 1
            ),
            local_ep_size=atom_config.parallel_config.data_parallel_size_local
            * tp_size_,
            requested_all2all_backend=requested_all2all_backend,
            low_latency=low_latency,
        )


def naive_multicast_fake(
    x: torch.Tensor, cu_tokens_across_dp_cpu: torch.Tensor
) -> torch.Tensor:
    assert len(x.shape) == 2
    # print(f"cu_tokens_across_dp_cpu: {cu_tokens_across_dp_cpu}")
    buffer = torch.empty(
        (cu_tokens_across_dp_cpu[-1], x.size(1)), device=x.device, dtype=x.dtype
    )
    return buffer


@torch_compile_guard()
def naive_multicast(
    x: torch.Tensor, cu_tokens_across_dp_cpu: torch.Tensor
) -> torch.Tensor:
    dp_rank = get_dp_group().rank_in_group
    assert len(x.shape) == 2
    # print(f"cu_tokens_across_dp_cpu: {cu_tokens_across_dp_cpu}")
    buffer = torch.empty(
        (cu_tokens_across_dp_cpu[-1], x.size(1)), device=x.device, dtype=x.dtype
    )

    start = 0 if dp_rank == 0 else cu_tokens_across_dp_cpu[dp_rank - 1]
    end = cu_tokens_across_dp_cpu[dp_rank]
    buffer[start:end, :].copy_(x)
    for idx in range(get_dp_group().world_size):
        start = 0 if idx == 0 else cu_tokens_across_dp_cpu[idx - 1]
        end = cu_tokens_across_dp_cpu[idx]
        get_dp_group().broadcast(buffer[start:end, :], idx)
    return buffer


def pad_for_all_gather(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Pad ``x`` up to the uniform all-gather height, and return both.

    The height comes off the context, never off ``x``: this pads for a
    collective, so guessing it from the tensor turns a caller's mistake into a
    DP-wide hang instead of a local failure. A decode forward arrives already
    at ``running_tokens`` (the runner padded ``input_ids``), a prefill at
    ``scheduled_tokens``; a draft states its own pair before it runs
    (`Drafter._publish_draft_shape`), so it needs no case here.

    ATOM therefore never reaches the copy below -- it only runs uniform decode,
    where the two are equal. The bridges can, on a prefill they call uniform.

    Pad rows are left UNINITIALIZED: the expert GEMM is per-token and
    ``reduce_scatter_with_unpadding`` drops them. They DO reach the EPLB
    counter, since ``get_moe_input`` gathers ``router_logits`` the same way --
    exclude them there rather than zeroing here, which skews it just as much.
    """
    ctx = get_forward_context().context
    running_tokens = ctx.running_tokens
    local_tokens = ctx.scheduled_tokens if ctx.is_prefill else running_tokens
    assert x.shape[0] == local_tokens, (
        f"MoE was handed {x.shape[0]} rows on a "
        f"{'prefill' if ctx.is_prefill else 'decode'} step expecting "
        f"{local_tokens} (scheduled_tokens={ctx.scheduled_tokens}, "
        f"running_tokens={running_tokens})"
    )
    if local_tokens >= running_tokens:
        return x, local_tokens

    padding_shape = list(x.shape)
    padding_shape[0] = running_tokens
    padded_x = torch.empty(padding_shape, device=x.device, dtype=x.dtype)
    padded_x[:local_tokens, :].copy_(x)
    # padded_x[local_tokens:, :].zero_() # keep for debug
    return padded_x, local_tokens


def all_gather_with_padding(
    x: torch.Tensor, use_cag: bool = True
) -> tuple[torch.Tensor, int]:
    padded_x, local_tokens = pad_for_all_gather(x)
    # use_custom=True routes through CA IPC (outplace_all_gather). Default
    # use_custom=False falls back to torch.distributed.all_gather_into_tensor
    # (NCCL), whose WorkNCCL end-event recorded inside CUDAGraph capture is
    # later queried by the watchdog thread -> hipErrorCapturedEvent crash.
    gathered_hidden_states = get_dp_group().all_gather(
        padded_x, use_custom=use_cag, dim=0
    )
    return gathered_hidden_states, local_tokens


def repeat_rows(x: torch.Tensor, times: int) -> torch.Tensor:
    """Repeat `x` along dim 0 so it holds `times` copies of what was gathered.

    Stands in for the token shards of the DP ranks a simulated run did not
    launch, giving the expert GEMMs the full deployment's token volume. Under
    `--fake-eplb` the gathered rows cover a whole number of turns of the
    balanced expert ring, so copying them keeps every expert's count equal.

    The caller drops the copies before the reduce scatter.
    """
    if times <= 1:
        return x
    return x.repeat(times, *([1] * (x.dim() - 1)))


def reduce_scatter_with_unpadding(x: torch.Tensor, local_tokens: int) -> torch.Tensor:
    dp_group = get_dp_group()
    scattered_output = dp_group.reduce_scatter_tensor(x)

    # Drop the rows pad_for_all_gather appended (padding is on dim 0). Their
    # contents were never initialized, so they must not survive past here.
    if scattered_output.shape[0] > local_tokens:
        scattered_output = scattered_output[:local_tokens]

    return scattered_output


def all_gatherv(x: torch.Tensor, sizes: list[int], group) -> torch.Tensor:
    return group.device_communicator.all_gatherv(x, dim=0, sizes=list(sizes))


def reduce_scatterv(x: torch.Tensor, sizes: list[int], group) -> torch.Tensor:
    return group.device_communicator.reduce_scatterv(x, dim=0, sizes=list(sizes))


def dp_gather_hidden_and_router(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    dp_eager_mode: bool,
    ctx,
    dp_group,
):
    """All-gather ``hidden_states`` + ``router_logits`` across DP ranks.

    - **Eager (variable-length) mode**: per-rank token counts may differ, so
      we ``all_gatherv`` with per-rank sizes. Both tensors are concatenated
      along the last dim and gathered in a single collective to avoid the
      cost of two separate rounds (caller splits them back out).
    - **Uniform mode** (DP-decode-only / CUDAGraph path): every rank has the
      same token count, so a plain padded ``all_gather`` per tensor is
      enough.

    Returns ``(hidden_states, router_logits, local_tokens, sizes)``, where
    ``local_tokens`` is this rank's own height -- what ``reduce_scatterv`` /
    ``reduce_scatter_with_unpadding`` must trim back to. ``sizes`` is non-None
    only in eager mode (needed later for ``reduce_scatterv``).
    """
    if dp_eager_mode:
        sizes = ctx.dp_metadata.get_sizes_across_dp()
        # Eager means nothing was padded, so the context's two heights agree
        # and either is the row count to unpad back to. Taken from the context
        # rather than the tensor for the reason `pad_for_all_gather` gives.
        local_tokens = ctx.context.scheduled_tokens
        assert hidden_states.shape[0] == local_tokens, (
            f"MoE was handed {hidden_states.shape[0]} rows on an eager step "
            f"expecting {local_tokens}"
        )
        h_dim = hidden_states.shape[-1]
        r_dim = router_logits.shape[-1]
        r_dtype = router_logits.dtype
        # Cast router_logits to hidden_states.dtype for the single fused
        # gather; cast back after split. (all_gatherv requires uniform dtype.)
        r_for_gather = (
            router_logits
            if r_dtype == hidden_states.dtype
            else router_logits.to(hidden_states.dtype)
        )
        combined = torch.cat([hidden_states, r_for_gather], dim=-1)
        combined = all_gatherv(combined, sizes, dp_group)
        hidden_states, router_logits = combined.split([h_dim, r_dim], dim=-1)
        hidden_states = hidden_states.contiguous()
        if router_logits.dtype != r_dtype:
            router_logits = router_logits.to(r_dtype)
        else:
            router_logits = router_logits.contiguous()
        return hidden_states, router_logits, local_tokens, sizes

    hidden_states, local_tokens = all_gather_with_padding(hidden_states)
    router_logits, _ = all_gather_with_padding(router_logits)
    return hidden_states, router_logits, local_tokens, None


@torch_compile_guard()
def get_max_tokens_across_dispatchers(input: torch.Tensor) -> int:
    return input.item()


class FusedMoEMethodBase(QuantizeMethodBase):
    def __init__(self, moe: FusedMoEConfig):
        super().__init__()
        self.moe = moe
        self.moe_quant_config: FusedMoEQuantConfig | None = None
        self.fused_experts: FusedMoEModularKernel | None = None
        self.topk_indices_dtype = None

    @abstractmethod
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: torch.Tensor | None = None,
        fused_shared_experts_scoring_func: str | None = None,
        apply_router_weight_on_input: bool = False,
        activation: ActivationType = ActivationType.Silu,
        prefix: str = "",
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        raise NotImplementedError

    def select_experts_with_record(
        self,
        *,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        use_grouped_topk: bool = False,
        top_k: int,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: torch.Tensor | None = None,
        fused_shared_experts_scoring_func: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_weights, topk_logical = FusedMoE.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            num_routing_experts=layer.expert_layout.num_routed,
            num_fused_shared_experts=(
                0
                if layer.expert_layout.uses_all2all_fusion
                else layer.num_fused_shared_experts
            ),
            fused_shared_experts_scoring_func=fused_shared_experts_scoring_func,
            routed_scaling_factor=layer.routed_scaling_factor,
        )
        if layer.expert_layout.shared_is_routed:
            # EPLB places and records shared experts with routed experts.
            topk_weights, topk_logical = layer.append_shared_logical_column(
                topk_weights, topk_logical
            )
            return topk_weights, eplb_map_and_record_fused(layer, topk_logical)
        topk_physical = eplb_map_and_record_fused(layer, topk_logical)
        return layer.to_dispatch_space(topk_weights, topk_physical)

    @staticmethod
    def _maybe_make_prepare_finalize(
        moe: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig | None,
    ) -> FusedMoEPrepareAndFinalize | None:
        from aiter.dist.parallel_state import get_ep_group

        prepare_finalize: FusedMoEPrepareAndFinalize | None = None
        ep_group = get_ep_group()

        if moe.use_rccl_kernels:
            from atom.model_ops.fused_moe.rccl_prepare_finalize import (
                RcclPrepareAndFinalize,
            )

            return RcclPrepareAndFinalize(
                ep_group,
                num_local_experts=moe.num_local_experts,
                max_tokens_per_rank=moe.max_num_tokens,
                num_replicated_shared_experts=(
                    moe.expert_layout.num_fused_shared_experts
                    if moe.expert_layout.uses_dispatch_remap
                    else 0
                ),
                num_routed_experts_per_rank=(
                    moe.expert_layout.routed_physical_per_rank
                ),
            )

        all2all_manager = ep_group.device_communicator.all2all_manager
        assert all2all_manager is not None

        # TODO: could allow this now
        # assert not moe.use_flashinfer_cutlass_kernels, "Must be created in modelopt.py"
        if moe.use_mori_kernels:
            from atom.utils import envs as _atom_envs

            # gfx1250: use mori dispatch_combine_v2 (cco/FlyDSL) instead of the
            # gfx942/950-only v1 kernels. Gated by ATOM_MORI_V2.
            if _atom_envs.ATOM_MORI_V2:
                from atom.model_ops.fused_moe.mori_v2_prepare_finalize import (
                    make_mori_v2_prepare_finalize,
                )

                return make_mori_v2_prepare_finalize(moe, all2all_manager)

            assert quant_config is not None

            # One branch decides the whole wire format: the dtype dispatch() will
            # see (which is what selects the MoRI kernel), the pre-dispatch
            # quantizer, and the staging scale geometry. Keeping these together is
            # load-bearing -- a scale_dim/scale_type_size that disagrees with what
            # prepare() sends makes MoRI stride the staging scale buffer wrong and
            # walk off the end of it (GPU memory fault on the first real batch).
            dispatch_format = resolve_mori_dispatch(
                in_dtype=moe.in_dtype,
                hidden_dim=moe.hidden_dim,
                quant_config=quant_config,
            )
            mori_dtype = dispatch_format.dtype
            scale_type_size = dispatch_format.scale_type_size
            # For PTPC (per token per channel) quant, the scale dim for each token is 1
            # For 1x128 quant, the scale dim for each token is hidden_dim // 128
            scale_dim = (
                dispatch_format.scale_dim
                if dispatch_format.is_fp4
                else (1 if quant_config.is_per_act_token else moe.hidden_dim // 128)
            )
            # Combine-side codec, passed through aiter's all2all manager into the
            # MoRI config. "none" (the MoRI default) sends bf16 back; "fp8_blockwise"
            # selects EpCombineIntraNodeKernel_*_fp8bwq_*. Independent of the dispatch
            # dtype above -- MoRI infers dispatch from the tensor, combine from this.
            mori_combine_quant_type = envs.ATOM_MORI_COMBINE_QUANT

            from atom.utils.tbo.ubatching import tbo_enabled

            tbo_is_enabled = tbo_enabled()
            low_latency = moe.moe_parallel_config.low_latency
            if all2all_manager.internode and mori_combine_quant_type != "none":
                # MoRI registers no blockwise combine kernel for these; without
                # this it fails later, at kernel lookup.
                raise ValueError(
                    f"ATOM_MORI_COMBINE_QUANT={mori_combine_quant_type!r} is "
                    "unsupported on the inter-node MoRI kernels."
                )

            all_to_all_args = {
                "rank": all2all_manager.rank,
                "num_ep_ranks": all2all_manager.world_size,
                # Must be dispatch_format.dtype: MoRI sizes its staging buffers
                # from this and picks the kernel from the tensor prepare() sends.
                "quant_dtype": mori_dtype,
                "token_hidden_size": moe.hidden_dim,
                "scale_dim": scale_dim,
                "scale_type_size": scale_type_size,
                "max_num_tokens_per_dp_rank": moe.max_num_tokens,
                # input_dtype=moe.in_dtype,
                "input_dtype": moe.in_dtype,
                "num_local_experts": moe.num_local_experts,
                "num_experts_per_token": moe.experts_per_token,
                "gpu_per_node": moe.moe_parallel_config.local_ep_size,
                # aiter crosses this with its own internode probe.
                "low_latency": low_latency,
            }
            if mori_combine_quant_type != "none":
                # Only inject when actually requested: aiter's _make_all2all_kwargs
                # takes fixed named params, so an unpatched aiter raises TypeError on
                # an unexpected "quant_type" kwarg.
                all_to_all_args["quant_type"] = mori_combine_quant_type

            # TBO and regular forwards are mutually exclusive. Slot 0 serves
            # regular forwards and TBO ubatch 0; slot 1 exists only for the
            # concurrently running second ubatch.
            num_mori_ops = _NUM_TBO_UBATCHES if tbo_is_enabled else 1
            mori_ops = [
                all2all_manager.get_handle(all_to_all_args, index=slot)
                for slot in range(num_mori_ops)
            ]

            # Off the op, not re-derived: the mapping is aiter's.
            kernel_name = mori_ops[0].config.kernel_type.name
            logger.info(
                "[MORI] EP all2all kernel=%s internode=%s low_latency=%s "
                "tbo_ubatch_ops=%d",
                kernel_name,
                all2all_manager.internode,
                low_latency,
                len(mori_ops) if tbo_is_enabled else 0,
            )

            prepare_finalize = MoriPrepareAndFinalize(
                mori_ops,
                max_tokens_per_rank=moe.max_num_tokens,
                num_dispatchers=all2all_manager.world_size,
                dispatch_format=dispatch_format,
                low_latency=low_latency,
                internode=all2all_manager.internode,
            )

        return prepare_finalize

    def maybe_make_prepare_finalize(self) -> FusedMoEPrepareAndFinalize | None:
        # if True:
        if self.moe.moe_parallel_config.use_all2all_kernels:
            return FusedMoEMethodBase._maybe_make_prepare_finalize(
                self.moe, self.moe_quant_config
            )
        else:
            return None

    # Note: init_prepare_finalize should only be called by
    # prepare_communication_buffer_for_model.
    def init_prepare_finalize(self, layer: torch.nn.Module):
        # print("init_prepare_finalize")
        assert self.moe is not None

        # We must get the quant config here so that the layer is
        # completely initialized, i.e. all weights loaded and post
        # processed.
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

        prepare_finalize = self.maybe_make_prepare_finalize()

        if prepare_finalize is not None:
            # logger.debug(
            #     "%s for %s(%s)", prepare_finalize.__class__.__name__, self, id(self)
            # )
            assert self.topk_indices_dtype is None
            assert (
                self.fused_experts is None
            ), f"Attempt to override experts for {id(self)}!"
            self.topk_indices_dtype = prepare_finalize.topk_indices_dtype()
            # experts = self.select_gemm_impl(prepare_finalize, layer)
            from atom.model_ops.fused_moe.mori_v2_prepare_finalize import (
                MoriV2ModularKernel,
                MoriV2PrepareAndFinalize,
            )

            modular_cls = (
                MoriV2ModularKernel
                if isinstance(prepare_finalize, MoriV2PrepareAndFinalize)
                else FusedMoEModularKernel
            )
            self.fused_experts = modular_cls(
                prepare_finalize,
                # experts,
                # layer.shared_experts,
                quant_config=self.moe_quant_config,
            )

            # The v2 fused transport (MegaMoE) runs the whole layer, so it must be
            # told the expert-GEMM recipe that only the layer + this quant method
            # know. Here is also the last point before any cudagraph capture, and
            # it allocates a cco arena and JIT-compiles its kernels.
            bind_mega = getattr(prepare_finalize, "bind_mega_transport", None)
            if bind_mega is not None:
                bind_mega(layer, self)

    @property
    def using_modular_kernel(self) -> bool:
        return self.fused_experts is not None


class UnquantizedFusedMoEMethod(FusedMoEMethodBase):
    """MoE method without quantization."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Streaming source weights start on meta; target weights remain real.
        stream_online_quant = getattr(layer, "_stream_online_quant", False)
        weight_device = "meta" if stream_online_quant else None

        # Fused gate_up_proj (column parallel)
        w13_weight = atom_parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
                device=weight_device,
            )
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = atom_parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
                device=weight_device,
            )
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

    def _maybe_pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return weight

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

        layer.w13_weight = atom_parameter(self._maybe_pad_weight(layer.w13_weight.data))
        layer.w2_weight = atom_parameter(self._maybe_pad_weight(layer.w2_weight.data))
        # reshaping weights is required for aiter moe kernel.
        shuffle_weights(layer.w13_weight, layer.w2_weight)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return FUSED_MOE_UNQUANTIZED_CONFIG

    @mark_trace(prefix="unquantized_moe", torch_compile=False)
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: torch.Tensor | None = None,
        fused_shared_experts_scoring_func: str | None = None,
        apply_router_weight_on_input: bool = False,
        activation: ActivationType = ActivationType.Silu,
        prefix: str = "",
    ) -> torch.Tensor:
        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            num_routing_experts=global_num_experts,
            num_fused_shared_experts=layer.num_fused_shared_experts,
            fused_shared_experts_scoring_func=fused_shared_experts_scoring_func,
            routed_scaling_factor=layer.routed_scaling_factor,
        )
        if self.fused_experts:
            return self.fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=False,
                activation=activation,
                quant_type=QuantType.No,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                expert_mask=layer.expert_mask,
            )
        return fused_moe(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weight=topk_weights,
            topk_ids=topk_ids,
            expert_mask=layer.expert_mask,
            activation=activation,
        )


def rocm_asm_moe_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: torch.Tensor | None = None,
    activation: int = ActivationType.Silu.value,
    quant_type: int = QuantType.No.value,
    doweight_stage1: bool = False,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    from aiter import ActivationType, QuantType
    from aiter.fused_moe_bf16_asm import asm_moe

    activation_ = ActivationType(activation)
    quant_type_ = QuantType(quant_type)

    # - fc1_scale: [E, inter_dim*2, 1]
    # - fc2_scale: [E, model_dim, 1]
    # - fc1_smooth_scale: [E, model_dim]
    # - fc2_smooth_scale: [E, inter_dim]
    fc1_scale_fixed = w1_scale
    fc2_scale_fixed = w2_scale
    fc1_smooth_scale_fixed = a1_scale
    fc2_smooth_scale_fixed = a2_scale

    a16_mode = (
        quant_type_ in [QuantType.per_Token, QuantType.per_1x128]
        and hidden_states.dtype in [torch.float16, torch.bfloat16]
        and w1.dtype in [torch.int8, torch.uint8, torch.float8_e4m3fnuz]
        and fc1_smooth_scale_fixed is not None
        and fc2_smooth_scale_fixed is not None
    )

    return asm_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        fc1_scale=fc1_scale_fixed,
        fc2_scale=fc2_scale_fixed,
        fc1_smooth_scale=fc1_smooth_scale_fixed,
        fc2_smooth_scale=fc2_smooth_scale_fixed,
        a16=a16_mode,
        per_tensor_quant_scale=None,
        block_shape=None,
        expert_mask=expert_mask,
        activation=activation_,
    )


def rocm_aiter_fused_moe_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: torch.Tensor | None = None,
    activation: int = ActivationType.Silu.value,
    quant_type: int = QuantType.No.value,
    doweight_stage1: bool = False,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    swiglu_limit: float = 0.0,
    gate_mode: str = GateMode.SEPARATED.value,
) -> torch.Tensor:
    from aiter import ActivationType, QuantType

    activation_ = ActivationType(activation)
    quant_type_ = QuantType(quant_type)

    return fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        expert_mask,
        activation_,
        quant_type_,
        doweight_stage1,
        w1_scale,
        w2_scale,
        a1_scale,
        a2_scale,
        swiglu_limit=swiglu_limit,
        gate_mode=gate_mode,
    )


def rocm_aiter_fused_moe_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: torch.Tensor | None = None,
    activation: int = ActivationType.Silu.value,
    quant_type: int = QuantType.No.value,
    doweight_stage1: bool = False,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    swiglu_limit: float = 0.0,
    gate_mode: str = GateMode.SEPARATED.value,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="rocm_aiter_fused_moe",
    op_func=rocm_aiter_fused_moe_impl,
    mutates_args=[],
    fake_impl=rocm_aiter_fused_moe_fake,
)


class Nvfp4MoEMethod(FusedMoEMethodBase):
    """Quark / NVIDIA ModelOpt NVFP4 MoE source representation.

    This deliberately stops before any kernel-specific shuffle. The raw
    group-16 E4M3 scales and their per-tensor global scales remain in checkpoint
    layout until the online pass converts them to MXFP4.
    Direct NVFP4 inference is intentionally unsupported.
    """

    def __init__(self, quant_config: LayerQuantConfig, moe: FusedMoEConfig):
        super().__init__(moe)
        self.quant_config = quant_config
        self.quant_type = quant_config.quant_type
        self.quant_dtype = quant_config.quant_dtype
        self.quant_method = quant_config.quant_method or ""
        self.group_size = NVFP4_GROUP_SIZE
        self.pad_align = 1

    @staticmethod
    def _register_parameter(
        layer: torch.nn.Module,
        name: str,
        tensor: torch.Tensor,
        extra_weight_attrs: dict,
    ) -> None:
        param = atom_parameter(tensor)
        layer.register_parameter(name, param)
        set_weight_attrs(param, extra_weight_attrs)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del params_dtype
        if hidden_size % self.group_size != 0:
            raise ValueError(
                "NVFP4 hidden size "
                f"({hidden_size}) must be divisible by group_size={self.group_size}."
            )
        if intermediate_size_per_partition % self.group_size != 0:
            raise ValueError(
                "NVFP4 intermediate size per TP partition "
                f"({intermediate_size_per_partition}) must be divisible by "
                f"group_size={self.group_size}."
            )
        # The only way out of NVFP4 is MXFP4, whose 32-wide blocks run along
        # hidden for w13 and along the TP-local intermediate for w2. A size
        # that is only 16-aligned otherwise loads the whole checkpoint and
        # then trips a message-less `cols % group_size` assert in aiter.
        if hidden_size % 32 != 0:
            raise ValueError(
                f"NVFP4 hidden size ({hidden_size}) is valid for the source "
                "group size 16 but cannot be converted to MXFP4, which "
                "requires a multiple of 32."
            )
        if intermediate_size_per_partition % 32 != 0:
            raise ValueError(
                "NVFP4 intermediate size per TP partition "
                f"({intermediate_size_per_partition}) is valid for the source "
                "group size 16 but cannot be converted to MXFP4, which "
                "requires a multiple of 32. Choose a TP size that keeps it "
                "aligned."
            )
        # An unregistered bias is skipped by the loader without a warning, and
        # the MXFP4 target would then allocate zeros for it.
        if layer.has_bias:
            raise ValueError(
                f"{getattr(layer, 'prefix', '<fused_moe>')}: NVFP4 MoE expert "
                "biases are not supported. Supporting them requires registering "
                "w13_bias/w2_bias in Nvfp4MoEMethod.create_weights and copying "
                "them into the MXFP4 buffers in FusedMoE._online_quant."
            )

        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size_per_partition
        self.hidden_pad = 0
        # None until a model pins the MXFP4 target's padding here.
        self.intermediate_pad = None
        scale_dtype = torch.float8_e4m3fn
        weight_device = (
            "meta" if getattr(layer, "_stream_online_quant", False) else None
        )

        self._register_parameter(
            layer,
            "w13_weight",
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.uint8,
                device=weight_device,
            ),
            extra_weight_attrs,
        )
        self._register_parameter(
            layer,
            "w2_weight",
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
                device=weight_device,
            ),
            extra_weight_attrs,
        )
        self._register_parameter(
            layer,
            "w13_weight_scale",
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=scale_dtype,
                device=weight_device,
            ),
            extra_weight_attrs,
        )
        self._register_parameter(
            layer,
            "w2_weight_scale",
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=scale_dtype,
                device=weight_device,
            ),
            extra_weight_attrs,
        )

        # w13 combines w1 and w3, so retain both source projections' global
        # weight scales independently. Input scales are not consumed by the
        # dynamic MXFP4 target and are intentionally not registered.
        for name, shape in (
            ("w13_weight_scale_2", (num_experts, 2)),
            ("w2_weight_scale_2", (num_experts,)),
        ):
            self._register_parameter(
                layer,
                name,
                torch.zeros(shape, dtype=torch.float32, device=weight_device),
                extra_weight_attrs,
            )

        layer.register_parameter("w13_bias", None)
        layer.register_parameter("w2_bias", None)

        # Names read by common FusedMoE plumbing.
        layer.w13_input_scale = None
        layer.w2_input_scale = None
        layer.w13_swizzle_layout = None
        layer.w2_swizzle_layout = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        raise RuntimeError(
            f"{getattr(layer, 'prefix', '<fused_moe>')}: NVFP4 expert weights "
            "were not converted to MXFP4 by FusedMoE online quantization."
        )

    def init_prepare_finalize(self, layer: torch.nn.Module) -> None:
        # No communication/kernel buffers are valid until conversion to a
        # supported runtime format has happened.
        return

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return None

    def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        raise RuntimeError(
            f"{getattr(layer, 'prefix', '<fused_moe>')}: NVFP4 checkpoint "
            "weights are loaded, but direct NVFP4 inference is intentionally "
            "blocked. Convert this layer to MXFP4 during online quantization "
            "before forward."
        )


class Mxfp4MoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config: LayerQuantConfig, moe: FusedMoEConfig):
        super().__init__(moe)
        self.quant_config = quant_config
        self.quant_type = quant_config.quant_type
        self.quant_dtype = quant_config.quant_dtype
        self.quant_method = quant_config.quant_method or ""
        self.static_input_scales = not quant_config.is_dynamic
        self.fp8_activations = moe.mxfp4_fp8_activations
        self.is_guinterleave = envs.ATOM_MOE_GU_ITLV or self.fp8_activations
        self.pad_align = 256
        self.block_quant = (
            self.quant_type == QuantType.per_1x128
            or self.quant_type == QuantType.per_1x32
        )
        gfx = get_gfx()
        self.is_gfx1250 = gfx == "gfx1250"
        self.act_quant = MoEActivationQuant.from_model_config(moe.a_quant_dtype)
        ep_moe = self.moe.use_ep
        if envs.is_set("ATOM_USE_TRITON_MOE"):
            use_triton_moe = envs.ATOM_USE_TRITON_MOE
        else:
            use_triton_moe = not ep_moe and (
                gfx.startswith("gfx94")
                or (gfx.startswith("gfx95") and envs.ATOM_USE_TRITON_GEMM)
            )
        # One env flag, two disjoint paths. TP runs the experts inside apply();
        # EP runs them from the modular kernel, between the mori dispatch and
        # combine, over rows that arrived from other ranks. They share the aiter
        # kernels but nothing else -- different weight prep, different routing
        # source -- so they are tracked separately rather than by one `use_triton`
        # that each call site has to re-interpret.
        self.use_triton = use_triton_moe and not ep_moe
        self.use_triton_ep = use_triton_moe and ep_moe
        # Phase split: keep the weights in the FlyDSL layout and run the Triton
        # /gluon kernel on decode only, over a zero-copy view rebuilt in apply().
        # Narrows ATOM_USE_TRITON_MOE, so it is inert unless that path is on.
        #
        # Now arms under EP as well as TP. The weight-sharing premise is the same
        # either way -- one FlyDSL copy, a zero-copy Triton view over it -- and it
        # rests on gfx1250 + GUGU + SiLU + zero pad, which the asserts in
        # _process_weight_layout_after_loading enforce for both.
        #
        # Both EP entry points honour it: the modular-kernel (transport) path
        # publishes `triton_experts` built from _triton_views_of_flydsl_weights,
        # and the local no-transport path builds the same views. Neither reads
        # the branch-A layout under this flag, because the prep never wrote one.
        self.use_triton_decode = (
            self.use_triton or self.use_triton_ep
        ) and envs.ATOM_USE_TRITON_MOE_DECODE

        # EPLB owns logical-to-physical routing, load recording, and live expert
        # migration. The Triton forward paths bypass that routing flow, and their
        # weight layout is not migration-safe, so EPLB must use the standard path
        # even when no redundant experts are configured.
        if getattr(get_current_atom_config(), "eplb_enable", False):
            self.use_triton = False
            self.use_triton_decode = False
            self.use_triton_ep = False

        # Triton MoE under EP exists only for gfx95x and gfx125x -- those are the
        # arches whose branch-A weight prep and gluon experts are wired up. The
        # arch test that computes `use_triton_moe` above is bypassed whenever
        # ATOM_USE_TRITON_MOE is set explicitly, so a fleet-wide
        # ATOM_USE_TRITON_MOE=1 would otherwise reach the modular kernel's Triton
        # experts on gfx94x over weights that were never prepared for them.
        #
        # Declined rather than asserted, so one env setting can span mixed
        # arches. Sits with the EPLB force-off and BEFORE the A4W4 assert below
        # on purpose: that assert has to see the final value, or a4w4 would be
        # accepted here and then silently dropped on an arch that cannot serve
        # it. `use_triton` needs no reset -- it is already False under EP.
        if self.use_triton_ep and not gfx.startswith(("gfx95", "gfx125")):
            logger.warning(
                "ATOM_USE_TRITON_MOE=1 asks for Triton MoE, but the EP path is "
                "supported only on gfx95x and gfx125x (arch=%s). Falling back to "
                "the FlyDSL EP path for the routed experts.",
                gfx,
            )
            self.use_triton_ep = False
            self.use_triton_decode = False

        assert not (
            envs.ATOM_USE_TRITON_MOE_A4W4
            and not (self.use_triton or self.use_triton_ep)
        ), (
            "ATOM_USE_TRITON_MOE_A4W4=1 requires a Triton MoE path, but it is off "
            f"(ATOM_USE_TRITON_MOE={int(use_triton_moe)}, use_ep={ep_moe}, "
            f"eplb_enable={getattr(get_current_atom_config(), 'eplb_enable', False)})."
        )

        # Selecting Triton for the routed experts under EP is a change of
        # backend, not a tuning knob, and nothing downstream announces it -- so
        # say so once at construction where it is decided. Warning rather than
        # info because it diverges from what an EP deployment got before this
        # path existed.
        if self.use_triton_ep:
            logger.warning(
                "Triton MoE selected for the routed experts under EP (arch=%s); "
                "this replaces the FlyDSL EP path.",
                gfx,
            )

        # Which wrapper the Triton path will use. a4w4 has a third trigger that
        # only exists at call time -- an already-MXFP4 activation, which is what
        # a mori fp4 dispatch hands over -- so this reports the two reasons
        # decided here, not all three. See _fused_experts_silu_gugu.
        if self.use_triton or self.use_triton_ep:
            logger.info(
                "Triton MoE experts: %s (ATOM_USE_TRITON_MOE_A4W4=%d, "
                "act_quant=%s, ep=%d)",
                (
                    "a4w4"
                    if envs.ATOM_USE_TRITON_MOE_A4W4
                    or self.act_quant == MoEActivationQuant.FP4
                    else "a8w4"
                ),
                int(envs.ATOM_USE_TRITON_MOE_A4W4),
                self.act_quant.value,
                int(self.use_triton_ep),
            )

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.num_experts = num_experts
        weight_dtype = params_dtype
        scale_dtype = torch.uint8

        mxfp4_block = 32
        pad_align = self.pad_align

        intermediate_size_per_partition_after_pad = (
            (intermediate_size_per_partition + pad_align - 1) // pad_align * pad_align
        )
        hidden_size = (hidden_size + pad_align - 1) // pad_align * pad_align
        self.intermediate_size = intermediate_size_per_partition_after_pad
        self.hidden_size = hidden_size
        self.hidden_pad = self.hidden_size - layer.hidden_size
        # Update moe.hidden_dim to match the padded hidden size for Mori kernels
        self.moe.hidden_dim = hidden_size
        self.intermediate_pad = (
            self.intermediate_size - layer.intermediate_size_per_partition
        )
        # MXFP4 is target-only: it is never dequantized as an online source.
        # Fused gate_up_proj (column parallel)
        w13_weight = atom_parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,  # TP included
                hidden_size // 2,
                dtype=weight_dtype,
            )
        )
        layer.register_parameter("w13_weight", w13_weight)
        # Zero-fill padding region: FP4 dtype doesn't support torch.zeros,
        # so we zero the underlying bytes to avoid garbage in padded rows.
        w13_weight.data.view(torch.uint8).zero_()
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_weight_scale = atom_parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                hidden_size // mxfp4_block,
                dtype=scale_dtype,
            )
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        if layer.has_bias:
            w13_bias = atom_parameter(
                torch.zeros(
                    num_experts,
                    2 * intermediate_size_per_partition_after_pad,
                    dtype=torch.bfloat16,
                )
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
        else:
            layer.register_parameter("w13_bias", None)

        # down_proj (row parallel)
        w2_weight = atom_parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // 2,  # TP included
                dtype=weight_dtype,
            )
        )
        layer.register_parameter("w2_weight", w2_weight)
        w2_weight.data.view(torch.uint8).zero_()
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale = atom_parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // mxfp4_block,
                dtype=scale_dtype,
            )
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        if layer.has_bias:
            w2_bias = atom_parameter(
                torch.zeros(
                    num_experts,
                    hidden_size,
                    dtype=torch.bfloat16,
                )
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)
        else:
            layer.register_parameter("w2_bias", None)

        layer.w13_swizzle_layout = None
        layer.w2_swizzle_layout = None

        if self.static_input_scales:
            w13_input_scale = atom_parameter(
                torch.ones(num_experts, dtype=torch.float32)
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = atom_parameter(
                torch.ones(num_experts, dtype=torch.float32)
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer):
        if layer.w13_bias is not None:
            layer.w13_bias.data = layer.w13_bias.data.to(torch.float32)
        if layer.w2_bias is not None:
            layer.w2_bias.data = layer.w2_bias.data.to(torch.float32)

        if self.static_input_scales:
            layer.w13_input_scale = atom_parameter(
                layer.w13_input_scale.max().to(torch.float32)
            )
            layer.w2_input_scale = atom_parameter(
                layer.w2_input_scale.max().to(torch.float32)
            )

        self._process_weight_layout_after_loading(layer)

    def _process_weight_layout_after_loading(self, layer) -> None:
        is_silu = (
            getattr(layer, "activation", ActivationType.Silu) == ActivationType.Silu
        )

        layer.use_fused_silu_gugu = self.is_gfx1250 and is_silu

        use_triton_gfx1250_silu = self.use_triton and self.is_gfx1250 and is_silu

        # Refuse a combination that is already fully decided here, instead of on
        # the first token. Only branch B stashes the dense shared-expert slices
        # that _apply_shared_experts_dense consumes in apply(); branch A (GUGU)
        # and branch C (FlyDSL) both return without them.
        #
        # Deliberately gated on `use_triton_gfx1250_silu` rather than on
        # `layer.use_fused_silu_gugu` alone: that flag is
        # `is_gfx1250 and is_silu` with no `use_triton` term, so asserting on it
        # by itself would also reject a plain ATOM_USE_TRITON_MOE=0 FlyDSL run on
        # gfx1250 -- where apply() never reads it, because the read sits inside
        # the `use_triton_now` branch. This form fires in exactly the cases
        # apply() would have, only at load time and off the per-forward path.
        #
        # Checked BEFORE the use_triton_decode override below, which forces this
        # local False (sending prep down branch C) while apply() still runs the
        # GUGU kernel on decode over a view of those FlyDSL weights.
        assert not (use_triton_gfx1250_silu and layer.num_fused_shared_experts > 0), (
            "the Triton GUGU MoE path cannot serve fused shared experts "
            f"(num_fused_shared_experts={layer.num_fused_shared_experts}): its "
            "weight prep does not stash the dense shared-expert slices that "
            "_apply_shared_experts_dense consumes."
        )

        # Decode-only Triton leaves the layout to the FlyDSL prep (branch C) and
        # rebuilds the Triton view per decode call in apply(). That single copy
        # can serve both kernels only where the two preps write the same bytes:
        # gfx1250, GUGU rows, SiLU. Off that config they are different layouts.
        if self.use_triton_decode:
            assert self.is_gfx1250 and self.is_guinterleave and is_silu, (
                "ATOM_USE_TRITON_MOE_DECODE=1 shares one weight copy between the "
                "FlyDSL and Triton kernels, which only matches on gfx1250 + "
                "ATOM_MOE_GU_ITLV=1 + SiLU (got gfx1250="
                f"{self.is_gfx1250}, gu_itlv={self.is_guinterleave}, silu={is_silu}). "
                "NOTE the ATOM_MOE_GU_ITLV requirement is INVERTED from earlier "
                "builds: this flag used to be skipped when GU interleaving was on "
                "and to run with ATOM_MOE_GU_ITLV=0, which is now the "
                "unsupported combination. A launch line carrying "
                "ATOM_MOE_GU_ITLV=0 with ATOM_USE_TRITON_MOE_DECODE=1 needs "
                "ATOM_MOE_GU_ITLV=1, or drop ATOM_USE_TRITON_MOE_DECODE."
            )
            # FlyDSL trims the create_weights padding via hidden_pad /
            # intermediate_pad; the Triton kernels take no such argument and read
            # the padded N/K straight off the weight. Equal only at zero pad --
            # otherwise prefill and decode would silently disagree.
            assert self.hidden_pad == 0 and self.intermediate_pad == 0, (
                "ATOM_USE_TRITON_MOE_DECODE needs an unpadded expert shape: "
                "FlyDSL trims the padding on prefill but the Triton decode "
                f"kernel cannot (hidden_pad={self.hidden_pad}, "
                f"intermediate_pad={self.intermediate_pad}). Lower pad_align."
            )
            use_triton_gfx1250_silu = False

        # ── A. Triton/gluon GUGU experts: TP (gfx1250 + SiLU) or EP ───────────
        if (use_triton_gfx1250_silu or self.use_triton_ep) and not (
            self.use_triton_decode
        ):
            from aiter.ops.triton.utils.shuffle import (
                moe_weight_decode_view,
                shuffle_scale_moe,
            )

            if self.is_gfx1250:
                layer.w13_weight.data = moe_shuffle_weight(
                    layer.w13_weight,
                    experts_cnt=self.num_experts,
                    is_guinterleave=True,
                    gate_up=True,
                )
                layer.w2_weight.data = moe_shuffle_weight(
                    layer.w2_weight,
                    experts_cnt=self.num_experts,
                    is_guinterleave=True,
                    gate_up=False,
                )
                layer.w13_weight.is_shuffled = True
                layer.w2_weight.is_shuffled = True
                # GUGU interleaves stage1 bias rows to [g0, u0, g1, u1, ...].
                # NOTE: the CDNA branch below interleaves the weight rows but
                # NOT the bias, so a biased EP layer on CDNA is inconsistent.
                # Pre-existing; DeepSeek-V4 has no MoE bias so it is latent.
                if layer.w13_bias is not None:
                    layer.w13_bias.data = interleave_gate_up_rows(layer.w13_bias.data)

            if self.is_gfx1250:
                w13_weight = moe_weight_decode_view(layer.w13_weight.data)
                w2_weight = moe_weight_decode_view(layer.w2_weight.data)
            else:
                w13_unshuffled = layer.w13_weight.data
                if w13_unshuffled.dtype != torch.uint8:
                    w13_unshuffled = w13_unshuffled.view(torch.uint8)
                assert w13_unshuffled.shape[1] == 2 * self.intermediate_size, (
                    "expected w13_weight (E, 2*intermediate, K_packed) so dim 1 is "
                    f"the gate/up axis, got {tuple(w13_unshuffled.shape)}"
                )
                w13_unshuffled = interleave_gate_up_rows(w13_unshuffled)
                w2_unshuffled = layer.w2_weight.data
                if w2_unshuffled.dtype != torch.uint8:
                    w2_unshuffled = w2_unshuffled.view(torch.uint8)
                w13_weight = w13_unshuffled.transpose(-2, -1)
                w2_weight = w2_unshuffled.transpose(-2, -1)

            raw_w13_scale = layer.w13_weight_scale.data
            raw_w2_scale = layer.w2_weight_scale.data
            assert raw_w13_scale.ndim == 3, (
                "expected a 3-D (E, N, K//32) w13 scale, got shape "
                f"{tuple(raw_w13_scale.shape)}"
            )
            assert raw_w13_scale.shape[1] == 2 * self.intermediate_size, (
                "w13 a8w4 scale dim 1 must be N = 2 * intermediate_size "
                f"({2 * self.intermediate_size}) for the GUGU row interleave, "
                f"but got shape {tuple(raw_w13_scale.shape)}. If dim 1 "
                "is K//32 the interleave must move after the transpose."
            )
            w13_scale_in = interleave_gate_up_rows(raw_w13_scale).transpose(-2, -1)
            w2_scale_in = raw_w2_scale.transpose(-2, -1)

            w13_scale, w13_swizzle_layout = shuffle_scale_moe(
                w13_scale_in,
                return_layout=True,
                scale_kwidth=4 if self.is_gfx1250 else 8,
            )
            w2_scale, w2_swizzle_layout = shuffle_scale_moe(
                w2_scale_in,
                return_layout=True,
                scale_kwidth=4 if self.is_gfx1250 else 8,
            )

            del layer.w13_weight
            del layer.w2_weight
            del layer.w13_weight_scale
            del layer.w2_weight_scale
            layer.w13_weight = w13_weight
            layer.w2_weight = w2_weight
            layer.w13_weight_scale = w13_scale
            layer.w2_weight_scale = w2_scale
            layer.w13_swizzle_layout = w13_swizzle_layout
            layer.w2_swizzle_layout = w2_swizzle_layout
            return

        # ── B. General Triton MXFP4 MoE ───────────────────────────────────────
        if self.use_triton and not self.use_triton_decode:
            from atom.model_ops.fused_moe_triton import _swizzle_mxfp4

            atom_config = get_current_atom_config()

            # Stash dense (pre-swizzle) shared-expert weights so the always-on
            # shared expert can be evaluated by a standalone dense MXFP4 GEMM
            # (gemm_a16wfp4, see Mxfp4MoEMethod._apply_shared_experts_dense)
            # instead of being fused into grouped_topk routing. The shared
            # experts occupy the last ``num_fused_shared_experts`` slots of the
            # routed weight tensors; their raw per-expert layout
            # (N, K // 2) + scale (N, K // 32) is exactly what gemm_a16wfp4
            # consumes, whereas _swizzle_mxfp4 below reorders the scales into
            # the MoE-kernel-only CDNA4 layout.
            n_shared = layer.num_fused_shared_experts
            if n_shared > 0:
                layer.shared_w13_weight = (
                    layer.w13_weight.data[-n_shared:].view(torch.uint8).contiguous()
                )
                layer.shared_w13_weight_scale = layer.w13_weight_scale.data[
                    -n_shared:
                ].contiguous()
                layer.shared_w2_weight = (
                    layer.w2_weight.data[-n_shared:].view(torch.uint8).contiguous()
                )
                layer.shared_w2_weight_scale = layer.w2_weight_scale.data[
                    -n_shared:
                ].contiguous()
                if layer.w13_bias is not None:
                    layer.shared_w13_bias = layer.w13_bias.data[-n_shared:].contiguous()
                else:
                    layer.shared_w13_bias = None
                if layer.w2_bias is not None:
                    layer.shared_w2_bias = layer.w2_bias.data[-n_shared:].contiguous()
                else:
                    layer.shared_w2_bias = None

            (
                w13_weight,
                w13_scale,
                w13_swizzle_layout,
                w2_weight,
                w2_scale,
                w2_swizzle_layout,
            ) = _swizzle_mxfp4(
                layer.w13_weight.view(torch.uint8),
                layer.w13_weight_scale,
                layer.w2_weight.view(torch.uint8),
                layer.w2_weight_scale,
                "mx4",
                self.intermediate_size * 2,  # N_1,
                self.hidden_size,  # K_1,
                self.hidden_size,  # N_2,
                self.intermediate_size,  # K_2,
                atom_config.tensor_parallel_size,
                # Selects the consuming kernel, which fixes the scale kwidth.
                act_quant=self.act_quant,
            )
            del layer.w13_weight
            del layer.w2_weight
            del layer.w13_weight_scale
            del layer.w2_weight_scale
            layer.w13_weight = w13_weight
            layer.w2_weight = w2_weight
            layer.w13_weight_scale = w13_scale
            layer.w2_weight_scale = w2_scale
            layer.w13_swizzle_layout = w13_swizzle_layout
            layer.w2_swizzle_layout = w2_swizzle_layout
            return

        # ── C. FlyDSL fused_moe ───────────────────────────────────────────────
        # shuffle weight (arch-aware: gfx1250 does the GUGU row interleave +
        # WMMA tile shuffle internally, other archs use the lane-level path)
        layer.w13_weight.data = moe_shuffle_weight(
            layer.w13_weight,
            experts_cnt=self.num_experts,
            is_guinterleave=self.is_guinterleave,
            gate_up=True,
        )
        layer.w2_weight.data = moe_shuffle_weight(
            layer.w2_weight,
            experts_cnt=self.num_experts,
            is_guinterleave=self.is_guinterleave,
            gate_up=False,
        )
        layer.w13_weight.is_shuffled = True
        layer.w2_weight.is_shuffled = True

        # On gfx1250, GUGU interleaves stage1 bias rows to [g0, u0, g1, u1, ...].
        if self.is_gfx1250 and self.is_guinterleave and layer.w13_bias is not None:
            layer.w13_bias.data = interleave_gate_up_rows(layer.w13_bias.data)

        # shuffle scale
        w13_scale_2d = layer.w13_weight_scale.reshape(
            -1, layer.w13_weight_scale.shape[-1]
        )
        w2_scale_2d = layer.w2_weight_scale.reshape(-1, layer.w2_weight_scale.shape[-1])

        shuffled_w13_scale = moe_shuffle_scale(
            w13_scale_2d,
            self.num_experts,
            is_guinterleave=self.is_guinterleave,
            gate_up=True,
        )
        shuffled_w2_scale = moe_shuffle_scale(
            w2_scale_2d,
            self.num_experts,
            is_guinterleave=self.is_guinterleave,
            gate_up=False,
        )
        layer.w13_weight_scale = atom_parameter(shuffled_w13_scale)
        layer.w2_weight_scale = atom_parameter(shuffled_w2_scale)

    def _triton_views_of_flydsl_weights(self, layer) -> tuple:
        """Zero-copy Triton/gluon views of the FlyDSL-shuffled MXFP4 weights.

        On gfx1250 under GUGU the FlyDSL prep (branch C of
        ``_process_weight_layout_after_loading``) and the Triton prep (branch A)
        write byte-identical buffers and differ only in the view laid over them:

        * weight ``(E, N, K/2)`` fp4  ->  ``(E, K/2*16, N/16)`` uint8
          (``moe_weight_decode_view``, a reshape + stride swap)
        * scale  ``(E, N/32, K/32*32)``  ->  ``(E, K/32*32, N/32)``
          (``shuffle_scale_moe``'s trailing transpose; the n32k4 permute
          underneath is the same one ``_shuffle_scale_tile_gfx1250`` applies)

        ``ATOM_USE_TRITON_MOE_DECODE`` keeps the FlyDSL view on the layer so
        prefill runs FlyDSL, and calls this to hand decode the Triton view of
        the same storage -- no second copy, no extra memory.

        Cached on the layer: the views alias the weight storage, so they survive
        an in-place EPLB weight swap, and rebuilding six view ops per MoE layer
        per decode step is pure overhead on a latency-critical path.
        """
        views = getattr(layer, "_triton_decode_views", None)
        if views is None:
            from aiter.ops.triton.utils.shuffle import moe_weight_decode_view

            views = (
                moe_weight_decode_view(layer.w13_weight.data),
                moe_weight_decode_view(layer.w2_weight.data),
                layer.w13_weight_scale.data.transpose(-1, -2),
                layer.w2_weight_scale.data.transpose(-1, -2),
                "GFX1250_SCALE",
                "GFX1250_SCALE",
            )
            layer._triton_decode_views = views
        return views

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:

        a1_scale = getattr(layer, "w13_input_scale", None)
        a2_scale = getattr(layer, "w2_input_scale", None)

        if self.act_quant == MoEActivationQuant.FP8:
            return mxfp4_w4a8_moe_quant_config(
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                w1_bias=layer.w13_bias,
                w2_bias=layer.w2_bias,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
            )
        return mxfp4_w4a16_moe_quant_config(
            w1_bias=layer.w13_bias,
            w2_bias=layer.w2_bias,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
        )

    @mark_trace
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        fused_shared_experts_scoring_func: str | None = None,
        activation: ActivationType = ActivationType.Silu,
        prefix: str = "",
    ) -> torch.Tensor:
        # ATOM_USE_TRITON_MOE_DECODE splits the two kernels by phase: the weights
        # sit in the FlyDSL layout, so prefill falls through to the FlyDSL tail
        # below and only decode takes the Triton block (over a view of the very
        # same storage, see _triton_views_of_flydsl_weights).
        # True only when EVERY rank is decoding this step.
        #
        # `not is_prefill` is only THIS rank's phase. On a mixed batch a peer can
        # be prefilling, and the ranks then disagree about which expert backend
        # to run on the same step; a batch that merely CONTAINS prefill tokens
        # can also read is_prefill False, handing the Triton experts a
        # prefill-sized M under a flag that says decode.
        #
        # `running_tokens_are_unified` is the collective answer -- "every rank of
        # the group is decoding, so running_tokens is the group's number" -- and
        # it is the SAME guard the shape narrowing uses (the M_eff trim in
        # modular_kernel.forward, and MoriV2ModularKernel._recv_bound), so
        # the phase split and the trims agree on what a decode step is.
        #
        # Asked only under use_triton_decode: every reader below is behind that
        # flag, and get_forward_context() ASSERTS when unset, so a bare call
        # would raise on any apply() outside a forward. No context means we
        # cannot tell, so: not a decode step.
        # NOTE `running_tokens_are_unified` is only the token-AGREEMENT half of
        # the old dp_uniform_decode. At dp_size 1 ForwardMode.decide sets it True
        # unconditionally ("a group of one is unified whatever it runs"), so it
        # stays True through prefill; alone it would hand the Triton experts a
        # prefill-sized M under a flag that says decode. Conjoin is_prefill, as
        # forward_context itself does right after computing `unified`.
        _ctx = get_forward_context().context if self.use_triton_decode else None
        all_ranks_decode = (
            _ctx is not None
            and not _ctx.is_prefill
            and bool(getattr(_ctx, "running_tokens_are_unified", True))
        )

        use_triton_now = self.use_triton
        # `and self.use_triton` guards the EP case: there self.use_triton is
        # False, and without it a decode step would flip use_triton_now True and
        # fall into the TP block below -- which routes from logits and reads the
        # branch-A weight names. EP does its phase split in the no-transport
        # block further down.
        if self.use_triton and self.use_triton_decode:
            use_triton_now = all_ranks_decode

        if use_triton_now:
            from atom.model_ops.fused_moe_triton import (
                triton_kernel_fused_experts,
                triton_kernel_moe_forward,
            )

            # Decided at weight prep, not re-derived here: see
            # _process_weight_layout_after_loading.
            use_triton_gfx1250_silu = layer.use_fused_silu_gugu
            if self.use_triton_decode:
                (
                    w13_weight,
                    w2_weight,
                    w13_weight_scale,
                    w2_weight_scale,
                    w13_swizzle_layout,
                    w2_swizzle_layout,
                ) = self._triton_views_of_flydsl_weights(layer)
            else:
                w13_weight = layer.w13_weight
                w2_weight = layer.w2_weight
                w13_weight_scale = layer.w13_weight_scale
                w2_weight_scale = layer.w2_weight_scale
                w13_swizzle_layout = layer.w13_swizzle_layout
                w2_swizzle_layout = layer.w2_swizzle_layout
            needs_custom_routing = (
                use_grouped_topk
                or scoring_func != "softmax"
                or e_score_correction_bias is not None
                or custom_routing_function is not None
            )
            if needs_custom_routing or use_triton_gfx1250_silu:
                # custom routing -- set for deepseek routing n expts act, for grouped topk
                n_expts_act = top_k

                # custom routing
                from aiter.ops.triton.moe.moe_routing.routing import (  # grouped topk included
                    routing,
                )

                routing_data, gather_idx, scatter_idx = routing(
                    router_logits,
                    n_expts_act,
                    score_mode=scoring_func,
                    bias=(
                        e_score_correction_bias.to(torch.float32)
                        if e_score_correction_bias is not None
                        else None
                    ),
                    renorm=renormalize,
                    routed_scaling_factor=layer.routed_scaling_factor,
                    use_grouped_topk=use_grouped_topk,
                    num_expert_group=num_expert_group,
                    topk_group=topk_group,
                )
                # Routed-only gate count (no shared-expert widening).
                n_expts_act = routing_data.n_expts_act

                # Convert to triton routing data structures
                _, n_expts_tot = router_logits.shape

                if global_num_experts > 0:
                    n_expts_tot = global_num_experts

                # Both preps publish the same six names, so there is nothing to
                # select here -- only which kernel consumes them.
                output = torch.empty_like(x)
                _moe_result = triton_kernel_fused_experts(
                    output,
                    x,
                    w13_weight,
                    w2_weight,
                    routing_data,
                    gather_idx,
                    scatter_idx,
                    topk=n_expts_act,
                    activation=activation,
                    w13_scale=w13_weight_scale,
                    w2_scale=w2_weight_scale,
                    w13_swizzle_layout=w13_swizzle_layout,
                    w2_swizzle_layout=w2_swizzle_layout,
                    a13_scale=layer.w13_input_scale,
                    a2_scale=layer.w2_input_scale,
                    w1_bias=layer.w13_bias,
                    w2_bias=layer.w2_bias,
                    swiglu_limit=getattr(layer, "swiglu_limit", 0.0),
                    apply_router_weight_on_input=apply_router_weight_on_input,
                    global_num_experts=n_expts_tot,
                    expert_map=expert_map,
                    act_quant=self.act_quant,
                    use_triton_gfx1250_silu=use_triton_gfx1250_silu,
                )

                # Always-on shared expert(s) via a standalone dense GEMM, added
                # to the routed output before the TP all-reduce. aiter's routing()
                # does not widen the top-k with the always-on slots, so without
                # this the shared expert is silently dropped.
                #
                # The GUGU-vs-fused-shared-experts refusal now lives in
                # _process_weight_layout_after_loading, where it is decided --
                # so this stays a plain add and costs no per-forward check.
                if layer.num_fused_shared_experts > 0:
                    _moe_result = _moe_result + self._apply_shared_experts_dense(
                        layer, x, activation
                    )
                return _moe_result
            assert (
                fused_shared_experts_scoring_func is None
            ), "triton kernel does not support fused shared experts func"

            # Takes directly from model dtype in config.json
            return triton_kernel_moe_forward(
                x,
                w13_weight,
                w2_weight,
                router_logits,
                topk=top_k,
                renormalize=renormalize,
                activation=activation,
                w13_scale=w13_weight_scale,
                w2_scale=w2_weight_scale,
                w13_swizzle_layout=w13_swizzle_layout,
                w2_swizzle_layout=w2_swizzle_layout,
                a13_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
                w1_bias=layer.w13_bias,
                w2_bias=layer.w2_bias,
                swiglu_limit=getattr(layer, "swiglu_limit", 7.0),
                expert_map=expert_map,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                act_quant=self.act_quant,
            )

        topk_weights, topk_ids = self.select_experts_with_record(
            layer=layer,
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            global_num_experts=global_num_experts,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            fused_shared_experts_scoring_func=fused_shared_experts_scoring_func,
        )
        a1_scale = getattr(layer, "w13_input_scale", None)
        a2_scale = getattr(layer, "w2_input_scale", None)
        moe_extra_args = {
            "gate_mode": (
                GateMode.INTERLEAVE.value
                if self.is_guinterleave
                else GateMode.SEPARATED.value
            ),
            "swiglu_limit": getattr(layer, "swiglu_limit", 0.0),
        }
        if self.fp8_activations:
            moe_extra_args["quant_dtype_a"] = dtypes.fp8
        if activation == ActivationType.Situv2:
            moe_extra_args["beta"] = getattr(layer, "activation_situ_beta", None)
            moe_extra_args["linear_beta"] = getattr(
                layer, "activation_situ_linear_beta", None
            )
        _ep_triton_now = False
        if self.use_triton_ep and self.using_modular_kernel:
            # Phase split, transport version. FusedMoEModularKernel.forward runs
            # its Triton block only when `triton_experts` is present and other-
            # wise falls through to its flydsl fused_moe tail -- so publishing
            # the dict on decode and withholding it on prefill IS the split, with
            # no change needed on the modular-kernel side. Dispatch and combine
            # are identical either way; only the expert GEMM in between moves.
            #
            # One weight copy in both phases: branch C's FlyDSL layout, with the
            # zero-copy Triton view laid over it per call. That is the same
            # premise as the TP path, and the asserts in
            # _process_weight_layout_after_loading enforce it.
            _ep_triton_now = True
            if self.use_triton_decode:
                _ep_triton_now = all_ranks_decode
            if self.use_triton_decode:
                (
                    _tw13,
                    _tw2,
                    _tw13_scale,
                    _tw2_scale,
                    _tw13_swz,
                    _tw2_swz,
                ) = self._triton_views_of_flydsl_weights(layer)
            else:
                assert getattr(layer, "w13_swizzle_layout", None) is not None, (
                    "use_triton_ep is on but this layer carries the FlyDSL weight "
                    "layout; flydsl fused_moe cannot serve as a fallback because "
                    "block A publishes the GUGU layout under the same names."
                )
                _tw13 = layer.w13_weight
                _tw2 = layer.w2_weight
                _tw13_scale = layer.w13_weight_scale
                _tw2_scale = layer.w2_weight_scale
                _tw13_swz = layer.w13_swizzle_layout
                _tw2_swz = layer.w2_swizzle_layout
        if _ep_triton_now:
            moe_extra_args["triton_experts"] = {
                "w13_weight": _tw13,
                "w2_weight": _tw2,
                "w13_scale": _tw13_scale,
                "w2_scale": _tw2_scale,
                "w13_swizzle_layout": _tw13_swz,
                "w2_swizzle_layout": _tw2_swz,
                "a13_scale": getattr(layer, "w13_input_scale", None),
                "a2_scale": getattr(layer, "w2_input_scale", None),
                "w1_bias": getattr(layer, "w13_bias", None),
                "w2_bias": getattr(layer, "w2_bias", None),
                "swiglu_limit": getattr(layer, "swiglu_limit", 10.0),
            }

        if self.fused_experts is None:
            # Serves prefill as well as decode: the gfx1250 gluon kernel's
            # prefill bug is fixed, and process_weights_after_loading no longer
            # builds the FlyDSL scale layout under use_triton_ep, so there is no
            # flydsl fallback left to drop back to.
            # Branch A of _process_weight_layout_after_loading publishes the
            # GUGU weights under the PLAIN names (w13_weight, w13_weight_scale,
            # w13_swizzle_layout) -- the `*_preshuffled` / `*_a8w4` names this
            # used to read belonged to the decode-only prep that branch replaced,
            # so nothing set them and this guard was never true.
            #
            # w13_swizzle_layout is the discriminator, not w13_weight: the latter
            # exists on every layer, while the former is initialised to None in
            # create_weights and set only by the Triton preps.
            #
            # Under ATOM_USE_TRITON_MOE_DECODE the layer kept the FlyDSL layout
            # (branch C), so w13_swizzle_layout is None and the weights come from
            # a zero-copy view instead; prefill falls through to the FlyDSL tail
            # below. Mirrors what the TP path does, one screen up.
            _ep_triton_now = self.use_triton_ep
            if self.use_triton_decode:
                _ep_triton_now = all_ranks_decode
            _ep_weights_ready = self.use_triton_decode or (
                getattr(layer, "w13_swizzle_layout", None) is not None
            )
            if _ep_triton_now and self.is_gfx1250 and _ep_weights_ready:
                from atom.model_ops.fused_moe_triton import (
                    routing_from_dispatched,
                    triton_kernel_fused_experts,
                )

                if self.use_triton_decode:
                    (
                        _w13,
                        _w2,
                        _w13_scale,
                        _w2_scale,
                        _w13_swz,
                        _w2_swz,
                    ) = self._triton_views_of_flydsl_weights(layer)
                else:
                    _w13 = layer.w13_weight
                    _w2 = layer.w2_weight
                    _w13_scale = layer.w13_weight_scale
                    _w2_scale = layer.w2_weight_scale
                    _w13_swz = layer.w13_swizzle_layout
                    _w2_swz = layer.w2_swizzle_layout

                M = topk_ids.shape[0]
                num_local_tokens = topk_ids.new_empty(1, dtype=torch.int32).fill_(M)
                # No scatter geometry: this path has no EP transport to deliver
                # into, so GEMM2 reduces locally and dst_row is unused.
                routing_data, gather_idx, scatter_idx, gate_valid, _dst_row = (
                    routing_from_dispatched(
                        topk_weights,
                        topk_ids,
                        expert_map,
                        self.num_experts,
                        num_local_tokens,
                    )
                )
                return triton_kernel_fused_experts(
                    None,  # output_tensor: the GUGU path allocates its own
                    x,
                    _w13,
                    _w2,
                    routing_data,
                    gather_idx,
                    scatter_idx,
                    topk=routing_data.n_expts_act,
                    use_triton_gfx1250_silu=True,
                    w13_scale=_w13_scale,
                    w2_scale=_w2_scale,
                    w13_swizzle_layout=_w13_swz,
                    w2_swizzle_layout=_w2_swz,
                    a13_scale=getattr(layer, "w13_input_scale", None),
                    a2_scale=getattr(layer, "w2_input_scale", None),
                    w1_bias=getattr(layer, "w13_bias", None),
                    w2_bias=getattr(layer, "w2_bias", None),
                    swiglu_limit=getattr(layer, "swiglu_limit", 10.0),
                    apply_router_weight_on_input=apply_router_weight_on_input,
                    gate_valid=gate_valid,
                )

            # Reaching flydsl with use_triton_ep on means the Triton branch
            # above was skipped (wrong arch, or the a8w4 prep never ran). The
            # FlyDSL scale layout was not built for this layer, so w13/w2
            # _weight_scale are None -- fail with the reason rather than let
            # fused_moe dereference None deep in a kernel launch.
            # `and not use_triton_decode`: under the phase split the FlyDSL
            # layout IS present, so reaching flydsl on prefill is the design, not
            # a failure.
            assert not (self.use_triton_ep and not self.use_triton_decode), (
                "use_triton_ep is on but the Triton EP path is unavailable here "
                f"(is_gfx1250={self.is_gfx1250}, w13_swizzle_layout="
                f"{getattr(layer, 'w13_swizzle_layout', None) is not None}); "
                "flydsl fused_moe cannot serve as a fallback because the FlyDSL "
                "scale layout is skipped under use_triton_ep."
            )
            return fused_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                expert_mask=layer.expert_mask,
                activation=activation,
                quant_type=self.quant_type,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                doweight_stage1=apply_router_weight_on_input,
                bias1=layer.w13_bias,
                bias2=layer.w2_bias,
                **moe_extra_args,
            )
        return self.fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
            activation=activation,
            quant_type=self.quant_type,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            expert_mask=layer.expert_mask,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            bias1=layer.w13_bias,
            bias2=layer.w2_bias,
            hidden_pad=self.hidden_pad,
            intermediate_pad=self.intermediate_pad,
            moe_extra_args=moe_extra_args,
        )

    def _apply_shared_experts_dense(self, layer, x, activation):
        """Standalone dense MXFP4 GEMM for the always-on shared expert(s).

        Functionally replaces fusing the shared expert into grouped_topk
        routing. The fused path appended each shared expert as an always-on
        routed slot with a fixed weight of ``SHARED_SCORE`` (== 1.0 here),
        placed *after* the routed renorm / ``routed_scaling_factor`` and run
        through the MoE GEMM as the last expert(s) of ``w13/w2``.

        Here we instead apply the shared expert to every token via two dense
        MXFP4 GEMM calls (gate_up -> SiLU-and-mul -> down) on the pre-swizzle
        weight slices stashed in ``process_weights_after_loading``. The dense
        GEMM is ``gemm_afp4wfp4`` (a4w4) when activations are MXFP4, otherwise
        ``gemm_a16wfp4`` (a16w4), matching the routed-expert activation dtype,
        and return the result so the caller adds it (weight 1.0) to the routed
        output before the TP all-reduce. The shared-expert intermediate is
        TP-partitioned exactly like the routed experts, so both partial outputs
        reduce together.
        """
        from aiter.ops.triton.fusions.fused_clamp_act_mul import fused_clamp_act_mul
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

        # The dense shared-expert GEMM only implements the SiLU activation
        # path; SwiGLU models have no fused shared experts, so this assert
        # documents the supported scope.
        assert (
            activation != ActivationType.Swiglu
        ), "dense shared-expert GEMM only supports the SiLU activation path"

        M = x.shape[0]
        swiglu_limit = getattr(layer, "swiglu_limit", 0.0)

        use_a4w4 = self.act_quant == MoEActivationQuant.FP4
        if use_a4w4:
            from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
            from aiter.ops.triton.moe.moe_op_gemm_a4w4 import mxfp4_quant

        def _shared_expert_gemm(act, weight, weight_scale):
            if use_a4w4:
                act_fp4, act_mx_scale = mxfp4_quant(act)
                return gemm_afp4wfp4(act_fp4, weight, act_mx_scale, weight_scale)
            return gemm_a16wfp4(act, weight, weight_scale)

        shared_w13_bias = getattr(layer, "shared_w13_bias", None)
        shared_w2_bias = getattr(layer, "shared_w2_bias", None)

        shared_out = None
        for e in range(layer.num_fused_shared_experts):
            gate_up = _shared_expert_gemm(
                x,
                layer.shared_w13_weight[e],
                layer.shared_w13_weight_scale[e],
            )
            if shared_w13_bias is not None:
                gate_up = gate_up + shared_w13_bias[e]
            half_n = gate_up.shape[-1] // 2
            intermediate = torch.empty((M, half_n), device=x.device, dtype=x.dtype)
            fused_clamp_act_mul(
                gate_up,
                out=intermediate,
                swiglu_limit=swiglu_limit,
                activation="silu",
                dtype_quant=None,
            )
            out_e = _shared_expert_gemm(
                intermediate,
                layer.shared_w2_weight[e],
                layer.shared_w2_weight_scale[e],
            )
            if shared_w2_bias is not None:
                out_e = out_e + shared_w2_bias[e]
            shared_out = out_e if shared_out is None else shared_out + out_e
        return shared_out


class MegaMxfp4MoEMethod(Mxfp4MoEMethod):
    """MXFP4 MoE method backed by the fused FlyDSL MegaMoE pipeline."""

    def __init__(self, quant_config: LayerQuantConfig, moe: FusedMoEConfig):
        super().__init__(quant_config, moe)
        # Mega owns the weight layout and the complete routed-MoE execution.
        # Standard Triton weight/forward paths must not preempt it.
        self.use_triton = False
        self.use_triton_decode = False

    def _process_weight_layout_after_loading(self, layer) -> None:
        from atom.model_ops.fused_moe.flydsl_mega_experts import build_mega_weights

        build_mega_weights(layer)

        # Mega reads only _mega_* weights. Release the raw AITER weight copies
        # and skip the standard shuffle so both layouts are not retained.
        layer.w13_weight.data = torch.empty(
            0, dtype=layer.w13_weight.dtype, device=layer.w13_weight.device
        )
        layer.w2_weight.data = torch.empty(
            0, dtype=layer.w2_weight.dtype, device=layer.w2_weight.device
        )
        logger.info("Prepared MegaMoE weights for fused MoE layer")

    def init_prepare_finalize(self, layer: torch.nn.Module):
        # Mega includes dispatch, both GEMMs, and combine, so it must not
        # allocate the standard MORI prepare/finalize modular kernel. Instead it
        # installs itself as the whole-pipeline `fused_experts` backend, so the
        # post-routing tail of the inherited `apply` dispatches to it exactly the
        # way it dispatches to the MORI modular kernel.
        from atom.model_ops.fused_moe.flydsl_mega_experts import MegaFusedExperts

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        self.fused_experts = MegaFusedExperts(
            layer,
            model_dim=self.hidden_size,
            inter_dim=self.intermediate_size,
            mtpr=self.moe.max_num_tokens,
            quant="a8w4",
        )

    def get_eplb_weight_views(self, layer: torch.nn.Module) -> list[torch.Tensor]:
        """Return expert-major views owned by EPLB."""
        views: list[torch.Tensor] = []
        local_num_experts = layer.local_num_experts
        for name in (
            "_mega_w1",
            "_mega_w1_scale",
            "_mega_w2",
            "_mega_w2_scale",
        ):
            tensor = getattr(layer, name, None)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"MegaMoE weight {name!r} was not prepared")
            if not tensor.is_contiguous() or tensor.numel() % local_num_experts != 0:
                raise RuntimeError(
                    "MegaMoE weight must be contiguous and evenly divisible by "
                    "local_num_experts: "
                    f"name={name!r}, shape={tuple(tensor.shape)}, "
                    f"local_num_experts={local_num_experts}."
                )
            views.append(tensor.view(local_num_experts, -1))
        return views


def _make_mxfp4_moe_method(
    quant_config: LayerQuantConfig, moe: FusedMoEConfig
) -> Mxfp4MoEMethod:
    if get_current_atom_config().moe_backend == "mega":
        return MegaMxfp4MoEMethod(quant_config, moe)
    return Mxfp4MoEMethod(quant_config, moe)


# Refer to CompressedTensorsW8A8Fp8MoEMethod in vllm
class CompressedTensorsFp8MoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config: LayerQuantConfig, moe: FusedMoEConfig):
        super().__init__(moe)
        self.quant_config = quant_config
        self.quant_type = quant_config.quant_type
        self.quant_dtype = quant_config.quant_dtype

        # Check if we need to normalize e4m3fn to e4m3fnuz (AMD GPUs)
        self.need_normalize_e4m3fn_to_e4m3fnuz = (
            self.quant_dtype == torch.float8_e4m3fnuz
        )

        # Determine if this is block quantization
        self.block_quant = self.quant_type in [
            QuantType.per_1x128,
            QuantType.per_1x32,
        ]

        # For compressed-tensors, check if per-channel quantization
        self.per_channel = self.quant_type == QuantType.per_Token

        # Check if static input scales (activation quantization)
        self.static_input_scales = not quant_config.is_dynamic

        # Block sizes for block quantization
        if self.block_quant:
            if self.quant_type == QuantType.per_1x128:
                self.block_n = 128
                self.block_k = 128
            elif self.quant_type == QuantType.per_1x32:
                self.block_n = 1
                self.block_k = 32

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weight parameters for compressed-tensors FP8 MoE."""
        layer.num_experts = num_experts
        layer.params_dtype = params_dtype
        layer.hidden_size = hidden_size
        layer.intermediate_size_per_partition = intermediate_size_per_partition

        # Check block alignment for block quantization
        if self.block_quant:
            tp_size = get_tp_group().world_size
            if intermediate_size_per_partition % self.block_n != 0:
                raise ValueError(
                    f"intermediate_size_per_partition={intermediate_size_per_partition} "
                    f"must be divisible by block_n={self.block_n}"
                )
            if tp_size > 1 and intermediate_size_per_partition % self.block_k != 0:
                raise ValueError(
                    f"intermediate_size_per_partition={intermediate_size_per_partition} "
                    f"must be divisible by block_k={self.block_k}"
                )

        # WEIGHTS
        w13_weight = atom_parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            )
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = atom_parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            )
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES - different shapes based on quantization strategy
        if self.per_channel:
            # Per-channel quantization: shape [E, N, 1]
            # This is the key difference for compressed-tensors
            w13_weight_scale = atom_parameter(
                torch.ones(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    1,  # Important: dimension is 1, not omitted
                    dtype=torch.float32,
                )
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)

            w2_weight_scale = atom_parameter(
                torch.ones(
                    num_experts,
                    hidden_size,
                    1,  # Important: dimension is 1, not omitted
                    dtype=torch.float32,
                )
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)

            # Mark as per-channel quantization for weight loader
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
            )
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        elif self.block_quant:
            # Block quantization
            w13_weight_scale = atom_parameter(
                torch.ones(
                    num_experts,
                    2
                    * (
                        (intermediate_size_per_partition + self.block_n - 1)
                        // self.block_n
                    ),
                    (hidden_size + self.block_k - 1) // self.block_k,
                    dtype=torch.float32,
                )
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)

            w2_weight_scale = atom_parameter(
                torch.ones(
                    num_experts,
                    (hidden_size + self.block_n - 1) // self.block_n,
                    (intermediate_size_per_partition + self.block_k - 1)
                    // self.block_k,
                    dtype=torch.float32,
                )
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)

            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
            )
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        else:
            # Per-tensor quantization: shape [E, 2] for w13, [E] for w2
            w13_weight_scale = atom_parameter(
                torch.ones(num_experts, 2, dtype=torch.float32)
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)

            w2_weight_scale = atom_parameter(
                torch.ones(num_experts, dtype=torch.float32)
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)

            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
            )
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES (activation scales)
        if self.static_input_scales:
            w13_input_scale = atom_parameter(
                torch.ones(num_experts, dtype=torch.float32)
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = atom_parameter(
                torch.ones(num_experts, dtype=torch.float32)
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Post-process weights after loading from checkpoint."""
        # Get references to weights and scales
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        w13_input_scale = getattr(layer, "w13_input_scale", None)
        w2_input_scale = getattr(layer, "w2_input_scale", None)

        if self.need_normalize_e4m3fn_to_e4m3fnuz:
            (
                w13.data,
                w13_scale.data,
                w13_input_scale_data,
            ) = normalize_e4m3fn_to_e4m3fnuz(
                w13.data,
                w13_scale.data,
                w13_input_scale.data if w13_input_scale is not None else None,
            )
            if w13_input_scale is not None and w13_input_scale_data is not None:
                w13_input_scale.data = w13_input_scale_data

            w2.data, w2_scale.data, w2_input_scale_data = normalize_e4m3fn_to_e4m3fnuz(
                w2.data,
                w2_scale.data,
                w2_input_scale.data if w2_input_scale is not None else None,
            )
            if w2_input_scale is not None and w2_input_scale_data is not None:
                w2_input_scale.data = w2_input_scale_data

        # For per-tensor quantization, combine w1 and w3 scales
        # This is necessary for kernels that expect a single scale per expert
        if not self.per_channel and not self.block_quant:
            # w13_weight_scale has shape [E, 2] for w1 and w3
            # Use the max scale and requantize both w1 and w3 with it
            max_w13_scales = w13_scale.max(dim=1).values  # Shape: [E]
            num_experts = w13.shape[0]
            shard_size = layer.intermediate_size_per_partition

            # Requantize w1 and w3 with max scale per expert
            for expert_id in range(num_experts):
                max_scale = max_w13_scales[expert_id]

                # Process w1 (first shard)
                w1_scale = w13_scale[expert_id, 0]
                if w1_scale != max_scale:
                    # Dequantize: weight_fp32 = weight_fp8 * scale
                    w1_dq = per_tensor_dequantize(
                        w13[expert_id, :shard_size, :], w1_scale
                    )
                    # Quantize: weight_fp8 = weight_fp32 / scale
                    w1_q = (w1_dq / max_scale).clamp(
                        min=torch.finfo(w13.dtype).min,
                        max=torch.finfo(w13.dtype).max,
                    )
                    w13.data[expert_id, :shard_size, :] = w1_q.to(w13.dtype)

                # Process w3 (second shard)
                w3_scale = w13_scale[expert_id, 1]
                if w3_scale != max_scale:
                    # Dequantize: weight_fp32 = weight_fp8 * scale
                    w3_dq = per_tensor_dequantize(
                        w13[expert_id, shard_size:, :], w3_scale
                    )
                    # Quantize: weight_fp8 = weight_fp32 / scale
                    w3_q = (w3_dq / max_scale).clamp(
                        min=torch.finfo(w13.dtype).min,
                        max=torch.finfo(w13.dtype).max,
                    )
                    w13.data[expert_id, shard_size:, :] = w3_q.to(w13.dtype)

            # Update scale to single max scale per expert [E]
            layer.w13_weight_scale = atom_parameter(max_w13_scales)

        # Shuffle weights for asm moe (moved from inference to load time for better performance).
        if w13.dtype in [
            torch.int8,
            torch.uint8,
            torch.float8_e4m3fnuz,
            torch.float8_e4m3fn,
        ]:
            from aiter.ops.shuffle import shuffle_weight

            w13.data = shuffle_weight(w13.data)
            w2.data = shuffle_weight(w2.data)

        # Call parent class for any additional processing
        super().process_weights_after_loading(layer)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        """Get quantization config for compressed-tensors FP8."""
        from atom.model_ops.fused_moe.config import fp8_w8a8_moe_quant_config

        w1_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        a1_scale = getattr(layer, "w13_input_scale", None)
        a2_scale = getattr(layer, "w2_input_scale", None)

        # Determine block shape based on quantization type
        if self.block_quant:
            block_shape = [self.block_n, self.block_k]
        else:
            block_shape = None

        return fp8_w8a8_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=block_shape,
        )

    @mark_trace(prefix="compressed_fp8_moe", torch_compile=False)
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        fused_shared_experts_scoring_func: str | None = None,
        activation: ActivationType = ActivationType.Silu,
        prefix: str = "",
    ) -> torch.Tensor:
        """Apply compressed-tensors FP8 MoE computation."""
        # Select top-k experts using router logits
        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            num_fused_shared_experts=layer.num_fused_shared_experts,
            num_routing_experts=layer.global_num_experts,
            fused_shared_experts_scoring_func=fused_shared_experts_scoring_func,
            routed_scaling_factor=layer.routed_scaling_factor,
        )

        # Get activation scales (may be None for dynamic quantization)
        a1_scale = getattr(layer, "w13_input_scale", None)
        a2_scale = getattr(layer, "w2_input_scale", None)

        # Use modular kernel if available (for EP/DP setups).
        # Otherwise route through the standard AITER fused_moe path so
        # compressed-tensors FP8 shares the same tuned-kernel dispatch.
        if self.fused_experts is not None:
            return self.fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=False,
                activation=activation,
                quant_type=self.quant_type,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                expert_mask=layer.expert_mask,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                apply_router_weight_on_input=apply_router_weight_on_input,
                moe_extra_args={
                    "gate_mode": GateMode.SEPARATED.value,
                    "swiglu_limit": getattr(layer, "swiglu_limit", 0.0),
                },
            )
        else:
            return torch.ops.aiter.rocm_aiter_fused_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                expert_mask=layer.expert_mask,
                activation=activation.value,
                quant_type=self.quant_type.value,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                doweight_stage1=apply_router_weight_on_input,
                gate_mode=GateMode.SEPARATED.value,
                swiglu_limit=getattr(layer, "swiglu_limit", 0.0),
            )


class Fp8MoEMethod(FusedMoEMethodBase):
    """MoE method for FP8.
    Supports three quantization strategies:
      - per_Tensor:  per-tensor weight scale, static/dynamic activation scale
      - per_Token (PTPTC): per-channel weight scale, dynamic per-token activation
      - per_1x128 / per_1x32 (block): block-wise weight scale, dynamic activation

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config (LayerQuantConfig).
    """

    def __init__(self, quant_config: LayerQuantConfig, moe: FusedMoEConfig):
        super().__init__(moe)
        self.quant_config = quant_config
        self.quant_type = quant_config.quant_type
        self.quant_dtype = quant_config.quant_dtype
        self.block_quant = (
            self.quant_type == QuantType.per_1x128
            or self.quant_type == QuantType.per_1x32
        )
        self.channel_quant = self.quant_type == QuantType.per_Token
        self.need_normalize_e4m3fn_to_e4m3fnuz = (
            self.quant_dtype == torch.float8_e4m3fnuz
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.num_experts = num_experts
        intermediate_size_for_weight = intermediate_size_per_partition
        # Transient sources use checkpoint shapes; target-only kernel padding
        # would prevent streaming coverage from completing.
        stream_online_quant = getattr(layer, "_stream_online_quant", False)

        if self.block_quant:
            if self.quant_type == QuantType.per_1x128:
                block_n = 128
                block_k = 128
            elif self.quant_type == QuantType.per_1x32:
                block_n = 1
                block_k = 32
            tp_size = get_tp_group().world_size
            # NOTE: To ensure proper alignment of the block-wise quantization
            # scales, the output_size of the weights for both the gate and up
            # layers must be divisible by block_n.
            # Required by column parallel or enabling merged weights
            if intermediate_size_per_partition % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )
            if tp_size > 1 and intermediate_size_per_partition % block_k != 0:
                # Required by row parallel
                raise ValueError(
                    f"The input_size of down's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}."
                )
            if self.quant_type == QuantType.per_1x32 and not stream_online_quant:
                # aiter's GU-interleaved MXFP8 scale shuffle packs 8 scale
                # columns, i.e. 256 weight columns for 1x32 scales. TP8 on
                # MiniMax-M3 has local intermediate=384, so pad to 512.
                # For stream online quantization
                # Streaming sources skip this padding because they are temporary
                # checkpoint storage and never enter the shuffled kernel. Keeping
                # their logical shape also lets loaded coverage reach the expected
                # numel. This does not change the quantized target: _online_quant
                # clears the streaming flag before creating its target buffers.
                scale_pack_k = block_k * 8
                intermediate_size_for_weight = (
                    (intermediate_size_per_partition + scale_pack_k - 1)
                    // scale_pack_k
                    * scale_pack_k
                )

        # WEIGHTS
        weight_device = "meta" if stream_online_quant else None
        w13_weight = atom_parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_for_weight,
                hidden_size,
                dtype=params_dtype,
                device=weight_device,
            )
        )
        layer.register_parameter("w13_weight", w13_weight)
        # Only target buffers contain padding bytes.
        if self.quant_type == QuantType.per_1x32 and not stream_online_quant:
            w13_weight.data.view(torch.uint8).zero_()
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = atom_parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_for_weight,
                dtype=params_dtype,
                device=weight_device,
            )
        )
        layer.register_parameter("w2_weight", w2_weight)
        if self.quant_type == QuantType.per_1x32 and not stream_online_quant:
            w2_weight.data.view(torch.uint8).zero_()
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        if self.channel_quant:
            # Per-channel (PTPTC): one scale per output channel per expert.
            # w13: [E, 2*N], w2: [E, hidden_size]
            w13_weight_scale = atom_parameter(
                torch.ones(
                    num_experts,
                    2 * intermediate_size_for_weight,
                    dtype=torch.float32,
                )
            )
            w2_weight_scale = atom_parameter(
                torch.ones(num_experts, hidden_size, dtype=torch.float32)
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
        elif self.block_quant:
            scale_shape_w13 = (
                num_experts,
                2 * ((intermediate_size_for_weight + block_n - 1) // block_n),
                (hidden_size + block_k - 1) // block_k,
            )
            scale_shape_w2 = (
                num_experts,
                (hidden_size + block_n - 1) // block_n,
                (intermediate_size_for_weight + block_k - 1) // block_k,
            )
            # MXFP8 checkpoints store 1x32 scales as e8m0 bytes. Keep the
            # existing float32 initialization for the original 1x128 path.
            if self.quant_type == QuantType.per_1x32:
                w13_scale_data = torch.empty(scale_shape_w13, dtype=dtypes.fp8_e8m0)
                w2_scale_data = torch.empty(scale_shape_w2, dtype=dtypes.fp8_e8m0)
                w13_scale_data.view(torch.uint8).zero_()
                w2_scale_data.view(torch.uint8).zero_()
            else:
                w13_scale_data = torch.ones(scale_shape_w13, dtype=torch.float32)
                w2_scale_data = torch.ones(scale_shape_w2, dtype=torch.float32)
            w13_weight_scale = atom_parameter(w13_scale_data)
            w2_weight_scale = atom_parameter(w2_scale_data)
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            assert self.quant_config.is_dynamic
        else:
            # Per-tensor
            w13_weight_scale = atom_parameter(
                torch.ones(num_experts, 2, dtype=torch.float32)
            )
            w2_weight_scale = atom_parameter(
                torch.ones(num_experts, dtype=torch.float32)
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)

        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        # Per-channel uses dynamic per-token activation, no static input scales.
        if self.channel_quant or self.quant_config.is_dynamic:
            layer.w13_input_scale = None
            layer.w2_input_scale = None
        else:
            w13_input_scale = atom_parameter(
                torch.ones(num_experts, dtype=torch.float32)
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)
            w2_input_scale = atom_parameter(
                torch.ones(num_experts, dtype=torch.float32)
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)

    def _normalize_weights_and_scales(self, layer: nn.Module):
        if not self.need_normalize_e4m3fn_to_e4m3fnuz:
            return
        w13_weight, w13_weight_scale, w13_input_scale = normalize_e4m3fn_to_e4m3fnuz(
            layer.w13_weight, layer.w13_weight_scale, layer.w13_input_scale
        )
        w2_weight, w2_weight_scale, w2_input_scale = normalize_e4m3fn_to_e4m3fnuz(
            layer.w2_weight, layer.w2_weight_scale, layer.w2_input_scale
        )
        layer.w13_weight = atom_parameter(w13_weight)
        layer.w13_weight_scale = atom_parameter(w13_weight_scale)
        layer.w2_weight = atom_parameter(w2_weight)
        layer.w2_weight_scale = atom_parameter(w2_weight_scale)
        if w13_input_scale is not None:
            layer.w13_input_scale = atom_parameter(w13_input_scale)
        if w2_input_scale is not None:
            layer.w2_input_scale = atom_parameter(w2_input_scale)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if self.block_quant:
            self._process_block_quant(layer)
        elif self.channel_quant:
            self._process_channel_quant(layer)
        else:
            self._process_tensor_quant(layer)

    def _process_block_quant(self, layer: nn.Module) -> None:
        assert self.quant_config.is_dynamic
        self._normalize_weights_and_scales(layer)

        if not self.need_normalize_e4m3fn_to_e4m3fnuz:
            layer.w13_weight = atom_parameter(layer.w13_weight.data)
            layer.w13_weight_scale = atom_parameter(layer.w13_weight_scale.data)
            layer.w2_weight = atom_parameter(layer.w2_weight.data)
            layer.w2_weight_scale = atom_parameter(layer.w2_weight_scale.data)

        if self.quant_type == QuantType.per_1x32:
            # aiter's MXFP8 MoE kernels consume the same gate/up interleaved
            # layout used by their 1x32 shuffle helpers. Keep this branch
            # isolated so the existing 1x128 FP8 path still uses shuffle_weights.
            # moe_shuffle_weight mirrors moe_shuffle_scale: arch-aware GUGU
            # handling (row interleave + WMMA tile shuffle on gfx1250).
            layer.w13_weight.data = moe_shuffle_weight(
                layer.w13_weight,
                experts_cnt=self.num_experts,
                is_guinterleave=True,
                gate_up=True,
            )
            layer.w2_weight.data = moe_shuffle_weight(
                layer.w2_weight,
                experts_cnt=self.num_experts,
                is_guinterleave=True,
                gate_up=False,
            )
            layer.w13_weight.is_shuffled = True
            layer.w2_weight.is_shuffled = True

            w13_scale_2d = layer.w13_weight_scale.reshape(
                -1, layer.w13_weight_scale.shape[-1]
            )
            w2_scale_2d = layer.w2_weight_scale.reshape(
                -1, layer.w2_weight_scale.shape[-1]
            )
            layer.w13_weight_scale = atom_parameter(
                moe_shuffle_scale(
                    w13_scale_2d,
                    self.num_experts,
                    is_guinterleave=True,
                    gate_up=True,
                )
            )
            layer.w2_weight_scale = atom_parameter(
                moe_shuffle_scale(
                    w2_scale_2d,
                    self.num_experts,
                    is_guinterleave=True,
                    gate_up=False,
                )
            )
            return

        shuffle_weights(layer.w13_weight, layer.w2_weight)

    def _process_channel_quant(self, layer: nn.Module) -> None:
        """PTPTC"""
        self._normalize_weights_and_scales(layer)

        if layer.w13_weight.data.dtype in (torch.bfloat16, torch.float16):
            quant_func = get_hip_quant(QuantType.per_Token)
            for expert_id in range(layer.local_num_experts):
                w13_q, w13_s = quant_func(
                    layer.w13_weight.data[expert_id], quant_dtype=dtypes.fp8
                )
                layer.w13_weight.data[expert_id] = w13_q
                layer.w13_weight_scale.data[expert_id] = w13_s.squeeze(-1)

                w2_q, w2_s = quant_func(
                    layer.w2_weight.data[expert_id], quant_dtype=dtypes.fp8
                )
                layer.w2_weight.data[expert_id] = w2_q
                layer.w2_weight_scale.data[expert_id] = w2_s.squeeze(-1)

        shuffle_weights(layer.w13_weight, layer.w2_weight)

    def _process_tensor_quant(self, layer: nn.Module) -> None:
        if not self.quant_config.is_dynamic:
            if layer.w13_input_scale is None or layer.w2_input_scale is None:
                raise ValueError(
                    "QuantConfig has static quantization, but found "
                    "activation scales are None."
                )
            layer.w13_input_scale = atom_parameter(layer.w13_input_scale.max())
            layer.w2_input_scale = atom_parameter(layer.w2_input_scale.max())

        self._normalize_weights_and_scales(layer)

        assert layer.w13_weight_scale is not None
        shard_size = layer.intermediate_size_per_partition
        max_w13_scales = layer.w13_weight_scale.max(dim=1).values
        for expert_id in range(layer.local_num_experts):
            start = 0
            for shard_id in range(2):
                dq_weight = per_tensor_dequantize(
                    layer.w13_weight[expert_id][start : start + shard_size, :],
                    layer.w13_weight_scale[expert_id][shard_id],
                )
                quant_func = get_hip_quant(self.quant_type)
                (
                    layer.w13_weight[expert_id][start : start + shard_size, :],
                    _,
                ) = quant_func(dq_weight, max_w13_scales[expert_id])
                start += shard_size

        shuffle_weights(layer.w13_weight, layer.w2_weight)

        layer.w13_weight_scale = atom_parameter(max_w13_scales)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        if self.channel_quant:
            return fp8_w8a8_moe_quant_config(
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
                per_act_token_quant=True,
            )
        else:
            # block_quant (per_1x128 / per_1x32) MUST hand the kernel its block
            # shape — otherwise fp8_w8a8 broadcasts the [N/128, K/128] scale as
            # if it were per-tensor / per-channel and produces garbage.
            # V4-Flash-Base on gfx942 hits this: routed experts are FP8 e4m3
            # per_1x128 + UE8M0 block-scale.
            if self.block_quant:
                if self.quant_type == QuantType.per_1x128:
                    block_shape = [128, 128]
                elif self.quant_type == QuantType.per_1x32:
                    block_shape = [1, 32]
                else:
                    block_shape = None
            else:
                block_shape = None
            return fp8_w8a8_moe_quant_config(
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
                block_shape=block_shape,
            )

    @mark_trace(prefix="fp8_moe", torch_compile=False)
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        fused_shared_experts_scoring_func: str | None = None,
        activation: ActivationType = ActivationType.Silu,
        prefix: str = "",
    ) -> torch.Tensor:
        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            fused_shared_experts_scoring_func=fused_shared_experts_scoring_func,
            num_routing_experts=global_num_experts,
            num_fused_shared_experts=layer.num_fused_shared_experts,
            routed_scaling_factor=layer.routed_scaling_factor,
        )
        # Match the 1x32 preshuffled layout above; other FP8 quant modes keep
        # the historical separated gate/up layout.
        gate_mode = (
            GateMode.INTERLEAVE.value
            if self.quant_type == QuantType.per_1x32
            else GateMode.SEPARATED.value
        )
        moe_extra_args = {
            "gate_mode": gate_mode,
            "swiglu_limit": getattr(layer, "swiglu_limit", 0.0),
        }
        if self.quant_type == QuantType.per_Tensor or self.fused_experts is None:
            return torch.ops.aiter.rocm_aiter_fused_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                expert_mask=layer.expert_mask,
                activation=activation.value,
                quant_type=self.quant_type.value,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
                doweight_stage1=apply_router_weight_on_input,
                **moe_extra_args,
            )
        return self.fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
            activation=activation,
            quant_type=self.quant_type,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            expert_mask=layer.expert_mask,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            apply_router_weight_on_input=apply_router_weight_on_input,
            moe_extra_args=moe_extra_args,
        )


def moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    atom_config = get_current_atom_config()
    self = atom_config.compilation_config.static_forward_context[layer_name]
    return self.forward_impl(hidden_states, router_logits)


def moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="moe_forward",
    op_func=moe_forward,
    mutates_args=["hidden_states"],
    fake_impl=moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


@FusedMoEDecoratorForPluginMode
class FusedMoE(torch.nn.Module):
    """FusedMoE layer for MoE models.

    This layer contains both MergedColumnParallel weights (gate_up_proj /
    w13) and RowParallelLinear weights (down_proj/ w2).

    Note: Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We
    copy that naming convention here and handle any remapping in the
    load_weights function in each model implementation.

    Args:
        num_experts: Number of experts in the model
        top_k: Number of experts selected for each token
        hidden_size: Input hidden state size of the transformer
        intermediate_size: Intermediate size of the experts
        params_dtype: Data type for the parameters.
        reduce_results: Whether to all all_reduce on the output of the layer
        renomalize: Whether to renormalize the logits in the fused_moe kernel
        quant_config: Quantization configure.
    """

    def __init__(
        self,
        num_experts: int,  # Global number of experts
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype | None = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: int | None = None,
        topk_group: int | None = None,
        quant_config: QuantizationConfig | None = None,
        tp_size: int | None = None,
        ep_size: int | None = None,
        dp_size: int | None = None,
        layer_id: int | None = None,
        prefix: str = "",
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        has_bias: bool = False,
        activation: ActivationType = ActivationType.Silu,
        shared_expert_scoring_func: str | None = None,
        config: PretrainedConfig | None = None,
        shared_expert_prefix: str | None = None,
        pad_align: int | None = None,
        enable_comm_fused: bool = False,
        mxfp4_fp8_activations: bool = False,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.prefix = prefix
        layer_quant_config = (
            quant_config.get_layer_quant_config(prefix, check_children=True)
            if quant_config
            else None
        )
        self.params_dtype = (
            layer_quant_config.quant_dtype
            if layer_quant_config
            else torch.get_default_dtype()
        )
        self.layer_quant_config = layer_quant_config
        self.source_is_nvfp4 = (
            layer_quant_config is not None
            and layer_quant_config.quant_dtype == NVFP4_DTYPE
        )
        if self.source_is_nvfp4:
            validate_nvfp4_online_target(quant_config, prefix)
        self.has_bias = has_bias

        # Note: here we guard against accessing the TP and DP groups when
        # uninitialized (this happens when testing)
        # self.tp_size = 1
        tp_size = tp_size if tp_size is not None else get_tp_group().world_size
        dp_size = dp_size if dp_size is not None else get_dp_group().world_size

        atom_config = get_current_atom_config()
        self.moe_parallel_config = FusedMoEParallelConfig.make(
            tp_size, dp_size, atom_config
        )
        self._comm_fused_moe = None
        if enable_comm_fused:
            from atom.model_ops.fused_moe.comm_fused_moe import (
                create_comm_fused_moe_backend,
            )

            self._comm_fused_moe = create_comm_fused_moe_backend(
                layer_quant_config=layer_quant_config,
                online_quant=quant_config is not None and quant_config.online_quant,
                parallel_config=self.moe_parallel_config,
                model_dim=hidden_size,
                inter_dim=intermediate_size // self.tp_size,
                experts=num_experts,
                topk=top_k,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
        if shared_expert_prefix is None and prefix.endswith(".experts"):
            shared_expert_prefix = prefix[: -len(".experts")] + ".shared_experts"
        fuse_shared_experts = (
            self._comm_fused_moe is None
            and is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config(
                quant_config,
                shared_expert_prefix=shared_expert_prefix,
                routed_expert_prefix=prefix,
            )
        )
        num_fused_shared_experts = (
            getattr(config, "n_shared_experts", 0) if fuse_shared_experts else 0
        )

        eplb_enabled = self.use_ep and getattr(atom_config, "eplb_enable", False)
        num_redundant_experts = (
            int(getattr(atom_config.eplb_config, "num_redundant_experts", 0))
            if eplb_enabled
            else 0
        )
        self.expert_layout = MoEExpertLayout.make(
            num_routed=num_experts,
            num_fused_shared_experts=num_fused_shared_experts,
            num_configured_redundant=num_redundant_experts,
            ep_size=self.ep_size,
            use_all2all=self.moe_parallel_config.use_all2all_kernels,
            eplb_enabled=eplb_enabled,
        )
        # Dispatch space: the id bound the kernels see, not the routed width.
        self.global_num_experts = self.expert_layout.num_physical
        self.num_redundant_experts = self.expert_layout.num_redundant
        if self.use_ep:
            assert self.global_num_experts % self.ep_size == 0, (
                "EPLB physical experts must be divisible by ep_size: "
                f"num_logical={self.expert_layout.num_logical}, "
                f"num_redundant={self.num_redundant_experts}, ep_size={self.ep_size}"
            )
        self.register_buffer("expert_map", None, persistent=False)
        self.register_buffer("expert_mask", None, persistent=False)
        if self.use_ep:
            # Routed width: expert_map is indexed by checkpoint expert ids.
            self.local_num_experts, self.expert_map = determine_expert_map(
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_num_experts=self.expert_layout.num_routed_physical,
            )
        else:
            self.local_num_experts = self.global_num_experts
        self.top_k = top_k
        self.shared_expert_scoring_func = shared_expert_scoring_func
        self.routed_scaling_factor = (
            getattr(config, "routed_scaling_factor", 1.0)
            if config is not None and atom_config.torch_dtype != torch.float16
            else 1.0
        )
        if (
            self.expert_layout.uses_dispatch_remap
            and custom_routing_function is not None
        ):
            raise NotImplementedError(
                "Fusing shared experts into the all2all dispatch does not "
                "support custom routing functions; those still depend on "
                "the legacy AITER shared-expert metadata."
            )
        if self.use_ep and not self.expert_layout.shared_is_routed:
            self.expert_map = torch.cat(
                (
                    self.expert_map,
                    torch.tensor(
                        [
                            self.local_num_experts + i
                            for i in range(self.num_fused_shared_experts)
                        ],
                        dtype=torch.int32,
                    ),
                    # Sentinel used by aiter topK for non-local tokens.
                    torch.tensor([-1], dtype=torch.int32),
                ),
                dim=0,
            )
        if self.expert_layout.mode in (
            SharedExpertMode.LEGACY_AITER,
            SharedExpertMode.LOCAL_REPLICA,
        ):
            self.local_num_experts += self.num_fused_shared_experts
        if self.expert_layout.uses_all2all_fusion and self.layer_id in (None, 0):
            logger.info(
                "Shared expert fusion: mode=%s, local_num_experts=%d, topk=%d",
                self.expert_layout.mode.name,
                self.local_num_experts,
                self.top_k + self.num_fused_shared_experts,
            )
        if self.use_ep:
            if self.expert_layout.uses_dispatch_remap:
                self.expert_mask = torch.zeros(
                    self.global_num_experts,
                    dtype=torch.int32,
                    device=self.expert_map.device,
                )
                start = self.ep_rank * self.local_num_experts
                self.expert_mask[start : start + self.local_num_experts] = 1
            else:
                self.expert_mask = (self.expert_map > -1).to(torch.int32)
        moe_token_capacity = moe_kernel_token_capacity(
            atom_config,
            dp_size=self.moe_parallel_config.dp_size,
            use_all2all=self.moe_parallel_config.use_all2all_kernels,
            dp_logical_ratio=self.moe_parallel_config.dp_logical_ratio,
        )
        if self.expert_layout.mode is SharedExpertMode.LEGACY_AITER:
            init_aiter_topK_meta_data(
                n_routed_experts=num_experts,
                n_shared_experts=self.num_fused_shared_experts,
                top_k=self.top_k,
                tp_rank=self.ep_rank if self.use_ep else self.tp_rank,
                tp_size=self.ep_size if self.use_ep else self.tp_size,
                shared_experts_score=(
                    1.0
                    if is_rocm_aiter_fuse_routed_scaling_factor()
                    else 1 / self.routed_scaling_factor
                ),
                max_num_tokens=moe_token_capacity,
                is_EP=self.use_ep,
            )
        assert intermediate_size % self.tp_size == 0
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.e_score_correction_bias = e_score_correction_bias
        if atom_config.fake_eplb and e_score_correction_bias is not None:
            # The noaux_tc bias is added on top of the score, so a bias wider than
            # the synthetic logits' gap (~3.16 after sqrtsoftplus) picks the top-k
            # by itself -- and `--load_dummy` never writes it, leaving recycled
            # garbage that collapses every token onto one EP rank. `del` first: a
            # plain tensor cannot overwrite the parameter registered above.
            del self.e_score_correction_bias
            self.e_score_correction_bias = torch.zeros_like(e_score_correction_bias)
        self.activation = activation
        if config is not None:
            self.activation_situ_beta = getattr(config, "activation_situ_beta", None)
            self.activation_situ_linear_beta = getattr(
                config, "activation_situ_linear_beta", None
            )
        else:
            self.activation_situ_beta = None
            self.activation_situ_linear_beta = None

        self.use_chunked = get_dp_group().world_size > 1

        if self.source_is_nvfp4:
            # NVFP4 is currently always converted to MXFP4 by online
            # quantization; its input stages (FP4 values, FP8 scales) do not
            # describe the converted layer's activations.
            a_quant_dtype = None
        else:
            try:
                a_quant_dtype = (
                    config.quantization_config.get("global_quant_config", "")
                    .get("input_tensors", "")
                    .get("dtype", "")
                )
            except AttributeError:
                # global quant config does not exist, no activation loaded
                a_quant_dtype = None

        moe = FusedMoEConfig(
            num_experts=self.global_num_experts,
            experts_per_token=self.top_k + self.num_fused_shared_experts,
            hidden_dim=hidden_size,
            num_local_experts=self.local_num_experts,
            moe_parallel_config=self.moe_parallel_config,
            expert_layout=self.expert_layout,
            in_dtype=atom_config.torch_dtype,
            a_quant_dtype=a_quant_dtype,
            mxfp4_fp8_activations=mxfp4_fp8_activations,
            max_num_tokens=moe_token_capacity,
            has_bias=self.has_bias,
            # is_act_and_mul=True,
            is_lora_enabled=False,
        )
        self.moe_config = moe
        self.quant_config = quant_config
        self.online_quant = quant_config is not None and quant_config.online_quant

        quant_method_str = (
            layer_quant_config.quant_method if layer_quant_config else None
        )
        if layer_quant_config is None or layer_quant_config.quant_type == QuantType.No:
            self.quant_method: QuantizeMethodBase | None = UnquantizedFusedMoEMethod(
                moe
            )
        elif (
            quant_method_str == "compressed-tensors"
            and layer_quant_config.quant_dtype == dtypes.fp8
        ):
            # Use CompressedTensorsFp8MoEMethod for compressed-tensors format
            self.quant_method = CompressedTensorsFp8MoEMethod(layer_quant_config, moe)
        elif layer_quant_config.quant_dtype == NVFP4_DTYPE:
            self.quant_method = Nvfp4MoEMethod(layer_quant_config, moe)
        elif layer_quant_config.quant_dtype == dtypes.fp8:
            self.quant_method = Fp8MoEMethod(layer_quant_config, moe)
        elif layer_quant_config.quant_dtype == dtypes.fp4x2:
            self.quant_method = _make_mxfp4_moe_method(layer_quant_config, moe)
        else:
            raise ValueError(
                f"Unsupported quant dtype: {layer_quant_config.quant_dtype}"
            )

        assert self.quant_method is not None
        if self.expert_layout.uses_all2all_fusion and not isinstance(
            self.quant_method, (Mxfp4MoEMethod, Nvfp4MoEMethod)
        ):
            raise NotImplementedError(
                "Shared-expert fusion is only wired up for the MXFP4 MoE path, got "
                f"{type(self.quant_method).__name__}"
            )

        # Override weight padding alignment before create_weights consumes it.
        # Must happen here (pre-create_weights) — setting it on quant_method
        # after FusedMoE construction is a no-op since weights are already sized.
        if pad_align is not None:
            self.quant_method.pad_align = pad_align

        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": hidden_size,
            "intermediate_size_per_partition": self.intermediate_size_per_partition,
            "params_dtype": self.params_dtype,
            "weight_loader": self.weight_loader,
        }
        # Set before create_weights so streaming sources allocate on meta.
        self._stream_online_quant = should_stream_online_quant(
            quant_config,
            prefix,
            layer_quant_config.quant_type if layer_quant_config else QuantType.No,
            self.params_dtype,
        )
        self._load_device = torch.empty(0).device if self._stream_online_quant else None
        self.quant_method.create_weights(layer=self, **self.moe_quant_params)
        compilation_config = atom_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.layer_name = prefix
        # Interleave the synthetic logits across the token-shard (DP) axis only
        # when DP-attention actually shards tokens per DP group; otherwise every
        # device sees the same tokens and must share one table (dp_size=1).
        _dp_shard = self.use_ep and atom_config.enable_dp_attention
        self.balance_router_logits = (
            init_balance_router_logits(
                self.expert_layout.num_routed,
                top_k,
                self.ep_size if self.use_ep else 1,
                self.dp_size if _dp_shard else 1,
                self.dp_rank if _dp_shard else 0,
                (self.num_expert_group or 1) if self.use_grouped_topk else 1,
                (self.topk_group or 1) if self.use_grouped_topk else 1,
                torch.get_default_dtype(),
                atom_config.max_num_batched_tokens,
            )
            if atom_config.fake_eplb
            else None
        )

    @property
    def num_fused_shared_experts(self) -> int:
        return self.expert_layout.num_fused_shared_experts

    @property
    def shared_expert_weight(self) -> float:
        if is_rocm_aiter_fuse_routed_scaling_factor():
            return 1.0
        return 1.0 / self.routed_scaling_factor

    @property
    def shared_dispatch_base(self) -> int:
        """This rank's first pinned shared slot, in dispatch space."""
        return (
            self.ep_rank * self.local_num_experts
            + self.expert_layout.routed_physical_per_rank
        )

    def to_dispatch_space(
        self, topk_weights: torch.Tensor, topk_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Widen physical ids and pin this rank's shared column onto them."""
        if not self.expert_layout.uses_dispatch_remap:
            return topk_weights, topk_ids
        return remap_topk_to_dispatch(
            topk_weights,
            topk_ids,
            self.expert_layout.num_routed_physical,
            self.expert_layout.routed_physical_per_rank,
            self.shared_dispatch_base,
            self.num_fused_shared_experts,
            self.shared_expert_weight,
        )

    def append_shared_logical_column(
        self,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append shared experts to routed logical columns."""
        num_tokens = topk_ids.shape[0]
        num_fused_shared_experts = self.num_fused_shared_experts
        shared_ids = torch.arange(
            self.expert_layout.num_routed,
            self.expert_layout.num_routed + num_fused_shared_experts,
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        ).expand(num_tokens, num_fused_shared_experts)
        shared_w = torch.full(
            (num_tokens, num_fused_shared_experts),
            self.shared_expert_weight,
            dtype=topk_weights.dtype,
            device=topk_weights.device,
        )
        return (
            torch.cat((topk_weights, shared_w), dim=1),
            torch.cat((topk_ids, shared_ids), dim=1),
        )

    def process_weights_after_loading(self):
        self._online_quant()
        self._validate_moe_backend()
        if self._comm_fused_moe is not None:
            self._comm_fused_moe.initialize(self)

    def _validate_moe_backend(self) -> None:
        if get_current_atom_config().moe_backend != "mega":
            return
        if not isinstance(self.quant_method, MegaMxfp4MoEMethod):
            raise TypeError(
                "moe_backend='mega' currently supports only MXFP4/A8W4 MoE, "
                f"got {type(self.quant_method).__name__}"
            )
        if self.expert_layout.mode is SharedExpertMode.LEGACY_AITER:
            raise NotImplementedError(
                "moe_backend='mega' requires the shared expert to be fused into "
                "the dispatch space; the legacy AITER fusion is unsupported."
            )

    def _online_quant_row_local_batched(
        self,
        old_w13_data: torch.Tensor,
        old_w2_data: torch.Tensor,
        old_w13_scale: torch.Tensor | None,
        old_w2_scale: torch.Tensor | None,
        source_quant_type: QuantType,
        source_quant_dtype: torch.dtype | None,
        online_quant_type: QuantType,
        online_quant_dtype: torch.dtype,
    ) -> None:
        """Batch row-local experts to accelerate streaming online quantization."""
        batch_size = 8

        def _quantize(source, scale):
            bf16 = dequant_moe_weight_online(
                source,
                scale,
                source_quant_type,
                source_quant_dtype,
            )
            quantized, quant_scale = quant_weight_online(
                bf16,
                online_quant_type=online_quant_type,
                online_quant_dtype=online_quant_dtype,
            )
            del bf16
            return quantized, quant_scale

        def _reshape_scale(scale: torch.Tensor, count: int) -> torch.Tensor:
            assert scale.dim() == 2, (
                "row-local online quantization must return a 2D scale, "
                f"got shape={tuple(scale.shape)}"
            )
            assert scale.shape[0] % count == 0
            return scale.reshape(count, scale.shape[0] // count, scale.shape[1])

        def _copy_scale(
            target: torch.Tensor,
            source: torch.Tensor,
            *,
            split_gate_up: bool,
        ) -> None:
            # PTPC stores [E, rows], while block schemes store
            # [E, scale_rows, scale_cols].
            if online_quant_type == QuantType.per_Token:
                assert source.shape[2] == 1
                source = source.squeeze(2)
            assert source.dim() == target.dim()

            def _copy_rows(
                target_start: int,
                source_start: int,
                rows: int,
            ) -> None:
                if target.dim() == 2:
                    target_region = target[:, target_start : target_start + rows]
                    source_region = source[:, source_start : source_start + rows]
                else:
                    target_region = target[
                        :,
                        target_start : target_start + rows,
                        : source.shape[2],
                    ]
                    source_region = source[
                        :,
                        source_start : source_start + rows,
                    ]
                FusedMoE._copy_quant_storage(target_region, source_region)

            if split_gate_up:
                source_half_rows = source.shape[1] // 2
                target_half_rows = target.shape[1] // 2
                _copy_rows(0, 0, source_half_rows)
                _copy_rows(
                    target_half_rows,
                    source_half_rows,
                    source_half_rows,
                )
            else:
                _copy_rows(0, 0, source.shape[1])

        for start in range(0, self.local_num_experts, batch_size):
            end = min(start + batch_size, self.local_num_experts)
            count = end - start

            # Copy gate/up separately to support target-side padding between
            # the two halves; without padding their offsets are identical.
            source_w13 = old_w13_data[start:end]
            source_w13_scale = (
                old_w13_scale[start:end] if old_w13_scale is not None else None
            )
            w13_q, w13_s = _quantize(source_w13, source_w13_scale)
            source_w13_rows = source_w13.shape[1]
            source_half_rows = source_w13_rows // 2
            w13_q = w13_q.reshape(count, source_w13_rows, w13_q.shape[1])
            w13_s = _reshape_scale(w13_s, count)

            target_w13 = self.w13_weight.data[start:end]
            target_w13_scale = self.w13_weight_scale.data[start:end]
            target_half_rows = target_w13.shape[1] // 2
            FusedMoE._copy_quant_storage(
                target_w13[:, :source_half_rows, : w13_q.shape[2]],
                w13_q[:, :source_half_rows],
            )
            FusedMoE._copy_quant_storage(
                target_w13[
                    :,
                    target_half_rows : target_half_rows + source_half_rows,
                    : w13_q.shape[2],
                ],
                w13_q[:, source_half_rows:],
            )
            _copy_scale(target_w13_scale, w13_s, split_gate_up=True)
            del w13_q, w13_s

            source_w2 = old_w2_data[start:end]
            source_w2_scale = (
                old_w2_scale[start:end] if old_w2_scale is not None else None
            )
            w2_q, w2_s = _quantize(source_w2, source_w2_scale)
            source_w2_rows = source_w2.shape[1]
            w2_q = w2_q.reshape(count, source_w2_rows, w2_q.shape[1])
            w2_s = _reshape_scale(w2_s, count)

            target_w2 = self.w2_weight.data[start:end]
            target_w2_scale = self.w2_weight_scale.data[start:end]
            FusedMoE._copy_quant_storage(
                target_w2[:, :source_w2_rows, : w2_q.shape[2]],
                w2_q,
            )
            _copy_scale(target_w2_scale, w2_s, split_gate_up=False)
            del w2_q, w2_s

    def _commit_online_quant_state(
        self,
        online_quant_config,
        online_quant_type,
        online_quant_dtype,
        *,
        source_is_nvfp4: bool,
    ) -> None:
        """Point the layer at the online target after weights have been converted.

        ``weight_loader`` and a later ``_online_quant`` re-entry both read
        ``layer_quant_config`` / ``params_dtype``. Leaving the source NVFP4
        spec in place keeps the merged-load guard armed and would AttributeError
        on the delattr'd ``*_weight_scale_2`` names. Match Linear: rewrite the
        spec, and drop NVFP4 scalars by assignment rather than deleting them.
        """
        self.layer_quant_config = online_quant_config
        self.params_dtype = online_quant_dtype
        self._stream_online_quant = False
        if source_is_nvfp4:
            self.source_is_nvfp4 = False
            self.w13_weight_scale_2 = None
            self.w2_weight_scale_2 = None
        self._online_quant_info = {
            "layer": self.layer_name,
            "quant_type": online_quant_type.name,
            "quant_dtype": str(online_quant_dtype),
        }

    def _online_quant(self):
        """Handle online quantization: (optionally dequant →) quantize weights,
        then switch quant_method.

        Called by the loader BEFORE quant_method.process_weights_after_loading().
        Flow:
          1. If source is already quantized (e.g. per_1x128 FP8), dequant → float
          2. Switch quant_method and allocate target quantized buffers
          3. Per-expert: quantize float → write into buffers via
             _load_model_weight_or_group_weight_scale (reuses TP-shard + padding)
             Row-local targets may batch experts before writing the same layout.
          4. Loader then calls target method's process_weights_after_loading
             which does fn→fnuz normalization and shuffle on the already-FP8 weights.
        """
        if not self.online_quant:
            return

        online_quant_config = self.quant_config.get_layer_quant_config(
            self.layer_name, use_online_quant=True
        )
        online_quant_type = online_quant_config.quant_type
        online_quant_dtype = online_quant_config.quant_dtype
        source_quant_type = self.layer_quant_config.quant_type
        source_quant_dtype = self.layer_quant_config.quant_dtype
        source_is_nvfp4 = self.source_is_nvfp4
        if should_skip_online_quant(
            source_quant_type, self.params_dtype, online_quant_config
        ):
            return
        # Re-quantize any source we can dequantize back to float first:
        # unquantized (No), per-output-channel FP8 (per_Token / ptpc_fp8),
        # 128x128 block FP8 (per_1x128), MXFP8 (per_1x32) and NVFP4 (per_1x32
        # with NVFP4_DTYPE). Other sources (e.g. per_Tensor) are rejected up
        # front rather than silently producing garbage.
        assert source_quant_type in (
            QuantType.No,
            QuantType.per_Token,
            QuantType.per_1x128,
            QuantType.per_1x32,
        ), (
            f"Unsupported source quant_type for MoE online quantization: "
            f"{source_quant_type} (layer={self.layer_name}). Supported sources: "
            f"No, per_Token, per_1x128, per_1x32."
        )
        need_dequant = source_quant_type in (
            QuantType.per_Token,
            QuantType.per_1x128,
            QuantType.per_1x32,
        )

        def _dequant_func(
            w: torch.Tensor,
            sc: torch.Tensor,
            global_scale: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if source_is_nvfp4:
                # Both branches return the model dtype, so the w1_bf16 /
                # w3_bf16 names below hold for every source format.
                return dequantize_nvfp4(
                    w.contiguous(),
                    sc.contiguous(),
                    global_scale,
                )
            return dequant_weight_online(
                w.contiguous(), sc.contiguous(), source_quant_type, source_quant_dtype
            )

        # Online target quantization uses the same shared dispatch for every
        # source format, including NVFP4 after it is dequantized.
        def _quant_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return quant_weight_online(
                w,
                online_quant_type=online_quant_type,
                online_quant_dtype=online_quant_dtype,
            )

        # Determine whether each weight needs all_gather to match offline quantization.
        # w13 (column parallel): (E, (2*intermediate/tp, hidden)) — TP dim 0
        # w2  (row parallel):    (E, (hidden, intermediate/tp)) — TP dim 1
        # w13 [e, m, n]->[e, m//tp, n//2]->[e, m//tp, n//32]
        # Streaming tails must remain collective-free across ranks.
        def check_need_allgather():
            if self.use_ep:
                assert self.tp_size == 1, "EP MoE should not TP-shard expert weights"
                return False
            if source_is_nvfp4:
                # There is no offline MXFP4 layout to reproduce: the blocks are
                # made here, per rank, so the local shard is the only one that
                # matters. `Nvfp4MoEMethod.create_weights` rejects shards that
                # are not 32-aligned, which is what the `% 32` rule below would
                # otherwise catch -- and gathering cannot fix those anyway,
                # since a globally quantized block would straddle two ranks.
                return False

            need_gather_w2 = False
            if not self._stream_online_quant and self.tp_size > 1:
                # self.intermediate_size_per_partition = intermediate_size // self.tp_size
                w2_in = self.intermediate_size_per_partition
                if online_quant_type == QuantType.per_Token:
                    need_gather_w2 = True
                elif online_quant_type == QuantType.per_1x32:
                    need_gather_w2 = w2_in % 32 != 0
            return need_gather_w2

        need_gather_w2 = check_need_allgather()
        tp_group = get_tp_group() if need_gather_w2 else None
        load_full_w2 = not need_gather_w2

        # Save references to old weights before create_weights overwrites them.
        # For per_1x128 source we also need the old scales for dequantization.
        old_w13_data = self.w13_weight.data
        old_w2_data = self.w2_weight.data
        old_w13_scale = self.w13_weight_scale.data if need_dequant else None
        old_w2_scale = self.w2_weight_scale.data if need_dequant else None
        old_w13_scale_2 = self.w13_weight_scale_2.data if source_is_nvfp4 else None
        old_w2_scale_2 = self.w2_weight_scale_2.data if source_is_nvfp4 else None
        if source_is_nvfp4:
            # These buffers are zero-filled at allocation, so a checkpoint that
            # omits *_weight_scale_2 leaves them zero and dequantizes every
            # expert to all zeros -- a silent accuracy loss, not a crash.
            # EPLB redundant replicas are exempt: fill_redundant copies them
            # from their base experts only after this conversion has run, so
            # their slots still hold the zero placeholder here.
            n_base = self.num_local_base_experts
            replicas_end = self.expert_layout.routed_physical_per_rank
            for what, scale_2 in (
                ("w13_weight_scale_2", old_w13_scale_2),
                ("w2_weight_scale_2", old_w2_scale_2),
            ):
                validate_nvfp4_global_scales(
                    torch.cat((scale_2[:n_base], scale_2[replicas_end:])),
                    self.layer_name,
                    what,
                )
        device = old_w13_data.device
        # Nvfp4MoEMethod pads nothing itself, so an intermediate_pad on it was
        # pinned by the model for the MXFP4 target, whose create_weights resets it.
        pinned_intermediate_pad = (
            getattr(self.quant_method, "intermediate_pad", None)
            if source_is_nvfp4
            else None
        )

        # Switch quant_method and allocate target quantized-type buffers.
        if online_quant_dtype == dtypes.fp8:
            self.quant_method = Fp8MoEMethod(online_quant_config, self.moe_config)
        elif online_quant_dtype == dtypes.fp4x2:
            self.quant_method = _make_mxfp4_moe_method(
                online_quant_config, self.moe_config
            )
        else:
            raise ValueError(
                f"Unsupported online quant_dtype for MoE: {online_quant_dtype}"
            )
        self.moe_quant_params["params_dtype"] = online_quant_dtype
        # Target buffers must use real storage.
        self._stream_online_quant = False
        with torch.device(device):
            self.quant_method.create_weights(layer=self, **self.moe_quant_params)
        if pinned_intermediate_pad is not None and hasattr(
            self.quant_method, "intermediate_pad"
        ):
            self.quant_method.intermediate_pad = pinned_intermediate_pad

        self.w13_input_scale = None
        self.w2_input_scale = None

        # Batch row-local formats; per-tensor and gathered formats stay expertwise.
        row_local_target = online_quant_type in (
            QuantType.per_Token,
            QuantType.per_1x32,
            QuantType.per_1x128,
        )
        target_rows_are_batchable = online_quant_type != QuantType.per_1x128 or (
            old_w13_data.shape[1] // 2 % 128 == 0 and old_w2_data.shape[1] % 128 == 0
        )
        if (
            not source_is_nvfp4
            and row_local_target
            and target_rows_are_batchable
            and not need_gather_w2
        ):
            self._online_quant_row_local_batched(
                old_w13_data=old_w13_data,
                old_w2_data=old_w2_data,
                old_w13_scale=old_w13_scale,
                old_w2_scale=old_w2_scale,
                source_quant_type=source_quant_type,
                source_quant_dtype=source_quant_dtype,
                online_quant_type=online_quant_type,
                online_quant_dtype=online_quant_dtype,
            )
            FusedMoE._commit_online_quant_state(
                self,
                online_quant_config,
                online_quant_type,
                online_quant_dtype,
                source_is_nvfp4=source_is_nvfp4,
            )
            return

        for expert_id in range(self.local_num_experts):
            # --- w13 column-parallel ---
            w13_local = old_w13_data[expert_id]
            w1_size = w13_local.shape[0] // 2

            if need_dequant:
                w13_scale = old_w13_scale[expert_id]
                s1_size = w13_scale.shape[0] // 2
                w1_bf16 = _dequant_func(
                    w13_local[:w1_size],
                    w13_scale[:s1_size],
                    (old_w13_scale_2[expert_id, 0] if source_is_nvfp4 else None),
                )
                w3_bf16 = _dequant_func(
                    w13_local[w1_size:],
                    w13_scale[s1_size:],
                    (old_w13_scale_2[expert_id, 1] if source_is_nvfp4 else None),
                )
            else:
                w1_bf16 = w13_local[:w1_size]
                w3_bf16 = w13_local[w1_size:]

            w1_q, w1_s = _quant_weight(w1_bf16)
            w3_q, w3_s = _quant_weight(w3_bf16)
            del w1_bf16, w3_bf16

            w13_expert = self.w13_weight.data[expert_id]
            w13_scale_expert = self.w13_weight_scale.data[expert_id]
            if source_is_nvfp4:
                target_half_rows = w13_expert.shape[0] // 2
                self._copy_quant_storage(
                    w13_expert[: w1_q.shape[0], : w1_q.shape[1]],
                    w1_q,
                )
                self._copy_quant_storage(
                    w13_expert[
                        target_half_rows : target_half_rows + w3_q.shape[0],
                        : w3_q.shape[1],
                    ],
                    w3_q,
                )
                self._copy_quant_storage(
                    w13_scale_expert[: w1_s.shape[0], : w1_s.shape[1]],
                    w1_s,
                )
                self._copy_quant_storage(
                    w13_scale_expert[
                        target_half_rows : target_half_rows + w3_s.shape[0],
                        : w3_s.shape[1],
                    ],
                    w3_s,
                )
            else:
                for shard_id, wq, ws in (
                    ("w1", w1_q, w1_s),
                    ("w3", w3_q, w3_s),
                ):
                    self._load_model_weight_or_group_weight_scale(
                        shard_dim=0,
                        expert_data=w13_expert,
                        shard_id=shard_id,
                        loaded_weight=wq,
                        tp_rank=self.tp_rank,
                        load_full=True,
                    )
                    self._load_quant_weight_scale(
                        expert_data=w13_scale_expert,
                        shard_dim=0,
                        shard_id=shard_id,
                        loaded_weight=ws,
                        tp_rank=self.tp_rank,
                        quant_type=online_quant_type,
                        load_full=True,
                    )
            del w1_q, w3_q, w1_s, w3_s

            # w2 row-parallel: optionally gather before quantization
            # w2 mxfp4    [e, m, n]->[e, m, n//2//tp]->[e, m, n//32//tp]
            # w2 ptpc_fp8 [e, m, n]->[e, m, n//tp]->[e, m, 1]
            w2_local = old_w2_data[expert_id]
            if need_dequant:
                w2_local = _dequant_func(
                    w2_local,
                    old_w2_scale[expert_id],
                    old_w2_scale_2[expert_id] if source_is_nvfp4 else None,
                )
            if need_gather_w2:
                w2_full = tp_group.all_gather(w2_local, dim=1)
                w2_q, w2_s = _quant_weight(w2_full)
                del w2_full
            else:
                w2_q, w2_s = _quant_weight(w2_local)

            if source_is_nvfp4:
                self._copy_quant_storage(
                    self.w2_weight.data[expert_id, : w2_q.shape[0], : w2_q.shape[1]],
                    w2_q,
                )
                self._copy_quant_storage(
                    self.w2_weight_scale.data[
                        expert_id, : w2_s.shape[0], : w2_s.shape[1]
                    ],
                    w2_s,
                )
            else:
                self._load_model_weight_or_group_weight_scale(
                    shard_dim=1,
                    expert_data=self.w2_weight.data[expert_id],
                    shard_id="w2",
                    loaded_weight=w2_q,
                    tp_rank=self.tp_rank,
                    load_full=load_full_w2,
                )
                # per_Token scale is along output dim (not TP-split), never needs shard
                w2_scale_load_full = (
                    False if online_quant_type == QuantType.per_Token else load_full_w2
                )
                self._load_quant_weight_scale(
                    expert_data=self.w2_weight_scale.data[expert_id],
                    shard_dim=1,
                    shard_id="w2",
                    loaded_weight=w2_s,
                    tp_rank=self.tp_rank,
                    quant_type=online_quant_type,
                    load_full=w2_scale_load_full,
                )
            del w2_q, w2_s

        del old_w13_data, old_w2_data
        if need_dequant:
            del old_w13_scale, old_w2_scale
        if source_is_nvfp4:
            del old_w13_scale_2, old_w2_scale_2

        FusedMoE._commit_online_quant_state(
            self,
            online_quant_config,
            online_quant_type,
            online_quant_dtype,
            source_is_nvfp4=source_is_nvfp4,
        )

    @property
    def tp_size(self):
        return self.moe_parallel_config.tp_size

    @property
    def dp_size(self):
        return self.moe_parallel_config.dp_size

    @property
    def ep_size(self):
        return self.moe_parallel_config.ep_size

    @property
    def tp_rank(self):
        return self.moe_parallel_config.tp_rank

    @property
    def dp_rank(self):
        return self.moe_parallel_config.dp_rank

    @property
    def ep_rank(self):
        return self.moe_parallel_config.ep_rank

    @property
    def use_ep(self):
        return self.moe_parallel_config.use_ep

    def _load_per_tensor_weight_scale(
        self,
        shard_id: str,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_id: int,
    ):
        param_data = param.data
        # for per tensor weight quantization
        if shard_id in ("w1", "w3"):
            # We have to keep the weight scales of w1 and w3 because
            # we need to re-quantize w1/w3 weights after weight loading.
            idx = 0 if shard_id == "w1" else 1
            param_data[expert_id][idx] = loaded_weight
        # If we are in the row parallel case (down_proj)
        elif shard_id == "w2":
            param_data[expert_id] = loaded_weight

    def _load_model_weight_or_group_weight_scale(
        self,
        shard_dim: int,
        expert_data: torch.Tensor,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full: bool = False,
    ):
        """
        Load grouped weight scales for group quantization or model weights
            :param shard_dim: dimension to shard
            :param expert_data: parameter for a particular expert
            :param shard_id: either w1, w2, or w3
            :param loaded_weight: checkpoint weight to load into the param
            :param tp_rank: tensor parallel rank
            :param load_full_w2: whether or not the w2 loaded should be sharded.
        """
        if shard_id == "w2":
            # In the case where we have actorder/g_idx, we do not partition the
            # w2 scales, as indicated by `load_full` argument, for all tp cases
            self._load_w2(
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
                load_full=load_full,
            )
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
                load_full=load_full,
            )

    def _load_quant_weight_scale(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        quant_type,
        load_full: bool = False,
    ):
        """Dispatch weight-scale loading by quant_type."""
        if quant_type == QuantType.per_Token:
            self._load_per_channel_weight_scale(
                expert_data=expert_data,
                shard_dim=shard_dim,
                shard_id=shard_id,
                loaded_weight=loaded_weight.squeeze(-1),
                tp_rank=tp_rank,
                load_full=load_full,
            )
        else:
            # mxfp4 (per_1x32) returns a byte-encoded FP4x2 / e8m0 scale that
            # must be byte-viewed before loading. Other schemes (e.g. per_1x128
            # 128x128 block FP8) carry a float32 scale that is copied as-is.
            if quant_type == QuantType.per_1x32:
                loaded_weight = loaded_weight.view(torch.uint8)
            self._load_model_weight_or_group_weight_scale(
                shard_dim=shard_dim,
                expert_data=expert_data,
                shard_id=shard_id,
                loaded_weight=loaded_weight,
                tp_rank=tp_rank,
                load_full=load_full,
            )

    def _load_per_channel_weight_scale(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full: bool = False,
    ):
        # for per channel weight quantization
        if load_full:
            if shard_id == "w2":
                load_size = loaded_weight.shape[shard_dim]
                if load_size != expert_data.shape[shard_dim]:
                    expert_data = expert_data.narrow(shard_dim, 0, load_size)
                self._copy_quant_storage(expert_data, loaded_weight)
            elif shard_id in ("w1", "w3"):
                self._load_w13(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                    load_full=True,
                )
            return
        if shard_id == "w2":
            self._copy_quant_storage(expert_data, loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    @staticmethod
    def _copy_quant_storage(dst: torch.Tensor, src: torch.Tensor) -> None:
        """Copy quantized tensors without numeric conversion of byte formats."""
        if src.device.type == "cpu" and not src.is_contiguous():
            src = src.contiguous()

        if dst.dtype == dtypes.fp4x2:
            dst.view(torch.uint8).copy_(src.view(torch.uint8))
            return
        fp8_storage_dtypes = (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e8m0fnu,
            dtypes.fp8,
            dtypes.fp8_e8m0,
        )
        if dst.dtype in fp8_storage_dtypes and src.dtype in fp8_storage_dtypes:
            # Offline FP8 checkpoints encode raw FP8 bytes. Avoid dtype-to-dtype
            # numeric conversion when destination storage uses a different FP8
            # variant; later scale fixups expect the original bytes.
            dst.view(torch.uint8).copy_(src.view(torch.uint8))
            return
        if dst.dtype == dtypes.fp8_e8m0 and src.dtype == torch.uint8:
            # e8m0 microscale tensors are byte-encoded; copy_ would convert the
            # uint8 values numerically instead of preserving the scale bits.
            dst.view(torch.uint8).copy_(src)
            return
        if dst.dtype == torch.uint8 and src.dtype in (
            torch.float8_e8m0fnu,
            torch.float8_e4m3fn,
        ):
            src = src.view(torch.uint8)
        dst.copy_(src)

    def _load_w13(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full: bool = False,
    ):
        # for online local quantizaiton
        if load_full:
            expert_shard_size = expert_data.shape[shard_dim] // 2
            expert_data = expert_shard_view(expert_data, shard_id, shard_dim)
            load_size = loaded_weight.shape[shard_dim]
            if load_size != expert_shard_size:
                expert_data = expert_data.narrow(shard_dim, 0, load_size)
            self._copy_quant_storage(expert_data, loaded_weight)
            return

        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
        expert_shard_size = expert_data.shape[shard_dim] // 2
        # Derive shard size from loaded_weight (unpadded checkpoint) to avoid
        # out-of-bounds when expert_data is padded (e.g. MXFP4 alignment).
        load_shard_size = loaded_weight.shape[shard_dim] // self.tp_size
        loaded_weight = loaded_weight.narrow(
            shard_dim, load_shard_size * tp_rank, load_shard_size
        )
        # Narrow parameter and load: w1 (gate_proj) into the first logical
        # weight of w13, w3 (up_proj) into the second.
        expert_data = expert_shard_view(expert_data, shard_id, shard_dim)
        # When expert_data is padded beyond the actual weight size, narrow to
        # the loaded weight size so the copy shape matches.
        if load_shard_size != expert_shard_size:
            expert_data = expert_data.narrow(shard_dim, 0, load_shard_size)
        self._copy_quant_storage(expert_data, loaded_weight)

    def _load_w2(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full: bool = False,
    ):
        # # for online local quantizaiton
        if load_full:
            shard_size = expert_data.shape[shard_dim]
            load_size = loaded_weight.shape[shard_dim]
            if load_size != shard_size:
                expert_data = expert_data.narrow(shard_dim, 0, load_size)
            self._copy_quant_storage(expert_data, loaded_weight)
            return

        # Index the loaded weight for tp sharding.
        # down_proj: "RowParallel" so tp sharding on input_dim
        # Narrow parameter and load.
        shard_size = expert_data.shape[shard_dim]
        load_shard_size = loaded_weight.shape[shard_dim] // self.tp_size
        loaded_weight = loaded_weight.narrow(
            shard_dim, load_shard_size * tp_rank, load_shard_size
        )
        if load_shard_size != shard_size:
            expert_data = expert_data.narrow(shard_dim, 0, load_shard_size)
        # w2, down_proj: Load into only logical weight of w2.
        self._copy_quant_storage(expert_data, loaded_weight)

    def _load_single_value(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int
    ):
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        param_data[expert_id] = loaded_weight

    def _load_g_idx(
        self,
        shard_id: str,
        expert_data: torch.Tensor,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        tp_rank: int,
    ):

        if shard_id == "w2":
            self._load_w2(
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
        else:
            assert shard_id in ("w1", "w3")
            expert_data.copy_(loaded_weight)

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        if self.expert_map is None:
            return expert_id
        return self.expert_map[
            physical_expert_id(
                expert_id,
                self.global_num_experts,
                self.num_redundant_experts,
                self.num_fused_shared_experts,
            )
        ].item()

    def mxf4_merged_weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_id: int | None = None,
    ):
        target_param = param
        # single_expert means gate_up_proj.shape=[2880*2, 1440] from quark
        maybe_single_expert_input = loaded_weight.dim() == param.dim() - 1
        if expert_id is not None and maybe_single_expert_input:
            local_expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
            if local_expert_id == -1:
                return
            # Support loading a split/single expert tensor while reusing the
            # original merged loading logic.
            if loaded_weight.dim() == param.dim() - 1:
                loaded_weight = loaded_weight.unsqueeze(0)
                target_param = param[local_expert_id : local_expert_id + 1]
        # (FIXME) for gpt-oss all experts are combined
        mxfp4_block = 32
        ep_rank_start = self.ep_rank * self.local_num_experts
        ep_rank_end = ep_rank_start + self.local_num_experts
        tp_rank_start = self.tp_rank * self.intermediate_size_per_partition
        tp_rank_end = tp_rank_start + self.intermediate_size_per_partition
        if param is getattr(self, "w13_bias", None):
            if self.use_ep:
                if loaded_weight.shape[0] == target_param.shape[0]:
                    narrow_weight = loaded_weight
                else:
                    narrow_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = loaded_weight[:, 2 * tp_rank_start : 2 * tp_rank_end]
            dim1 = narrow_weight.shape[1]
            target_param[:, :dim1].copy_(narrow_weight)
        elif param is getattr(self, "w2_bias", None):
            if self.use_ep:
                if loaded_weight.shape[0] == target_param.shape[0]:
                    narrow_weight = loaded_weight
                else:
                    narrow_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = loaded_weight
                if self.tp_rank != 0:
                    narrow_weight.zero_()
            dim1 = narrow_weight.shape[1]
            target_param[:, :dim1].copy_(narrow_weight)
        elif param is getattr(self, "w13_weight", None):
            loaded_weight = loaded_weight.view(*loaded_weight.shape[:2], -1)
            if self.use_ep:
                if loaded_weight.shape[0] == target_param.shape[0]:
                    narrow_weight = loaded_weight
                else:
                    narrow_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = loaded_weight[
                    :, 2 * tp_rank_start : 2 * tp_rank_end, ...
                ]
            dim1, dim2 = narrow_weight.shape[1:]
            target_param.view(torch.uint8)[:, :dim1, :dim2].copy_(
                narrow_weight.view(torch.uint8)
            )
        elif param is getattr(self, "w2_weight", None):
            loaded_weight = loaded_weight.view(*loaded_weight.shape[:2], -1)
            if self.use_ep:
                if loaded_weight.shape[0] == target_param.shape[0]:
                    narrow_weight = loaded_weight
                else:
                    narrow_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = loaded_weight[
                    ..., tp_rank_start // 2 : tp_rank_end // 2
                ]
            dim1, dim2 = narrow_weight.shape[1:]
            target_param.view(torch.uint8)[:, :dim1, :dim2].copy_(
                narrow_weight.view(torch.uint8)
            )
        elif param is getattr(self, "w13_weight_scale", None):
            if self.use_ep:
                if loaded_weight.shape[0] == target_param.shape[0]:
                    narrow_weight = loaded_weight
                else:
                    narrow_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = loaded_weight[
                    :, 2 * tp_rank_start : 2 * tp_rank_end, ...
                ]
            dim1, dim2 = narrow_weight.shape[1:]
            target_param[:, :dim1, :dim2].copy_(narrow_weight)
        elif param is getattr(self, "w2_weight_scale", None):
            if self.use_ep:
                if loaded_weight.shape[0] == target_param.shape[0]:
                    narrow_weight = loaded_weight
                else:
                    narrow_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = loaded_weight[
                    ..., tp_rank_start // mxfp4_block : tp_rank_end // mxfp4_block
                ]
            dim1, dim2 = narrow_weight.shape[1:]
            target_param[:, :dim1, :dim2].copy_(narrow_weight)
        elif param is getattr(self, "w13_input_scale", None) or param is getattr(
            self, "w2_input_scale", None
        ):
            # input_scale is scalar per expert.
            if loaded_weight.dim() == 0:
                loaded_weight = loaded_weight.unsqueeze(0)
            if self.use_ep and loaded_weight.shape[0] != target_param.shape[0]:
                narrow_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = loaded_weight
            target_param[: narrow_weight.shape[0]].copy_(narrow_weight)

    def _copy_expert_shard(
        self,
        param: torch.nn.Parameter,
        expert_data: torch.Tensor,
        shard_id: str,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        weight_name: str,
    ) -> bool:
        """Copy one expert's shard into ``expert_data``. Shared by
        ``weight_loader`` (GPU dest) and ``stage_expert_weight`` (CPU staging
        dest) so the two paths can't drift. Returns False for cases the batched
        path can't handle (per-Tensor scale, ``weight_shape``); caller falls back.
        """
        # scales / zero-points / offset (group + per-channel)
        if "scale" in weight_name or "zero" in weight_name or "offset" in weight_name:
            quant_method = self.layer_quant_config.quant_type
            if quant_method == QuantType.per_Token:
                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                )
                return True
            if quant_method in (QuantType.per_1x128, QuantType.per_1x32):
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                    load_full=getattr(param, "load_full_w2", False),
                )
                return True
            return False

        # model weights; `weight_shape` shares the substring but isn't batchable
        if "weight" in weight_name and "weight_shape" not in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.tp_rank,
            )
            return True

        return False

    def stage_expert_weight(
        self,
        param: torch.nn.Parameter,
        staging: torch.Tensor,
        loaded_weight: torch.Tensor,
        local_expert_id: int,
        shard_id: str,
        weight_name: str,
    ) -> bool:
        # Cases not handled by the batched path — caller falls back.
        if (
            "input_scale" in weight_name
            or "g_idx" in weight_name
            or "weight_shape" in weight_name
            or weight_name == ""
        ):
            return False
        if shard_id not in ("w1", "w2", "w3"):
            return False

        # compressed-tensors packed-weight flip (matches weight_loader)
        if self.quant_method.__class__.__name__ in (
            "CompressedTensorsWNA16MarlinMoEMethod",
            "CompressedTensorsWNA16MoEMethod",
        ):
            loaded_weight = loaded_weight.t().contiguous()

        shard_dim = expert_shard_dim(shard_id, getattr(param, "is_transposed", False))

        if len(loaded_weight.shape) == 3:
            return False

        expert_data = staging[local_expert_id]

        if (
            staging.dtype == torch.uint8
            and param.data.dtype == dtypes.fp4x2
            and loaded_weight.dtype == dtypes.fp4x2
        ):
            loaded_weight = loaded_weight.view(torch.uint8)

        return self._copy_expert_shard(
            param=param,
            expert_data=expert_data,
            shard_id=shard_id,
            shard_dim=shard_dim,
            loaded_weight=loaded_weight,
            weight_name=weight_name,
        )

    def expected_batched_arrivals(self, param: torch.nn.Parameter) -> int | None:
        w13_batchable = [getattr(self, "w13_weight", None)]
        w2_batchable = [getattr(self, "w2_weight", None)]
        if self.layer_quant_config.quant_type in (
            QuantType.per_Token,
            QuantType.per_1x128,
            QuantType.per_1x32,
        ):
            w13_batchable.append(getattr(self, "w13_weight_scale", None))
            w2_batchable.append(getattr(self, "w2_weight_scale", None))
        # Counted over routed BASE slots only. EPLB redundant replicas are
        # filled by fill_redundant after loading, and fused shared experts are
        # delivered by the checkpoint separately and never staged
        # (`is_batched_expert_slot`). Counting either would leave the entry
        # short of its threshold forever: it is never flushed or freed, so
        # staging leaks for every layer and the rank never finishes loading.
        n_local_base = self.num_local_base_experts
        if any(param is p for p in w13_batchable if p is not None):
            return n_local_base * 2
        if any(param is p for p in w2_batchable if p is not None):
            return n_local_base
        return None

    def batched_expert_region_numel(
        self,
        param: torch.nn.Parameter,
        local_expert_id: int,
        shard_id: str,
    ) -> int:
        """Return the physical size of one staged ``(expert, shard)`` region."""
        return expert_region(
            param.data,
            local_expert_id,
            shard_id,
            getattr(param, "is_transposed", False),
        ).numel()

    @property
    def num_local_base_experts(self) -> int:
        """Local slots holding a routed base expert, i.e. slots `[0, n)`."""
        return count_local_base_experts(
            expert_map=self.expert_map,
            global_num_experts=self.expert_layout.num_routed_physical,
            num_redundant_experts=self.num_redundant_experts,
            local_num_experts=self.local_num_experts,
            num_fused_shared_experts=self.num_fused_shared_experts,
        )

    def is_batched_expert_slot(self, local_expert_id: int) -> bool:
        """Whether arrivals for this local slot should go through staging.

        Only routed base experts benefit: they arrive one tensor per (expert,
        shard) and coalescing them is the entire point. A fused shared expert
        is three tensors per layer, and in checkpoints that store routed
        experts as one stacked tensor it is the *only* thing that would be
        staged -- leaving a parameter-sized buffer alive for a handful of rows.
        """
        return local_expert_id < self.num_local_base_experts

    def flush_staged(
        self,
        param: torch.nn.Parameter,
        staging: torch.Tensor,
        filled: set[tuple[int, str]],
    ) -> None:
        """Write staged (slot, shard) regions of `staging` back into `param`.

        Only the regions in `filled` are written. Other loader paths may own
        the remaining slots of the same parameter -- a checkpoint that stores
        routed experts as one stacked tensor loads them directly while the
        shared expert comes through the per-expert path -- and a whole-buffer
        copy would overwrite their work with zeros.
        """
        dst = (
            param.data.view(torch.uint8)
            if staging.dtype != param.data.dtype
            else param.data
        )
        n_base = self.num_local_base_experts
        if len(filled) == self.expected_batched_arrivals(param):
            # Every base slot arrived: one large copy, which is the reason the
            # staging path exists at all.
            dst[:n_base].copy_(staging[:n_base])
        else:
            is_transposed = getattr(param, "is_transposed", False)
            for local_expert_id, shard_id in sorted(filled):
                expert_region(dst, local_expert_id, shard_id, is_transposed).copy_(
                    expert_region(staging, local_expert_id, shard_id, is_transposed)
                )
        # EPLB redundant replicas belong to fill_redundant, which runs after
        # process_weights_after_loading has already read every local slot.
        # create_weights hands them out as torch.empty, so zero them here --
        # the whole-buffer flush this replaced used to do it as a side effect.
        for slot in range(n_base, self.expert_layout.routed_physical_per_rank):
            dst[slot].zero_()

    def _load_nvfp4_global_scale(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: str,
        expert_id: int,
    ) -> bool:
        """Load one per-expert NVFP4 per-tensor global scale.

        Returns whether ``param`` is one of the two NVFP4 scalar-metadata
        parameters. These values are replicated across TP ranks and selected
        only by the local expert id (plus w1/w3 half for w13).
        """
        w13_param = getattr(self, "w13_weight_scale_2", None)
        w2_param = getattr(self, "w2_weight_scale_2", None)
        if param is not w13_param and param is not w2_param:
            return False

        value = loaded_weight.reshape(-1)
        if value.numel() != 1:
            raise RuntimeError(
                "Each NVFP4 expert projection must provide exactly one "
                f"global scale, got shape {tuple(loaded_weight.shape)}."
            )
        if param is w13_param:
            if shard_id not in ("w1", "w3"):
                raise ValueError(
                    f"NVFP4 w13 global scale requires w1/w3, got {shard_id!r}."
                )
            shard_index = 0 if shard_id == "w1" else 1
            target = param.data[expert_id, shard_index]
        else:
            if shard_id != "w2":
                raise ValueError(
                    f"NVFP4 w2 global scale requires w2, got {shard_id!r}."
                )
            target = param.data[expert_id]
        target.copy_(value[0])
        return True

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str = "",
        shard_id: str = "",
        expert_id: int = 0,
    ) -> None:
        is_nvfp4 = bool(
            self.layer_quant_config is not None
            and self.layer_quant_config.quant_dtype == NVFP4_DTYPE
        )
        if is_nvfp4 and weight_name == "":
            raise RuntimeError(
                "Merged NVFP4 expert loading is not supported; this loader "
                "expects split per-expert w1/w2/w3 checkpoint tensors."
            )
        if self.layer_quant_config.quant_dtype == dtypes.fp4x2 and weight_name == "":
            self.mxf4_merged_weight_loader(param, loaded_weight, expert_id)
            return

        expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
        if expert_id == -1:
            return
        if is_nvfp4 and self._load_nvfp4_global_scale(
            param, loaded_weight, shard_id, expert_id
        ):
            return

        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO (mgoin): check self.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        if self.quant_method.__class__.__name__ in (
            "CompressedTensorsWNA16MarlinMoEMethod",
            "CompressedTensorsWNA16MoEMethod",
        ):
            loaded_weight = loaded_weight.t().contiguous()

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(
                f"shard_id must be ['w1','w2','w3'] but " f"got {shard_id}."
            )

        # Fetch the dim to shard the parameter/loaded weight based on the shard
        # id; `is_transposed` (GPTQ, compressed-tensors) flips it.
        shard_dim = expert_shard_dim(shard_id, getattr(param, "is_transposed", False))

        full_load = len(loaded_weight.shape) == 3
        if full_load:
            shard_dim += 1

        expert_data = param.data if full_load else param.data[expert_id]
        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            if (
                param.data[expert_id] != 1
                and (param.data[expert_id] - loaded_weight).abs() > 1e-5
            ):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}"
                )

            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return

        # Case g_idx
        if "g_idx" in weight_name:
            self._load_g_idx(
                shard_dim=0,
                shard_id=shard_id,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.tp_rank,
            )
            return

        # Group/per-channel scales and model weights share one copy path with
        # the batched loader (_copy_expert_shard).
        # TODO @dsikka: once hardened, refactor to use vLLM Parameters.
        if self._copy_expert_shard(
            param=param,
            expert_data=expert_data,
            shard_id=shard_id,
            shard_dim=shard_dim,
            loaded_weight=loaded_weight,
            weight_name=weight_name,
        ):
            return

        # Case per-Tensor weight scale
        if "scale" in weight_name or "zero" in weight_name or "offset" in weight_name:
            if self.layer_quant_config.quant_type == QuantType.per_Tensor:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
            return

        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return

    @staticmethod
    def select_experts(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        use_grouped_topk: bool,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: torch.Tensor | None = None,
        num_routing_experts: int = 0,
        num_fused_shared_experts: int = 0,
        fused_shared_experts_scoring_func: str | None = None,
        routed_scaling_factor: float = 1.0,
    ):

        # custom_routing_function takes precedence (e.g. DeepSeek-V4 hash routing
        # in the first 3 layers, where topk_ids are looked up from a per-token
        # hash table instead of computed from gate logits).
        if custom_routing_function is not None:
            topk_weights, topk_ids = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
            )
            return topk_weights, topk_ids

        # DeekSeekv2 uses grouped_top_k
        if use_grouped_topk:
            assert topk_group is not None
            assert num_expert_group is not None
            assert fused_shared_experts_scoring_func is None
            topk_weights, topk_ids = grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias,
                routed_scaling_factor=routed_scaling_factor,
                num_fused_shared_experts=num_fused_shared_experts,
            )
        else:
            if scoring_func == "softmax":
                topk_weights, topk_ids = fused_topk(
                    gating_output=router_logits,
                    topk=top_k,
                    renormalize=renormalize,
                    num_fused_shared_experts=num_fused_shared_experts,
                    num_routing_experts=num_routing_experts,
                    fused_shared_experts_scoring_func=fused_shared_experts_scoring_func,
                )
            elif scoring_func in ("sigmoid", "sqrtsoftplus"):
                tokens_num = router_logits.shape[0]
                fuse_shared = num_fused_shared_experts > 0
                if fuse_shared:
                    import atom.model_ops.topK as _topK_mod

                    assert _topK_mod.aiter_topK_meta_data is not None, (
                        "AITER topK meta data is not initialized. "
                        "Please ensure that init_aiter_topK_meta_data is called "
                        "before this function."
                    )
                    total_topk_weights, total_topk_ids = _topK_mod.aiter_topK_meta_data
                    assert total_topk_weights.shape[0] >= tokens_num
                    n_extra = total_topk_ids.shape[1] - top_k
                    topk_ids, _ = torch.split(
                        total_topk_ids[:tokens_num],
                        [top_k, n_extra],
                        dim=1,
                    )
                    topk_weights, _ = torch.split(
                        total_topk_weights[:tokens_num],
                        [top_k, n_extra],
                        dim=1,
                    )
                else:
                    topk_ids = torch.empty(
                        tokens_num,
                        top_k,
                        dtype=torch.int32,
                        device=router_logits.device,
                    )
                    topk_weights = torch.empty(
                        tokens_num,
                        top_k,
                        dtype=torch.float32,
                        device=router_logits.device,
                    )

                # MiniMax-M3 applies the routed scale outside MoE when shared
                # experts are not fused; DeepSeek-V4 folds it into routing.
                route_scale = (
                    routed_scaling_factor
                    if fuse_shared or scoring_func == "sqrtsoftplus"
                    else 1.0
                )
                topk_gating(
                    topk_weights,
                    topk_ids,
                    router_logits,
                    e_score_correction_bias,
                    renormalize,
                    route_scale,
                    score_func=scoring_func,
                )
                if fuse_shared:
                    # Switch from the routed view back to the full buffer
                    # (routed + shared cols) for the fused MoE kernel.
                    topk_weights = total_topk_weights[:tokens_num]
                    topk_ids = total_topk_ids[:tokens_num]
            else:
                raise ValueError(
                    f"Unsupported scoring function for non-grouped topk: {scoring_func}"
                )

        return topk_weights, topk_ids

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        if self.balance_router_logits is not None:
            # Match the dtype of the logits being replaced. The table is built
            # once at the default dtype, but aiter's `biased_grouped_topk`
            # dispatches on `gating_output.dtype()` and then reinterpret_casts
            # `correction_bias` to that same scalar_t -- so handing it a bf16
            # table for a router whose bias is fp32 makes the kernel read that
            # fp32 buffer at half width, i.e. garbage bias over half the
            # experts. Only fake-EPLB reaches this, so the cast costs nothing
            # on a real run.
            router_logits = self.balance_router_logits[: hidden_states.shape[0]].to(
                router_logits.dtype
            )
        return torch.ops.aiter.moe_forward(
            hidden_states, router_logits, self.layer_name
        )

    def forward_maybe_comm_fused(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_partial: torch.Tensor | None,
        before_stage2: Callable[[], torch.Tensor] | None = None,
        stage2_stream: torch.cuda.Stream | None = None,
    ) -> tuple[torch.Tensor, bool]:
        """Return ``(output, complete)`` after fused or ordinary dispatch.

        A complete output already contains the shared expert and TP reduction.
        """
        backend = self._comm_fused_moe
        if backend is not None and backend.supports(hidden_states.shape[0]):
            return (
                backend.forward(
                    self,
                    hidden_states,
                    router_logits,
                    shared_partial,
                    before_stage2=before_stage2,
                    stage2_stream=stage2_stream,
                ),
                True,
            )
        return self(hidden_states, router_logits), False

    def forward_impl_graph(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ):
        # There are three mode
        # 1. Pure DP mode: only DP is used
        # 2. DP attention + EP mori Moe
        # 3. DP attention + TP All_gahter/reduce Moe
        # `local_tokens` / `sizes` are bound inside the gather block below and
        # read only inside the scatter block, which tests the same unchanged
        # condition -- pre-seeding them with None would make a state the code
        # cannot reach look representable.
        # Use all_gather/reduce_scatter when DP > 1 but not using mori all2all kernels
        use_dp_gather_scatter = (
            self.dp_size > 1
            and not self.moe_parallel_config.use_all2all_kernels
            and get_current_atom_config().enable_dp_attention
        )
        if use_dp_gather_scatter:
            ctx = get_forward_context()
            dp_group = get_dp_group()
            dp_eager_mode = not ctx.context.running_tokens_are_unified

            from atom.utils.tbo.ubatching import tbo_active

            _tbo = tbo_active()
            if _tbo:
                from atom.utils.tbo.ubatching import (
                    tbo_switch_to_compute_sync,
                    tbo_yield_and_switch_from_compute_to_comm,
                )

                tbo_yield_and_switch_from_compute_to_comm()

            (
                hidden_states,
                router_logits,
                local_tokens,
                sizes,
            ) = dp_gather_hidden_and_router(
                hidden_states, router_logits, dp_eager_mode, ctx, dp_group
            )

            # Stand in for the DP ranks a simulated run did not launch. Same row
            # order for both, so a copied token keeps its own routing.
            dp_repeat = self.moe_parallel_config.dp_logical_ratio
            gathered_rows = hidden_states.shape[0]
            if dp_repeat > 1:
                hidden_states = repeat_rows(hidden_states, dp_repeat)
                router_logits = repeat_rows(router_logits, dp_repeat)

            if _tbo:
                tbo_switch_to_compute_sync()

        # Matrix multiply.
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self.expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            fused_shared_experts_scoring_func=self.shared_expert_scoring_func,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            prefix=f"{self.prefix}.fused_moe",
        )

        # Use reduce_scatter when DP > 1 but not using mori all2all kernels
        if use_dp_gather_scatter:
            # `sizes` and the scatter below are sized for the real ranks.
            if dp_repeat > 1:
                final_hidden_states = final_hidden_states[:gathered_rows]
            if _tbo:
                tbo_yield_and_switch_from_compute_to_comm()
            if dp_eager_mode:
                final_hidden_states = reduce_scatterv(
                    final_hidden_states, sizes, dp_group
                )
            else:
                final_hidden_states = reduce_scatter_with_unpadding(
                    final_hidden_states, local_tokens
                )
            if _tbo:
                tbo_switch_to_compute_sync()

        if self.reduce_results and (self.tp_size > 1 or self.ep_size > 1):
            # Default set to False. (May have to add shared expert outputs.)
            final_hidden_states = get_tp_group().all_reduce(
                final_hidden_states, ca_fp8_quant=False
            )

        return final_hidden_states

    def forward_impl(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        assert self.quant_method is not None

        if get_dp_group().world_size > 1:
            return self.forward_impl_graph(hidden_states, router_logits)

        dp_group = get_dp_group()
        if dp_group.world_size > 1:
            cu_tokens_across_dp_cpu = (
                get_forward_context().dp_metadata.cu_tokens_across_dp_cpu
            )

            hidden_states = naive_multicast(hidden_states, cu_tokens_across_dp_cpu)
            router_logits = naive_multicast(router_logits, cu_tokens_across_dp_cpu)

        # Simulated DP with no peers to gather from: the absent ranks' token
        # shards are stood in for locally, same as after the gather in
        # forward_impl_graph. Dropped again below.
        dp_repeat = self.moe_parallel_config.dp_logical_ratio
        local_rows = hidden_states.shape[0]
        if dp_repeat > 1:
            hidden_states = repeat_rows(hidden_states, dp_repeat)
            router_logits = repeat_rows(router_logits, dp_repeat)

        # Matrix multiply.
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self.expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            fused_shared_experts_scoring_func=self.shared_expert_scoring_func,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            prefix=f"{self.prefix}.fused_moe",
        )

        if dp_repeat > 1:
            final_hidden_states = final_hidden_states[:local_rows]

        dp_group = get_dp_group()
        if dp_group.world_size > 1:
            dp_rank = dp_group.rank_in_group
            start = 0 if dp_rank == 0 else cu_tokens_across_dp_cpu[dp_rank - 1]
            end = cu_tokens_across_dp_cpu[dp_rank]

            all_hidden_states = get_dp_group().all_reduce(final_hidden_states)
            final_hidden_states = all_hidden_states[start:end, :]

        if self.reduce_results and (self.tp_size > 1 or self.ep_size > 1):
            # Default set to False. (May have to add shared expert outputs.)
            final_hidden_states = get_tp_group().all_reduce(
                final_hidden_states, ca_fp8_quant=False
            )

        return final_hidden_states

    @classmethod
    def make_expert_params_mapping(
        cls,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
        has_bias: bool = False,
    ) -> list[tuple[str, str, int, str]]:

        return [
            # (param_name, weight_name, expert_id, shard_id)
            (
                (
                    "experts.w13_"
                    if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]
                    else "experts.w2_"
                ),
                f"experts.{expert_id}.{weight_name}.",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

    def extra_repr(self) -> str:

        s = (
            f"global_num_experts={self.global_num_experts}, "
            f"local_num_experts={self.local_num_experts}, "
            f"top_k={self.top_k}, "
            f"intermediate_size_per_partition={self.intermediate_size_per_partition}, "
            f"tp_size={self.tp_size},\n"
            f"ep_size={self.ep_size}, "
            f"reduce_results={self.reduce_results}, "
            f"renormalize={self.renormalize}, "
            f"use_grouped_topk={self.use_grouped_topk}"
        )

        if self.use_grouped_topk:
            s += f", num_expert_group={self.num_expert_group}, topk_group={self.topk_group}"

        s += f", scoring_func='{self.scoring_func}', activation='{self.activation}'"

        return s
