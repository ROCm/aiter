# SPDX-License-Identifier: MIT
# Fused MoE routing preamble (topk + sort + MXFP4 quant) op bindings.
import functools

import torch

from ..jit.core import compile_ops

MD_NAME = "module_fused_moe_router"


def _fused_moe_router_impl_fake(*args, **kwargs) -> None:
    """No-op fake: every output is an in-place out-param, and the op returns
    None. Without it, tracing falls through to the real JIT kernel."""
    return


# The C++ entry takes torch::Tensor directly (see fused_moe_router_entry.cu),
# so no develop=True aiter_tensor_t conversion.
@compile_ops("module_fused_moe_router", gen_fake=_fused_moe_router_impl_fake)
def fused_moe_router_impl(
    gating: torch.Tensor,
    bias: torch.Tensor,
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    out_fp4: torch.Tensor,
    out_scale: torch.Tensor,
    num_experts: int,
    topk: int,
    unit_size: int,
    group_size: int,
    need_renorm: bool,
    routed_scaling_factor: float,
    workspace: torch.Tensor,
    expert_mask: torch.Tensor | None = None,
    moe_buf: torch.Tensor | None = None,
    num_fused_shared_experts: int = 0,
    shared_expert_weight: float = 1.0,
    ep_rank: int = 0,
    ep_size: int = 1,
) -> None:
    """Single-barrier fused routing preamble: biased grouped topk + moe_sort +
    MXFP4 activation quant, in one (or, above M~104, two) launches.

    Replaces the 4-kernel sequence ``biased_grouped_topk`` -> ``moe_sorting``
    -> ``fused_dynamic_mxfp4_quant_moe_sort``. All outputs are written in
    place and must be preallocated with exactly the shapes the stock path
    produces.

    Constraints checked by the C++ entry: ``hidden.shape[1] == 4096``,
    ``num_experts <= 512``, ``unit_size`` a power of two, bf16 inputs,
    ``num_expert_group == topk_group == 1``.

    Args:
        gating: ``[M, num_experts]`` bf16 router logits.
        bias: ``[num_experts]`` bf16 or fp32 e_score_correction_bias. Read as
            given; unlike the stock wrapper this path does not coerce it.
        hidden: ``[M, cols]`` bf16 activations to quantize.
        topk_ids: ``[M, topk + num_fused_shared_experts]`` int32 output.
        topk_weights: same shape, fp32 output.
        sorted_ids: ``[max_num_tokens_padded]`` int32 output.
        sorted_weights: ``[max_num_tokens_padded]`` fp32 output.
        sorted_expert_ids: ``[max_num_m_blocks]`` int32 output.
        num_valid_ids: ``[2]`` int32 output.
        out_fp4: ``[M, cols // 2]`` fp4x2 quantized activation output.
        out_scale: ``[pad32(max_num_tokens_padded), pad8(cols // group_size)]``
            e8m0 output, swizzled to the GEMM tile layout.
        num_experts: global expert count (mask length under EP).
        topk: experts per token.
        unit_size: sort block size (``block_size_M``).
        group_size: MX scale group, 32.
        need_renorm: renormalize the topk weights.
        routed_scaling_factor: post-renorm weight scale.
        workspace: uint8 scratch of at least
            ``fused_moe_router_workspace_size(M)`` bytes, first 4 bytes zeroed
            once at allocation. Use :func:`get_fused_moe_router_workspace`.
        expert_mask: ``[>= num_experts + num_fused_shared_experts + 1]`` int32
            under shared fusion, ``[>= num_experts]`` otherwise; nonzero iff
            this rank owns the expert. ``None`` disables EP; when given,
            ``sorted_expert_ids`` carries local ids.
        moe_buf: optional GEMM output buffer to zero-fill while routing runs.
        num_fused_shared_experts: shared experts appended after the routed ids,
            taking global ids ``num_experts .. num_experts + n - 1``. 0 or the
            model's shared-expert count; at most 1.
        shared_expert_weight: weight every token gives each shared expert.
        ep_rank: this rank's index, used only under shared fusion.
        ep_size: EP world size. Shared weights are replicated on every rank
            while every rank sees every token, so token ownership round-robins
            over ``ep_size`` and non-owners park the shared row on an
            always-masked sentinel. 1 (the default) makes every rank an owner,
            which is the non-EP case.
    """


@compile_ops("module_fused_moe_router")
def fused_moe_router_workspace_size(max_tokens: int) -> int:
    """Workspace bytes needed to serve every ``M`` up to ``max_tokens``."""


# Unbounded: an evicted entry frees a workspace that a captured CUDA graph may
# still hold a baked pointer to
@functools.cache
def _get_fused_moe_router_workspace_keyed(
    device: torch.device, stream_id: int, nbytes: int
) -> torch.Tensor:
    # zeros, not empty: the barrier semaphore in the first 4 bytes is
    # self-resetting, so this is the only time it is ever cleared.
    return torch.zeros(nbytes, dtype=torch.uint8, device=device)


def get_fused_moe_router_workspace(
    device: torch.device, max_tokens: int
) -> torch.Tensor:
    """Return a per-(device, stream) workspace sized for ``max_tokens``.

    Keyed by stream because nothing serializes launches on different streams,
    and both the semaphore and the token-indexed scale scratch are written by
    one phase and read by the next.

    Sizes are bucketed to the next power of two, so a caller sweeping ``M``
    reuses one buffer instead of allocating per shape. The C++ side reads only
    the first ``fused_moe_router_workspace_size(M)`` bytes, so a larger buffer
    is fine.

    Args:
        device: device the routing inputs live on.
        max_tokens: highest ``M`` this caller will pass.

    Returns:
        uint8 workspace, zero-initialized on first use for this key.
    """
    nbytes = fused_moe_router_workspace_size(max_tokens)
    alloc = 1 if nbytes <= 1 else 1 << (int(nbytes) - 1).bit_length()
    stream = torch.cuda.current_stream(device)
    return _get_fused_moe_router_workspace_keyed(device, stream.cuda_stream, alloc)
