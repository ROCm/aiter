import torch
import triton

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.triton._triton_kernels.moe.small_m_sort import _small_m_sort_kernel
from aiter.ops.triton.utils.moe_config_utils import get_moe_dispatch


def small_m_sort(
    topk_ids, topk_weights, num_experts, model_dim, dtype, block_size, output=None
):
    """Sort packed token/slot IDs into padded expert blocks and clear output.

    Accept contiguous int32 IDs and float32 weights with shape [tokens, 6].
    Return IDs, weights, expert blocks, counts and a BF16 output buffer.
    Return None outside gfx942, 384 experts, 1..64 tokens and block size 16/32.
    Negative and out-of-range expert IDs do not contribute. Output must not
    share storage with either input. Tail capacity beyond counts is undefined.
    """
    if topk_ids.ndim != 2 or model_dim <= 0:
        return None
    tokens, topk = topk_ids.shape
    arch = get_gfx_runtime()
    if not (
        arch == "gfx942"
        and 1 <= tokens <= 64
        and num_experts == 384
        and topk == 6
        and block_size in (16, 32)
        and dtype == torch.bfloat16
        and topk_ids.dtype == torch.int32
        and topk_weights.dtype == torch.float32
        and topk_ids.is_cuda
        and topk_weights.device == topk_ids.device
        and topk_ids.is_contiguous()
        and topk_weights.is_contiguous()
        and topk_weights.shape == topk_ids.shape
    ):
        return None
    if output is not None and (
        output.shape != (tokens, model_dim)
        or output.dtype != dtype
        or output.device != topk_ids.device
        or not output.is_contiguous()
        or any(
            output.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
            for tensor in (topk_ids, topk_weights)
        )
    ):
        raise ValueError("output must be a distinct contiguous BF16 matrix")
    config = get_moe_dispatch("SMALL-M-SORT", arch, "triton")["any"]
    capacity = (
        triton.cdiv(tokens * topk + num_experts * block_size - topk, block_size)
        * block_size
    )
    sorted_ids = torch.empty(capacity, device=topk_ids.device, dtype=torch.int32)
    sorted_weights = torch.empty(capacity, device=topk_ids.device, dtype=torch.float32)
    expert_ids = torch.empty(
        capacity // block_size, device=topk_ids.device, dtype=torch.int32
    )
    valid = torch.empty(2, device=topk_ids.device, dtype=torch.int32)
    if output is None:
        output = torch.empty((tokens, model_dim), device=topk_ids.device, dtype=dtype)
    _small_m_sort_kernel[(num_experts,)](
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        expert_ids,
        valid,
        output,
        tokens,
        topk,
        num_experts,
        block_size,
        triton.next_power_of_2(max(tokens * topk, block_size)),
        triton.next_power_of_2(num_experts),
        output.numel(),
        triton.next_power_of_2(triton.cdiv(output.numel(), num_experts)),
        num_warps=config["num_warps"],
    )
    return sorted_ids, sorted_weights, expert_ids, valid, output
