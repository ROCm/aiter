import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@triton.jit(
    repr=make_kernel_repr(
        "_small_m_sort_kernel", ["TOKENS", "BM", "ROUTES", "ZERO_BLOCK"]
    )
)
def _small_m_sort_kernel(
    ids,
    weights,
    sorted_ids,
    sorted_weights,
    expert_ids,
    valid,
    output,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    EXPERTS: tl.constexpr,
    BM: tl.constexpr,
    ROUTES: tl.constexpr,
    BINS: tl.constexpr,
    OUTPUT_SIZE: tl.constexpr,
    ZERO_BLOCK: tl.constexpr,
):
    expert = tl.program_id(0)
    offsets = tl.arange(0, ROUTES)
    route_valid = offsets < TOKENS * TOPK
    route_ids = tl.load(ids + offsets, route_valid, -1)
    route_valid = route_valid & (route_ids >= 0) & (route_ids < EXPERTS)
    counts = tl.histogram(route_ids, BINS, mask=route_valid)
    padded = tl.cdiv(counts, BM) * BM
    expert_offsets = tl.arange(0, BINS)
    start = tl.sum(tl.where(expert_offsets < expert, padded, 0), 0)
    count = tl.sum(tl.where(expert_offsets == expert, counts, 0), 0)
    extent = tl.cdiv(count, BM) * BM
    selected = route_valid & (route_ids == expert)
    positions = tl.cumsum(selected.to(tl.int32), 0) - 1
    encoded = ((offsets % TOPK) << 24) | (offsets // TOPK)
    route_weights = tl.load(weights + offsets, route_valid, 0)
    tl.store(sorted_ids + start + positions, encoded, selected)
    tl.store(sorted_weights + start + positions, route_weights, selected)
    padding = (offsets >= count) & (offsets < extent)
    tl.store(sorted_ids + start + offsets, (TOPK << 24) | TOKENS, padding)
    tl.store(sorted_weights + start + offsets, 0, padding)
    tl.store(expert_ids + start // BM + offsets, expert, offsets < extent // BM)
    zero_offsets = expert * ZERO_BLOCK + tl.arange(0, ZERO_BLOCK)
    tl.store(output + zero_offsets, 0, zero_offsets < OUTPUT_SIZE)
    if expert == 0:
        tl.store(valid, tl.sum(padded, 0))
        tl.store(valid + 1, TOKENS)
