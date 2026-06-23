import triton
import triton.language as tl


@triton.jit
def _local_argmax_pack_kernel(
    logits_ptr,
    packed_ptr,
    vocab_start_idx,
    N: tl.constexpr,
    M: tl.constexpr,
    stride_logits_n: tl.constexpr,
    stride_logits_m: tl.constexpr,
    stride_packed_n: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_M)
    mask = offs < M
    vals = tl.load(
        logits_ptr + row * stride_logits_n + offs * stride_logits_m,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)

    max_val = tl.max(vals, axis=0)
    idxs = offs.to(tl.int64)
    masked_idxs = tl.where((vals == max_val) & mask, idxs, idxs + BLOCK_M)
    local_idx = tl.min(masked_idxs, axis=0)
    global_idx = local_idx + vocab_start_idx

    tl.store(packed_ptr + row * stride_packed_n, max_val)
    tl.store(packed_ptr + row * stride_packed_n + 1, global_idx.to(tl.float32))
