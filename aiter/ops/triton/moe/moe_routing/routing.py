import torch
import triton
from dataclasses import dataclass, field
from aiter.ops.triton._triton_kernels.moe.moe_routing.routing import (
    _combined_routing,
    _combined_routing_fused,
)
from aiter.ops.triton.utils._triton.arch_info import is_tdm_avail


@dataclass
class ExptData:
    # hist[i] is the number of tokens routed to expert i
    hist: torch.Tensor
    # token_offs_raw[i] is the offset of the first token routed
    # to expert i in an expert-sorted array
    token_offs_raw: torch.Tensor
    # token_offs_pad[i] is the offset of the first token routed
    # to expert i in an expert-sorted array, assuming histogram
    # rounded to the next multiple of `block_m`
    token_offs_pad: torch.Tensor
    # block_id_map contain one value for each `pid`` launched by
    # the matrix multiplication kernel launched with block_m:
    # - the value is -1 if the `pid` has no work to do
    # - otherwise, the value is two int16 (packed as an int32) that
    #   correspond respectively to (1) the expert assigned to
    #   the tokens processed by this pid; (2) the block assigned to the
    #   tokens processed by this pid (think `pid_m` in a regular matmul)
    # see `test_routing.py` for a reference implementation and more details
    block_pid_map: torch.Tensor

    def __post_init__(self):
        if self.hist is not None:
            assert self.hist.dtype == torch.int32
        if self.token_offs_raw is not None:
            assert self.token_offs_raw.dtype == torch.int32
        if self.token_offs_pad is not None:
            assert self.token_offs_pad.dtype == torch.int32
        if self.block_pid_map is not None:
            assert self.block_pid_map.dtype == torch.int32


@dataclass
class RoutingData:
    block_m: int = field()
    gate_scal: torch.Tensor = field()
    expt_hist: torch.Tensor = field()
    n_expts_tot: int = field()
    n_expts_act: int = field()
    expt_data: ExptData = None

    def n_blocks(self, n_rows, block_m):
        if n_rows <= self.n_expts_tot:
            return n_rows
        else:
            return (
                triton.cdiv(max(n_rows - self.n_expts_tot + 1, 0), block_m)
                + self.n_expts_tot
                - 1
            )


# --------------------------
# sort tokens by expert
# --------------------------


def sort_tokens(expt_scal, expt_indx, n_expts_tot, bitmatrix, block_m, HIST_BLOCK_M):
    cdiv = triton.cdiv

    device = expt_scal.device
    dtype = expt_scal.dtype
    n_tokens, n_expts_act = expt_scal.shape
    n_gates = n_tokens * n_expts_act
    n_expts_act_pad = triton.next_power_of_2(n_expts_act)

    hist, partial_hist = bitmatrix.sum(partials_block_size=HIST_BLOCK_M)
    hist = hist[:n_expts_tot]
    assert hist.dtype == torch.int32
    # scratchpad
    combined_indx = torch.empty(n_gates * 2, dtype=torch.int32, device=device)
    # output
    topk_indx = combined_indx[:n_gates]
    gate_indx = combined_indx[n_gates:]
    gate_scal = torch.empty(n_gates, dtype=dtype, device=device)

    token_offs_raw, token_offs_pad, block_pid_map, blocks1a, BLOCK_A, block_m_log2 = (
        _compute_expt_data_internal(n_expts_tot, n_gates, block_m, device)
    )

    blocks1b = cdiv(n_tokens, HIST_BLOCK_M)

    indx_offs = partial_hist

    _combined_routing[(blocks1a + blocks1b,)](
        topk_indx,
        gate_indx,
        gate_scal,  # outputs
        expt_scal,
        expt_indx,
        indx_offs,
        indx_offs.stride(0),
        indx_offs.stride(1),  # inputs
        n_gates,  # input shape
        HIST_BLOCK_M,
        n_tokens % HIST_BLOCK_M == 0,
        n_expts_act,  # constants
        n_expts_act_pad,
        hist,
        n_expts_tot,
        token_offs_raw,
        token_offs_pad,  #
        blocks1a,
        block_pid_map,
        block_pid_map.shape[0],  #
        block_m_log2,
        BLOCK_A=BLOCK_A,
        EQUAL_A=(hist.shape[0] == BLOCK_A),  # optimization parameters
        USE_TDM=is_tdm_avail(),
        num_warps=1,
    )

    return (
        hist,
        topk_indx,
        gate_indx,
        gate_scal,
        token_offs_raw,
        token_offs_pad,
        block_pid_map,
    )


def sort_tokens_fused(
    expt_scal, expt_indx, n_expts_tot, bitmatrix, block_m, HIST_BLOCK_M
):
    cdiv = triton.cdiv

    device = expt_scal.device
    dtype = expt_scal.dtype
    n_tokens, n_expts_act = expt_scal.shape
    n_gates = n_tokens * n_expts_act
    n_expts_act_pad = triton.next_power_of_2(n_expts_act)

    hist = bitmatrix.scratchpad
    hist = hist[:n_expts_tot]
    assert hist.dtype == torch.int32
    num_blocks_bitmatrix = cdiv(bitmatrix.shape[1], 32)
    # scratchpad
    combined_indx = torch.empty(n_gates * 2, dtype=torch.int32, device=device)
    # output
    topk_indx = combined_indx[:n_gates]
    gate_indx = combined_indx[n_gates:]
    gate_scal = torch.empty(n_gates, dtype=dtype, device=device)

    token_offs_raw, token_offs_pad, block_pid_map, blocks1a, BLOCK_A, block_m_log2 = (
        _compute_expt_data_internal(n_expts_tot, n_gates, block_m, device)
    )

    blocks1b = cdiv(n_tokens, HIST_BLOCK_M)

    _combined_routing_fused[(blocks1a + blocks1b,)](
        topk_indx,
        gate_indx,
        gate_scal,  # outputs
        expt_scal,
        expt_indx,
        bitmatrix.data,
        bitmatrix.shape[0],
        bitmatrix.data.stride(0),
        bitmatrix.data.stride(1),
        num_blocks_bitmatrix,
        n_gates,  # input shape
        HIST_BLOCK_M,
        n_tokens % HIST_BLOCK_M == 0,
        n_expts_act,  # constants
        n_expts_act_pad,
        n_expts_tot,
        hist,
        token_offs_raw,
        token_offs_pad,  #
        blocks1a,
        block_pid_map,
        block_pid_map.shape[0],  #
        block_m_log2,
        BLOCK_A=BLOCK_A,
        EQUAL_A=(hist.shape[0] == BLOCK_A),  # optimization parameters
        USE_TDM=is_tdm_avail(),
        num_warps=1,
    )

    return (
        hist,
        topk_indx,
        gate_indx,
        gate_scal,
        token_offs_raw,
        token_offs_pad,
        block_pid_map,
    )


# --------------------------
# expt_data
# --------------------------


def log2_power_of_two(x):
    assert x > 0 and (x & (x - 1)) == 0, "x must be a power of two"
    return x.bit_length() - 1


def _compute_expt_data_internal(n_expts_tot, n_gates, block_m, device):
    BLOCK = 128
    cdiv = triton.cdiv
    block_m_log2 = log2_power_of_two(block_m)
    if n_gates <= n_expts_tot:
        max_n_tiles = n_gates
    else:
        max_n_tiles = n_expts_tot - 1 - ((n_expts_tot - n_gates - 1) // block_m)

    # allocate memory
    def pad(x):
        return cdiv(x, BLOCK) * BLOCK

    dtype = torch.int32

    token_offs_combined = torch.empty(
        (2, pad(n_expts_tot + 1)), dtype=dtype, device=device
    )

    token_offs_raw = token_offs_combined[0][: n_expts_tot + 1]
    token_offs_pad = token_offs_combined[1][: n_expts_tot + 1]

    # block_pid_map = torch.empty((pad(max_n_tiles),), dtype=dtype, device=device)
    block_pid_map = torch.empty((max_n_tiles,), dtype=dtype, device=device)
    # block_pid_map = block_pid_map[:max_n_tiles]

    blocks1 = n_expts_tot
    return token_offs_raw, token_offs_pad, block_pid_map, blocks1, BLOCK, block_m_log2


# --------------------------
# routing
# --------------------------


def routing(logits, n_expts_act, sm_first=False):
    HIST_BLOCK_M = 32

    from .topk import topk

    if sm_first:
        logits = torch.softmax(logits, dim=-1)
    expt_scal, expt_indx, bitmatrix = topk(
        logits,
        n_expts_act,
        apply_softmax=not sm_first,
        HIST_BLOCK_M=HIST_BLOCK_M,
    )

    num_tokens, n_expts_tot = logits.shape
    m = num_tokens * n_expts_act
    tokens_per_expt = max(1, m // n_expts_tot)
    block_m = max(16, min(triton.next_power_of_2(tokens_per_expt), 128))
    if num_tokens <= 16:
        HIST_BLOCK_M = triton.next_power_of_2(num_tokens)
        (
            hist,
            topk_indx,
            gate_indx,
            gate_scal,
            token_offs_raw,
            token_offs_pad,
            block_pid_map,
        ) = sort_tokens_fused(
            expt_scal, expt_indx, n_expts_tot, bitmatrix, block_m, HIST_BLOCK_M
        )
    else:
        (
            hist,
            topk_indx,
            gate_indx,
            gate_scal,
            token_offs_raw,
            token_offs_pad,
            block_pid_map,
        ) = sort_tokens(
            expt_scal, expt_indx, n_expts_tot, bitmatrix, block_m, HIST_BLOCK_M
        )
    expt_data = ExptData(hist, token_offs_raw, token_offs_pad, block_pid_map)

    # pack the matmul data structure
    gather_indx = topk_indx
    scatter_indx = gate_indx
    return (
        RoutingData(block_m, gate_scal, hist, n_expts_tot, n_expts_act, expt_data),
        gather_indx,
        scatter_indx,
    )


def routing_a8w4(
    logits: torch.Tensor,
    n_expts_act: int,
    block_m: int,
    *,
    score_mode: str = "sqrtsoftplus",
    bias: torch.Tensor | None = None,
    renorm: bool = True,
    routed_scaling_factor: float = 1.0,
):
    """All-Triton routing for the a8w4 path: fused V4 routing math + sort.

    One-shot pipeline:
      1. aiter `_topk` (extended): pre-transform (sqrtsoftplus) + bias + topk
         + bitmatrix + renorm + scale — single Triton kernel.
      2. aiter `sort_tokens` (or `sort_tokens_fused` for tiny M): sort tokens by
         expert and produce ExptData specialized for the given ``block_m``.

    Returns (RoutingData, gather_indx, scatter_indx) where gather_indx and
    scatter_indx are raw int32 tensors (no GatherIndx/ScatterIndx wrappers) —
    consumed directly by ``moe_gemm_a8w4``.

    No multi-block_m dict, no triton_kernels wrapper, no Python bridge step.
    """
    from .topk import topk

    n_tokens, n_expts_tot = logits.shape

    # Step 1: extended topk does sqrtsoftplus + bias + topk + bitmatrix + renorm + scale.
    expt_scal, expt_indx, bitmatrix = topk(
        logits,
        n_expts_act,
        apply_softmax=False,
        score_mode=score_mode,
        bias=bias,
        renorm=renorm,
        routed_scaling_factor=routed_scaling_factor,
        HIST_BLOCK_M=32,
    )

    # Step 2: sort tokens by expert and build ExptData for the chosen block_m.
    if n_tokens <= 16:
        HIST_BLOCK_M = triton.next_power_of_2(max(n_tokens, 1))
        sort_fn = sort_tokens_fused
    else:
        HIST_BLOCK_M = 32
        sort_fn = sort_tokens
    (
        hist,
        topk_indx,
        gate_indx,
        gate_scal,
        token_offs_raw,
        token_offs_pad,
        block_pid_map,
    ) = sort_fn(expt_scal, expt_indx, n_expts_tot, bitmatrix, block_m, HIST_BLOCK_M)
    expt_data = ExptData(hist, token_offs_raw, token_offs_pad, block_pid_map)
    routing_data = RoutingData(
        block_m=block_m,
        gate_scal=gate_scal,
        expt_hist=hist,
        n_expts_tot=n_expts_tot,
        n_expts_act=n_expts_act,
        expt_data=expt_data,
    )
    return routing_data, topk_indx, gate_indx


def routing_a8w4_from_hash(
    router_logits: torch.Tensor,
    tid2eid: torch.Tensor,
    input_ids: torch.Tensor,
    n_expts_act: int,
    block_m: int,
    *,
    score_mode: str = "sqrtsoftplus",
    renorm: bool = True,
    routed_scaling_factor: float = 1.0,
):
    """All-Triton routing for the a8w4 path on DeepSeek-V4 hash layers.

    Single fused kernel ``hash_routing`` does tid2eid lookup + score transform
    + gather + renorm + scale + bitmatrix in one launch, then
    ``sort_tokens_fused`` (same as :func:`routing_a8w4`) produces ExptData.

    Replaces the Python ``_hash_topk`` + multi-kernel ``fused_routing_from_topk``
    counting-sort + ``compute_expt_data`` (with memset) chain entirely.
    """
    from .topk import hash_routing

    n_tokens, n_expts_tot = router_logits.shape

    expt_scal, expt_indx, bitmatrix = hash_routing(
        router_logits,
        tid2eid,
        input_ids,
        n_expts_act=n_expts_act,
        HIST_BLOCK_M=32,
        score_mode=score_mode,
        renorm=renorm,
        routed_scaling_factor=routed_scaling_factor,
    )

    if n_tokens <= 16:
        HIST_BLOCK_M = triton.next_power_of_2(max(n_tokens, 1))
        sort_fn = sort_tokens_fused
    else:
        HIST_BLOCK_M = 32
        sort_fn = sort_tokens
    (
        hist,
        topk_indx,
        gate_indx,
        gate_scal,
        token_offs_raw,
        token_offs_pad,
        block_pid_map,
    ) = sort_fn(expt_scal, expt_indx, n_expts_tot, bitmatrix, block_m, HIST_BLOCK_M)
    expt_data = ExptData(hist, token_offs_raw, token_offs_pad, block_pid_map)
    routing_data = RoutingData(
        block_m=block_m,
        gate_scal=gate_scal,
        expt_hist=hist,
        n_expts_tot=n_expts_tot,
        n_expts_act=n_expts_act,
        expt_data=expt_data,
    )
    return routing_data, topk_indx, gate_indx


def routing_a8w4_from_topk(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    n_expts_tot: int,
    block_m: int,
):
    """Routing for the a8w4 path when topk has been pre-computed externally
    (e.g. DeepSeek-V4 hash layers with tid2eid lookup).

    Mirrors ``routing_a8w4`` but skips the score+topk math step. Pipeline:
      1. aiter ``fused_routing_from_topk``: 3-kernel counting-sort over the
         supplied ``(topk_weights, topk_ids)``. Allocates only via
         ``torch.empty`` — no histogram memset.
      2. aiter ``_expt_data_only_kernel``: standalone stage1+stage2 launch
         that materialises ExptData (token_offs_raw, token_offs_pad,
         block_pid_map) from the histogram for the chosen ``block_m``.

    Returns ``(RoutingData, gather_indx, scatter_indx)`` where ``gather_indx``
    and ``scatter_indx`` are raw int32 tensors — same contract as
    ``routing_a8w4`` — so ``_a8w4_fused_experts`` consumes them unchanged.
    """
    from aiter.ops.triton.fusions.fused_routing_from_topk import (
        fused_routing_from_topk,
    )
    from aiter.ops.triton._triton_kernels.moe.moe_routing.expt_data import (
        _expt_data_only_kernel,
    )

    n_tokens, n_expts_act = topk_weights.shape
    n_gates = n_tokens * n_expts_act

    hist, topk_indx, gate_indx, gate_scal = fused_routing_from_topk(
        topk_weights, topk_ids, n_expts_tot
    )

    token_offs_raw, token_offs_pad, block_pid_map, blocks1a, BLOCK_A, block_m_log2 = (
        _compute_expt_data_internal(n_expts_tot, n_gates, block_m, topk_weights.device)
    )

    _expt_data_only_kernel[(blocks1a,)](
        hist,
        n_expts_tot,
        token_offs_raw,
        token_offs_pad,
        block_pid_map,
        block_pid_map.shape[0],
        n_gates,
        block_m_log2,
        BLOCK_A,
        (hist.shape[0] == BLOCK_A),
        num_warps=1,
    )

    expt_data = ExptData(hist, token_offs_raw, token_offs_pad, block_pid_map)
    routing_data = RoutingData(
        block_m=block_m,
        gate_scal=gate_scal,
        expt_hist=hist,
        n_expts_tot=n_expts_tot,
        n_expts_act=n_expts_act,
        expt_data=expt_data,
    )
    return routing_data, topk_indx, gate_indx


# --------------------------
# torch reference
# --------------------------


def compute_expt_data_torch(hist, n_expts_tot, n_gates, block_m):
    # offset for each experts
    device = hist.device
    token_offs_raw = torch.cumsum(hist, dim=0)
    token_offs_raw = torch.cat((torch.zeros(1, device=device), token_offs_raw))
    token_offs_raw = token_offs_raw.int()
    # maximum number of tiles for all values of `block_m` considered
    if n_gates <= n_expts_tot:
        max_n_tiles = n_gates
    else:
        # ceil_div(n_gates - n_experts + 1, d_tile) + n_experts - 1
        # ceil_div(x, y): -(-x // y)
        max_n_tiles = n_expts_tot - 1 - ((n_expts_tot - n_gates - 1) // block_m)
    # fill up tile offset/infos for each block
    n_tiles = (hist + block_m - 1) // block_m  # matmul blocks needed
    token_offs_pad = torch.cumsum(n_tiles, dim=0)
    token_offs_pad = torch.cat((torch.zeros(1, device=device), token_offs_pad))
    token_offs_pad = token_offs_pad.int()
    # compute data required to drive ragged batch matmul
    block_pid_map = -torch.ones(max_n_tiles, device=device)
    for e in range(n_expts_tot):
        offset = token_offs_pad[e]
        for b in range(n_tiles[e]):
            block_pid_map[offset + b] = (b << 16) + e
    block_pid_map = block_pid_map.int()
    return ExptData(hist, token_offs_raw, token_offs_pad, block_pid_map)


def routing_torch(logits, n_expts_act, sm_first=False):
    n_gates_pad = logits.shape[0] * n_expts_act

    def topk(vals, k):
        # topk of experts
        tk_indx = torch.argsort(vals, dim=1, stable=True)[:, -k:]
        tk_indx = tk_indx.long()
        tk_val = torch.take_along_dim(vals, tk_indx, dim=1)
        tk_indx = tk_indx.int()
        return tk_val, tk_indx

    _, n_expts_tot = logits.shape
    if sm_first:
        logits = torch.softmax(logits, dim=-1)
    expt_scal, expt_indx = topk(logits, n_expts_act)
    if not sm_first:
        expt_scal = torch.softmax(expt_scal, dim=-1)
    # sort each token's selections by expert
    expt_indx, sort_indices = torch.sort(expt_indx, dim=1)
    expt_scal = torch.gather(expt_scal, 1, sort_indices)
    # flatten topk data
    expt_scal = expt_scal.reshape(-1)
    expt_indx = expt_indx.reshape(-1).to(torch.int32)
    # sort by expert_id so experts are contiguous for the matmul
    topk_indx = torch.argsort(expt_indx, stable=True)
    gate_indx = torch.argsort(topk_indx, stable=True)
    gate_scal = expt_scal[topk_indx]
    hist = torch.histc(
        expt_indx, bins=n_expts_tot, max=n_expts_tot - 1
    ).int()  # histogram of tokens over experts
    # pack the matmul data structure
    gather_indx = topk_indx.int()
    scatter_indx = gate_indx.int()
    # compute expt_data
    tokens_per_expt = max(1, n_gates_pad // n_expts_tot)
    block_m = max(16, min(triton.next_power_of_2(tokens_per_expt), 128))
    expt_data = compute_expt_data_torch(hist, n_expts_tot, n_gates_pad, block_m)
    return (
        RoutingData(block_m, gate_scal, hist, n_expts_tot, n_expts_act, expt_data),
        gather_indx,
        scatter_indx,
    )
