import os
from dataclasses import dataclass, field, replace

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.moe_routing.routing import (
    _combined_routing,
    _combined_routing_fused,
    _ep_gate_hist_scan_kernel,
    _ep_sort_fused_kernel,
    _ep_sorted_scatter_kernel,
    _ep_topk_kernel,
)
from aiter.ops.triton.moe.moe_routing.topk import grouped_topk
from aiter.ops.triton.utils._triton.arch_info import is_tdm_avail

# HERD (Hot-Expert Routing for Decode): when AITER_TRITON_USE_HERD is set, the flat-topk
# path uses fused min-unique routing (top-(k+1) -> drop least-batch-popular -> keep-k)
# for batch sizes in [MIN_M (default 16), MAX_M (default 128)]. Unset, or outside that
# window (tiny M where the expert union can't shrink, or prefill), falls through to stock.
_USE_HERD = os.environ.get("AITER_TRITON_USE_HERD", "") not in (
    "",
    "0",
    "false",
    "False",
)
_HERD_MIN_M = int(os.environ.get("AITER_TRITON_HERD_MIN_M", "16"))
_HERD_MAX_M = int(os.environ.get("AITER_TRITON_HERD_MAX_M", "128"))


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
    # Per-token selected expert ids [n_tokens, n_expts_act] (int16, as produced by
    # topk), in the original topk order (matches scatter_indx grouping). Used by
    # callers to build an expert-parallel gate-validity mask, e.g.
    # `expert_map[topk_ids.long()] >= 0`.
    topk_ids: torch.Tensor = None
    # ExptData of the same sorted rows for other tile heights, by block_m: lets
    # two GEMMs over one routing use different block_m (see with_block_m).
    extra_expt_data: dict = None

    def with_block_m(self, block_m):
        """This routing tiled by `block_m` rows (one of extra_expt_data's keys).

        Only the tile map changes: the sorted rows, and so every GEMM's input
        and output rows, are the same for every block_m.
        """
        if block_m == self.block_m:
            return self
        return replace(
            self,
            block_m=block_m,
            expt_data=self.extra_expt_data[block_m],
            extra_expt_data=None,
        )

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
    if n_gates <= 65536 - 1:  # save one token id for OOB checks
        combined_indx = torch.empty(n_gates * 2, dtype=torch.uint16, device=device)
    else:
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
        token_offs_pad,
        blocks1a,
        block_pid_map,
        block_pid_map.shape[0],
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
    if n_gates <= 65536:
        combined_indx = torch.empty(n_gates * 2, dtype=torch.uint16, device=device)
    else:
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
        token_offs_pad,
        blocks1a,
        block_pid_map,
        block_pid_map.shape[0],
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
    BLOCK = triton.next_power_of_2(n_expts_tot)
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


def routing(
    logits: torch.Tensor,
    n_expts_act: int,
    *,
    score_mode: str | None = None,
    sm_first: bool = False,
    bias: torch.Tensor | None = None,
    renorm: bool = True,
    routed_scaling_factor: float = 1.0,
    use_grouped_topk: bool = False,
    num_expert_group: int | None = None,
    topk_group: int | None = None,
    expert_group: torch.Tensor | None = None,
):
    """Routing entry point. ``score_mode`` selects the path:

    * ``score_mode is None`` (default) -> the plain flat top-k routing process:
      flat top-k with softmax. ``sm_first`` controls whether softmax is applied
      to the logits before the top-k (``True``) or inside the top-k
      (``False``). The fused-V4-only arguments (``bias``, ``use_grouped_topk``
      ...) are ignored on this path.
    * ``score_mode is not None`` -> the fused V4 (DeepSeek) routing process:
      fused score transform + (optionally grouped) top-k. ``sm_first`` is
      ignored on this path.

    ``block_m`` is not supplied by the caller: it is derived internally from the
    raw ``logits`` shape and the originally requested ``n_expts_act``.

    Returns ``(RoutingData, gather_indx, scatter_indx)``.
    """
    num_tokens, n_expts_tot = logits.shape

    # block_m heuristic from the raw logits shape and the originally requested
    # n_expts_act.
    m = num_tokens * n_expts_act
    tokens_per_expt = max(1, m // n_expts_tot)
    block_m = max(16, min(triton.next_power_of_2(tokens_per_expt), 128))

    # ------------------------------------------------------------------
    # flat top-k path: plain top-k + softmax (score_mode is None)
    # ------------------------------------------------------------------
    if score_mode is None:
        # HERD: env-gated fused min-unique routing (decode-sized batches only;
        # prefill / large M falls through to the stock top-k path below).
        if _USE_HERD and _HERD_MIN_M <= num_tokens <= _HERD_MAX_M:
            from aiter.ops.triton.moe.moe_routing.minunique import routing_minunique

            return routing_minunique(logits, n_expts_act, sm_first=sm_first)
        from aiter.ops.triton.moe.moe_routing.topk import topk

        HIST_BLOCK_M = 32
        if sm_first:
            logits = torch.softmax(logits, dim=-1)
        expt_scal, expt_indx, bitmatrix = topk(
            logits,
            n_expts_act,
            apply_softmax=not sm_first,
            HIST_BLOCK_M=HIST_BLOCK_M,
        )
        if num_tokens <= 16:
            HIST_BLOCK_M = triton.next_power_of_2(num_tokens)
            sort_fn = sort_tokens_fused
        else:
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
        return (
            RoutingData(
                block_m,
                gate_scal,
                hist,
                n_expts_tot,
                n_expts_act,
                expt_data,
                topk_ids=expt_indx,
            ),
            topk_indx,
            gate_indx,
        )

    # ------------------------------------------------------------------
    # fused path: fused routing math + sort (score_mode given)
    # ------------------------------------------------------------------

    # HERD: env-gated fused min-unique routing for decode-sized batches.
    # Only for non-grouped-topk (DSv4 flat topk with sqrtsoftplus).
    if _USE_HERD and _HERD_MIN_M <= num_tokens <= _HERD_MAX_M and not use_grouped_topk:
        from aiter.ops.triton.moe.moe_routing.minunique import routing_minunique_fused

        return routing_minunique_fused(
            logits,
            n_expts_act,
            score_mode=score_mode,
            bias=bias,
            renorm=renorm,
            routed_scaling_factor=routed_scaling_factor,
        )

    if use_grouped_topk and num_expert_group != 1:
        assert (
            num_expert_group is not None and topk_group is not None
        ), "use_grouped_topk requires num_expert_group and topk_group"

        expt_scal, expt_indx, bitmatrix = grouped_topk(
            logits,
            n_expts_act,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            expert_group=expert_group,
            apply_softmax=False,
            score_mode=score_mode,
            bias=bias,
            renorm=renorm,
            routed_scaling_factor=routed_scaling_factor,
            HIST_BLOCK_M=32,
        )
    else:
        from aiter.ops.triton.moe.moe_routing.topk import topk

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

    if num_tokens <= 16:
        HIST_BLOCK_M = triton.next_power_of_2(max(num_tokens, 1))
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
        topk_ids=expt_indx,
    )
    return routing_data, topk_indx, gate_indx


def routing_from_hash(
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
    from aiter.ops.triton.moe.moe_routing.topk import hash_routing

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
        topk_ids=expt_indx,
    )
    return routing_data, topk_indx, gate_indx


# --------------------------
# expert-parallel sort
# --------------------------


@dataclass(frozen=True)
class EpScatterGeometry:
    """What the sort needs to also emit the EP combine-scatter destination map.

    ``src_token_map[recv_slot] = origin_pe * max_tokens_per_rank + origin_lid``
    is the dispatch's reverse map; ``peer_rows`` is the stride, in staging-slot
    rows, between one peer's slot region and the next in the symmetric window.
    Pass this to ``ep_sort_routing`` and it returns ``dst_row`` alongside the
    usual five, at no extra launch -- the scatter kernel already holds every
    input the map needs.
    """

    src_token_map: torch.Tensor
    max_tokens_per_rank: int
    peer_rows: int

    def __post_init__(self):
        if (
            self.src_token_map.dtype != torch.int32
            or not self.src_token_map.is_contiguous()
        ):
            raise ValueError("src_token_map must be contiguous int32")
        if self.max_tokens_per_rank <= 0:
            raise ValueError("max_tokens_per_rank must be positive")
        if self.peer_rows <= 0:
            raise ValueError("peer_rows must be positive")


# Persistent per-(device, n_bins) scratch for the EP sort's histogram and join
# counter. Both are left zeroed by _ep_gate_hist_scan_kernel, so the torch.zeros
# below is paid once per process rather than once per layer per step.
_EP_SORT_SCRATCH = {}

# Up to _EP_SORT_FUSED_MAX gates one launch does the sort: one CTA up to
# _EP_SORT_ONE_CTA_MAX gates, else CTAs of _EP_SORT_FUSED_BLOCK gates that each
# count all gates in steps of _EP_SORT_FUSED_HIST. Measured on gfx1250, DP MoE with
# 256 experts: the two-kernel sort is 0.7-0.8x from 1k to 4k gates.
_EP_SORT_ONE_CTA_MAX = 512
_EP_SORT_FUSED_MAX = 512
_EP_SORT_FUSED_BLOCK = 256
_EP_SORT_FUSED_HIST = 2048
_EP_SORT_FUSED_WARPS = 4
# Two-kernel sort: (largest n_gates, gates per CTA, num_warps), measured on
# gfx1250. A CTA sorts its gates, so smaller blocks are faster, but each CTA also
# sums the histograms of the CTAs in front of it.
_EP_SORT_TILES = ((4096, 256, 4), (32768, 512, 4), (65536, 1024, 8), (None, 2048, 16))
# Up to this many CTAs each CTA of kernel B sums all per-CTA histograms itself:
# kernel A then needs no grid-wide join (no atomics, no ticket).
_EP_SORT_NO_JOIN_MAX_CTAS = 1024
# ep_sort_routing(quant_x=...): up to this many gates the MXFP4 quant of the rows
# runs as extra CTAs of the sort launch; above it, the standalone quant kernel (as
# fast from 32k gates up, 256 experts top-8).
_EP_SORT_QUANT_FUSED_MAX = 16384
# The same for the MXFP8 quant (quant_dtype=float8_e4m3fn), None: no cap. The
# Triton downcast_to_mxfp it replaces is slower at every size measured (4 .. 524288
# gates).
_EP_SORT_QUANT8_FUSED_MAX = None
# Tiles of the sort launch's quant CTAs: _EP_QUANT_BN columns and
# _EP_QUANT_ROWS_PER_WARP rows per wave.
_EP_QUANT_BN = 256
_EP_QUANT_ROWS_PER_WARP = 8
# ep_sort_routing(logits=...): the top-k takes a wave a row. One launch up to
# _EP_SORT_LOGITS_ONE_CTA_MAX gates, else the two-kernel sort with
# _EP_SORT_LOGITS_TILES (kernel A runs the top-k: a wave a row). Above
# _EP_SORT_LOGITS_MAX gates a top-k kernel of _EP_TOPK_ROWS rows a CTA runs first,
# then the plain sort.
_EP_SORT_LOGITS_ONE_CTA_MAX = 128
_EP_SORT_LOGITS_TILES = ((1024, 32, 4), (4096, 64, 8), (None, 128, 16))
_EP_SORT_LOGITS_MAX = 16384
_EP_TOPK_ROWS = 8
# Which launch of the two-kernel sort carries the quant CTAs: "A" or "B".
_EP_QUANT_KERNEL = "B"


def _ep_sort_scratch(device, n_bins):
    key = (device, n_bins)
    bufs = _EP_SORT_SCRATCH.get(key)
    if bufs is None:
        bufs = (
            torch.zeros(n_bins, dtype=torch.int32, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
        )
        _EP_SORT_SCRATCH[key] = bufs
    return bufs


def ep_sort_routing(
    dispatch_weights,
    dispatch_ids,
    expert_map,
    num_local_experts,
    num_local_tokens,
    M,
    topk,
    n_gates,
    expt_data_bufs,
    ep_scatter_geometry=None,
    quant_x=None,
    quant_live_only=False,
    expt_data_bufs2=None,
    quant_dtype=torch.uint8,
    logits=None,
):
    """Expert-sort rows that an expert-parallel all-to-all has already dispatched.

    ``routing()`` cannot serve this: it starts from router logits, but after the
    all-to-all the top-k choice is made and the rows have been permuted across
    ranks, so each row carries a full top-k tuple of which only *some* entries
    belong to this rank. Non-local entries go to a sentinel bin that the caller
    slices off the histogram, so the matmul schedules no block for them.

    Up to ``_EP_SORT_FUSED_MAX`` gates one kernel does all of it: each CTA counts
    every gate itself, so no CTA waits for another. Above, two kernels and no
    memset:

    A  : gating + per-CTA histogram                       (grid: n_ctas)
    B  : sorted scatter + ExptData | ExptData stage2      (grid: E + n_ctas)

    Each CTA of B sums the per-CTA histograms itself. Above
    ``_EP_SORT_NO_JOIN_MAX_CTAS`` CTAs, A joins the histograms with atomics and
    a ticket instead (last CTA: scan + ExptData stage1).

    Each expert's rows are in ascending gate order, so the outputs do not depend
    on the CTA schedule.

    ``expt_data_bufs`` is what ``_compute_expt_data_internal`` returns; the caller
    allocates it so the sort can fill it in the same launches. ``expt_data_bufs2``
    (same, for a second block_m) gets its token_offs_pad and block_pid_map filled
    in the same launches too: the tile map of a second GEMM with another block_m. Returns ``(hist,
    topk_indx, gate_indx, gate_scal, gate_valid, dst_row)`` with the ExptData
    buffers filled in place. ``hist`` has ``next_pow2(E + 1)`` entries -- the
    caller wants ``hist[:num_local_experts]``, since the tail holds no rows.

    ``gate_valid`` marks which gate slots this rank owns, in flat gate order
    (``row * topk + slot``), so a grouped reduce can skip the slots no GEMM here
    ever wrote.

    With ``ep_scatter_geometry`` (an ``EpScatterGeometry``) the returned tuple
    gains a sixth entry, ``dst_row``: where each sorted row must be delivered for
    a scatter-fused combine (-1 past the live rows). It is None otherwise.

    With ``quant_x`` (the (M, K) bf16 rows) the tuple gains ``x_fp4`` (M, K // 2)
    uint8 and ``x_scale`` (M, K // 32) uint8: the bytes ``mxfp4_quant(quant_x)``
    returns, made by extra CTAs of the sort launch (the sort never reads the
    rows). ``quant_live_only`` leaves rows past ``num_local_tokens`` unwritten:
    only for consumers that read the rows through ``gather_indx`` alone.
    ``quant_dtype=torch.float8_e4m3fn`` makes it MXFP8 instead: ``x_fp8`` (M, K)
    and ``x_scale`` (M, K // 32) uint8, the bytes of
    ``downcast_to_mxfp(quant_x, torch.float8_e4m3fn, axis=-1)``.

    With ``logits`` (the (M, E) bf16 router logits; no expert parallelism: the
    map is the identity) the sort launch also runs the renormalized top-k
    softmax and writes ``dispatch_ids`` / ``dispatch_weights``, the bits
    ``topk_softmax(renormalize=True)`` writes (ids -1 and weights 0 past
    ``num_local_tokens``).

    SINGLE STREAM ONLY above ``_EP_SORT_NO_JOIN_MAX_CTAS`` CTAs: the histogram and
    join counter are process-persistent and re-armed by the kernel that drains
    them, so two concurrent calls sharing a device would interleave into them.
    """
    device = dispatch_ids.device
    sentinel = num_local_experts
    # No bin for SENTINEL: the dead gates are never counted.
    n_bins = triton.next_power_of_2(sentinel)
    token_offs_raw, token_offs_pad, block_pid_map, blocks1, _, block_m_log2 = (
        expt_data_bufs
    )
    # The fused grid recovers the gate CTA as pid - N_EXPTS, so the expt_data
    # half must be exactly num_local_experts CTAs wide.
    assert blocks1 == num_local_experts, f"{blocks1} != {num_local_experts}"
    if expt_data_bufs2 is None:
        token_offs_pad2 = block_pid_map2 = max_num_tiles2 = None
        block_m2_log2 = 0
    else:
        _, token_offs_pad2, block_pid_map2, _, _, block_m2_log2 = expt_data_bufs2
        max_num_tiles2 = block_pid_map2.shape[0]

    gate_valid = torch.empty(n_gates, dtype=torch.int32, device=device)
    hist = torch.empty(n_bins, dtype=torch.int32, device=device)
    topk_indx = torch.empty(n_gates, dtype=torch.int32, device=device)
    gate_indx = torch.empty(n_gates, dtype=torch.int32, device=device)
    gate_scal = torch.empty(n_gates, dtype=torch.float32, device=device)
    if ep_scatter_geometry is None:
        dst_row = None
        src_token_map = None
        max_tok = 0
        peer_rows = 0
    else:
        dst_row = torch.empty(n_gates, dtype=torch.int32, device=device)
        src_token_map = ep_scatter_geometry.src_token_map
        max_tok = ep_scatter_geometry.max_tokens_per_rank
        peer_rows = ep_scatter_geometry.peer_rows

    if logits is not None:
        assert logits.dtype == torch.bfloat16 and logits.stride(1) == 1
        assert logits.shape == (M, num_local_experts)
        assert expert_map.numel() == num_local_experts + 1
    quant_args = _ep_quant_args(quant_x, n_gates, device, quant_dtype)
    if logits is not None and n_gates > _EP_SORT_LOGITS_MAX:
        _ep_topk_kernel[(triton.cdiv(M, _EP_TOPK_ROWS),)](
            logits,
            dispatch_ids,
            dispatch_weights,
            num_local_tokens,
            logits.stride(0),
            n_gates,
            N_EXPTS=num_local_experts,
            TOPK=topk,
            ROWS=_EP_TOPK_ROWS,
            num_warps=_EP_TOPK_ROWS,
        )
        logits = None
    logits_kw = {
        "Logits": logits,
        "stride_logits": 0 if logits is None else logits.stride(0),
        "LOGITS": logits is not None,
    }
    # The multi-CTA one-launch sort counts every gate in each CTA: no top-k there.
    one_launch_max = (
        _EP_SORT_FUSED_MAX
        if logits is None
        else min(_EP_SORT_FUSED_MAX, _EP_SORT_ONE_CTA_MAX, _EP_SORT_LOGITS_ONE_CTA_MAX)
    )
    if n_gates <= one_launch_max:
        if n_gates <= _EP_SORT_ONE_CTA_MAX:
            block = triton.next_power_of_2(max(n_gates, 16))
            hist_block = block
            num_warps = 1 if block <= 128 else 4 if block <= 512 else 8
        else:
            block = _EP_SORT_FUSED_BLOCK
            hist_block = min(_EP_SORT_FUSED_HIST, triton.next_power_of_2(n_gates))
            num_warps = _EP_SORT_FUSED_WARPS
        if logits is not None:
            # The top-k takes a wave a row.
            num_warps = max(num_warps, block // topk)
        n_sort_ctas = triton.cdiv(n_gates, block)
        quant_kw, n_quant_ctas = _ep_quant_kwargs(
            quant_args, num_warps, n_sort_ctas, quant_live_only
        )
        _ep_sort_fused_kernel[(n_sort_ctas + n_quant_ctas,)](
            dispatch_ids,
            expert_map,
            num_local_tokens,
            dispatch_weights,
            src_token_map,
            gate_valid,
            hist,
            token_offs_raw,
            token_offs_pad,
            block_pid_map,
            topk_indx,
            gate_indx,
            gate_scal,
            dst_row,
            block_pid_map.shape[0],
            n_gates,
            expert_map.numel(),
            token_offs_pad2,
            block_pid_map2,
            max_num_tiles2,
            N_EXPTS=num_local_experts,
            TOPK=topk,
            SENTINEL=sentinel,
            N_BINS=n_bins,
            BLOCK=block,
            LOG2_BLOCK=block.bit_length() - 1,
            HIST_BLOCK=hist_block,
            tile_dim_log2=block_m_log2,
            tile2_dim_log2=block_m2_log2,
            HAS_TILES2=expt_data_bufs2 is not None,
            TILE_BLOCK=4,
            MEMSET_BLOCK=256,
            HAS_DST_ROW=dst_row is not None,
            MAX_TOK=max_tok,
            PEER_ROWS=peer_rows,
            ONE_CTA=n_gates <= block,
            num_warps=num_warps,
            **quant_kw,
            **logits_kw,
        )
        return _ep_sort_result(
            (hist, topk_indx, gate_indx, gate_scal, gate_valid, dst_row), quant_args
        )

    tiles = _EP_SORT_TILES if logits is None else _EP_SORT_LOGITS_TILES
    block, num_warps = next(
        (b, w) for top, b, w in tiles if top is None or n_gates <= top
    )
    n_ctas = triton.cdiv(n_gates, block)
    join = n_ctas > _EP_SORT_NO_JOIN_MAX_CTAS
    hist_atomic, ticket = _ep_sort_scratch(device, n_bins)
    cursor = torch.empty(n_bins, dtype=torch.int32, device=device)
    cta_hist = torch.empty((n_ctas, n_bins), dtype=torch.int32, device=device)
    # The join counts every CTA of the grid: there the quant runs on its own.
    in_a = _EP_QUANT_KERNEL == "A" and not join
    quant_kw, n_quant_ctas = _ep_quant_kwargs(
        quant_args if in_a else None, num_warps, n_ctas, quant_live_only
    )
    quant_kw_b, n_quant_ctas_b = _ep_quant_kwargs(
        quant_args if _EP_QUANT_KERNEL == "B" else None,
        num_warps,
        n_ctas,
        quant_live_only,
    )
    _ep_gate_hist_scan_kernel[(n_ctas + n_quant_ctas,)](
        dispatch_ids,
        expert_map,
        num_local_tokens,
        gate_valid,
        cta_hist,
        hist_atomic,
        ticket,
        hist,
        cursor,
        token_offs_raw,
        token_offs_pad,
        block_pid_map,
        dst_row,
        block_pid_map.shape[0],
        n_gates,
        expert_map.numel(),
        token_offs_pad2,
        block_pid_map2,
        max_num_tiles2,
        N_EXPTS=num_local_experts,
        TOPK=topk,
        SENTINEL=sentinel,
        N_BINS=n_bins,
        BLOCK=block,
        tile_dim_log2=block_m_log2,
        tile2_dim_log2=block_m2_log2,
        HAS_TILES2=expt_data_bufs2 is not None,
        MEMSET_BLOCK=1024,
        HAS_DST_ROW=dst_row is not None,
        JOIN=join,
        num_warps=num_warps,
        **quant_kw,
        **logits_kw,
        DispatchWeights=dispatch_weights,
    )
    if join and _EP_QUANT_KERNEL == "A" and quant_args is not None:
        quant_args = quant_args[:3] + (False,)
    _ep_sorted_scatter_kernel[(num_local_experts + n_ctas + n_quant_ctas_b,)](
        dispatch_ids,
        expert_map,
        num_local_tokens,
        dispatch_weights,
        cta_hist,
        cursor,
        topk_indx,
        gate_indx,
        gate_scal,
        hist,
        token_offs_raw,
        token_offs_pad,
        block_pid_map,
        dst_row,
        src_token_map,
        block_pid_map.shape[0],
        n_gates,
        expert_map.numel(),
        token_offs_pad2,
        block_pid_map2,
        max_num_tiles2,
        N_EXPTS=num_local_experts,
        TOPK=topk,
        SENTINEL=sentinel,
        N_BINS=n_bins,
        BLOCK=block,
        LOG2_BLOCK=block.bit_length() - 1,
        PREFIX_ROWS=min(256, triton.next_power_of_2(n_ctas)),
        tile_dim_log2=block_m_log2,
        tile2_dim_log2=block_m2_log2,
        HAS_TILES2=expt_data_bufs2 is not None,
        MEMSET_BLOCK=256,
        HAS_DST_ROW=dst_row is not None,
        JOIN=join,
        MAX_TOK=max_tok,
        PEER_ROWS=peer_rows,
        num_warps=num_warps,
        **quant_kw_b,
    )
    return _ep_sort_result(
        (hist, topk_indx, gate_indx, gate_scal, gate_valid, dst_row), quant_args
    )


def _ep_quant_args(quant_x, n_gates, device, quant_dtype):
    """(x, x_q, x_scale, fused) for the sort launch's quant CTAs, or None.
    MXFP8 (quant_dtype float8_e4m3fn) above its cap allocates nothing here: the
    standalone downcast_to_mxfp makes its own outputs."""
    if quant_x is None:
        return None
    assert quant_x.ndim == 2 and quant_x.dtype == torch.bfloat16
    assert quant_x.stride(1) == 1, "quant_x needs unit column stride"
    m, k = quant_x.shape
    assert k % _EP_QUANT_BN == 0, f"K={k} must be a multiple of {_EP_QUANT_BN}"
    if quant_dtype == torch.float8_e4m3fn:
        cap = _EP_SORT_QUANT8_FUSED_MAX
        if cap is not None and n_gates > cap:
            return quant_x, None, None, False
        x_q = torch.empty((m, k), dtype=quant_dtype, device=device)
        x_scale = torch.empty((m, k // 32), dtype=torch.uint8, device=device)
        return quant_x, x_q, x_scale, True
    assert quant_dtype == torch.uint8, f"quant_dtype {quant_dtype}: uint8 or fp8 e4m3"
    x_fp4 = torch.empty((m, k // 2), dtype=torch.uint8, device=device)
    x_scale = torch.empty((m, k // 32), dtype=torch.uint8, device=device)
    return quant_x, x_fp4, x_scale, n_gates <= _EP_SORT_QUANT_FUSED_MAX


def _ep_quant_kwargs(quant_args, num_warps, n_sort_ctas, live_only):
    """Kernel arguments of the quant CTAs and their count (0 without quant)."""
    if quant_args is None or not quant_args[3]:
        kw = {
            "X": None,
            "XQ": None,
            "XS": None,
            "n_sort_ctas": n_sort_ctas,
            "x_rows": 0,
            "stride_x": 0,
            "stride_q": 0,
            "stride_s": 0,
            "QUANT": False,
            "QK": 0,
            "QBM": 1,
            "QBN": 1,
            "QUANT_LIVE_ONLY": False,
            "QFP8": False,
        }
        return kw, 0
    x, x_fp4, x_scale, _ = quant_args
    m, k = x.shape
    qbm = _EP_QUANT_ROWS_PER_WARP * num_warps
    kw = {
        "X": x,
        "XQ": x_fp4,
        "XS": x_scale,
        "n_sort_ctas": n_sort_ctas,
        "x_rows": m,
        "stride_x": x.stride(0),
        "stride_q": x_fp4.stride(0),
        "stride_s": x_scale.stride(0),
        "QUANT": True,
        "QK": k,
        "QBM": qbm,
        "QBN": _EP_QUANT_BN,
        "QUANT_LIVE_ONLY": live_only,
        "QFP8": x_fp4.dtype != torch.uint8,
    }
    return kw, triton.cdiv(m, qbm) * (k // _EP_QUANT_BN)


def _ep_sort_result(outs, quant_args):
    """The sort's outputs, plus (x_q, x_scale) with quant_x."""
    if quant_args is None:
        return outs
    x, x_q, x_scale, fused = quant_args
    if not fused:
        if x_q is None or x_q.dtype != torch.uint8:
            from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp

            return outs + downcast_to_mxfp(x, torch.float8_e4m3fn, axis=-1)
        from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant

        dynamic_mxfp4_quant(x, x_fp4=x_q, blockscale_e8m0=x_scale)
    return outs + (x_q, x_scale)


# --------------------------
# torch reference
# --------------------------


def compute_expt_data_torch(hist, n_expts_tot, n_gates, block_m):
    # offset for each experts
    device = hist.device
    token_offs_raw = torch.cumsum(hist, dim=0)
    token_offs_raw = torch.cat(
        (torch.zeros(1, dtype=torch.float32, device=device), token_offs_raw)
    )
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
    token_offs_pad = torch.cat(
        (torch.zeros(1, dtype=torch.float32, device=device), token_offs_pad)
    )
    token_offs_pad = token_offs_pad.int()
    # compute data required to drive ragged batch matmul
    block_pid_map = -torch.ones(max_n_tiles, dtype=torch.float32, device=device)
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
