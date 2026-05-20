import triton
import torch
from aiter.ops.triton._triton_kernels.moe.moe_routing.topk import _topk, _hash_routing
from aiter.ops.triton.moe.moe_routing.bitmatrix import Bitmatrix


def topk(
    x,
    k,
    apply_softmax=True,
    dim=1,
    return_bitmatrix=True,
    HIST_BLOCK_M=32,
    score_mode: str = "softmax",
    bias=None,
    renorm: bool = False,
    routed_scaling_factor: float = 1.0,
):
    """Top-k expert selection with bitmatrix.

    score_mode:
      - "softmax" (default): no pre-transform; APPLY_SOFTMAX may renormalize.
      - "sqrtsoftplus": pre-transform `scores = sqrt(softplus(logits))` before
        adding the optional `bias` and running topk. Selected weights are the
        UNBIASED sqrt(softplus(logits)). DeepSeek-V4 noaux_tc router.

    bias (fp32, [n_expts_tot]): added to scores for selection only, not for
    returned weights. Only meaningful with score_mode='sqrtsoftplus'.

    renorm: renormalize weights to sum=1 per row before multiplying by
    routed_scaling_factor.
    """
    # print(f"topk {x.dtype=} {x.shape=} {k=} {apply_softmax=} {dim=} {return_bitmatrix=} {HIST_BLOCK_M=} {score_mode=} {bias.dtype} {bias.shape} {renorm=} {routed_scaling_factor=}")
    assert len(x.shape) == 2
    n_rows, n_cols = x.shape

    # BLOCK_M=1 for small n_rows keeps the grid wide enough to overlap with
    # the bitmatrix-scratchpad memset pids (see /app/_test/verify-perf-topk.py).
    BLOCK_M = 1 if n_rows <= 256 else 32
    BLOCK_N = 128  # triton.next_power_of_2(x_shape[1])  # 128
    BLOCK_S = 128
    BLOCK_SP = 128
    assert n_cols < 32768
    assert dim == 1
    assert return_bitmatrix
    assert score_mode in (
        "softmax",
        "sqrtsoftplus",
    ), f"score_mode must be 'softmax' or 'sqrtsoftplus', got {score_mode!r}"
    if score_mode != "softmax":
        assert not apply_softmax, "apply_softmax only valid with score_mode='softmax'"
    has_bias = bias is not None
    if has_bias:
        assert bias.dim() == 1
        assert bias.shape[0] == x.shape[-1]
        assert bias.dtype == torch.float32
        assert (
            score_mode == "sqrtsoftplus"
        ), "bias currently only supported with score_mode='sqrtsoftplus'"
    dev = x.device
    # scratchpad tensors
    # NOTE: these are not returned
    y_vals = torch.empty((n_rows, k), dtype=x.dtype, device=dev)
    y_indx = torch.empty((n_rows, k), dtype=torch.int16, device=dev)
    k_pow2 = triton.next_power_of_2(k)
    # create bitmatrix in transposed memory layout:
    n_cols_pad = triton.cdiv(n_cols, BLOCK_N) * BLOCK_N
    n_cols_words = n_cols_pad // 32
    bitmatrix = torch.empty(
        (n_cols_words, triton.cdiv(n_rows, 32) * 32), dtype=torch.uint32, device=dev
    )
    bitmatrix = torch.transpose(bitmatrix, 0, 1)[:n_rows]
    s_blocks = triton.cdiv(n_cols, BLOCK_S)
    s_cols = s_blocks * BLOCK_S
    scratchpad = torch.empty((s_cols,), dtype=torch.int32, device=dev)
    TILE_SIZE = 8
    BLOCK_MM = HIST_BLOCK_M * TILE_SIZE
    pids_x = triton.cdiv(n_rows, BLOCK_MM)
    scratchpad_partials = torch.empty(
        (n_cols_pad, pids_x * TILE_SIZE), device=dev, dtype=torch.int32
    )
    scratchpad_partials = torch.transpose(scratchpad_partials, 0, 1)
    sp_size = torch.numel(scratchpad_partials)
    sp_blocks = triton.cdiv(sp_size, BLOCK_SP)
    pids = max(triton.cdiv(n_rows, BLOCK_M), s_blocks + sp_blocks)
    _topk[(pids,)](
        x,
        x.stride(0),  # inputs
        y_vals,  # output [topk]
        y_indx,
        y_vals.stride(0),
        bitmatrix,
        bitmatrix.stride(0),
        bitmatrix.stride(1),  # output [bitmatrix]
        n_rows,
        n_cols,  # shapes
        scratchpad,
        BLOCK_S,
        s_blocks,  # thing to memset to zero
        scratchpad_partials,
        BLOCK_SP,
        sp_blocks,
        sp_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,  # tunable parameter
        APPLY_SOFTMAX=apply_softmax,
        N_EXPTS_PAD=n_cols_pad,
        N_EXPTS_ACT=k,  # constants
        N_EXPTS_ACT_PAD=k_pow2,
        num_warps=8,
        Bias=bias,
        SCORE_MODE=score_mode,
        HAS_BIAS=has_bias,
        APPLY_RENORM=renorm,
        ROUTED_SCALING=routed_scaling_factor,
    )
    bitmatrix_shape = [n_rows, n_cols_words * 32]
    bitmatrix = Bitmatrix(
        bitmatrix,
        shape=bitmatrix_shape,
        scratchpad=scratchpad,
        scratchpad_partials=scratchpad_partials,
    )
    return y_vals, y_indx, bitmatrix


def hash_routing(
    router_logits: torch.Tensor,  # [n_rows, n_expts_tot] bf16/fp32
    tid2eid: torch.Tensor,  # [vocab_size, K] int32 per-token-id expert table
    input_ids: torch.Tensor,  # [n_rows] int32 token ids (post DP gather, clamped)
    n_expts_act: int,
    HIST_BLOCK_M: int = 32,
    score_mode: str = "sqrtsoftplus",
    renorm: bool = True,
    routed_scaling_factor: float = 1.0,
):
    """Fused hash routing: tid2eid lookup + score transform + gather + renorm
    + scale + bitmatrix construction. Output contract matches :func:`topk` so
    downstream :func:`sort_tokens_fused` consumes it unchanged.

    Replaces the Python ``_hash_topk`` + ``fused_routing_from_topk``
    counting-sort + bitmatrix-build chain with one Triton kernel launch.
    """

    BLOCK_M = 32
    BLOCK_N = 128
    BLOCK_S = 128
    BLOCK_SP = 128
    assert router_logits.dim() == 2
    assert input_ids.dim() == 1
    assert tid2eid.dim() == 2
    assert input_ids.shape[0] == router_logits.shape[0]
    assert (
        tid2eid.shape[1] == n_expts_act
    ), f"tid2eid second dim {tid2eid.shape[1]} must equal n_expts_act {n_expts_act}"
    assert tid2eid.dtype == torch.int32
    assert input_ids.dtype in (torch.int32, torch.int64)
    assert score_mode in ("sqrtsoftplus",)

    n_rows, n_cols = router_logits.shape
    dev = router_logits.device
    k = n_expts_act

    y_vals = torch.empty((n_rows, k), dtype=router_logits.dtype, device=dev)
    y_indx = torch.empty((n_rows, k), dtype=torch.int16, device=dev)
    k_pow2 = triton.next_power_of_2(k)

    n_cols_pad = triton.cdiv(n_cols, BLOCK_N) * BLOCK_N
    n_cols_words = n_cols_pad // 32
    bitmatrix = torch.empty(
        (n_cols_words, triton.cdiv(n_rows, 32) * 32), dtype=torch.uint32, device=dev
    )
    bitmatrix = torch.transpose(bitmatrix, 0, 1)[:n_rows]
    s_blocks = triton.cdiv(n_cols, BLOCK_S)
    s_cols = s_blocks * BLOCK_S
    scratchpad = torch.empty((s_cols,), dtype=torch.int32, device=dev)
    TILE_SIZE = 8
    BLOCK_MM = HIST_BLOCK_M * TILE_SIZE
    pids_x = triton.cdiv(n_rows, BLOCK_MM)
    scratchpad_partials = torch.empty(
        (n_cols_pad, pids_x * TILE_SIZE), device=dev, dtype=torch.int32
    )
    scratchpad_partials = torch.transpose(scratchpad_partials, 0, 1)
    sp_size = torch.numel(scratchpad_partials)
    sp_blocks = triton.cdiv(sp_size, BLOCK_SP)
    pids = max(triton.cdiv(n_rows, BLOCK_M), s_blocks + sp_blocks)

    # int32 cast for input_ids if int64
    input_ids_i32 = (
        input_ids.to(torch.int32) if input_ids.dtype != torch.int32 else input_ids
    )

    _hash_routing[(pids,)](
        input_ids_i32,
        tid2eid,
        tid2eid.stride(0),
        router_logits,
        router_logits.stride(0),
        y_vals,
        y_indx,
        y_vals.stride(0),
        bitmatrix,
        bitmatrix.stride(0),
        bitmatrix.stride(1),
        n_rows,
        n_cols,
        scratchpad,
        BLOCK_S,
        s_blocks,
        scratchpad_partials,
        BLOCK_SP,
        sp_blocks,
        sp_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        N_EXPTS_PAD=n_cols_pad,
        N_EXPTS_ACT=k,
        N_EXPTS_ACT_PAD=k_pow2,
        SCORE_MODE=score_mode,
        APPLY_RENORM=renorm,
        ROUTED_SCALING=routed_scaling_factor,
        num_warps=8,
    )

    bitmatrix_shape = [n_rows, n_cols_words * 32]
    bitmatrix = Bitmatrix(
        bitmatrix,
        shape=bitmatrix_shape,
        scratchpad=scratchpad,
        scratchpad_partials=scratchpad_partials,
    )
    return y_vals, y_indx, bitmatrix
