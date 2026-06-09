import triton
import torch
from aiter.ops.triton._triton_kernels.moe.moe_routing.topk import (
    _topk,
    _hash_routing,
    _grouped_topk,
)
from aiter.ops.triton.moe.moe_routing.bitmatrix import Bitmatrix


def grouped_topk(
    x: torch.Tensor,
    k: int,
    num_expert_group: int,
    topk_group: int,
    *,
    expert_group: torch.Tensor | None = None,
    apply_softmax: bool = False,  # accepted for parity with topk(); ignored
    HIST_BLOCK_M: int = 32,
    score_mode: str = "softmax",
    bias: torch.Tensor | None = None,
    renorm: bool = False,
    routed_scaling_factor: float = 1.0,
):
    """Triton grouped top-k expert selection. See module docstring.

    Returns ``(y_vals, y_indx, bitmatrix)`` matching the contract of
    ``aiter.ops.triton.moe.moe_routing.topk.topk``:

      - y_vals: ``(n_rows, k)`` in ``x.dtype``.
      - y_indx: ``(n_rows, k)`` ``int16``.

      - bitmatrix: real :class:`Bitmatrix`; same uint32
        ``(n_cols_words, n_rows_pad32).T`` storage / scratchpad layout the
        ``_topk`` kernel emits, so ``sort_tokens`` and ``sort_tokens_fused``
        consume it unchanged.
    """
    assert x.dim() == 2
    n_rows, n_cols = x.shape
    assert n_cols <= 256, f"grouped_topk n_expts_tot ({n_cols}) only supported <= 256"
    n_total = n_cols  # experts (bitmatrix width)
    k_out = k  # output width (routed top-k)
    assert num_expert_group > 1
    assert (
        num_expert_group <= 16
    ), f"NUM_EXPERT_GROUP ({num_expert_group}) > 16 not supported"
    assert 0 < topk_group <= num_expert_group
    assert 0 < k <= 16
    assert score_mode in (
        "softmax",
        "sigmoid",
        "sqrtsoftplus",
        "none",
    ), f"unknown score_mode {score_mode!r}"
    has_bias = bias is not None
    if has_bias:
        assert bias.dim() == 1 and bias.shape[0] == n_cols
        assert bias.dtype == torch.float32
        assert score_mode in (
            "sqrtsoftplus",
            "sigmoid",
        ), "bias only supported with sqrtsoftplus / sigmoid"

    dev = x.device

    # Default expert→group mapping = contiguous DeepSeek layout.
    if expert_group is None:
        assert n_cols % num_expert_group == 0, (
            f"n_expts_tot ({n_cols}) not divisible by num_expert_group "
            f"({num_expert_group}); pass an explicit expert_group table."
        )
        g_size = n_cols // num_expert_group
        expert_group = (
            torch.arange(n_cols, device=dev, dtype=torch.int32) // g_size
        ).to(torch.int32)
    else:
        assert expert_group.dim() == 1 and expert_group.shape[0] == n_cols
        assert expert_group.dtype == torch.int32

    # Block sizes — single BLOCK_N pass for DeepSeek envelope. BLOCK_N must
    # cover the shared-expert columns too so their bits fit in the bitmatrix.
    BLOCK_M = 1
    BLOCK_N = max(32, triton.next_power_of_2(n_total))
    N_EXPTS_PAD = BLOCK_N
    # Mirror topk(): pad to ≥ 2 to dodge tl.argmax/topk(k=1) compile quirks.
    N_EXPTS_ACT_PAD = max(2, triton.next_power_of_2(k_out))
    BLOCK_S = 128
    BLOCK_SP = 128
    TILE_SIZE = 8

    # Outputs (same shapes / dtypes as topk(...)), widened by the shared slots.
    y_vals = torch.empty((n_rows, k_out), dtype=x.dtype, device=dev)
    y_indx = torch.empty((n_rows, k_out), dtype=torch.int16, device=dev)

    # Bitmatrix in transposed-uint32 storage layout (identical to topk()).
    n_cols_pad = triton.cdiv(n_total, BLOCK_N) * BLOCK_N
    n_cols_words = n_cols_pad // 32
    bitmatrix_data = torch.empty(
        (n_cols_words, triton.cdiv(n_rows, 32) * 32),
        dtype=torch.uint32,
        device=dev,
    )
    bitmatrix_data = torch.transpose(bitmatrix_data, 0, 1)[:n_rows]

    # Scratchpads. The per-column sum buffer consumed by Bitmatrix.sum() /
    # sort_tokens must cover the full padded column count (n_cols_pad), which
    # widens with the shared experts; sizing by n_total alone can under-allocate
    # (e.g. n_total=257 -> n_cols_pad=512 but cdiv(257,128)*128=384).
    s_blocks = triton.cdiv(n_cols_pad, BLOCK_S)
    s_cols = s_blocks * BLOCK_S
    scratchpad = torch.empty((s_cols,), dtype=torch.int32, device=dev)
    BLOCK_MM = HIST_BLOCK_M * TILE_SIZE
    pids_x = triton.cdiv(n_rows, BLOCK_MM)
    scratchpad_partials = torch.empty(
        (n_cols_pad, pids_x * TILE_SIZE), dtype=torch.int32, device=dev
    )
    scratchpad_partials = torch.transpose(scratchpad_partials, 0, 1)
    sp_size = scratchpad_partials.numel()
    sp_blocks = triton.cdiv(sp_size, BLOCK_SP)

    pids = max(triton.cdiv(n_rows, BLOCK_M), s_blocks + sp_blocks)

    _grouped_topk[(pids,)](
        x,
        x.stride(0),
        expert_group,
        y_vals,
        y_indx,
        y_vals.stride(0),
        bitmatrix_data,
        bitmatrix_data.stride(0),
        bitmatrix_data.stride(1),
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
        N_EXPTS_PAD=N_EXPTS_PAD,
        BLOCK_N=BLOCK_N,
        N_EXPTS_ACT=k,
        N_EXPTS_ACT_PAD=N_EXPTS_ACT_PAD,
        NUM_EXPERT_GROUP=num_expert_group,
        TOPK_GROUP=topk_group,
        Bias=bias,
        SCORE_MODE=score_mode,
        HAS_BIAS=has_bias,
        APPLY_RENORM=renorm,
        ROUTED_SCALING=routed_scaling_factor,
        num_warps=4,
    )

    bitmatrix = Bitmatrix(
        bitmatrix_data,
        shape=[n_rows, n_cols_words * 32],
        scratchpad=scratchpad,
        scratchpad_partials=scratchpad_partials,
    )
    return y_vals, y_indx, bitmatrix


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
    assert len(x.shape) == 2
    n_rows, n_cols = x.shape

    # BLOCK_M=1 for small n_rows keeps the grid wide enough to overlap with
    BLOCK_M = 1 if n_rows <= 256 else 32
    BLOCK_N = 128
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
    # Triton's tl.topk fails to compile for k=1 (log_k=0 reduces the hypercube
    # to a 0-D tensor; the final reshape hits dtype.numel). Pad to ≥ 2 — the
    # kernel already masks N_EXPTS_ACT < N_EXPTS_ACT_PAD on store.
    k_pow2 = max(2, triton.next_power_of_2(k))
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
    # See note in topk(): pad to ≥ 2 to dodge tl.topk(k=1) compile bug.
    k_pow2 = max(2, triton.next_power_of_2(k))

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
