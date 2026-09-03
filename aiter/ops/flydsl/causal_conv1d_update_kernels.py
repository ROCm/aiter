# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL causal-conv1d update host wrappers (decode + speculative verify).

One algorithm behind two upstream interfaces, both maintained:
``causal_conv1d_update_flydsl`` is a drop-in for vLLM's (and aiter's Triton)
``causal_conv1d_update``; ``causal_conv1d_update_sglang_flydsl`` is SGLang's,
which adds per-step ``intermediate_conv_window`` snapshots and an EAGLE tree
traversal. Both launch the same grid shape, so one tile/occupancy policy
(``_pick_block_n`` / ``_pick_cpt``) serves both.

Neither is selected by default; the in-tree seam in
``aiter.ops.triton.conv.causal_conv1d`` opts in via
``AITER_CONV1D_UPDATE_FLYDSL=1`` and keeps Triton for anything outside the
port's scope. The ``_..._supported`` predicates are private so that a caller
which gets the scope wrong hits a ``NotImplementedError`` instead of silently
mis-executing.

Warning: the two upstreams' skip sentinels differ in *value*, not just in name
-- vLLM's ``null_block_id`` is ``0`` (a valid index, since block 0 is reserved)
while SGLang's ``pad_slot_id`` is ``-1``. Mixing them up does not raise: under
``null_block_id=0``, ``conv_state_indices`` must start at 1 or sequence 0 is
silently skipped.
"""

from __future__ import annotations

import os

import torch

from .kernels.causal_conv1d_update import (
    compile_causal_conv1d_update,
    compile_causal_conv1d_update_sglang,
)
from .kernels.tensor_shim import _run_compiled

#: Private, and absent from torch builds without a CUDA/ROCm backend, so the
#: public spelling stays as the fallback -- same idiom as aiter/jit/core.py.
_RAW_STREAM = getattr(torch._C, "_cuda_getCurrentRawStream", None)


def _raw_stream(device: torch.device) -> int:
    """Current stream handle, without building a ``torch.cuda.Stream`` per launch.

    Keyed off the tensors' own device rather than the ambient one, so a caller
    that never entered a ``torch.cuda.device`` context still enqueues where its
    pointers live.
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    if _RAW_STREAM is None:
        return torch.cuda.current_stream(index).cuda_stream
    return _RAW_STREAM(index)


#: SGLang's padded-slot sentinel (also aiter's Triton conv1d convention).
PAD_SLOT_ID = -1

#: vLLM's null cache block, a *valid* index unlike ``PAD_SLOT_ID``.
NULL_BLOCK_ID = 0

#: Anything outside these (notably fp32) is reinterpreted as fp16 downstream.
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)

#: Implemented widths. The narrower SGLang range matches its own Triton kernel.
_VLLM_WIDTHS = range(2, 7)
_SGLANG_WIDTHS = range(2, 5)

#: Targets the kernels are built for and the test suite runs on. An exact list
#: rather than a ``gfx9`` prefix: the older gfx9 parts are never exercised, so
#: admitting them would dispatch where nothing has been validated.
_SUPPORTED_ARCHS = ("gfx942", "gfx950")

#: Narrowest channel tile.
_WAVEFRONT = 64

#: Widest first; :func:`_pick_block_n` scans in order.
_BLOCK_N_CANDIDATES = (256, 128, _WAVEFRONT)

_CPT_MAX = 2

#: Longest speculative window that still gains from ``_CPT_MAX``.
_CPT_MAX_SEQLEN = 4

#: ``batch * dim`` at or below which :func:`_pick_block_n` inverts and takes the
#: widest tile.
_MIN_CHANNELS_TO_SPLIT = 1024

#: A ratio, not a count, so it carries across parts unchanged.
_TARGET_WG_PER_CU = 2

_CU_COUNT_CACHE: dict[int, int | None] = {}
_ENV_OVERRIDE_CACHE: dict[str, int | None] = {}


__all__ = [
    "NULL_BLOCK_ID",
    "PAD_SLOT_ID",
    "causal_conv1d_update_flydsl",
    "causal_conv1d_update_sglang_flydsl",
]


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _env_int(name: str | None) -> int | None:
    """Read an integer launch override, or ``None`` to keep the heuristic.

    Unset, empty and ``"auto"`` all mean "no override"; anything unparseable
    raises rather than being ignored. Cached, so a mid-process change is ignored.
    """
    if name is None:
        return None
    if name not in _ENV_OVERRIDE_CACHE:
        raw = os.environ.get(name, "")
        _ENV_OVERRIDE_CACHE[name] = None if raw in ("", "auto") else max(1, int(raw))
    return _ENV_OVERRIDE_CACHE[name]


def _n_cu(device: torch.device) -> int | None:
    """Compute-unit count of ``device``, or ``None`` if it could not be queried."""
    key = device.index if device.index is not None else 0
    if key not in _CU_COUNT_CACHE:
        try:
            count = torch.cuda.get_device_properties(device).multi_processor_count
        except Exception:  # noqa: BLE001 - no live device, or a meta/CPU tensor
            count = None
        _CU_COUNT_CACHE[key] = count
    return _CU_COUNT_CACHE[key]


def _target_wg(device: torch.device) -> int | None:
    """Workgroup count the launch should reach, or ``None`` if unknown.

    Callers treat ``None`` as "assume the largest machine" rather than guessing a
    mid-range CU count, since over-decomposing only queues extra workgroups while
    under-occupying exposes memory latency.
    """
    count = _n_cu(device)
    return None if count is None else _TARGET_WG_PER_CU * count


def _pick_cpt(
    batch: int,
    dim: int,
    device: torch.device,
    *,
    seqlen: int = 1,
    env: str | None = None,
) -> int:
    """Pick channels-per-thread, spending it only when the launch can spare it.

    More channels per thread puts more loads in flight but halves the workgroup
    count, so it only pays once the grid already reaches the occupancy target.
    Long speculative windows are excluded outright: they hold more state per
    thread, and past ``_CPT_MAX_SEQLEN`` the knob turns into a loss.

    Occupancy is estimated at the widest span the tile search could pick, to stay
    consistent with :func:`_pick_block_n`.
    """
    override = _env_int(env)
    if override is not None:
        return override
    if seqlen > _CPT_MAX_SEQLEN:
        return 1
    target = _target_wg(device)
    if target is None:
        return 1
    widest_span = _BLOCK_N_CANDIDATES[0] * _CPT_MAX
    return _CPT_MAX if batch * _ceil_div(dim, widest_span) >= target else 1


def _pick_block_n(
    batch: int, dim: int, device: torch.device, cpt: int = 1, *, env: str | None = None
) -> int:
    """Pick the channel-tile width so the launch fills the GPU.

    Grid size is ``batch * cdiv(dim, BLOCK_N * cpt)``, so a wide tile is cheapest
    per workgroup but under-occupies at small batch. Returns the widest candidate
    that still reaches the occupancy target, else the narrowest.

    Below ``_MIN_CHANNELS_TO_SPLIT`` channels the rule inverts and takes the
    widest tile: splitting the axis buys parallelism to hide latency behind, and
    once no tile can fill the part there is no latency left to hide, only the
    per-workgroup cost paid more times. The threshold is on ``batch * dim``
    because the choice does not care how that product factorizes.
    """
    override = _env_int(env)
    if override is not None:
        return override
    if batch * dim <= _MIN_CHANNELS_TO_SPLIT:
        return _BLOCK_N_CANDIDATES[0]
    target = _target_wg(device)
    if target is not None:
        for cand in _BLOCK_N_CANDIDATES:
            if batch * _ceil_div(dim, cand * cpt) >= target:
                return cand
    return _BLOCK_N_CANDIDATES[-1]


def _is_supported_arch(device: torch.device) -> bool:
    try:
        arch = str(torch.cuda.get_device_properties(device).gcnArchName)
    except Exception:  # noqa: BLE001 - no live device, meta/CPU tensor
        return False
    return arch.split(":")[0] in _SUPPORTED_ARCHS


def _dtype_out_of_scope(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> str | None:
    """Why no kernel can be built for these dtypes, or ``None`` if one can.

    One ``dtype_str`` specializes the whole kernel, so every tensor it touches
    is interpreted as that one dtype. Were any of them allowed to differ, its
    bytes would be reinterpreted rather than converted, which is a wrong answer
    instead of a refusal; casting instead would cost the caller the update twice
    over, since the kernel would write into the cast copy and leave the buffer
    upstream overwrites in place untouched.

    The single source for both refusals: :func:`_shapes_supported` turns it into
    the ``False`` the dispatch seam falls through on, and the entry points raise
    it. Stating the rule once is what keeps those two from disagreeing.
    """
    if conv_state.dtype not in _SUPPORTED_DTYPES:
        return (
            f"`conv_state` dtype {conv_state.dtype} is outside the specialized "
            f"{list(_SUPPORTED_DTYPES)}"
        )
    for name, t in (("x", x), ("weight", weight), ("bias", bias)):
        if t is not None and t.dtype != conv_state.dtype:
            return (
                f"`{name}` dtype {t.dtype} must match `conv_state` dtype "
                f"{conv_state.dtype}, which is what the kernel would read it as"
            )
    return None


def _require_in_scope(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    fn: str,
) -> None:
    """Refuse at the entry point what the seam would have fallen through on.

    The ``_..._supported`` predicates are private, so a caller holding only the
    public entry point has no way to ask whether its problem is in scope. That
    makes the entry point the one place left to say no, as it already does for
    an out-of-range width.
    """
    reason = _dtype_out_of_scope(x, conv_state, weight, bias)
    if reason is None and not _is_supported_arch(x.device):
        reason = f"{x.device} is not one of the built {list(_SUPPORTED_ARCHS)}"
    if reason is not None:
        raise NotImplementedError(f"{fn}: {reason}")


def _shapes_supported(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    widths: range,
    is_spec: bool,
    is_varlen: bool = False,
    max_query_len: int = -1,
    bias: torch.Tensor | None = None,
) -> bool:
    """Shape / dtype / placement checks shared by both interfaces."""
    if x.dim() not in (2, 3) or conv_state.dim() != 3 or weight.dim() != 2:
        return False
    if is_varlen and (x.dim() != 2 or max_query_len <= 0):
        return False
    if _dtype_out_of_scope(x, conv_state, weight, bias) is not None:
        return False
    if not (x.is_cuda and conv_state.is_cuda and weight.is_cuda):
        return False
    if x.device != conv_state.device or x.device != weight.device:
        return False

    dim = x.size(1)
    # Packed x has no sequence axis, so the token budget comes from the caller.
    seqlen = max_query_len if is_varlen else (x.size(2) if x.dim() == 3 else 1)
    width = weight.size(1)
    if width not in widths:
        return False
    if weight.size(0) != dim or conv_state.size(1) != dim:
        return False

    state_len_eff = (width - 1 + (seqlen - 1)) if is_spec else (width - 1)
    return conv_state.size(2) >= state_len_eff and _is_supported_arch(x.device)


def _causal_conv1d_update_flydsl_supported(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
) -> bool:
    """Whether ``causal_conv1d_update_flydsl`` can serve this problem.

    Every mode of vLLM's Triton kernel is covered, so only shapes and dtypes are
    screened.
    """
    del block_idx_last_scheduled_token, initial_state_idx  # supported
    return _shapes_supported(
        x,
        conv_state,
        weight,
        _VLLM_WIDTHS,
        num_accepted_tokens is not None,
        is_varlen=query_start_loc is not None,
        max_query_len=max_query_len,
        bias=bias,
    )


def _causal_conv1d_update_sglang_flydsl_supported(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    num_accept_tokens: torch.Tensor | None = None,
    cache_seqlens: torch.Tensor | None = None,
) -> bool:
    """Whether ``causal_conv1d_update_sglang_flydsl`` can serve this problem.

    ``cache_seqlens`` (circular conv_state buffer) is unimplemented here exactly
    as it is in SGLang's own Triton kernel.
    """
    if cache_seqlens is not None:
        return False
    return _shapes_supported(
        x,
        conv_state,
        weight,
        _SGLANG_WIDTHS,
        num_accept_tokens is not None,
        bias=bias,
    )


def _resolve_activation(activation: bool | str | None) -> bool:
    if isinstance(activation, bool):
        activation = "silu" if activation else None
    elif activation is not None:
        assert activation in ("silu", "swish")
    return activation in ("silu", "swish")


def causal_conv1d_update_flydsl(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    null_block_id: int = NULL_BLOCK_ID,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    validate_data: bool = False,
    out: torch.Tensor | None = None,
    block_n: int = 0,
    channels_per_thread: int = 0,
):
    """FlyDSL decode / chain-verify causal_conv1d update (vLLM-aligned).

    Drop-in for vLLM's ``causal_conv1d_update`` (same parameter names and order),
    covering all of its modes, plus trailing FlyDSL ``block_n`` /
    ``channels_per_thread`` knobs where ``0`` means auto.

    - ``x``:                ``(batch, dim)`` decode, ``(batch, dim, seqlen)``
                            verify, or ``(cu_tokens, dim)`` varlen.
    - ``conv_state``:       ``(num_cache_lines, dim, state_len)`` with
                            ``state_len >= width - 1 + (seqlen - 1)`` for verify.
                            Updated **in place**.
    - ``weight``:           ``(dim, width)``.
    - ``bias``:             ``(dim,)`` or ``None``.
    - ``conv_state_indices``: ``(batch,)`` int32 cache line per sequence,
                            ``(batch, num_blocks)`` under APC. Defaults to
                            ``arange(batch)``.
    - ``num_accepted_tokens``: ``(batch,)`` int32; enables the chain speculative
                            rollback (``offset = num_accepted - 1``).
    - ``null_block_id``:    sequences on this cache line are skipped (**0**
                            upstream, not ``-1``); ``None`` disables the check.
    - ``query_start_loc`` / ``max_query_len``: ``(batch + 1,)`` int32 cumulative
                            token counts, turning on the packed layout.
                            ``max_query_len`` is the compile-time budget; a
                            sequence's real count is the successive difference
                            and may be anything from ``0`` up to it.
    - ``block_idx_last_scheduled_token`` / ``initial_state_idx``: ``(batch,)``
                            int32 enabling prefix-cache copy-on-write, which
                            reads the history from one block of
                            ``conv_state_indices[i]`` and writes the rolled
                            window to another, so a shared prefix is copied
                            rather than clobbered.
    - ``out``:              optional, shaped like ``x``; omitted overwrites the
                            input as upstream does.

    Returns an output tensor with the same shape as ``x``.
    """
    if weight.size(1) not in _VLLM_WIDTHS:
        raise NotImplementedError(
            f"causal_conv1d_update_flydsl: width={weight.size(1)} is outside "
            f"vLLM's implemented {list(_VLLM_WIDTHS)}"
        )
    is_varlen = query_start_loc is not None
    if is_varlen:
        if conv_state_indices is None:
            # The only source of batch here: x is packed and query_start_loc may
            # be padded longer than the batch.
            raise ValueError(
                "`conv_state_indices` is required when `query_start_loc` is given."
            )
        if max_query_len <= 0:
            raise ValueError(
                "`max_query_len` must be positive when `query_start_loc` is given,"
                f" got {max_query_len}."
            )
    is_apc = block_idx_last_scheduled_token is not None
    if is_apc and initial_state_idx is None:
        # A null deref upstream, which dereferences it unconditionally once the
        # mode is on; say so instead.
        raise ValueError(
            "`initial_state_idx` is required when `block_idx_last_scheduled_token`"
            " is given."
        )
    silu = _resolve_activation(activation)

    _require_in_scope(x, conv_state, weight, bias, "causal_conv1d_update_flydsl")

    if out is None:
        out = x  # upstream overwrites the input rather than allocating
    else:
        if out.shape != x.shape:
            raise ValueError(
                f"`out` shape {tuple(out.shape)} must match `x` shape {tuple(x.shape)}."
            )
        if out.dtype != x.dtype or out.device != x.device:
            raise ValueError("`out` must have the same dtype and device as `x`.")

    unsqueeze = not is_varlen and x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
        out = out.unsqueeze(-1)
    if is_varlen:
        # x is (cu_tokens, dim): no sequence axis, hence no sequence stride.
        batch = conv_state_indices.shape[0]
        seqlen = int(max_query_len)
        stride_x_tok, stride_x_dim = x.stride()
        stride_o_tok, stride_o_dim = out.stride()
        stride_x_seq = stride_o_seq = 0
        dim = x.shape[1]
    else:
        batch, dim, seqlen = x.shape
        stride_x_seq, stride_x_dim, stride_x_tok = x.stride()
        stride_o_seq, stride_o_dim, stride_o_tok = out.stride()
    _, width = weight.shape
    num_cache_lines, cs_dim, state_len_phys = conv_state.shape

    is_spec = num_accepted_tokens is not None
    state_len_eff = (width - 1 + (seqlen - 1)) if is_spec else (width - 1)

    if validate_data:
        assert dim == weight.size(0)
        assert cs_dim == dim
        assert (
            state_len_phys >= state_len_eff
        ), f"conv_state state_len={state_len_phys} < required {state_len_eff}"
        assert weight.stride(1) == 1

    if conv_state_indices is None:
        conv_state_indices = torch.arange(batch, dtype=torch.int32, device=x.device)

    has_null_block = null_block_id is not None
    null_block_arg = null_block_id if has_null_block else -1

    if channels_per_thread <= 0:
        # Deliberately not _pick_cpt(), unlike the SGLang sibling.
        channels_per_thread = 1
    if block_n <= 0:
        block_n = _pick_block_n(batch, dim, x.device, channels_per_thread)

    # Vectorize the per-channel stores when the token axis is contiguous; an even
    # channel stride is not required.
    cs_vec = bool(conv_state.stride(2) == 1)
    # Never under varlen: the packed layout leaves a channel's tokens dim apart.
    o_vec = bool(stride_o_tok == 1)

    dtype_str = "bf16" if x.dtype == torch.bfloat16 else "fp16"
    launcher = compile_causal_conv1d_update(
        int(width),
        int(seqlen),
        bias is not None,
        bool(silu),
        bool(is_spec),
        bool(has_null_block),
        int(block_n),
        dtype_str,
        bool(weight.stride(1) == 1),
        cs_vec,
        o_vec,
        int(channels_per_thread),
        bool(is_apc),
        bool(is_varlen),
    )
    span = launcher._bn * launcher._cpt
    grid_y_dim = (dim + span - 1) // span

    stride_w_dim, stride_w_width = weight.stride()
    stride_cs_seq, stride_cs_dim, stride_cs_tok = conv_state.stride()
    stride_csi = conv_state_indices.stride(0)

    bias_arg = bias if bias is not None else x  # dummy ptr when HAS_BIAS=False
    nacc_arg = num_accepted_tokens if is_spec else x  # dummy ptr when not spec
    # dummy ptrs when the mode is off; the kernel builds no descriptor for them
    qsl_arg = query_start_loc if is_varlen else x
    blst_arg = block_idx_last_scheduled_token if is_apc else x
    isi_arg = initial_state_idx if is_apc else x

    _run_compiled(
        launcher,
        x.data_ptr(),
        weight.data_ptr(),
        bias_arg.data_ptr(),
        conv_state.data_ptr(),
        conv_state_indices.data_ptr(),
        nacc_arg.data_ptr(),
        qsl_arg.data_ptr(),
        blst_arg.data_ptr(),
        isi_arg.data_ptr(),
        out.data_ptr(),
        int(dim),
        int(num_cache_lines),
        int(null_block_arg),
        int(stride_x_seq),
        int(stride_x_dim),
        int(stride_x_tok),
        int(stride_w_dim),
        int(stride_w_width),
        int(stride_cs_seq),
        int(stride_cs_dim),
        int(stride_cs_tok),
        int(stride_csi),
        int(stride_o_seq),
        int(stride_o_dim),
        int(stride_o_tok),
        int(batch),
        int(grid_y_dim),
        _raw_stream(x.device),
    )

    if unsqueeze:
        out = out.squeeze(-1)
    return out


def _is_dedup_conv_window(window: torch.Tensor, width: int) -> bool:
    """Whether ``window`` is SGLang's overlapping (deduplicated) snapshot view.

    On the linear draft chain the snapshot is an ``as_strided`` view over a
    compact ``(cache_lines, dim, seqlen+width-2)`` buffer whose step axis
    advances by one tap, so consecutive windows alias and writing them one at a
    time stores every element ``width-1`` times. A dense snapshot instead
    advances a whole window, so the strides can only coincide under aliasing.

    ``width > 2`` guards against a single-tap window, which has nothing to
    deduplicate and whose size-1 trailing axis carries an arbitrary stride.
    """
    return width > 2 and window.stride(1) == window.stride(3)


def causal_conv1d_update_sglang_flydsl(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    cache_seqlens: torch.Tensor | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accept_tokens: torch.Tensor | None = None,
    intermediate_conv_window: torch.Tensor | None = None,
    intermediate_state_indices: torch.Tensor | None = None,
    retrieve_next_token: torch.Tensor | None = None,
    retrieve_next_sibling: torch.Tensor | None = None,
    retrieve_parent_token: torch.Tensor | None = None,
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data: bool = False,
    block_n: int = 0,
    channels_per_thread: int = 0,
    out: torch.Tensor | None = None,
):
    """FlyDSL decode / verify causal_conv1d update (SGLang-aligned).

    Drop-in for SGLang's ``causal_conv1d_update`` (same parameter names and
    order), plus trailing FlyDSL ``block_n`` / ``channels_per_thread`` knobs
    where ``0`` means auto, overridable via ``AITER_FLYDSL_CONV1D_BN`` /
    ``AITER_FLYDSL_CONV1D_CPT``.

    - ``x``:                ``(batch, dim)`` or ``(batch, dim, seqlen)``.
    - ``conv_state``:       ``(num_cache_lines, dim, state_len)``, updated in
                            place; ``state_len >= width-1 + (seqlen-1)`` when
                            ``num_accept_tokens`` is given.
    - ``num_accept_tokens``: ``(batch,)`` int32; enables the speculative
                            rollback (``offset = num_accept_tokens - 1``).
    - ``intermediate_conv_window``: ``(cache_lines, seqlen, dim, width-1)``;
                            snapshots each step's window so an accepted prefix
                            can be restored. An overlapping view is detected and
                            written once per element -- see
                            ``_is_dedup_conv_window``.
    - ``retrieve_next_token`` / ``retrieve_next_sibling``: ``(batch, seqlen)``
                            int32 EAGLE tree links; the convolution then walks
                            each token's parent chain and
                            ``retrieve_parent_token`` receives the parent map.

    ``cache_seqlens`` (circular buffer) is not implemented, matching SGLang.

    Returns ``out`` if given, else a fresh tensor shaped like ``x``. ``out`` is a
    FlyDSL extension: SGLang always allocates, but aiter's Triton kernel writes
    over ``x``, so the dispatch seam passes ``out=x`` to keep that contract.
    """
    if cache_seqlens is not None:
        raise NotImplementedError(
            "causal_conv1d_update_sglang_flydsl: cache_seqlens (circular "
            "buffer) is not supported, matching SGLang's Triton kernel"
        )
    if weight.size(1) not in _SGLANG_WIDTHS:
        raise NotImplementedError(
            f"causal_conv1d_update_sglang_flydsl: width={weight.size(1)} is "
            f"outside SGLang's implemented {list(_SGLANG_WIDTHS)}"
        )
    silu = _resolve_activation(activation)

    _require_in_scope(x, conv_state, weight, bias, "causal_conv1d_update_sglang_flydsl")

    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    num_cache_lines, cs_dim, state_len_phys = conv_state.shape

    is_spec = num_accept_tokens is not None
    state_len_eff = (width - 1 + (seqlen - 1)) if is_spec else (width - 1)
    save_inter = intermediate_conv_window is not None
    has_tree = retrieve_next_token is not None
    # Purely internal: same bytes at the same addresses, so the caller keeps
    # passing (and reading back) intermediate_conv_window either way.
    save_stream = save_inter and _is_dedup_conv_window(intermediate_conv_window, width)
    save_window = save_inter and not save_stream

    if validate_data:
        assert dim == weight.size(0)
        assert cs_dim == dim
        assert (
            state_len_phys >= state_len_eff
        ), f"conv_state state_len={state_len_phys} < required {state_len_eff}"
        assert weight.stride(1) == 1
        if save_inter:
            assert intermediate_state_indices is not None
            assert intermediate_conv_window.shape[1:] == (seqlen, dim, width - 1), (
                f"intermediate_conv_window {tuple(intermediate_conv_window.shape)} "
                f"!= (cache_lines, {seqlen}, {dim}, {width - 1})"
            )
        if has_tree:
            assert retrieve_next_sibling is not None
            assert retrieve_parent_token is not None

    if conv_state_indices is None:
        conv_state_indices = torch.arange(batch, dtype=torch.int32, device=x.device)
    if save_inter and intermediate_state_indices is None:
        # aiter's Triton addresses both through conv_state_indices, so reuse it.
        # An arange would silently write the snapshot to other rows whenever
        # conv_state_indices is not itself an arange.
        intermediate_state_indices = conv_state_indices

    if out is None:
        out = torch.empty_like(x)  # SGLang allocates rather than overwriting x
    else:
        # Held to the contract vLLM defines for its own `out`: same shape, dtype
        # and device as the input. x is already unsqueezed by now, so compare
        # against the caller's shape.
        want_shape = x.shape[:-1] if unsqueeze else x.shape
        if out.shape != want_shape:
            raise ValueError(
                f"`out` shape {tuple(out.shape)} must match `x` shape "
                f"{tuple(want_shape)}."
            )
        if out.dtype != x.dtype or out.device != x.device:
            raise ValueError("`out` must have the same dtype and device as `x`.")
        if unsqueeze:
            out = out.unsqueeze(-1)

    has_null_block = pad_slot_id is not None
    null_block_arg = pad_slot_id if has_null_block else -1

    if channels_per_thread <= 0:
        channels_per_thread = _pick_cpt(
            batch, dim, x.device, seqlen=seqlen, env="AITER_FLYDSL_CONV1D_CPT"
        )
    if block_n <= 0:
        block_n = _pick_block_n(
            batch, dim, x.device, channels_per_thread, env="AITER_FLYDSL_CONV1D_BN"
        )

    # As in the vLLM sibling above; i_vec covers the snapshot's W-1 tap slots.
    cs_vec = bool(conv_state.stride(2) == 1)
    o_vec = bool(out.stride(2) == 1)
    i_vec = bool(save_inter and intermediate_conv_window.stride(3) == 1)

    dtype_str = "bf16" if x.dtype == torch.bfloat16 else "fp16"
    launcher = compile_causal_conv1d_update_sglang(
        int(width),
        int(seqlen),
        bias is not None,
        bool(silu),
        bool(is_spec),
        bool(has_null_block),
        int(block_n),
        dtype_str,
        bool(weight.stride(1) == 1),
        cs_vec,
        o_vec,
        i_vec,
        bool(save_window),
        bool(save_stream),
        bool(has_tree),
        int(channels_per_thread),
    )
    span = launcher._bn * launcher._cpt
    grid_y_dim = (dim + span - 1) // span

    stride_x_seq, stride_x_dim, stride_x_tok = x.stride()
    stride_w_dim, stride_w_width = weight.stride()
    stride_cs_seq, stride_cs_dim, stride_cs_tok = conv_state.stride()
    stride_o_seq, stride_o_dim, stride_o_tok = out.stride()
    stride_csi = conv_state_indices.stride(0)

    if save_inter:
        # si_step goes unused on the stream path, which walks the tap axis across
        # the whole run instead of restarting it per step.
        si_seq, si_step, si_dim, si_win = intermediate_conv_window.stride()
        sisi = intermediate_state_indices.stride(0)
    else:
        si_seq = si_step = si_dim = si_win = sisi = 0

    if has_tree:
        srnt_seq, srnt_tok = retrieve_next_token.stride()
        srns_seq, srns_tok = retrieve_next_sibling.stride()
        srpt_seq, srpt_tok = retrieve_parent_token.stride()
    else:
        srnt_seq = srnt_tok = srns_seq = srns_tok = srpt_seq = srpt_tok = 0

    bias_arg = bias if bias is not None else x  # dummy ptr when HAS_BIAS=False
    nacc_arg = num_accept_tokens if is_spec else x
    inter_arg = intermediate_conv_window if save_inter else x
    isi_arg = intermediate_state_indices if save_inter else conv_state_indices
    rnt_arg = retrieve_next_token if has_tree else conv_state_indices
    rns_arg = retrieve_next_sibling if has_tree else conv_state_indices
    rpt_arg = retrieve_parent_token if has_tree else conv_state_indices

    _run_compiled(
        launcher,
        x.data_ptr(),
        weight.data_ptr(),
        bias_arg.data_ptr(),
        conv_state.data_ptr(),
        conv_state_indices.data_ptr(),
        nacc_arg.data_ptr(),
        out.data_ptr(),
        inter_arg.data_ptr(),
        isi_arg.data_ptr(),
        rnt_arg.data_ptr(),
        rns_arg.data_ptr(),
        rpt_arg.data_ptr(),
        int(dim),
        int(num_cache_lines),
        int(null_block_arg),
        int(stride_x_seq),
        int(stride_x_dim),
        int(stride_x_tok),
        int(stride_w_dim),
        int(stride_w_width),
        int(stride_cs_seq),
        int(stride_cs_dim),
        int(stride_cs_tok),
        int(stride_csi),
        int(stride_o_seq),
        int(stride_o_dim),
        int(stride_o_tok),
        int(si_seq),
        int(si_step),
        int(si_dim),
        int(si_win),
        int(sisi),
        int(srnt_seq),
        int(srnt_tok),
        int(srns_seq),
        int(srns_tok),
        int(srpt_seq),
        int(srpt_tok),
        int(batch),
        int(grid_y_dim),
        _raw_stream(x.device),
    )

    if unsqueeze:
        out = out.squeeze(-1)
    return out
