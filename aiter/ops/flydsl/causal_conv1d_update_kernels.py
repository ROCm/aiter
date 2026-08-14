# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL causal-conv1d update host wrappers (decode + speculative verify).

The ``torch``-facing entry points for the two decode-stage conv1d update
kernels. They are the same algorithm behind two upstream interfaces, and both
are maintained -- neither supersedes the other:

* ``causal_conv1d_update_flydsl`` -- vLLM's interface, a drop-in for
  ``aiter.ops.triton.conv.causal_conv1d.causal_conv1d_update``, covering that
  kernel's decode, chain-verify, varlen-packing and prefix-caching modes.
* ``causal_conv1d_update_sglang_flydsl`` -- SGLang's interface, which adds
  per-step ``intermediate_conv_window`` snapshots and an EAGLE tree traversal.

Both kernels are built by ``kernels.causal_conv1d_update``.

Both wrappers prepare tensors, resolve the launch shape, manage the
compiled-kernel cache and hand the current stream to the launcher, keeping the
kernel-compile module ``kernels.causal_conv1d_update`` free of any ``torch``
dependency -- the same split as ``kernels.mla_reduce`` and this module's
counterpart ``mla_reduce_kernels``.

Neither is selected by default. The in-tree caller reaches the SGLang-shaped
wrapper through the opt-in seam in
``aiter.ops.triton.conv.causal_conv1d.causal_conv1d_update``
(``AITER_CONV1D_UPDATE_FLYDSL=1``), which consults the module-private
``_causal_conv1d_update_sglang_flydsl_supported`` and silently keeps Triton for
anything outside the port's scope -- the same shape as the FlyDSL MLA-reduce seam
in ``aiter/mla.py``. External callers (vLLM, SGLang) import a wrapper directly;
the predicates are private because a caller that gets the scope wrong should hit
the wrapper's own ``NotImplementedError`` rather than silently mis-execute.

Launch policy
-------------
Both kernels launch the same grid shape, so one tile/occupancy policy
(``_pick_block_n`` / ``_pick_cpt``) serves both. It assumes CDNA: wavefronts are
64 lanes, which is why the channel tile never goes below 64, as a narrower
workgroup is a partial wave and wastes lanes outright. The CU count is always
read from the live device and never assumed -- the supported parts differ by more
than 3x (MI30X vs MI35X) and the same part reports fewer CUs when partitioned
(CPX/NPS modes) -- so the policy is expressed as a *ratio* to the queried count
and a new part needs no change here.

The policy is deliberately coarse: the tile only has to be wide enough to keep
the GPU busy. A sweep of the three candidates found them indistinguishable above
the machine's noise floor -- the ranking even flipped between sessions -- while
dropping below one wavefront was reproducibly bad. So the rule is "don't
under-occupy, don't go sub-wave" and nothing finer; a lookup table at this
granularity would be fitting noise. That verdict came from a ~20% noise floor on
small grids, so it says what could be resolved rather than that the candidates
are equal. Anyone revisiting it should re-measure with the two arms alternating
inside one process: on these parts a process lands in a fast or slow state at
startup that scales small-grid dispatch by ~1.2x for its whole lifetime, so
separate-process A/B comparisons of a few percent measure the draw, not the
change.

Sentinel warning
----------------
The two upstreams spell the skip sentinel differently and the *values* differ,
not just the names. vLLM main passes ``null_block_id``, whose value is ``0``
because block 0 is the reserved null block; SGLang and the rest of aiter's
conv1d modules pass ``pad_slot_id``, whose value is ``-1``. Each wrapper below
follows its own upstream. Mixing them up does not raise: with
``null_block_id=0``, ``conv_state_indices`` must start at 1, or sequence 0 is
silently skipped and only shows up as wrong numerics.
"""

from __future__ import annotations

import os

import torch

from .kernels.causal_conv1d_update import (
    compile_causal_conv1d_update,
    compile_causal_conv1d_update_sglang,
)
from .kernels.tensor_shim import _run_compiled

#: SGLang's padded-slot sentinel (also aiter's Triton conv1d convention).
PAD_SLOT_ID = -1

#: vLLM's null cache block. Block 0 is reserved, so unlike ``PAD_SLOT_ID`` this
#: sentinel is a *valid* index; see the module docstring.
NULL_BLOCK_ID = 0

#: Element types the kernels are specialized for. Anything else (notably fp32)
#: would be silently reinterpreted as fp16 by the dtype dispatch below.
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)

#: Widths implemented by each kernel. SGLang's Triton kernel only covers 2/3/4,
#: and the port matches it; the vLLM one generalizes.
_VLLM_WIDTHS = range(2, 7)
_SGLANG_WIDTHS = range(2, 5)

#: The launch policy assumes 64-lane wavefronts and the kernels address memory
#: through buffer resources, so the supported parts are the CDNA (gfx9) family.
_SUPPORTED_ARCH_PREFIX = "gfx9"

#: Lanes per wavefront on CDNA. Also the channel-tile floor.
_WAVEFRONT = 64

#: Channel-tile widths to consider, widest first. The narrowest is one full
#: wavefront; see the module docstring for why there is nothing below it.
_BLOCK_N_CANDIDATES = (256, 128, _WAVEFRONT)

#: Largest channels-per-thread the kernels implement.
_CPT_MAX = 2

#: Longest speculative window that still measured a gain from ``_CPT_MAX``. Past
#: it the knob reverses sign, and the EAGLE tree path runs at S=8 by default.
_CPT_MAX_SEQLEN = 4

#: Workgroups per CU the launch aims for. Two gives the scheduler something to
#: swap in while a workgroup is stalled on memory without over-subscribing.
#: A ratio, not a count, so it carries across parts unchanged.
_TARGET_WG_PER_CU = 2

#: ``None`` caches "this device could not be queried".
_CU_COUNT_CACHE: dict[int, int | None] = {}

#: ``None`` caches "this override is not set".
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

    Unset, empty and ``"auto"`` all mean "no override". These overrides exist for
    parameter sweeps, so a bogus value should be loud rather than silently
    ignored -- only the clamp to a positive tile is applied.

    Read once per name and cached: this sits on the per-launch path of a kernel
    that completes in single-digit microseconds, and ``os.environ.get`` costs more
    than the rest of the heuristic put together. The overrides are set before the
    process starts, so changing one mid-process has no effect.
    """
    if name is None:
        return None
    if name not in _ENV_OVERRIDE_CACHE:
        raw = os.environ.get(name, "")
        _ENV_OVERRIDE_CACHE[name] = None if raw in ("", "auto") else max(1, int(raw))
    return _ENV_OVERRIDE_CACHE[name]


def _n_cu(device: torch.device) -> int | None:
    """Compute-unit count of ``device``, cached per device index.

    ``None`` means the device could not be queried (no live GPU, a meta/CPU
    tensor). Callers must degrade rather than substitute a guess: see
    :func:`_target_wg`.
    """
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

    An unknown CU count degrades to "assume the largest machine" rather than to
    an invented mid-range constant, because the two failure directions are not
    symmetric: guessing too high just over-decomposes (extra workgroups queue,
    and the tile candidates measured as indistinguishable in noise), while
    guessing too low under-occupies, which is the one effect that reproducibly
    cost time. Callers spell that as taking the fall-through branch, which is the
    same code path an unreachable target already takes.
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
    """Pick channels-per-thread.

    More channels per thread means more independent loads in flight per wave
    (better latency hiding) but halves the workgroup count. That trade flips
    around batch 16-32: below it the launch is occupancy-bound and the lost
    workgroups cost more than the extra memory-level parallelism buys.

    The crossover therefore tracks the queried CU count instead of sitting at a
    fixed batch: a part with fewer CUs fills up sooner and starts preferring the
    extra memory-level parallelism at a correspondingly smaller batch.

    Occupancy is estimated at the widest span the tile search could pick, so this
    stays consistent with :func:`_pick_block_n`.

    ``seqlen`` gates the whole thing, because the knob's sign turns over with the
    speculative window rather than with the batch: paired A/B measured +4.1% at
    S=4 but -2.4% at S=8, and the same reversal at S=8 on the vLLM-shaped sibling.
    The plausible reading is that a longer window already keeps more live
    registers per thread, so doubling the channels costs occupancy that the extra
    memory-level parallelism no longer repays -- plausible but unverified, it
    needs a VGPR count to confirm. Windows of 5 to 7 were never measured, so they
    take the conservative side of the reversal: giving up an unmeasured gain is
    preferable to a default that may carry an unmeasured regression. The
    ``AITER_FLYDSL_CONV1D_CPT`` override stays the way to re-tune any of this.
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

    Grid size is ``batch * cdiv(dim, BLOCK_N * cpt)``. At large batch a wide tile
    already yields plenty of workgroups and is the most efficient per workgroup.
    At small batch that leaves the GPU under-occupied and memory latency is
    exposed, so a narrower tile spawns more workgroups.

    Returns the widest candidate whose workgroup count reaches the occupancy
    target, else the narrowest. A bigger part raises the target and so shifts the
    choice towards narrower tiles on its own.
    """
    override = _env_int(env)
    if override is not None:
        return override
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
    return arch.split(":")[0].startswith(_SUPPORTED_ARCH_PREFIX)


def _shapes_supported(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    widths: range,
    is_spec: bool,
    is_varlen: bool = False,
    max_query_len: int = -1,
) -> bool:
    """Shape / dtype / placement checks shared by both interfaces."""
    if x.dim() not in (2, 3) or conv_state.dim() != 3 or weight.dim() != 2:
        return False
    if is_varlen and (x.dim() != 2 or max_query_len <= 0):
        return False
    if conv_state.dtype not in _SUPPORTED_DTYPES or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if not (x.is_cuda and conv_state.is_cuda and weight.is_cuda):
        return False
    if x.device != conv_state.device or x.device != weight.device:
        return False

    dim = x.size(1)
    # Packed x carries no sequence axis, so the token budget comes from the caller.
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
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
) -> bool:
    """Whether ``causal_conv1d_update_flydsl`` can serve this problem.

    The kernel covers every mode of vLLM's Triton ``causal_conv1d_update`` --
    decode, chain speculative-verify, varlen packing and Automatic-Prefix-Caching
    copy-on-write -- so this only screens the shapes and dtypes.
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
    )


def _causal_conv1d_update_sglang_flydsl_supported(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    *,
    num_accept_tokens: torch.Tensor | None = None,
    cache_seqlens: torch.Tensor | None = None,
) -> bool:
    """Whether ``causal_conv1d_update_sglang_flydsl`` can serve this problem.

    Covers SGLang's decode, chain-verify, ``SAVE_INTERMEDIATE`` and EAGLE tree
    paths. ``cache_seqlens`` (circular conv_state buffer) is unimplemented here
    exactly as it is in SGLang's own Triton kernel.
    """
    if cache_seqlens is not None:
        return False
    return _shapes_supported(
        x, conv_state, weight, _SGLANG_WIDTHS, num_accept_tokens is not None
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

    Drop-in for vLLM's ``causal_conv1d_update`` (same parameter names and
    order), plus trailing FlyDSL-specific ``block_n`` (workgroup width along the
    channel axis) and ``channels_per_thread`` knobs, both ``0`` = auto. Auto
    ``block_n`` narrows the tile at small batch so more workgroups launch and
    hide memory latency; auto ``channels_per_thread`` is 1, since measurement
    did not support spending it (see the call site). Pass a positive value to
    force either. Covers every mode of the upstream kernel -- decode, chain
    speculative-verify, varlen packing and Automatic-Prefix-Caching
    copy-on-write:

    - ``x``:                ``(batch, dim)`` (single-token decode),
                            ``(batch, dim, seqlen)`` (multi-token / verify) or
                            ``(cu_tokens, dim)`` (varlen packing).
    - ``conv_state``:       ``(num_cache_lines, dim, state_len)``,
                            ``state_len >= width - 1 + (seqlen - 1)`` for verify.
                            Updated **in place** (matches vLLM).
    - ``weight``:           ``(dim, width)``.
    - ``bias``:             ``(dim,)`` or ``None``.
    - ``conv_state_indices``: ``(batch,)`` int32; selects the cache line per
                            sequence. ``(batch, num_blocks)`` under APC. Defaults
                            to ``arange(batch)``.
    - ``num_accepted_tokens``: ``(batch,)`` int32. If given, enables the chain
                            speculative rollback (``offset = num_accepted-1``).
    - ``null_block_id``:    sequences whose cache line equals this id are skipped
                            (null/padding block, **0** upstream, not ``-1``);
                            pass ``None`` to disable the check.
    - ``query_start_loc`` / ``max_query_len``: ``(batch + 1,)`` int32 cumulative
                            token counts, turning on the packed ``(cu_tokens,
                            dim)`` layout. ``max_query_len`` is the compile-time
                            token budget; each sequence's real count is
                            ``query_start_loc[i+1] - query_start_loc[i]`` and may
                            be anything from ``0`` to it.
    - ``block_idx_last_scheduled_token`` / ``initial_state_idx``: ``(batch,)``
                            int32 Automatic-Prefix-Caching copy-on-write. Giving
                            the former turns it on: the history is read from
                            ``conv_state_indices[i, initial_state_idx[i]]`` and
                            the rolled window written to
                            ``conv_state_indices[i, block_idx_last_scheduled_token[i]]``,
                            so a shared prefix block is copied, not clobbered.
    - ``out``:              optional output tensor shaped like ``x``. When
                            omitted the input is overwritten, as upstream does.

    Returns an output tensor with the same shape as ``x``.
    """
    is_varlen = query_start_loc is not None
    if is_varlen:
        if conv_state_indices is None:
            # batch is only recoverable from the index tensor here: x is packed
            # and query_start_loc may be padded longer than the batch.
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
        # Upstream keys the mode off block_idx_last_scheduled_token alone and then
        # dereferences initial_state_idx unconditionally, so this combination is a
        # null deref there; say so instead.
        raise ValueError(
            "`initial_state_idx` is required when `block_idx_last_scheduled_token`"
            " is given."
        )
    silu = _resolve_activation(activation)

    original_dtype = x.dtype
    x = x.to(conv_state.dtype)

    if out is None:
        out = x  # upstream overwrites the input rather than allocating
    else:
        if out.shape != x.shape:
            raise ValueError(
                f"`out` shape {tuple(out.shape)} must match `x` shape {tuple(x.shape)}."
            )
        if out.dtype != original_dtype or out.device != x.device:
            raise ValueError("`out` must have the same dtype and device as `x`.")

    unsqueeze = not is_varlen and x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)  # (batch, dim, 1)
        out = out.unsqueeze(-1)
    if is_varlen:
        # x is (cu_tokens, dim): the sequence axis is gone and the token axis is
        # the outer one, so there is no per-sequence stride to walk.
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
        # Deliberately NOT _pick_cpt(). The kernel supports CPT > 1, but a paired
        # A/B (both arms alternating inside one loop, which is the only way to
        # get a usable ratio on these parts) put it at a wash here: geomean
        # 0.99x for decode, 1.01x at S=2 (the vLLM MTP shape), 1.02x for verify
        # S=4, 0.98x for verify S=8. The SGLang sibling does gain on verify
        # S=4, so the knob stays for re-tuning; this default just does not
        # spend it.
        channels_per_thread = 1
    if block_n <= 0:
        block_n = _pick_block_n(batch, dim, x.device, channels_per_thread)

    # Vectorize the per-channel conv_state / output stores whenever the token
    # axis is contiguous. Odd channel strides leave half the lanes' bases 2-byte
    # (not dword) aligned, but MUBUF buffer stores tolerate unaligned addresses
    # (verified bit-exact), and the measured cost of the split lanes stays below
    # the win from halving the store-instruction count -- e.g. verify S=3
    # (state_len=5, odd) sped up ~7-9% at batch >= 32 and decode (state_len=3)
    # ~7% at batch 128. So an even channel stride is not required.
    cs_vec = bool(conv_state.stride(2) == 1)
    # Under varlen the packed layout puts the channel axis innermost, so one
    # channel's tokens are dim apart and there is no run to vectorize; the
    # per-slot store predication that path needs would rule it out anyway.
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
        x,
        weight,
        bias_arg,
        conv_state,
        conv_state_indices,
        nacc_arg,
        qsl_arg,
        blst_arg,
        isi_arg,
        out,
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
        torch.cuda.current_stream(),
    )

    if unsqueeze:
        out = out.squeeze(-1)
    return out.to(original_dtype)


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
    order), plus trailing FlyDSL-specific ``block_n`` / ``channels_per_thread``
    knobs. Both default to ``0`` (auto-select from the batch); pass a positive
    value to force one, or set ``AITER_FLYDSL_CONV1D_BN`` /
    ``AITER_FLYDSL_CONV1D_CPT``.

    - ``x``:                ``(batch, dim)`` or ``(batch, dim, seqlen)``.
    - ``conv_state``:       ``(num_cache_lines, dim, state_len)``, updated
                            in place; ``state_len >= width-1 + (seqlen-1)``
                            when ``num_accept_tokens`` is given.
    - ``num_accept_tokens``: ``(batch,)`` int32; enables the speculative
                            rollback (``offset = num_accept_tokens - 1``).
    - ``intermediate_conv_window``: ``(cache_lines, seqlen, dim, width-1)``;
                            when given, each step's convolution window is
                            snapshotted so any accepted prefix can be restored.
    - ``retrieve_next_token`` / ``retrieve_next_sibling``: ``(batch, seqlen)``
                            int32 EAGLE tree links; when given, the
                            convolution walks each token's parent chain and
                            ``retrieve_parent_token`` receives the parent map.

    ``cache_seqlens`` (circular buffer) is not implemented, matching SGLang.

    Returns ``out`` if given, else a freshly allocated tensor shaped like ``x``.
    ``out`` is a trailing FlyDSL extension: SGLang always allocates, but aiter's
    Triton ``causal_conv1d_update`` writes the output over ``x``, so its dispatch
    seam passes ``out=x`` to keep that in-place contract.
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

    original_dtype = x.dtype
    x = x.to(conv_state.dtype)

    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)  # (batch, dim, 1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    num_cache_lines, cs_dim, state_len_phys = conv_state.shape

    is_spec = num_accept_tokens is not None
    state_len_eff = (width - 1 + (seqlen - 1)) if is_spec else (width - 1)
    save_inter = intermediate_conv_window is not None
    has_tree = retrieve_next_token is not None

    if validate_data:
        assert dim == weight.size(0)
        assert cs_dim == dim
        assert (
            state_len_phys >= state_len_eff
        ), f"conv_state state_len={state_len_phys} < required {state_len_eff}"
        assert weight.stride(1) == 1
        if save_inter:
            assert intermediate_state_indices is not None
        if has_tree:
            assert retrieve_next_sibling is not None
            assert retrieve_parent_token is not None

    if conv_state_indices is None:
        conv_state_indices = torch.arange(batch, dtype=torch.int32, device=x.device)
    if save_inter and intermediate_state_indices is None:
        intermediate_state_indices = torch.arange(
            batch, dtype=torch.int32, device=x.device
        )

    if out is None:
        out = torch.empty_like(x)  # SGLang allocates rather than overwriting x
    else:
        # SGLang has no `out` parameter, so hold this extension to the contract
        # vLLM defines for its own: same shape, dtype and device as the input.
        # x has already been cast to the conv_state dtype, which is what the
        # kernel writes, so that is what the buffer has to be.
        # x is already unsqueezed here, so compare against the caller's shape.
        want_shape = x.shape[:-1] if unsqueeze else x.shape
        if out.shape != want_shape:
            raise ValueError(
                f"`out` shape {tuple(out.shape)} must match `x` shape "
                f"{tuple(want_shape)}."
            )
        if out.dtype != x.dtype or out.device != x.device:
            raise ValueError(
                "`out` must have the conv_state dtype and the device of `x`."
            )
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

    # Vectorize the per-channel conv_state / output stores whenever the token
    # axis is contiguous. Odd channel strides misalign half the lanes but MUBUF
    # tolerates that and the win from halving the store count dominates; see the
    # vLLM sibling above for the measured numbers.
    cs_vec = bool(conv_state.stride(2) == 1)
    o_vec = bool(out.stride(2) == 1)

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
        bool(save_inter),
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
        x,
        weight,
        bias_arg,
        conv_state,
        conv_state_indices,
        nacc_arg,
        out,
        inter_arg,
        isi_arg,
        rnt_arg,
        rns_arg,
        rpt_arg,
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
        torch.cuda.current_stream(),
    )

    if unsqueeze:
        out = out.squeeze(-1)
    return out.to(original_dtype)
