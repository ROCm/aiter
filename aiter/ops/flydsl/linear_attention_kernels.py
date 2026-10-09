# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL Linear Attention APIs."""

from __future__ import annotations

import collections
import functools

import torch
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.triton.utils.device_info import get_num_sms

from .kernels.gdr_decode import (
    MTP_MODE_CHAIN,
    MTP_MODE_SNAPSHOT,
    create_vk_gdr_decode_kernel,
    create_vk_gdr_mtp_kernel,
)
from .kernels.tensor_shim import _run_compiled, get_dtype_str

__all__ = [
    "flydsl_gdr_decode",
    "flydsl_gdr_mtp",
    "flydsl_gdr_mtp_sglang",
]


# _MTP_BY_ARCH is keyed on the base name; the hardware path already drops the
# feature suffix, but an arch set by hand through the environment keeps it.
GDR_GPU_ARCH = get_rocm_arch().split(":")[0]


def _mtp_variant(mode, has_tree):
    """Which contract a launch is running; the rung branches on it."""
    return f"{mode}_tree" if has_tree else mode


def _tile_warps(tile_v, warp_threads_v, max_warps=4):
    """Most warps up to ``max_warps`` whose group tiles ``tile_v``, or None.

    The group has to divide the tile exactly, and a head width that is a
    multiple of 32 without being a power of two has splits where no warp count
    does.
    """
    for num_warps in range(min(max_warps, tile_v // warp_threads_v), 0, -1):
        if tile_v % (num_warps * warp_threads_v) == 0:
            return num_warps
    return None


# Decode launches one block per (batch, v-head) pair, so a short batch leaves
# most of the part idle and splitting the value dimension is the only way to
# make more blocks. Each split adds a reduction across the blocks that share a
# head, so it stops paying once the grid covers the machine: these are the
# grids to split up to, in blocks per CU. The fp32 state moves twice the bytes
# per block and so tolerates one split more.
_DECODE_GRID_PER_CU = 1
_DECODE_GRID_PER_CU_F32_STATE = 2

# Splits worth considering, largest first. Not every power of two earns a
# place: with an fp32 state the 32-wide value tile is beaten at every grid by
# either the 16-wide tile above it or the 64-wide one below, so that ladder
# steps straight from 8 to 2 and never lands on it.
_DECODE_V_SPLITS = (8, 4, 2, 1)
_DECODE_V_SPLITS_F32_STATE = (8, 2, 1)

# Warp shape for the value tile a split leaves, as (NUM_WARPS,
# WARP_THREADS_K), keyed on (tile width, fp32 state). A tile of 64 or wider is
# covered best by four warps over a narrow K group; from 32 down there is no
# longer enough value width to spread four warps across, and the coverage has
# to be bought from a wider K group instead, at the price of one more stage in
# the cross-lane reduction.
_DECODE_WARP_SHAPE = {
    (128, False): (4, 8),
    (128, True): (4, 8),
    (64, False): (4, 4),
    (64, True): (4, 8),
    (32, False): (4, 8),
    (32, True): (4, 16),
    (16, False): (2, 8),
    (16, True): (4, 16),
}


def _decode_warp_shape(head_k_dim, tile_v, f32_state):
    """The warp shape for a value tile: the measured entry, or a derived one.

    A K group of ``WARP_THREADS_K`` lanes reads ``values_per_thread_k`` values
    each, so it only divides some head widths. The measured entry is used where
    it divides, and the rest fall back to the widest group that fits.
    """
    values_per_thread_k = 4 if f32_state else 8
    shape = _DECODE_WARP_SHAPE.get((tile_v, f32_state))
    if shape is not None:
        num_warps, warp_threads_k = shape
        if head_k_dim % (warp_threads_k * values_per_thread_k) == 0 and (
            tile_v % (num_warps * (64 // warp_threads_k)) == 0
        ):
            return shape
    best = None
    for warp_threads_k in (8, 16, 4):
        if head_k_dim % (warp_threads_k * values_per_thread_k):
            continue
        num_warps = _tile_warps(tile_v, 64 // warp_threads_k)
        if num_warps is not None and (best is None or num_warps > best[0]):
            best = (num_warps, warp_threads_k)
    return best


def _decode_tiling(batch_size, num_v_heads, head_k_dim, head_v_dim, state_dtype_str):
    """Split the value dimension until the grid covers the machine.

    The split is a function of ``batch_size * num_v_heads`` alone -- the grid
    the launch would have without one -- and the warp shape then follows from
    the value tile the split leaves. Neither depends on the head counts beyond
    that product.
    """
    f32_state = state_dtype_str == "torch.float32"
    target = get_num_sms() * (
        _DECODE_GRID_PER_CU_F32_STATE if f32_state else _DECODE_GRID_PER_CU
    )
    splits = _DECODE_V_SPLITS_F32_STATE if f32_state else _DECODE_V_SPLITS
    for num_blocks in splits:
        if head_v_dim % num_blocks or batch_size * num_v_heads * num_blocks > target:
            continue
        shape = _decode_warp_shape(head_k_dim, head_v_dim // num_blocks, f32_state)
        if shape is not None:
            return {
                "NUM_BLOCKS_PER_V_DIM": num_blocks,
                "NUM_WARPS": shape[0],
                "WARP_THREADS_K": shape[1],
            }
    # Either the grid already covers the machine or nothing tiles the head; hand
    # back the old default and let the builder be the one to refuse it, with its
    # own message.
    return {"NUM_BLOCKS_PER_V_DIM": 1, "NUM_WARPS": 4, "WARP_THREADS_K": 8}


def get_default_kwargs(
    dtype_str,
    state_dtype_str,
    batch_size,
    seq_length,
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
):
    """The decode tiling for this launch.

    Takes the whole launch shape, not just the part the rule reads, so a caller
    does not have to know which part that is.
    """
    return _decode_tiling(
        batch_size, num_v_heads, head_k_dim, head_v_dim, state_dtype_str
    )


_MTP_WARPS = 4
# Tilings, as (blocks, warps, K group, waves per EU). The last spends a block on
# a single warp over a quarter of the value dimension, thin enough that it needs
# the occupancy hint with it.
_MTP_SPLIT_8 = (8, _MTP_WARPS, 16, 0)
_MTP_SPLIT_4 = (4, _MTP_WARPS, 8, 0)
_MTP_SPLIT_2 = (2, _MTP_WARPS, 8, 0)
_MTP_WHOLE = (1, _MTP_WARPS, 8, 0)
_MTP_WIDE_K = (2, _MTP_WARPS, 16, 0)
_MTP_WIDE_4 = (4, _MTP_WARPS, 16, 0)
_MTP_THIN = (4, 1, 8, 3)
# Coverage past which the splits are close and a block's own shape decides.
_MTP_SATURATED = 8

# What the ladder takes from the part it runs on, and all it takes:
#
# ``fill``      the coverage each rung holds up to, in blocks per CU.
# ``covered``   the tiling per contract once the grid is covered; a contract
#               not named here wants the thin block, occupancy hint included.
# ``long_grid`` the contracts that keep that tiling once the grid is long,
#               instead of turning on the draft length.
# ``thin``      the thin block on a long grid, hint included or not.
_MtpArch = collections.namedtuple("_MtpArch", "fill covered long_grid thin")
_MTP_DEFAULT = _MtpArch(
    fill=((0.75, _MTP_SPLIT_4), (1, _MTP_SPLIT_2), (2, _MTP_WHOLE)),
    covered={MTP_MODE_SNAPSHOT: _MTP_SPLIT_2},
    long_grid={},
    thin=_MTP_THIN,
)
_MTP_BY_ARCH = {
    "gfx950": _MtpArch(
        fill=(
            (0.5, _MTP_SPLIT_8),
            (0.75, _MTP_SPLIT_4),
            (2, _MTP_SPLIT_2),
            (4, _MTP_WHOLE),
        ),
        covered={MTP_MODE_SNAPSHOT: _MTP_SPLIT_2, MTP_MODE_CHAIN: _MTP_WIDE_4},
        long_grid={MTP_MODE_CHAIN: _MTP_WIDE_4},
        thin=(4, 1, 8, 0),
    ),
}
# Longest draft that is still a single pair.
_MTP_PAIR = 2
# Tried in order when the head dims do not divide into the chosen tiling.
_MTP_FALLBACK = (_MTP_SPLIT_4, _MTP_SPLIT_2, _MTP_WHOLE)


def _mtp_rung(grid, seq_length, variant, num_sms, arch):
    """The tiling for the grid this launch would have.

    Verify runs at the batch that has draft tokens outstanding, so the grid
    leaves most of the part idle and splitting the value dimension is what
    fills it. Filling stops paying once the grid covers the part and reverses
    past it, which is why the ladder comes back down to one block.

    Past coverage what is left is how a block spends itself, and that turns on
    the contract -- the tree reads a parent for every token and vLLM's chain
    rolls the state back by the accepted count, so neither spends a block the
    way SGLang's chain does -- then, where the contract does not settle it, on
    the draft length.
    """
    part = _MTP_BY_ARCH.get(arch, _MTP_DEFAULT)
    for coverage, tiling in part.fill:
        if grid <= coverage * num_sms:
            return tiling
    if grid <= _MTP_SATURATED * num_sms:
        return part.covered.get(variant, _MTP_THIN)
    if variant in part.long_grid:
        return part.long_grid[variant]
    return _MTP_WIDE_K if seq_length <= _MTP_PAIR else part.thin


def _mtp_shape(head_k_dim, head_v_dim, state_dtype, num_blocks, warps, warp_threads_k):
    """One tiling, or None where the head dims do not divide into it.

    ``NUM_BLOCKS_PER_V_DIM`` and ``NUM_WARPS`` are not independent: a warp group
    covers ``NUM_WARPS * (64 // WARP_THREADS_K)`` of a value tile that is
    ``head_v_dim // NUM_BLOCKS_PER_V_DIM`` wide, so their product has to divide
    the tile. Splitting therefore costs warps, which this gives back, so every
    config it returns is one the builder accepts.
    """
    values_per_thread_k = 4 if state_dtype == torch.float32 else 8
    if head_k_dim % (warp_threads_k * values_per_thread_k) or head_v_dim % num_blocks:
        return None
    num_warps = _tile_warps(head_v_dim // num_blocks, 64 // warp_threads_k, warps)
    if num_warps is None:
        return None
    return {
        "NUM_BLOCKS_PER_V_DIM": num_blocks,
        "NUM_WARPS": num_warps,
        "WARP_THREADS_K": warp_threads_k,
    }


def _mtp_tiling(
    batch_size,
    num_v_heads,
    seq_length,
    head_k_dim,
    head_v_dim,
    state_dtype,
    variant,
    num_sms,
    arch,
):
    """The rung the launch lands on, dropped to one the head dims admit."""
    rung = _mtp_rung(batch_size * num_v_heads, seq_length, variant, num_sms, arch)
    for num_blocks, warps, warp_threads_k, waves_per_eu in (rung, *_MTP_FALLBACK):
        d = _mtp_shape(
            head_k_dim, head_v_dim, state_dtype, num_blocks, warps, warp_threads_k
        )
        if d is not None:
            if waves_per_eu:
                d["WAVES_PER_EU"] = waves_per_eu
            return d
    return None


def get_mtp_default_kwargs(*args):
    """Wrapper returning a fresh dict, so no caller can edit the cached one."""
    return dict(_mtp_kwargs(*args))


@functools.lru_cache(maxsize=1024)
def _mtp_kwargs(
    dtype_str,
    state_dtype_str,
    state_dtype,
    batch_size,
    seq_length,
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
    variant,
):
    """Pick a tiling for the MTP kernel.

    The decode default does not split the value dimension at all, which is right
    for it: decode is called at the batch a serving step accumulates, so
    ``batch * num_v_heads`` already covers the machine. Verify is called at the
    batch that has draft tokens outstanding, which is small by construction, and
    the same default would then launch ``num_v_heads`` blocks onto a part with
    hundreds of CUs.
    """
    d = _mtp_tiling(
        batch_size,
        num_v_heads,
        seq_length,
        head_k_dim,
        head_v_dim,
        state_dtype,
        variant,
        get_num_sms(),
        GDR_GPU_ARCH,
    )
    if d is None:
        # No tiling fits; hand back the decode default and let the builder be
        # the one to refuse it, with its own message.
        d = {"NUM_BLOCKS_PER_V_DIM": 1, "NUM_WARPS": 4, "WARP_THREADS_K": 8}
    return d


def flydsl_gdr_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    indices: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    use_qk_l2norm: bool,
    need_shuffle_state: bool,
    stream: torch.cuda.Stream = None,
    read_indices: torch.Tensor | None = None,
    write_indices: torch.Tensor | None = None,
):
    if stream is None:
        stream = torch.cuda.current_stream()
    device = query.device
    dtype = query.dtype
    read_indices = indices if read_indices is None else read_indices
    write_indices = indices if write_indices is None else write_indices
    for input in [
        query,
        key,
        value,
        a,
        b,
        dt_bias,
        A_log,
        read_indices,
        write_indices,
        out,
    ]:
        assert input.device == device
    assert state.data_ptr() % 16 == 0
    for input in [key, value, a, b, dt_bias, out]:
        assert input.dtype == dtype
    assert state.dtype in [torch.float, torch.bfloat16]
    assert A_log.dtype in [torch.float, torch.bfloat16]
    assert read_indices.dtype == torch.int32
    assert write_indices.dtype == torch.int32
    if query.stride(-1) != 1:
        raise ValueError(
            "`query` must have a contiguous last dimension for vectorized loads; "
            f"got stride {query.stride()}."
        )
    if key.stride(-1) != 1:
        raise ValueError(
            "`key` must have a contiguous last dimension for vectorized loads; "
            f"got stride {key.stride()}."
        )

    if need_shuffle_state:
        state_ = state.permute(0, 1, 3, 2).contiguous()
    else:
        state_ = state
    batch_size, seq_length, num_k_heads, head_k_dim = query.shape
    num_v_heads = value.shape[-2]
    head_v_dim = value.shape[-1]
    # The tiling reads the CU count off the current device. One fastmath setting
    # for every float op the body traces, rather than a flag per call site; the
    # jit compiles on first call, not on build, so it has to still be in scope
    # at the launch.
    with CompilationContext.compile_hints({"fastmath": "fast"}), torch.cuda.device(
        query.device.index
    ):
        kwargs_ = get_default_kwargs(
            str(dtype),
            str(state_.dtype),
            batch_size,
            seq_length,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
        )
        exe = create_vk_gdr_decode_kernel(
            get_dtype_str(query.dtype),
            get_dtype_str(A_log.dtype),
            get_dtype_str(state_.dtype),
            seq_length,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            query.stride(),
            key.stride(),
            value.stride(),
            state_.stride(),
            a.stride(),
            b.stride(),
            use_qk_l2norm,
            **kwargs_,
        )
        _run_compiled(
            exe,
            query,
            key,
            value,
            a,
            b,
            dt_bias.contiguous(),
            A_log.contiguous(),
            read_indices.contiguous(),
            write_indices.contiguous(),
            state_,
            out,
            batch_size,
            stream,
        )
    if need_shuffle_state:
        state_ = state_.permute(0, 1, 3, 2).contiguous()
        state.copy_(state_)


# Consumers may safely allocate ``out`` with ``torch.empty``: the fused kernel
# writes positive zero for every negative-index graph-padding row.
flydsl_gdr_decode.zeroes_invalid_output = True


_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)
_SUPPORTED_STATE_DTYPES = (torch.float32, torch.bfloat16)
_SUPPORTED_ARCHS = ("gfx942", "gfx950")


@functools.cache
def _is_supported_arch_index(index: int) -> bool:
    try:
        arch = str(torch.cuda.get_device_properties(index).gcnArchName)
    except Exception:  # noqa: BLE001 - no live device, meta/CPU tensor
        return False
    return arch.split(":")[0] in _SUPPORTED_ARCHS


def _is_supported_arch(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _is_supported_arch_index(index)


def _unit_strided(t: torch.Tensor) -> bool:
    """Whether FlyDSL can wrap ``t`` as a memref.

    It needs one axis it can call the fastest-moving one, and a length-1 tensor
    sliced out of a wider row has none even though torch calls it contiguous.
    The index vectors are where that arises, since a caller reaches for a
    column of its slot map.
    """
    return any(s == 1 for s in t.stride())


def _on_device(device: torch.device, *tensors: torch.Tensor | None) -> bool:
    """Whether every tensor given sits on ``device``.

    The launch hands the index vectors to the kernel as device addresses, and
    nothing downstream checks them: the operand assertions cover the value
    tensors only. A host tensor is then not a slow path but an invalid one, and
    these are the arguments a caller most easily leaves on the host.
    """
    return all(t.device == device for t in tensors if t is not None)


def _mtp_shapes_supported(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: torch.Tensor,
) -> bool:
    """Shape / dtype / placement screening shared by both MTP interfaces.

    The tiling the kernel builds assumes a head dimension it can cover with
    whole 16-byte vectors, and the state's K axis is what those vectors walk, so
    a state whose K is not the fastest-moving axis cannot be served at all.
    """
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4 or state.dim() != 4:
        return False
    if query.dtype not in _SUPPORTED_DTYPES:
        return False
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    if state.dtype not in _SUPPORTED_STATE_DTYPES:
        return False
    if not (query.is_cuda and state.is_cuda):
        return False
    if not _on_device(query.device, key, value, state):
        return False
    if query.stride(-1) != 1 or key.stride(-1) != 1:
        return False
    # state is [pool, HV, V, K]; the kernel vector-loads along K.
    if state.stride(-1) != 1:
        return False

    head_k_dim = query.shape[-1]
    head_v_dim = value.shape[-1]
    num_k_heads = query.shape[-2]
    num_v_heads = value.shape[-2]
    if num_v_heads % num_k_heads != 0:
        return False
    if state.shape[1] != num_v_heads or state.shape[2] != head_v_dim:
        return False
    if state.shape[3] != head_k_dim:
        return False

    # A 16-byte vector holds 4 fp32 or 8 bf16 state elements, and the default
    # 8-lane K split has to tile head_k_dim with whole vectors.
    values_per_thread_k = 4 if state.dtype == torch.float32 else 8
    if head_k_dim % (8 * values_per_thread_k) != 0:
        return False
    if head_v_dim % 32 != 0:
        return False
    return _is_supported_arch(query.device)


def _flydsl_gdr_mtp_supported(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: torch.Tensor,
    ssm_state_indices: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
) -> bool:
    """Whether ``flydsl_gdr_mtp`` can serve this problem.

    The chain interface needs both halves of vLLM's contract: a 2-D
    ``[batch, token]`` slot map to checkpoint into, and the accepted-token count
    that says which of those slots to roll back to.
    """
    if ssm_state_indices is None or num_accepted_tokens is None:
        return False
    if ssm_state_indices.dim() != 2:
        return False
    if (
        ssm_state_indices.shape[0] != query.shape[0]
        or ssm_state_indices.shape[1] < query.shape[1]
    ):
        return False
    if ssm_state_indices.dtype != torch.int32:
        return False
    if (
        num_accepted_tokens.dim() != 1
        or num_accepted_tokens.shape[0] != query.shape[0]
        or num_accepted_tokens.dtype != torch.int32
    ):
        return False
    if not _unit_strided(ssm_state_indices) or not _unit_strided(num_accepted_tokens):
        return False
    if not _on_device(query.device, ssm_state_indices, num_accepted_tokens):
        return False
    return _mtp_shapes_supported(query, key, value, state)


def _flydsl_gdr_mtp_sglang_supported(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: torch.Tensor,
    initial_state_indices: torch.Tensor | None,
    intermediate_states_buffer: torch.Tensor | None,
    intermediate_state_indices: torch.Tensor | None,
    retrieve_parent_token: torch.Tensor | None,
) -> bool:
    """Whether ``flydsl_gdr_mtp_sglang`` can serve this problem.

    The tree needs somewhere to read parents from, so a parent map without a
    snapshot buffer is unsupported.
    """
    if initial_state_indices is None or initial_state_indices.dim() != 1:
        return False
    if initial_state_indices.dtype != torch.int32:
        return False
    if not _unit_strided(initial_state_indices):
        return False
    if (intermediate_states_buffer is None) != (intermediate_state_indices is None):
        return False
    if intermediate_states_buffer is not None:
        if intermediate_states_buffer.dim() != 5:
            return False
        if intermediate_states_buffer.stride(-1) != 1:
            return False
        if intermediate_states_buffer.dtype not in _SUPPORTED_STATE_DTYPES:
            return False
        if intermediate_states_buffer.dtype.itemsize > state.dtype.itemsize:
            return False
        if intermediate_states_buffer.shape[1] < query.shape[1]:
            return False
        if intermediate_state_indices.dtype != torch.int32:
            return False
        if not _unit_strided(intermediate_state_indices):
            return False
    if retrieve_parent_token is not None:
        if intermediate_states_buffer is None:
            return False
        if retrieve_parent_token.dim() != 2:
            return False
        if retrieve_parent_token.dtype != torch.int32:
            return False
        if retrieve_parent_token.shape[1] < query.shape[1]:
            return False
        if not _unit_strided(retrieve_parent_token):
            return False
    if not _on_device(
        query.device,
        initial_state_indices,
        intermediate_states_buffer,
        intermediate_state_indices,
        retrieve_parent_token,
    ):
        return False
    return _mtp_shapes_supported(query, key, value, state)


def _snapshot_store_bytes(state_dtype, inter_dtype) -> int:
    """Width of one thread's snapshot store."""
    values_per_thread_k = 4 if state_dtype == torch.float32 else 8
    return values_per_thread_k * inter_dtype.itemsize


def _require_index(name, t, query, *, dims=None, layout="", covers=False):
    """Refuse an index operand, which the kernel reads as a bare device address."""
    if t.device != query.device:
        raise ValueError(
            f"`{name}` must sit on `query`'s device {query.device}; got {t.device}."
        )
    if t.dtype != torch.int32:
        raise ValueError(f"`{name}` must be int32; got {t.dtype}.")
    if dims is not None and t.dim() != dims:
        want = f"{dims}-D {layout}".rstrip()
        raise ValueError(f"`{name}` must be {want}; got shape {tuple(t.shape)}.")
    if t.shape[0] != query.shape[0]:
        raise ValueError(
            f"`{name}` must carry one row per sequence ({query.shape[0]}); got "
            f"shape {tuple(t.shape)}."
        )
    if covers and t.shape[1] < query.shape[1]:
        raise ValueError(
            f"`{name}` must reach every one of the {query.shape[1]} draft tokens; "
            f"got shape {tuple(t.shape)}."
        )


def _mtp_common_checks(query, key, value, a, b, dt_bias, A_log, state, out):
    """Refuse the operands the kernel would otherwise reinterpret.

    Raised rather than asserted, here and for the index operands: ``python -O``
    strips the statement form and takes the guard with it.
    """
    device = query.device
    dtype = query.dtype
    if not query.is_cuda:
        raise ValueError(f"`query` must be on a GPU; got {device}.")
    operands = {
        "key": key,
        "value": value,
        "a": a,
        "b": b,
        "dt_bias": dt_bias,
        "A_log": A_log,
        "state": state,
        "out": out,
    }
    for name, t in operands.items():
        if t.device != device:
            raise ValueError(
                f"every MTP operand must sit on one device; `{name}` is on "
                f"{t.device} and `query` on {device}."
            )
    for name in ("key", "value", "a", "b", "dt_bias", "out"):
        if operands[name].dtype != dtype:
            raise ValueError(
                f"`{name}` must carry `query`'s dtype {dtype}; got "
                f"{operands[name].dtype}."
            )
    if state.dtype not in _SUPPORTED_STATE_DTYPES:
        raise ValueError(
            f"`state` dtype must be one of {list(_SUPPORTED_STATE_DTYPES)}; got "
            f"{state.dtype}."
        )
    if A_log.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            f"`A_log` dtype must be float32 or bfloat16; got {A_log.dtype}."
        )
    if state.data_ptr() % 16 != 0:
        raise ValueError(
            "`state` must be 16-byte aligned for vectorized access; got address "
            f"{state.data_ptr():#x}."
        )
    if query.stride(-1) != 1:
        raise ValueError(
            "`query` must have a contiguous last dimension for vectorized loads; "
            f"got stride {query.stride()}."
        )
    if key.stride(-1) != 1:
        raise ValueError(
            "`key` must have a contiguous last dimension for vectorized loads; "
            f"got stride {key.stride()}."
        )
    if state.stride(-1) != 1:
        raise ValueError(
            "`state` must be [pool, HV, V, K] with K contiguous; got stride "
            f"{state.stride()}. Shuffling it here would copy the whole pool, "
            "which costs more than the kernel it feeds."
        )
    expected = (*query.shape[:2], value.shape[-2], value.shape[-1])
    if out.shape != expected or not out.is_contiguous():
        raise ValueError(f"`out` must be contiguous with shape {expected}.")


def _mtp_stream(query, stream):
    stream = torch.cuda.current_stream(query.device) if stream is None else stream
    if stream.device != query.device:
        raise ValueError(f"`stream` must be on {query.device}; got {stream.device}.")
    return stream


def _mtp_launch(
    *,
    mode,
    query,
    key,
    value,
    a,
    b,
    dt_bias,
    A_log,
    state,
    out,
    state_indices,
    num_accepted,
    inter_indices,
    parent_tokens,
    inter_buffer,
    use_qk_l2norm,
    min_live_slot,
    has_tree,
    disable_state_update,
    stream,
):
    batch_size, seq_length, num_k_heads, head_k_dim = query.shape
    num_v_heads = value.shape[-2]
    head_v_dim = value.shape[-1]

    # Unused operands are handed an existing tensor: the const_expr guards mean
    # the kernel never builds a descriptor for them, but the launch still needs
    # a valid address in the slot.
    filler = state_indices
    inter_strides = tuple(inter_buffer.stride()) if inter_buffer is not None else ()
    parent_strides = tuple(parent_tokens.stride()) if parent_tokens is not None else ()

    # The ladder reads the CU count off the current device.
    with torch.cuda.device(query.device.index):
        kwargs_ = get_mtp_default_kwargs(
            str(query.dtype),
            str(state.dtype),
            state.dtype,
            batch_size,
            seq_length,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            _mtp_variant(mode, has_tree),
        )

        # One setting for every float op the body traces, rather than a flag per
        # call site.
        build_hints = {"fastmath": "fast"}
        # Rides with the one tiling that wants it rather than being derived here.
        waves_per_eu = kwargs_.get("WAVES_PER_EU", 0)
        if waves_per_eu:
            build_hints["waves_per_eu"] = waves_per_eu

        # The jit compiles on first call, not on build, so the hint has to still
        # be in scope at the launch.
        with CompilationContext.compile_hints(build_hints):
            exe = create_vk_gdr_mtp_kernel(
                get_dtype_str(query.dtype),
                get_dtype_str(A_log.dtype),
                get_dtype_str(state.dtype),
                (
                    get_dtype_str(inter_buffer.dtype)
                    if inter_buffer is not None
                    else "f32"
                ),
                seq_length,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                query.stride(),
                key.stride(),
                value.stride(),
                state.stride(),
                a.stride(),
                b.stride(),
                tuple(state_indices.stride())
                + ((1,) if state_indices.dim() == 1 else ()),
                inter_strides,
                parent_strides,
                use_qk_l2norm,
                mode,
                has_tree,
                disable_state_update,
                min_live_slot=min_live_slot,
                **kwargs_,
            )

            _run_compiled(
                exe,
                query,
                key,
                value,
                a,
                b,
                dt_bias.contiguous(),
                A_log.contiguous(),
                state_indices,
                num_accepted if num_accepted is not None else filler,
                inter_indices if inter_indices is not None else filler,
                parent_tokens if parent_tokens is not None else filler,
                state,
                inter_buffer if inter_buffer is not None else state,
                out,
                batch_size,
                stream,
            )


def _launch_flydsl_gdr_mtp(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    use_qk_l2norm: bool = False,
    stream: torch.cuda.Stream = None,
    min_live_slot: int = 1,
):
    stream = _mtp_stream(query, stream)
    _mtp_launch(
        mode=MTP_MODE_CHAIN,
        query=query,
        key=key,
        value=value,
        a=a,
        b=b,
        dt_bias=dt_bias,
        A_log=A_log,
        state=state,
        out=out,
        state_indices=ssm_state_indices.contiguous(),
        num_accepted=num_accepted_tokens.contiguous(),
        inter_indices=None,
        parent_tokens=None,
        inter_buffer=None,
        use_qk_l2norm=use_qk_l2norm,
        min_live_slot=min_live_slot,
        has_tree=False,
        disable_state_update=False,
        stream=stream,
    )


def flydsl_gdr_mtp(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    use_qk_l2norm: bool = False,
    stream: torch.cuda.Stream = None,
    min_live_slot: int = 1,
):
    """Gated delta rule over a linear draft chain, vLLM's MTP contract.

    Rolls back to ``ssm_state_indices[n, num_accepted_tokens[n] - 1]`` and
    checkpoints each token into ``ssm_state_indices[n, t]``, so a later
    rejection has a slot per draft position to resume from. ``state`` is both
    the initial and the final store, as it is upstream.

    Slot 0 is vLLM's null block and a negative slot is the sentinel aiter's
    Triton kernel and SGLang pass instead; a sequence whose rollback slot is
    either is skipped entirely, so slot 0 must not be handed out as a live
    slot.
    """
    _mtp_common_checks(query, key, value, a, b, dt_bias, A_log, state, out)
    _require_index(
        "ssm_state_indices",
        ssm_state_indices,
        query,
        dims=2,
        layout="[batch, token]",
        covers=True,
    )
    _require_index("num_accepted_tokens", num_accepted_tokens, query)
    if min_live_slot not in (0, 1):
        raise ValueError(f"`min_live_slot` must be 0 or 1; got {min_live_slot}.")
    _launch_flydsl_gdr_mtp(
        query,
        key,
        value,
        a,
        b,
        dt_bias,
        A_log,
        state,
        out,
        ssm_state_indices,
        num_accepted_tokens,
        use_qk_l2norm,
        stream,
        min_live_slot,
    )


def flydsl_gdr_mtp_sglang(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    initial_state_indices: torch.Tensor,
    intermediate_states_buffer: torch.Tensor | None = None,
    intermediate_state_indices: torch.Tensor | None = None,
    retrieve_parent_token: torch.Tensor | None = None,
    disable_state_update: bool = False,
    use_qk_l2norm: bool = False,
    stream: torch.cuda.Stream = None,
):
    """Gated delta rule over a draft tree, SGLang's MTP contract.

    The sequence keeps one pool slot, ``initial_state_indices[n]``, and the
    per-token record lives in ``intermediate_states_buffer`` instead. With
    ``retrieve_parent_token`` the draft is an EAGLE tree and each token restarts
    from its parent's snapshot; without it the tokens run in a chain, which is
    the same computation as a parent map of ``t - 1``. ``disable_state_update``
    leaves the pool untouched, which is what a verify pass wants.
    """
    _mtp_common_checks(query, key, value, a, b, dt_bias, A_log, state, out)
    stream = _mtp_stream(query, stream)
    seqlen = query.shape[1]
    _require_index("initial_state_indices", initial_state_indices, query, dims=1)
    if (intermediate_states_buffer is None) != (intermediate_state_indices is None):
        raise ValueError(
            "`intermediate_states_buffer` and `intermediate_state_indices` are "
            "one feature: pass both or neither."
        )
    if retrieve_parent_token is not None and intermediate_states_buffer is None:
        raise ValueError(
            "`retrieve_parent_token` needs `intermediate_states_buffer`: the tree "
            "restarts each token from a snapshot, so there has to be one."
        )
    if intermediate_states_buffer is not None:
        if intermediate_states_buffer.device != query.device:
            raise ValueError(
                "`intermediate_states_buffer` must be on the same device as `query`."
            )
        if intermediate_states_buffer.dtype not in _SUPPORTED_STATE_DTYPES:
            raise ValueError(
                "`intermediate_states_buffer` must have one of "
                f"{_SUPPORTED_STATE_DTYPES}; got {intermediate_states_buffer.dtype}."
            )
        got = tuple(intermediate_states_buffer.shape)
        want = (value.shape[-2], value.shape[-1], query.shape[-1])
        if len(got) != 5 or got[1] < seqlen or got[2:] != want:
            raise ValueError(
                f"`intermediate_states_buffer` must be [slot, >={seqlen}, "
                f"{want[0]}, {want[1]}, {want[2]}]; got {got}."
            )
        if intermediate_states_buffer.stride(-1) != 1:
            raise ValueError(
                "`intermediate_states_buffer` must have K contiguous; got stride "
                f"{intermediate_states_buffer.stride()}."
            )
        if intermediate_states_buffer.dtype.itemsize > state.dtype.itemsize:
            raise ValueError(
                "`intermediate_states_buffer` cannot be wider than `state`: the "
                "snapshot is written with the lane count the state's dtype sets, "
                f"so a {state.dtype} state and a "
                f"{intermediate_states_buffer.dtype} snapshot ask for a "
                f"{_snapshot_store_bytes(state.dtype, intermediate_states_buffer.dtype)}"
                "-byte store that the buffer ops cannot express. Store the "
                "snapshot at `state.dtype` or narrower."
            )
        _require_index("intermediate_state_indices", intermediate_state_indices, query)
    if retrieve_parent_token is not None:
        _require_index(
            "retrieve_parent_token",
            retrieve_parent_token,
            query,
            dims=2,
            layout="[batch, token]",
            covers=True,
        )

    _mtp_launch(
        mode=MTP_MODE_SNAPSHOT,
        query=query,
        key=key,
        value=value,
        a=a,
        b=b,
        dt_bias=dt_bias,
        A_log=A_log,
        state=state,
        out=out,
        state_indices=initial_state_indices.contiguous(),
        num_accepted=None,
        inter_indices=(
            intermediate_state_indices.contiguous()
            if intermediate_state_indices is not None
            else None
        ),
        parent_tokens=(
            retrieve_parent_token.contiguous()
            if retrieve_parent_token is not None
            else None
        ),
        inter_buffer=intermediate_states_buffer,
        use_qk_l2norm=use_qk_l2norm,
        min_live_slot=0,
        has_tree=retrieve_parent_token is not None,
        disable_state_update=disable_state_update,
        stream=stream,
    )
