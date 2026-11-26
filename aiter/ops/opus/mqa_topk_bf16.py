# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""BF16 MQA logits fused with top-k (OPUS, gfx950): the DSA lightning indexer for training.

Per query row ``r`` over its window ``[row_starts[r], row_ends[r])`` of key rows::

    score[r, c] = sum_h relu(q[r, h] . k[c]) * weights[r, h]
    indices[r]  = the top-``topk`` keys of row ``r``

The score matrix is never written: the op samples a threshold, emits the keys at or above it and
selects exactly per row (rows it cannot finish that way are rescored densely and selected the
same way). ``mqa_logits_bf16`` writes the dense scores (reference and debug)::

    cfg = mqa_topk_bf16_config(dev, max_rows=..., max_keys=...)   # once per shape bucket
    indices, values = mqa_topk_bf16(q, k, weights, row_starts, row_ends, cfg=cfg)
    logits = mqa_logits_bf16(q, k, weights, row_starts, row_ends, cfg=cfg)

LAYOUTS (contiguous)::

    q        [rows, 32, 128] bf16     weights  [rows, 32]  fp32 (scales folded in)
    k        [keys, 128]     bf16     row_starts, row_ends [rows] int32, half-open

Windows are in key-row space and are clamped to ``[0, keys)``; causality is never inferred from
the row order, so a context-parallel caller passes its local query rows with the gathered keys.

OUTPUT: ``indices [rows, min(topk, keys)]`` int32 absolute key rows. A row with ``n`` valid keys
holds ``min(n, W)`` of them in ascending order, no duplicates, then ``-1``. Rows with more than
``W`` keys keep the top ``W`` by (score desc, index asc). ``values`` (optional, fp32) are the
selected scores (``-0`` as ``+0``), ``-inf`` at ``-1``. Scores use BF16 MFMA with FP32
accumulation in a fixed order, the same in every stage and in ``mqa_logits_bf16``, so the results
are bitwise deterministic. NaN or Inf inputs are undefined.
"""

import functools
from dataclasses import dataclass

import torch

from ...jit.core import compile_ops
from ...jit.utils.torch_guard import torch_compile_guard
from ._arch import GFX950, _device_arch

_MD_NAME = "module_mqa_topk_bf16_opus"
SUPPORTED_GFX = (GFX950,)
HEADS = 32
HEAD_DIM = 128
MAX_KEYS = 1 << 24  # exclusive: keys x 256 B fits in 32 bits
MAX_TOPK = 4096  # select keeps a row's 3 x topk candidates in registers
_CHUNK_KEYS = 8192  # keys (samples, for the sample stage) a work item walks
_MIN_CAP = 1024


# == JIT stubs: signatures must match MQA_TOPK_BF16_PYBIND exactly ===============================
# Ops register by Python name, so the names carry the op prefix.
@compile_ops(_MD_NAME, fc_name="mqa_topk_bf16_instances", develop=True)
def _mqa_topk_bf16_instances_raw(device_id: int, out: torch.Tensor) -> int: ...


@compile_ops(_MD_NAME, fc_name="mqa_topk_bf16_workspace_bytes", develop=True)
def _mqa_topk_bf16_workspace_bytes_raw(
    rows: int, keys: int, cap_row: int, fb_slots: int
) -> int: ...


# Addresses and scalars, not tensors: the tensor op below owns the tensors and checks them.
@compile_ops(_MD_NAME, fc_name="mqa_topk_bf16_launch")
def _mqa_topk_bf16_launch_raw(
    q: int,
    k: int,
    weights: int,
    row_starts: int,
    row_ends: int,
    indices: int,
    values: int,
    workspace: int,
    workspace_bytes: int,
    rows: int,
    keys: int,
    topk: int,
    cap_row: int,
    chunk_keys: int,
    fb_slots: int,
    relu: int,
    instance: int,
    device: int,
    stream: int,
) -> None: ...


@compile_ops(_MD_NAME, fc_name="mqa_logits_bf16_launch")
def _mqa_logits_bf16_launch_raw(
    q: int,
    k: int,
    weights: int,
    row_starts: int,
    row_ends: int,
    out: int,
    rows: int,
    keys: int,
    chunk_keys: int,
    relu: int,
    instance: int,
    device: int,
    stream: int,
) -> None: ...


_topk_raw = getattr(_mqa_topk_bf16_launch_raw, "_eager_body", _mqa_topk_bf16_launch_raw)
_logits_raw = getattr(
    _mqa_logits_bf16_launch_raw, "_eager_body", _mqa_logits_bf16_launch_raw
)
_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None) or (
    lambda device: torch.cuda.current_stream(device).cuda_stream
)


def _launch_fake(*args, **kwargs) -> None:
    return None


@torch_compile_guard(
    mutates_args=["indices", "values", "workspace"],
    device="cuda",
    gen_fake=_launch_fake,
)
def _mqa_topk_bf16_launch(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    workspace: torch.Tensor,
    topk: int,
    cap_row: int,
    chunk_keys: int,
    fb_slots: int,
    relu: int,
    instance: int,
) -> None:
    device = q.get_device()
    _topk_raw(
        q.data_ptr(),
        k.data_ptr(),
        weights.data_ptr(),
        row_starts.data_ptr(),
        row_ends.data_ptr(),
        indices.data_ptr(),
        values.data_ptr() if values.numel() else 0,
        workspace.data_ptr(),
        workspace.numel(),
        q.shape[0],
        k.shape[0],
        topk,
        cap_row,
        chunk_keys,
        fb_slots,
        relu,
        instance,
        device,
        _raw_stream(device),
    )


@torch_compile_guard(mutates_args=["out"], device="cuda", gen_fake=_launch_fake)
def _mqa_logits_bf16_launch(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    out: torch.Tensor,
    chunk_keys: int,
    relu: int,
    instance: int,
) -> None:
    device = q.get_device()
    _logits_raw(
        q.data_ptr(),
        k.data_ptr(),
        weights.data_ptr(),
        row_starts.data_ptr(),
        row_ends.data_ptr(),
        out.data_ptr(),
        q.shape[0],
        k.shape[0],
        chunk_keys,
        relu,
        instance,
        device,
        _raw_stream(device),
    )


# == the registry ===================================================================================
@dataclass(frozen=True)
class _Instance:
    """One compiled kernel instance, as its registry entry describes it."""

    id: int
    rows_per_wave: int
    waves: int  # logits / emit: a workgroup's waves, sharing its keys
    sample_waves: int  # sample: a workgroup's waves
    tile_keys: int
    resident: tuple[
        int, int, int, int
    ]  # workgroups resident at once: logits/sample/emit/fallback


@functools.cache
def _instances(device: torch.device) -> tuple[_Instance, ...]:
    buf = torch.empty(256, dtype=torch.int32, device="cpu")
    need = _mqa_topk_bf16_instances_raw(device.index, buf)
    if need > buf.numel():
        buf = torch.empty(need, dtype=torch.int32, device="cpu")
        _mqa_topk_bf16_instances_raw(device.index, buf)
    flat = buf[:need].tolist()
    n, recs = flat[0], flat[1:]
    return tuple(
        _Instance(*recs[i : i + 5], resident=tuple(recs[i + 5 : i + n]))
        for i in range(0, len(recs), n)
    )


# == the configuration ==============================================================================
@dataclass(frozen=True)
class MqaTopkConfig:
    """A launch configuration for one shape bucket (:func:`mqa_topk_bf16_config`)."""

    device: torch.device
    heads: int
    head_dim: int
    topk: int
    max_rows: int
    max_keys: int
    relu: bool
    low_memory: bool
    cap_row: int  # candidate slots a row
    chunk_keys: int  # keys a work item walks
    fb_slots: int  # fallback workgroups, a scratch slice each
    workspace_bytes: int
    _instance: _Instance

    @property
    def width(self) -> int:
        """Columns of ``indices`` / ``values`` for a call with ``max_keys`` keys."""
        return min(self.topk, self.max_keys)

    def workspace(self) -> torch.Tensor:
        """Device workspace for one live call, sized from the bounds (caller-held)."""
        return torch.empty(self.workspace_bytes, dtype=torch.uint8, device=self.device)


def mqa_topk_bf16_config(
    device,
    *,
    max_rows: int,
    max_keys: int,
    heads: int = HEADS,
    head_dim: int = HEAD_DIM,
    topk: int = 2048,
    relu: bool = True,
    low_memory: bool = False,
    _instance: int | None = None,
    _chunk_keys: int | None = None,
) -> MqaTopkConfig:
    """Check the bounds against the device and return the configuration (cached).

    ``max_rows`` / ``max_keys`` bound every call made with it; the workspace is sized for them
    (about ``max_rows * (3 * topk * 8 + max_keys / 8)`` bytes plus the fallback's scratch, which
    ``low_memory`` cuts from one slice a CU to 32 slices). ``_instance`` and ``_chunk_keys``
    override the choice, for tuning and tests.
    """
    dev = (
        torch.device(device)
        if not isinstance(device, int)
        else torch.device("cuda", device)
    )
    if dev.type != "cuda":
        raise ValueError(f"mqa_topk_bf16 runs on a GPU, got {dev}")
    if dev.index is None:
        dev = torch.device(dev.type, torch.cuda.current_device())
    return _config(
        dev,
        int(max_rows),
        int(max_keys),
        int(heads),
        int(head_dim),
        int(topk),
        bool(relu),
        bool(low_memory),
        _instance,
        _chunk_keys,
    )


@functools.lru_cache(maxsize=256)
def _config(
    dev, max_rows, max_keys, heads, head_dim, topk, relu, low_memory, override, chunk
):
    arch = _device_arch(dev)
    if arch not in SUPPORTED_GFX:
        raise RuntimeError(f"mqa_topk_bf16 supports {SUPPORTED_GFX}, not {arch}")
    if (heads, head_dim) != (HEADS, HEAD_DIM):
        raise ValueError(f"heads x head_dim must be {HEADS} x {HEAD_DIM}")
    if not 1 <= topk <= MAX_TOPK or max_rows < 0 or not 1 <= max_keys < MAX_KEYS:
        raise ValueError(
            f"need 1 <= topk <= {MAX_TOPK}, max_rows >= 0, 1 <= max_keys < {MAX_KEYS}; "
            f"got {topk}, {max_rows}, {max_keys}"
        )
    insts = _instances(dev)
    if not insts:
        raise RuntimeError(f"no mqa_topk_bf16 instance compiled for {arch}")
    if override is not None:
        found = [i for i in insts if i.id == override]
        if not found:
            raise ValueError(f"no instance {override}; {arch} has {insts}")
        inst = found[0]
    else:
        inst = insts[0]
    chunk = _CHUNK_KEYS if chunk is None else int(chunk)
    if chunk < 16 or chunk % 16:
        raise ValueError(f"chunk_keys must be a positive multiple of 16, got {chunk}")
    width = min(topk, max_keys)
    # The tau stage aims the pool at about 32 * (e + 5 sqrt(e) + 8) keys, e = W / 32: 3584 of
    # 6144 slots at W = 2048, seven deviations under the cap.
    cap = max(-(-3 * width // 64) * 64, _MIN_CAP)
    cus = torch.cuda.get_device_properties(dev).multi_processor_count
    fb_slots = 32 if low_memory else cus
    ws = _mqa_topk_bf16_workspace_bytes_raw(max(max_rows, 1), max_keys, cap, fb_slots)
    return MqaTopkConfig(
        dev,
        heads,
        head_dim,
        topk,
        max_rows,
        max_keys,
        relu,
        low_memory,
        cap,
        chunk,
        fb_slots,
        ws,
        inst,
    )


# == the ops ========================================================================================
def _check_inputs(cfg, q, k, weights, row_starts, row_ends) -> tuple[int, int]:
    rows, keys = q.shape[0], k.shape[0]
    expect = (
        ("q", q, torch.bfloat16, (rows, cfg.heads, cfg.head_dim)),
        ("k", k, torch.bfloat16, (keys, cfg.head_dim)),
        ("weights", weights, torch.float32, (rows, cfg.heads)),
        ("row_starts", row_starts, torch.int32, (rows,)),
        ("row_ends", row_ends, torch.int32, (rows,)),
    )
    for name, t, dtype, shape in expect:
        if t.dtype != dtype or tuple(t.shape) != shape:
            raise ValueError(
                f"{name} must be {dtype} {shape}, got {t.dtype} {tuple(t.shape)}"
            )
        if t.device != cfg.device or not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous on {cfg.device}")
    if rows > cfg.max_rows or not 1 <= keys <= cfg.max_keys:
        raise ValueError(
            f"rows {rows} / keys {keys} exceed the config bounds "
            f"{cfg.max_rows} / {cfg.max_keys}"
        )
    return rows, keys


def _check_out(name, t, dtype, shape, device):
    if t.dtype != dtype or tuple(t.shape) != shape:
        raise ValueError(
            f"{name} must be {dtype} {shape}, got {t.dtype} {tuple(t.shape)}"
        )
    if t.device != device or not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous on {device}")


def mqa_topk_bf16(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    cfg: MqaTopkConfig,
    workspace: torch.Tensor | None = None,
    indices: torch.Tensor | None = None,
    values: torch.Tensor | None = None,
    return_values: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Top-k key rows of each query row (see the module docstring); returns (indices, values).

    ``values`` is returned when ``return_values`` is set or a ``values`` tensor is passed.
    ``workspace`` (from ``cfg.workspace()``) is allocated per call when not given.
    """
    rows, keys = _check_inputs(cfg, q, k, weights, row_starts, row_ends)
    width = min(cfg.topk, keys)
    dev = cfg.device
    if indices is None:
        indices = torch.empty(rows, width, dtype=torch.int32, device=dev)
    _check_out("indices", indices, torch.int32, (rows, width), dev)
    want_values = return_values or values is not None
    if want_values:
        if values is None:
            values = torch.empty(rows, width, dtype=torch.float32, device=dev)
        _check_out("values", values, torch.float32, (rows, width), dev)
    if workspace is None:
        workspace = cfg.workspace()
    if (
        workspace.dtype != torch.uint8
        or workspace.device != dev
        or not workspace.is_contiguous()
        or workspace.numel() < cfg.workspace_bytes
    ):
        raise ValueError(
            f"workspace must be contiguous uint8 on {dev} with >= {cfg.workspace_bytes} "
            "bytes (cfg.workspace())"
        )
    if rows:
        _mqa_topk_bf16_launch(
            q,
            k,
            weights,
            row_starts,
            row_ends,
            indices,
            values if want_values else workspace[:0],
            workspace,
            cfg.topk,
            cfg.cap_row,
            cfg.chunk_keys,
            cfg.fb_slots,
            int(cfg.relu),
            cfg._instance.id,
        )
    return indices, (values if want_values else None)


def mqa_logits_bf16(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    cfg: MqaTopkConfig,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense scores ``[rows, keys]`` fp32, ``-inf`` outside each window; returns ``out``."""
    rows, keys = _check_inputs(cfg, q, k, weights, row_starts, row_ends)
    if out is None:
        out = torch.empty(rows, keys, dtype=torch.float32, device=cfg.device)
    _check_out("out", out, torch.float32, (rows, keys), cfg.device)
    if rows:
        _mqa_logits_bf16_launch(
            q,
            k,
            weights,
            row_starts,
            row_ends,
            out,
            cfg.chunk_keys,
            int(cfg.relu),
            cfg._instance.id,
        )
    return out
