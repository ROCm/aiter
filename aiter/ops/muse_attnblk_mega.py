# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One-launch attention block for Muse Spark target verify (TP4, gfx950).

Each call issues ONE persistent HIP kernel per rank (``csrc/kernels/muse_attnblk_mega.hip``, 256 CTAs x 256
threads, one CTA per CU) that runs the whole decoder attention block of a DFLASH target-verify step:

    FP8 QKV GEMM -> per-head QK RMSNorm / q scale / GPT-J RoPE / FP8 K,V cache write -> token-balanced paged
    verify attention -> merge / o RMSNorm / sigmoid gate -> o_proj -> IPC reduce-scatter + all-gather ->
    attention boundary (post RMSNorm, gated residual, pre RMSNorm, BF16 output, FP32 stream, FP8 copy).

Shapes are fixed by the model: hidden 8192, 32 query heads / 4 KV heads of dim 64 per rank, at most 8 requests of
8 verify tokens (M <= 64), FP8 e4m3 KV cache with 16-token pages. Ranks synchronise through a peer-mapped IPC
inbox per instance; every rank must issue the calls of one instance in the same order (one instance per layer
kind is enough, the kernel's epoch counter keeps consecutive calls apart). See ``MuseAttnBlockMega`` for the
calling contract.

The kernel is compiled to a code object with ``hipcc --genco -O3 -ffp-contract=off`` (no FMA contraction, no
denormal flushing), the flags its numerics were validated with, and cached in the aiter JIT directory.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import subprocess
import tempfile

import torch
import torch.distributed as dist

from aiter.jit.core import AITER_CSRC_DIR, get_user_jit_dir

_SRC = os.path.join(AITER_CSRC_DIR, "kernels", "muse_attnblk_mega.hip")

NR = 4  # TP ranks
GRID, NTHR = 256, 256
HID = 8192
MAX_TOKENS = 64
MAX_BS = 8
DRAFT = 8  # verify tokens per request
NQKV = 4608  # q 2048 | k 256 | v 256 | gate 2048 (per rank)
NKV, HD, PAGE = 4, 64, 16  # KV heads per rank, head dim, KV page size (tokens)
_TPR, _NTILE, _NPART = 64, 256, 272 * 4
_IPC_BYTES = (
    NR * 2 * _TPR * MAX_TOKENS * 64 + MAX_TOKENS * HID * 2 + 4096 + 16 * _NTILE * 4
)

_P = ctypes.c_void_p


class _Args(ctypes.Structure):
    _fields_ = (
        [("peer", _P * NR)]
        + [
            (n, _P)
            for n in (
                "ctrl",
                "xq",
                "xs",
                "wq",
                "ws",
                "qkv_part",
                "qbuf",
                "gbuf",
                "kc",
                "vc",
                "out_loc",
                "swa_map",
                "positions",
                "cos_sin",
                "block_table",
                "seqused",
                "att_o",
                "att_ml",
                "x",
                "w",
                "stream",
                "alpha",
                "beta",
                "pre_w",
                "out",
                "new_stream",
                "out_fp8",
                "out_scale",
                "stamps",
            )
        ]
        + [
            (n, ctypes.c_float)
            for n in ("post_eps", "pre_eps", "norm_eps", "q_scale", "sm_scale_log2")
        ]
        + [
            (n, ctypes.c_int32)
            for n in (
                "M",
                "rank",
                "nchunk",
                "opt",
                "bs",
                "window",
                "rope",
                "bt_stride",
                "cs_stride",
                "qs_min",
            )
        ]
    )


if ctypes.sizeof(_Args) != 328:  # must match struct Args in the kernel source
    raise ImportError(
        f"muse_attnblk_mega: _Args is {ctypes.sizeof(_Args)} bytes, the kernel expects 328"
    )


class _FillArgs(ctypes.Structure):
    _fields_ = [("p", _P), ("n16", ctypes.c_uint64)]


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


_hip = None


def _hip_lib():
    global _hip
    if _hip is None:
        h = ctypes.CDLL("libamdhip64.so")
        h.hipModuleLoad.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        h.hipModuleGetFunction.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        h.hipModuleLaunchKernel.argtypes = (
            [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
        )
        h.hipMalloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        h.hipIpcGetMemHandle.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        h.hipIpcOpenMemHandle.argtypes = [ctypes.c_void_p, _IpcHandle, ctypes.c_uint]
        h.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        h.hipFree.argtypes = [ctypes.c_void_p]
        h.hipModuleUnload.argtypes = [ctypes.c_void_p]
        h.hipGetErrorString.restype = ctypes.c_char_p
        _hip = h
    return _hip


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(
            f"muse_attnblk_mega: {what} failed: {rc} ({_hip_lib().hipGetErrorString(rc).decode()})"
        )


def _build(device: torch.device, flags: tuple) -> ctypes.c_void_p:
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    with open(_SRC, "rb") as f:
        src = f.read()
    key = hashlib.sha256(src + repr((arch, flags)).encode()).hexdigest()[:20]
    cdir = os.path.join(get_user_jit_dir(), "build", "muse_attnblk_mega")
    os.makedirs(cdir, exist_ok=True)
    co = os.path.join(cdir, f"muse_attnblk_mega_{key}.co")
    if not os.path.exists(co):
        hipcc = os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "bin", "hipcc")
        fd, tmp = tempfile.mkstemp(dir=cdir, suffix=".co")
        os.close(fd)
        cmd = [
            hipcc,
            "--genco",
            f"--offload-arch={arch}",
            "-O3",
            "-ffp-contract=off",
            *flags,
            "-o",
            tmp,
            _SRC,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if r.returncode != 0:
            raise RuntimeError(f"hipcc failed for {_SRC}:\n{r.stderr[-4000:]}")
        os.replace(tmp, co)  # atomic: concurrent ranks may build the same key
    module = ctypes.c_void_p()
    _check(_hip_lib().hipModuleLoad(ctypes.byref(module), co.encode()), "hipModuleLoad")
    return module


def _fn(module, name: str) -> ctypes.c_void_p:
    f = ctypes.c_void_p()
    _check(
        _hip_lib().hipModuleGetFunction(ctypes.byref(f), module, name.encode()), name
    )
    return f


def _launch(fn, grid: int, block: int, args, stream: int) -> None:
    size = ctypes.c_size_t(ctypes.sizeof(args))
    extra = (ctypes.c_void_p * 5)(
        1, ctypes.addressof(args), 2, ctypes.addressof(size), 3
    )
    _check(
        _hip_lib().hipModuleLaunchKernel(
            fn, grid, 1, 1, block, 1, 1, 0, ctypes.c_void_p(stream), None,
            ctypes.cast(extra, ctypes.c_void_p),
        ),
        "hipModuleLaunchKernel",
    )  # fmt: skip


def _require(ok: bool, msg: str) -> None:
    # An explicit raise, not assert: the checks must also run under python -O.
    if not ok:
        raise ValueError(f"muse_attnblk_mega: {msg}")


def _all_ok(problems: list, what: str, cpu_group, exc=RuntimeError) -> None:
    """Gather every rank's problems and raise ``exc`` on every rank if any rank has one, so no rank is left
    waiting in a later collective or in the kernel. Every rank must reach this call."""
    gathered = [None] * NR
    dist.all_gather_object(gathered, list(problems), group=cpu_group)
    bad = {r: p for r, p in enumerate(gathered) if p}
    if bad:
        raise exc(f"muse_attnblk_mega: {what} failed: {bad}")


def _try(problems: list, what: str, fn):
    """Run ``fn``; on an exception record it in ``problems`` instead of raising (callers sync with _all_ok)."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - reported on every rank by _all_ok
        problems.append(f"{what}: {e!r}")
        return None


def _ipc_inbox(cpu_group, device: torch.device, nbytes: int, fill_fn):
    """Allocate this rank's zeroed inbox and map the peers' (all_gather of IPC handles over ``cpu_group``).
    Returns the 4 device addresses indexed by rank; entry ``rank`` is this rank's own allocation.
    Collective: a failure on any rank is raised on every rank, after this rank's partial setup is released.
    """
    h = _hip_lib()
    rank = dist.get_rank(group=cpu_group)
    problems = []
    ptr = ctypes.c_void_p()
    raw = b""
    if h.hipMalloc(ctypes.byref(ptr), nbytes) != 0:
        problems.append("hipMalloc")
        ptr = ctypes.c_void_p()
    else:
        handle = _IpcHandle()
        if h.hipIpcGetMemHandle(ctypes.byref(handle), ptr) != 0:
            problems.append("hipIpcGetMemHandle")
        else:
            raw = ctypes.string_at(ctypes.addressof(handle), 64)
    handles = [None] * NR
    dist.all_gather_object(handles, raw, group=cpu_group)
    opened = {}
    if all(handles):
        for i, peer_raw in enumerate(handles):
            if i == rank:
                continue
            peer, ph = ctypes.c_void_p(), _IpcHandle()
            ctypes.memmove(ctypes.addressof(ph), peer_raw, 64)
            if h.hipIpcOpenMemHandle(ctypes.byref(peer), ph, 0x1) != 0:
                problems.append(f"hipIpcOpenMemHandle(rank {i})")
                break
            opened[i] = peer.value
    elif not problems:
        problems.append("a peer has no IPC handle")
    if not problems:
        stream = torch.cuda.current_stream(device).cuda_stream
        _try(
            problems,
            "inbox fill",
            lambda: _launch(
                fill_fn, 1024, 256, _FillArgs(ptr.value, nbytes // 16), stream
            ),
        )
        _try(problems, "inbox fill sync", lambda: torch.cuda.synchronize(device))
    try:
        _all_ok(problems, "IPC inbox setup", cpu_group)
    except RuntimeError:
        for addr in opened.values():
            h.hipIpcCloseMemHandle(ctypes.c_void_p(addr))
        dist.barrier(group=cpu_group)  # no peer maps this rank's inbox any more
        if ptr.value:
            h.hipFree(ptr)
        raise
    peers = (ctypes.c_void_p * NR)()
    for i in range(NR):
        peers[i] = ptr.value if i == rank else opened[i]
    return peers


def _expect(
    t: torch.Tensor, shape: tuple, dtype: torch.dtype, device: torch.device, name: str
):
    _require(
        tuple(t.shape) == shape, f"{name} shape {tuple(t.shape)}, expected {shape}"
    )
    _require(t.dtype == dtype, f"{name} dtype {t.dtype}, expected {dtype}")
    _require(t.device == device, f"{name} on {t.device}, instance on {device}")
    _require(t.is_contiguous(), f"{name} must be contiguous")


def _index_problems(
    *,
    block_table,
    seq_lens,
    out_cache_loc,
    positions,
    swa_mapping,
    cos_sin_cache,
    slots,
):
    """Index values that would make the kernel read or write outside the pools (reads them to the host)."""
    problems = []
    pages = slots // PAGE
    lens = seq_lens.long()
    if int(lens.min()) < DRAFT:
        return ["seq_lens must include the 8 verify tokens"]
    need = (lens + PAGE - 1) // PAGE
    if int(need.max()) > block_table.shape[1]:
        return ["block_table has fewer pages than seq_lens needs"]
    cols = torch.arange(block_table.shape[1], device=block_table.device)
    used = block_table.long()[cols[None, :] < need[:, None]]
    if int(used.min()) < 0 or int(used.max()) >= pages:
        problems.append("block_table page outside the KV pool")
    if int(out_cache_loc.min()) < 0:
        problems.append("out_cache_loc has negative slots")
    elif swa_mapping is not None and int(out_cache_loc.max()) >= swa_mapping.shape[0]:
        problems.append("out_cache_loc outside swa_mapping")
    else:
        loc = out_cache_loc if swa_mapping is None else swa_mapping[out_cache_loc]
        if int(loc.min()) < 0 or int(loc.max()) >= slots:
            problems.append("out_cache_loc outside the KV pool")
    if cos_sin_cache is not None and (
        int(positions.min()) < 0 or int(positions.max()) >= cos_sin_cache.shape[0]
    ):
        problems.append("positions outside cos_sin_cache")
    return problems


class MuseAttnBlockMega:
    """One kernel instance (code object, IPC inbox, scratch) for one layer kind on one rank.

    Contract:
        * Construction and ``close()`` are collective over the 4 ranks of ``cpu_group``; a failure on any rank
          is raised on every rank, after the partial setup is released.
        * The instance is bound to ``device``; every input must be on it, and calls run on it whatever device
          is current.
        * Calls on one instance must be ordered on the GPU (one stream, or streams synchronized with each other),
          and all ranks must issue the same calls in the same order: the instance owns scratch buffers and an
          epoch counter that consecutive calls share. Use one instance per concurrent call site.
        * The per-call checks (shape, dtype, device, layout, scalar ranges) see only values that are identical
          on every rank of a TP group, so a bad call fails on all ranks before any launch. Index values are
          checked only with ``validate_indices=True``, which is collective (every rank must pass it).
        * The kernel waits for the peer ranks with a 600 s bound (NCCL's default timeout); on a timeout, or if
          an internal invariant breaks, it traps and the launch fails. As with an NCCL collective, a rank that
          fails after the checks (out of memory for the outputs, a launch error) leaves the peers waiting until
          that bound.

    Args:
        cpu_group: gloo group of the 4 TP ranks (setup, close and index validation only).
        device: this rank's device (gfx950, at least 256 CUs).
        sliding: build the sliding-window variant (per-request attention split, ``-DPSPLIT=1``).
        nchunk: all-gather row chunks (1, 2, 4, 8 or 16).
        qs_min: minimum attention piece length in 32-token steps, 1..512 (default 4 sliding, 1 full).
        extra_flags: extra hipcc flags (tuning / diagnostics macros of the kernel source).
    """

    def __init__(
        self,
        cpu_group,
        device: torch.device,
        sliding: bool,
        nchunk: int = 4,
        qs_min: int | None = None,
        extra_flags: tuple = (),
    ):
        self.device = torch.device(device)
        self._cpu_group = cpu_group
        self._closed = True  # until every resource exists
        _require(
            self.device.type == "cuda" and self.device.index is not None,
            f"device {device}",
        )
        _require(
            dist.get_world_size(group=cpu_group) == NR,
            "TP4 only (cpu_group must have 4 ranks)",
        )
        _require(nchunk in (1, 2, 4, 8, 16), f"nchunk {nchunk}")
        self.sliding = sliding
        self.rank = dist.get_rank(group=cpu_group)
        self.nchunk = nchunk
        self.qs_min = qs_min if qs_min is not None else (4 if sliding else 1)
        # an attention piece of qs_min 32-token steps keeps 2 page ids per step in a 1024-entry LDS table (PID_MAX)
        _require(
            1 <= self.qs_min <= 512, f"qs_min must be in 1..512, got {self.qs_min}"
        )
        flags = tuple(extra_flags)
        if sliding and not any(f.startswith("-DPSPLIT") for f in flags):
            flags = flags + ("-DPSPLIT=1",)

        # 1. the bound device itself (ranks of one group may sit on different archs)
        props = torch.cuda.get_device_properties(self.device)
        arch = props.gcnArchName.split(":")[0]
        problems = []
        if arch != "gfx950":
            problems.append(f"device {self.device} is {arch}, not gfx950")
        if props.multi_processor_count < GRID:
            problems.append(
                f"{GRID} co-resident CTAs needed, device has {props.multi_processor_count} CUs"
            )
        _all_ok(problems, "device check", cpu_group, exc=NotImplementedError)

        h = _hip_lib()
        with torch.cuda.device(self.device):
            # 2. code object
            module = _try(problems, "kernel build", lambda: _build(self.device, flags))
            fns = None
            if module is not None:
                fns = _try(problems, "kernel lookup", lambda: (_fn(module, "muse_attnblk_mega"),
                                                               _fn(module, "muse_attnblk_mega_fill")))  # fmt: skip
            # 3. scratch
            scratch = (
                _try(problems, "scratch allocation", self._alloc_scratch)
                if fns
                else None
            )
            try:
                _all_ok(problems, "kernel build / scratch allocation", cpu_group)
                # 4. IPC inbox (collective, releases its own partial setup on failure)
                self._peers = _ipc_inbox(cpu_group, self.device, _IPC_BYTES, fns[1])
            except Exception:
                if module is not None:
                    h.hipModuleUnload(module)
                raise
        self._module, self._fn = module, fns[0]
        (
            self.ctrl,
            self.qkv_part,
            self.qbuf,
            self.gbuf,
            self.obuf,
            self.att_o,
            self.att_ml,
        ) = scratch
        self._closed = False

    def _alloc_scratch(self):
        f32, bf, dev = torch.float32, torch.bfloat16, self.device
        return (
            torch.zeros(4096, dtype=torch.int32, device=dev),
            torch.zeros(8, MAX_TOKENS, NQKV, dtype=f32, device=dev),
            torch.zeros(MAX_TOKENS, 2048, dtype=bf, device=dev),
            torch.zeros(MAX_TOKENS, 2048, dtype=bf, device=dev),
            torch.zeros(MAX_TOKENS, 2048, dtype=bf, device=dev),
            torch.zeros(_NPART, 64, 64, dtype=f32, device=dev),
            torch.zeros(_NPART, 2, 64, dtype=f32, device=dev),
        )

    def close(self):
        """Release the IPC mappings, the inbox and the code object (collective; after the last call).
        Every rank reaches both barriers even if a local step fails; failures are raised afterwards.
        """
        if self._closed:
            return
        h = _hip_lib()
        errors = []
        with torch.cuda.device(self.device):
            _try(errors, "synchronize", lambda: torch.cuda.synchronize(self.device))
            dist.barrier(group=self._cpu_group)
            for i, addr in enumerate(self._peers):
                if (
                    i != self.rank
                    and h.hipIpcCloseMemHandle(ctypes.c_void_p(addr)) != 0
                ):
                    errors.append(f"hipIpcCloseMemHandle(rank {i})")
            dist.barrier(
                group=self._cpu_group
            )  # no peer maps this rank's inbox any more
            if h.hipFree(ctypes.c_void_p(self._peers[self.rank])) != 0:
                errors.append("hipFree")
            if h.hipModuleUnload(self._module) != 0:
                errors.append("hipModuleUnload")
        self._closed = True
        if errors:
            raise RuntimeError(f"muse_attnblk_mega.close: {errors}")

    @staticmethod
    def applies(rows: int, bs: int) -> bool:
        return 0 < rows <= MAX_TOKENS and rows == DRAFT * bs and bs <= MAX_BS

    def __call__(
        self,
        x_fp8: torch.Tensor,
        x_scale: torch.Tensor,
        stream: torch.Tensor,
        *,
        w_qkv: torch.Tensor,
        w_qkv_scale: torch.Tensor,
        w_o: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        positions: torch.Tensor,
        swa_mapping: torch.Tensor | None,
        cos_sin_cache: torch.Tensor | None,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        pre_norm_weight: torch.Tensor,
        norm_eps: float,
        q_scale: float,
        sm_scale: float,
        post_eps: float,
        pre_eps: float,
        window: int = 0,
        validate_indices: bool = False,
    ):
        """Run the attention block for ``M = 8 * bs`` verify rows.

        Args:
            x_fp8: e4m3 [M, 8192] FP8 copy of the block input; x_scale: fp32 [M] per-row scales.
            stream: fp32 [M, 8192] residual stream.
            w_qkv: e4m3 [4608, 8192] fused q | k | v | gate weight with per-row fp32 scales ``w_qkv_scale`` [4608].
            w_o: bf16 [8192, 2048] this rank's o_proj slice.
            k_cache / v_cache: e4m3 paged KV pools [slots, 4, 64] (page size 16).
            block_table: int32 [>= bs, pages], rows contiguous; seq_lens: int32 [bs] cached tokens per request
                including the 8 verify tokens; out_cache_loc: int64 [M] slots of the verify tokens (mapped
                through ``swa_mapping`` when given); positions: int64 [M].
            cos_sin_cache: fp32 [max_pos, 64] GPT-J rotary cache, or None for no RoPE.
            alpha / beta / pre_norm_weight: fp32 [8192] boundary gate coefficients and effective pre-norm weight.
            q_scale, sm_scale: query scale after the per-head norm and softmax scale; the kernel uses a static
                softmax bound, so 8 * q_scale * 8.56 * sm_scale * log2(e) must stay below 100.
            window: sliding window (0 = full attention); a query at q sees keys k with q - window < k <= q.
            validate_indices: also check the index values (pages, slots, positions) against the pools.
                Collective over the TP group and reads them back to the host: not for CUDA-graph capture or the
                serving loop.

        Returns:
            (ffn_in bf16 [M, 8192], new_stream fp32 [M, 8192], ffn_in_fp8 e4m3 [M, 8192], ffn_in_scale fp32 [M]).
        """
        _require(not self._closed, "instance is closed")
        m, bs = x_fp8.shape[0], seq_lens.shape[0]
        _require(
            self.applies(m, bs),
            f"rows {m} with {bs} requests (rows = 8 x requests, at most 8 requests)",
        )
        _require(
            (window > 0) == self.sliding,
            f"window {window} on a {'sliding' if self.sliding else 'full'} instance",
        )
        for name, v in (("norm_eps", norm_eps), ("post_eps", post_eps), ("pre_eps", pre_eps),
                        ("q_scale", q_scale), ("sm_scale", sm_scale)):  # fmt: skip
            _require(
                math.isfinite(v) and v > 0,
                f"{name} must be finite and positive, got {v}",
            )
        max_log2_score = 8.0 * q_scale * 8.0 * 1.07 * sm_scale * math.log2(math.e)
        _require(
            max_log2_score < 100.0,
            f"q_scale * sm_scale too large for the static softmax bound ({max_log2_score:.1f})",
        )
        _expect(x_fp8, (m, HID), torch.float8_e4m3fn, self.device, "x_fp8")
        _require(
            x_scale.is_contiguous() and x_scale.numel() == m,
            "x_scale: contiguous, M values",
        )
        xs = x_scale.view(
            -1
        )  # a view: the kernel reads the caller's storage in this order
        _expect(xs, (m,), torch.float32, self.device, "x_scale")
        _expect(stream, (m, HID), torch.float32, self.device, "stream")
        _expect(w_qkv, (NQKV, HID), torch.float8_e4m3fn, self.device, "w_qkv")
        _require(
            w_qkv_scale.is_contiguous() and w_qkv_scale.numel() == NQKV,
            "w_qkv_scale: contiguous, 4608 values",
        )
        ws = w_qkv_scale.view(-1)
        _expect(ws, (NQKV,), torch.float32, self.device, "w_qkv_scale")
        _expect(w_o, (HID, 2048), torch.bfloat16, self.device, "w_o")
        slots = k_cache.shape[0]
        _expect(k_cache, (slots, NKV, HD), torch.float8_e4m3fn, self.device, "k_cache")
        _expect(v_cache, (slots, NKV, HD), torch.float8_e4m3fn, self.device, "v_cache")
        _require(k_cache.numel() < 2**32, "KV pool must be < 4 GiB (uint32 offsets)")
        _require(
            block_table.dim() == 2 and block_table.shape[0] >= bs,
            "block_table must be [>= bs, pages]",
        )
        _require(
            block_table.dtype == torch.int32 and block_table.stride(1) == 1,
            "block_table: int32, rows contiguous",
        )
        _require(
            block_table.device == self.device, f"block_table on {block_table.device}"
        )
        _expect(seq_lens, (bs,), torch.int32, self.device, "seq_lens")
        _expect(out_cache_loc, (m,), torch.int64, self.device, "out_cache_loc")
        _expect(positions, (m,), torch.int64, self.device, "positions")
        if swa_mapping is not None:
            _expect(
                swa_mapping,
                (swa_mapping.shape[0],),
                torch.int64,
                self.device,
                "swa_mapping",
            )
        if cos_sin_cache is not None:
            _require(
                cos_sin_cache.dim() == 2 and cos_sin_cache.shape[1] == HD,
                "cos_sin_cache must be [max_pos, 64]",
            )
            _require(
                cos_sin_cache.dtype == torch.float32 and cos_sin_cache.stride(1) == 1,
                "cos_sin_cache: fp32 rows",
            )
            _require(
                cos_sin_cache.device == self.device,
                f"cos_sin_cache on {cos_sin_cache.device}",
            )
        for name, t in (
            ("alpha", alpha),
            ("beta", beta),
            ("pre_norm_weight", pre_norm_weight),
        ):
            _expect(t, (HID,), torch.float32, self.device, name)
        if validate_indices:
            with torch.cuda.device(self.device):
                problems = _index_problems(
                    block_table=block_table[:bs], seq_lens=seq_lens, out_cache_loc=out_cache_loc,
                    positions=positions, swa_mapping=swa_mapping, cos_sin_cache=cos_sin_cache, slots=slots,
                )  # fmt: skip
            _all_ok(problems, "index validation", self._cpu_group, exc=ValueError)
        # The launch goes to the current HIP device; make it the instance's device.
        with torch.cuda.device(self.device):
            dev = self.device
            out = torch.empty(m, HID, dtype=torch.bfloat16, device=dev)
            new_stream = torch.empty(m, HID, dtype=torch.float32, device=dev)
            fp8 = torch.empty(m, HID, dtype=torch.float8_e4m3fn, device=dev)
            scale = torch.empty(m, dtype=torch.float32, device=dev)
            args = _Args(
                self._peers, self.ctrl.data_ptr(), x_fp8.data_ptr(), xs.data_ptr(), w_qkv.data_ptr(),
                ws.data_ptr(), self.qkv_part.data_ptr(), self.qbuf.data_ptr(), self.gbuf.data_ptr(),
                k_cache.data_ptr(), v_cache.data_ptr(), out_cache_loc.data_ptr(),
                swa_mapping.data_ptr() if swa_mapping is not None else 0, positions.data_ptr(),
                cos_sin_cache.data_ptr() if cos_sin_cache is not None else 0, block_table.data_ptr(),
                seq_lens.data_ptr(), self.att_o.data_ptr(), self.att_ml.data_ptr(), self.obuf.data_ptr(),
                w_o.data_ptr(), stream.data_ptr(), alpha.data_ptr(), beta.data_ptr(), pre_norm_weight.data_ptr(),
                out.data_ptr(), new_stream.data_ptr(), fp8.data_ptr(), scale.data_ptr(), 0,
                post_eps, pre_eps, norm_eps, q_scale, sm_scale * math.log2(math.e),
                m, self.rank, self.nchunk, 3, bs, window, 1 if cos_sin_cache is not None else 0,
                block_table.stride(0), cos_sin_cache.stride(0) if cos_sin_cache is not None else 0, self.qs_min,
            )  # fmt: skip
            _launch(
                self._fn,
                GRID,
                NTHR,
                args,
                torch.cuda.current_stream(self.device).cuda_stream,
            )
        return out, new_stream, fp8, scale
