# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host side of the big-M path of MegaMoeTP.forward (the kernel: compile_bigm_kernel in
mega_moe_tp_kernel.py): MegaMoeTPBigM (the SDMA queues and hipIpc arena, buffers, per-batch
plans, launches), the disk cache of the compiled kernels, the bf16 row gather of the fused tail.
"""

import ast
import ctypes
import hashlib
import inspect
import logging
import os
import pickle
import re
import stat
import tempfile
import time

import torch
import torch.distributed as dist
from flydsl.utils import env as fenv

from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_gfx

from ....communication import SdmaComm, compile_signal_read, sdma_bitcode
from ..tensor_shim import _preload_compiled, _run_compiled
from .common import device_key, nearest, tuned_rows
from .gemm import bigm_lds
from .mega_moe_tp import LDS_LIMIT
from .mega_moe_tp_config import (
    BIGM_CTRL_INTS,
    C_ERR,
    CMAX,
    NQ,
    PAD,
    UNIT_ROWS,
    UQ,
    i8_row_bytes,
    unit_cap,
)
from .mega_moe_tp_kernel import compile_bigm_kernel

_log = logging.getLogger("aiter")

__all__ = ["MegaMoeTPBigM"]


# ---- disk cache of the compiled big-M kernels ----
# FlyDSL keeps kernels that link external bitcode (link_extern: the SDMA device functions here)
# out of its disk cache -- its key does not cover the bitcode -- so every process would compile
# every big-M variant again (~86 s each, all ranks waiting). The artifact itself pickles fine
# (no post-load processors), so this keeps it on disk keyed by the compile arguments plus the
# digest of everything else it is built from: the kernel's sources (mega_moe_tp_kernel.py and
# the modules it imports from aiter.ops.flydsl.kernels), the SDMA bitcode, the FlyDSL compiler
# (its own cache key) and the GPU arch.
_DIGEST = None


def _kernel_sources():
    """mega_moe_tp_kernel.py and every module of aiter.ops.flydsl.kernels it imports, transitively
    (relative imports, wherever in the file; MORI's bitcode is hashed itself)"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../kernels
    todo = [os.path.join(root, "mega_moe_tp", "mega_moe_tp_kernel.py")]
    seen = set()
    while todo:
        f = todo.pop()
        if f in seen or not os.path.isfile(f):
            continue
        seen.add(f)
        here = os.path.dirname(f)
        with open(f) as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            base = here
            for _ in range(node.level - 1):
                base = os.path.dirname(base)
            if not base.startswith(root):
                continue
            mods = (
                [os.path.join(base, *node.module.split("."))]
                if node.module
                else [os.path.join(base, a.name) for a in node.names]
            )
            for m in mods:
                todo += [m + ".py", os.path.join(m, "__init__.py")]
    return sorted(seen)


def _bigm_digest():
    global _DIGEST
    if _DIGEST is None:
        h = hashlib.sha256(get_gfx().encode())
        for f in _kernel_sources() + [sdma_bitcode()]:
            with open(f, "rb") as fh:
                h.update(fh.read())
        try:
            from flydsl.compiler.jit_function import _flydsl_key

            h.update(_flydsl_key().encode())
        except Exception:  # noqa: BLE001 (older / newer FlyDSL: its version instead)
            import flydsl

            h.update(str(getattr(flydsl, "__version__", "")).encode())
        _DIGEST = h.hexdigest()
    return _DIGEST


def _bigm_cache_dir():
    """FlyDSL's cache root (FLYDSL_RUNTIME_CACHE_DIR) / aiter_bigm; None when FlyDSL caching
    is off."""
    if not fenv.runtime.enable_cache:
        return None
    return os.path.join(fenv.runtime.cache_dir, "aiter_bigm")


_WARNED = set()


def _warn_once(key, msg, *args):
    if key not in _WARNED:
        _WARNED.add(key)
        _log.warning(msg, *args)


def _private(path):
    """path (the cache directory or an entry) is this user's and only writable by them: a
    pickled entry runs code when loaded, so a shared / writable-by-others cache is skipped
    """
    try:
        st = os.stat(path)
    except OSError:
        return False
    return st.st_uid == os.getuid() and not st.st_mode & (stat.S_IWGRP | stat.S_IWOTH)


def _bigm_cached(launch, args):
    """Collective-free: make launch(*args)'s compiled artifact resident -- from this process's
    memory, else the disk cache, else compiled (no kernel dispatch) and written to the disk
    cache -- and its code object loaded."""
    try:  # FlyDSL's in-process key of these arguments (private API: on any change, just compile)
        launch._ensure_sig()
        launch._ensure_cache_manager()
        bound = launch._sig.bind(*args)
        bound.apply_defaults()
        ck = launch._build_full_cache_key(
            bound.arguments, effective_hints=launch._effective_compile_hints()
        )
        mem = launch._mem_cache
    except Exception:  # noqa: BLE001
        _warn_once(
            "api",
            "MegaMoeTP big-M: FlyDSL's cache API changed (private names): no disk cache, "
            "every process compiles its big-M kernels (minutes each)",
        )
        _preload_compiled(launch, *args)
        return
    if ck in mem:
        mem[ck]._get_func_exe()
        return
    d = _bigm_cache_dir()
    path = None
    if d:
        sig = re.sub(r" at 0x[0-9a-f]+", "", str(ck))
        name = hashlib.sha256(
            f"{launch._bigm_spec}|{_bigm_digest()}|{sig}".encode()
        ).hexdigest()[:32]
        path = os.path.join(d, f"bigm_{name}.pkl")
        if os.path.isdir(d) and not _private(d):
            _warn_once(
                d,
                "MegaMoeTP big-M: disk cache %s is not private to this user "
                "(group / others writable or another owner): not used",
                d,
            )
            path = None
        elif os.path.isfile(path) and not _private(path):
            path = None
        if path and os.path.isfile(path):
            try:
                with open(path, "rb") as f:
                    art = pickle.load(f)
                art._get_func_exe()  # code object loaded here, not in the first launch
                mem[ck] = art
                return
            # a stale / corrupt entry: compile again
            except Exception as e:  # noqa: BLE001
                _log.warning("MegaMoeTP big-M: disk cache %s unusable (%s)", path, e)
    _preload_compiled(launch, *args)
    art = mem.get(ck)
    if art is None:
        return
    art._get_func_exe()
    _warn_spills(launch, art)
    if path:
        try:
            os.makedirs(d, mode=0o700, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as f:
                    pickle.dump(art, f)
                os.replace(tmp, path)  # atomic: concurrent ranks write the same bytes
            except BaseException:
                os.unlink(tmp)
                raise
        except Exception as e:  # noqa: BLE001
            _log.warning("MegaMoeTP big-M: disk cache write %s failed (%s)", path, e)


def _warn_spills(launch, art):
    """a freshly compiled kernel that spills to scratch: its register file is exhausted (seen
    with the waits timing out)"""
    m_ = re.search(r"private_segment_fixed_size = (\d+)", getattr(art, "ir", "") or "")
    if m_ and int(m_.group(1)):
        _log.warning(
            "MegaMoeTP big-M: kernel %s spills to scratch (%s B per lane): lower its "
            "register use (e.g. AITER_BIGM_R8=1 in the tuned csv's opts)",
            launch._bigm_spec[:120],
            m_.group(1),
        )


_HIP = None


def _hip():
    """the HIP runtime torch uses (ctypes): stream write / wait value"""
    global _HIP
    if _HIP is None:
        torch.cuda.init()
        with open("/proc/self/maps") as maps:
            path = next(l.split()[-1] for l in maps if "libamdhip64" in l)
        lib = ctypes.CDLL(path)
        lib.hipStreamWriteValue32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint,
        ]
        lib.hipStreamWaitValue32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint,
            ctypes.c_uint32,
        ]
        lib.hipStreamWriteValue32.restype = lib.hipStreamWaitValue32.restype = (
            ctypes.c_int
        )
        _HIP = lib
    return _HIP


def _hip_check(err):
    if err != 0:
        raise RuntimeError(f"MegaMoeTP big-M: HIP stream value op failed ({err})")


def _kcopy(dst, src):
    """dst <- src by a shader kernel (dst: a peer's memory). Not copy_: a contiguous copy_ is a
    hipMemcpyAsync, which ROCm runs on the SDMA engine of that xGMI link -- the engines the
    big-M kernel's MORI queues use; ROCr's own queue there slows every later big-M put.
    """
    torch.mul(src, 1, out=dst)


class _CudaBuf:
    """__cuda_array_interface__ of raw device bytes (a peer's arena as mapped here)."""

    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


class MegaMoeTPBigM:
    """Host side of the big-M dispatch: the SDMA queues and arena (same regions on every
    rank), scratch buffers and the launch.
    moe(x [m, H], topk_weights / topk_ids [m, K1], out, tail) (MegaMoeTP.forward, ag_rs)
    -> y [m, H], with tail = (res_in, res_out, norm_w) (y, q [T, H] fp8, scales [T]).
    Per batch the chunk plan and options come from the tuned csv, else defaults.
    """

    REGIONS = ("xq", "xs", "ids", "tw", "eq", "erecv", "q", "qs")

    def __init__(
        self,
        *,
        hidden,
        inter,
        experts,
        topk,
        max_tokens,
        eps,
        gemma,
        swiglu_limit,
        w1,
        w1_scale,
        w2,
        w2_scale,
        group=None,
        device=None,
        opts=None,
        act="swiglu",
        ygather=False,
    ):
        self.H, self.I, self.E, self.K = hidden, inter, experts, topk
        self.K1, self.NE = topk + 1, experts + 1
        self.tp = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.Tmax = max_tokens
        self.act = act  # the units' activation ("swiglu", "silu")
        self.ygather = bool(ygather)  # moe(gather=True): the "q" region holds bf16 rows
        self._plans = {}
        self.cfg = (float(eps), bool(gemma), float(swiglu_limit))
        self.opts = tuple(sorted((opts or {}).items()))  # tuned AITER_BIGM_* options
        self.w = (
            w1.view(torch.uint8),
            w1_scale.view(torch.uint8),
            w2.view(torch.uint8),
            w2_scale.view(torch.uint8),
        )
        H, T, K1, tp = hidden, max_tokens, self.K1, self.tp
        RB = i8_row_bytes(H)
        # the kernel's 32-bit offsets into these regions (the route rows: per chunk, _chunks)
        for what, n in (("E rows", T * RB), ("y gather rows", T * H * 2)):
            if n >= 1 << 31:
                raise ValueError(
                    f"MegaMoeTP big-M: {what} of {T} tokens exceed 32-bit offsets"
                )
        sizes = {
            "xq": T * H // 2,
            "xs": T * H // 32,
            "ids": T * K1 * 4,
            "tw": T * K1 * 4,
            "eq": T * RB,
            "erecv": T * RB,
            "q": T * H * (2 if self.ygather else 1),
            "qs": T * 4,
        }
        sd = self.sd = SdmaComm(group=group, num_queues=NQ, device=device)
        for r_ in self.REGIONS:
            sd.reserve(r_, sizes[r_])
        # gather_rows: every rank's bf16 tail rows (not a kernel region)
        sd.reserve("brows", 2 * T * H * 2)  # two buffers in turn
        sd.reserve("bflag", tp * 64)  # per-rank epoch flags (gather_rows)
        sd.commit()
        # the signal-base buffer each launch reads (the other: the next launch's, from GW)
        self._cpar = 0
        self._cons = tuple(int(c.data_ptr()) for c in sd.consumed)
        if self._cons[0] % (2 * tp * NQ * 4) or self._cons[1] != self._cons[0] ^ (
            tp * NQ * 4
        ):
            raise RuntimeError(f"MegaMoeTP big-M: misaligned signal bases {self._cons}")
        dev = self.device = sd.device
        self.offs = torch.tensor(
            [sd.offset(r_) for r_ in self.REGIONS], dtype=torch.int64, device=dev
        )
        m = T // tp
        bf = torch.bfloat16
        self.normed = torch.empty((m, H), dtype=bf, device=dev)
        self.resout = torch.empty((m, H), dtype=bf, device=dev)
        self.ybuf = torch.empty((m, H), dtype=bf, device=dev)
        # route rows / lists of every chunk packed (chunk k from route tp * o_k * K1)
        self.routes = torch.empty(T * K1 * H, dtype=bf, device=dev)
        self.lrix = torch.empty(T * K1, dtype=torch.int32, device=dev)
        self.lwt = torch.empty(T * K1, dtype=torch.float32, device=dev)
        # unit slots of every chunk ([C][8][ucap of the batch]); _chunks grows it if a plan
        # needs more
        self.uslots = 2 * (-(-tp * m * K1 // UNIT_ROWS) // 8) + CMAX * (
            -(-self.NE // 8) + UQ + 2
        )
        self.utab = torch.zeros(8 * self.uslots * 4, dtype=torch.int32, device=dev)
        self.ctrl = torch.zeros(BIGM_CTRL_INTS + 32, dtype=torch.int32, device=dev)
        self.chunks_t = {}
        # moe()'s padding buffers (m not a multiple of PAD), the largest padded size
        self._pads = (
            torch.zeros((m, H), dtype=bf, device=dev),
            torch.zeros((m, K1), dtype=torch.float32, device=dev),
            torch.full((m, K1), -1, dtype=torch.int32, device=dev),
            torch.zeros((m, H), dtype=bf, device=dev),
            torch.empty((T, H), dtype=torch.uint8, device=dev),
            torch.empty((T,), dtype=torch.float32, device=dev),
        )
        self._pad_dirty = 0  # rows below may hold an earlier call's tokens / routes
        self._ready = set()
        self._brows_peer = None  # gather_rows' state (set up on its first call)
        self.cu = torch.cuda.get_device_properties(dev).multi_processor_count
        self.nab = self._nab()

    def _nab(self):
        """the deepest A ring (dividing the GEMM1 chunk count) whose LDS (+ the kernel's
        row-phase scratch, < 4 KiB) fits a CU"""
        kch = self.H // 256
        for n in (4, 3, 2):
            if (
                kch % n == 0
                and bigm_lds(UNIT_ROWS, n, self.I)["BYTES"] + 4096 <= LDS_LIMIT
            ):
                return n
        raise ValueError(
            f"MegaMoeTP big-M: no A ring depth fits hidden {self.H} / inter {self.I}"
        )

    def gather_rows(self, rows, local=False):
        """Collective: every rank's rows [m, H] bf16 -> [tp * m, H] (rank r's at r * m), a
        view of the arena valid until the next call. local: this rank's rows only (placed
        at rank * m; no peer writes). Each rank writes its rows into every
        peer's arena (hipIpc mapping; one side stream per peer), then stream-ordered flags:
        hipStreamWriteValue32 of this call's epoch into every peer's flag slot of this rank,
        hipStreamWaitValue32 on every peer's slot here. No host sync (the host keeps running
        ahead of the GPU). Two buffers in turn: a peer's writes of call k + 2 follow its wait
        for this rank's flag of call k + 1, which this stream writes after whatever it ran on
        call k's rows."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("MegaMoeTP big-M: gather_rows is not graph-capturable")
        m, H = rows.shape
        T = m * self.tp
        if T > self.Tmax or rows.dtype != torch.bfloat16 or H != self.H:
            raise ValueError(
                f"MegaMoeTP big-M: gather_rows needs bf16 [m, {self.H}] rows, m * tp <= "
                f"{self.Tmax}; got {rows.dtype} {tuple(rows.shape)}"
            )
        stream = torch.cuda.current_stream()
        if self._brows_peer is None:
            off, half = self.sd.offset("brows"), self.Tmax * H * 2
            bases = self.sd.arena.base_ptrs
            self._brows_peer = [
                [
                    torch.as_tensor(
                        _CudaBuf(b_ + off + i * half, half), device=self.device
                    )
                    .view(torch.bfloat16)
                    .view(self.Tmax, H)
                    for b_ in bases
                ]
                for i in range(2)
            ]
            foff = self.sd.offset("bflag")
            # flag slot of rank r (64 B apart) in rank p's arena, as mapped here
            self._bflag = [[b_ + foff + r * 64 for r in range(self.tp)] for b_ in bases]
            self.sd.view("bflag", (self.tp * 16,), torch.int32).zero_()
            self._brows_streams = [
                torch.cuda.Stream(self.device) for _ in range(self.tp)
            ]
            self._bepoch = 0
            torch.cuda.synchronize(self.device)
            dist.barrier(group=self.sd.group)
        self._bepoch += 1
        ep = self._bepoch & 0x7FFFFFFF
        dst = self._brows_peer[self._bepoch & 1]
        src = rows.contiguous()
        lo, hi = self.rank * m, (self.rank + 1) * m
        if local:
            _kcopy(dst[self.rank][lo:hi], src)
            return dst[self.rank][:T]
        ev = torch.cuda.Event()
        ev.record(stream)
        for p in range(self.tp):
            if p == self.rank:
                continue
            s_ = self._brows_streams[p]
            s_.wait_event(ev)
            with torch.cuda.stream(s_):
                _kcopy(dst[p][lo:hi], src)
            src.record_stream(s_)
        _kcopy(dst[self.rank][lo:hi], src)
        hip, sh = _hip(), stream.cuda_stream
        for p in range(self.tp):
            if p != self.rank:
                stream.wait_stream(self._brows_streams[p])
        for p in range(self.tp):  # this rank's rows landed in p's arena
            if p != self.rank:
                _hip_check(
                    hip.hipStreamWriteValue32(sh, self._bflag[p][self.rank], ep, 0)
                )
        for p in range(self.tp):  # every peer's rows landed here
            if p != self.rank:
                _hip_check(
                    hip.hipStreamWaitValue32(
                        sh, self._bflag[self.rank][p], ep, 0, 0x7FFFFFFF
                    )
                )
        return dst[self.rank][:T]

    def poll_errors(self) -> int:
        """Nonzero if a wait on a peer inside the kernel gave up since the last clear."""
        return int(self.ctrl[C_ERR].item())

    def clear_errors(self) -> None:
        self.ctrl[C_ERR:].zero_()

    def error_flag(self):
        """The sticky watchdog word as a 1-element device tensor (no sync): 1 if a wait gave
        up since the last clear."""
        return self.ctrl[C_ERR : C_ERR + 1]

    def reset(self) -> None:
        """Collective: re-base the per-(peer, queue) signal bases on the received signals and
        clear the watchdog word -- after a wait gave up the bases may be out of step with the
        peers' puts (every later wait on that queue would time out)."""
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.sd.group)
        time.sleep(0.2)  # the puts of the peers' finished launches land
        got = torch.zeros(self.tp * NQ, dtype=torch.int32, device=self.device)
        _run_compiled(
            compile_signal_read(self.tp, NQ),
            self.sd.dev_comm.ptr,
            got.data_ptr(),
            torch.cuda.current_stream(),
        )
        self.sd.consumed[self._cpar].copy_(got)
        self.ctrl.zero_()
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.sd.group)

    def close(self) -> None:
        """Collective: free the SDMA queues and the arena (unusable afterwards)."""
        self.sd.close()

    def _plan(self, T):
        """(chunk weights, AITER_BIGM_* options) of a batch of T tokens (T / tp a multiple
        of PAD): the tuned csv's, else the defaults; at most one chunk per PAD local rows
        """
        p = self._plans.get(T)
        if p is None:
            o = dict(self.opts)
            key = {
                **device_key(self.device),
                "model_dim": self.H,
                "inter_dim": self.I,
                "experts": self.NE,
                "topk": self.K1,
                "tp": self.tp,
            }
            path = AITER_CONFIGS.AITER_CONFIG_MEGA_MOE_TP_BIGM_FILE
            tu = nearest(tuned_rows(path, key, "tokens"), T, clamp=True)
            if tu is not None:
                w = [int(x) for x in tu["chunks"].split(",")]
                opts = (kv.split("=", 1) for kv in tu.get("opts", "").split(";") if kv)
                o = {**dict(opts), **o}
            else:
                w = [1, 1] if T <= 16384 else [1, 2, 2, 1]
            if not 0 < len(w) <= CMAX or min(w) <= 0:
                raise ValueError(f"MegaMoeTP big-M: bad chunk weights {w} (T={T})")
            n = max(1, T // self.tp // PAD)
            if len(w) > n:  # few rows: fewer chunks (the largest weights)
                w = sorted(w, reverse=True)[:n]
            p = self._plans[T] = (tuple(w), tuple(sorted(o.items())))
        return p

    def _split(self, m):
        w = self._plan(m * self.tp)[0]
        tot = sum(w)
        # multiples of 64 rows (moe() pads m to PAD): whole input row items (ZX) in every head
        # sub-block (AITER_BIGM_NSBA <= 4)
        g = PAD
        sizes = [max(g, (m * x // tot) // g * g) for x in w]
        sizes[w.index(max(w))] += m - sum(sizes)
        if m % g or sum(sizes) != m or min(sizes) <= 0 or any(x % g for x in sizes):
            raise RuntimeError(f"MegaMoeTP big-M: cannot split {m} rows by {w}")
        return sizes

    def _chunks(self, T):
        m = T // self.tp
        sizes = tuple(self._split(m))
        c = self.chunks_t.get(T)
        if c is None:
            # the kernel's 32-bit offsets of a chunk's route rows (rows * H bf16)
            if self.tp * max(sizes) * self.K1 * self.H * 2 >= 1 << 31:
                raise ValueError(
                    f"MegaMoeTP big-M: chunks {sizes} of {T} tokens: a chunk's route rows "
                    "exceed 32-bit offsets (more chunks: the tuned csv)"
                )
            need = len(sizes) * self._ucap(T)
            if need > self.uslots:
                self.uslots = need
                self.utab = torch.zeros(
                    8 * need * 4, dtype=torch.int32, device=self.device
                )
            os_ = [sum(sizes[:i]) for i in range(len(sizes))]
            c = self.chunks_t[T] = torch.tensor(
                [[o, s] for o, s in zip(os_, sizes)],
                dtype=torch.int32,
                device=self.device,
            )
        return c

    def set_weights(self, w1, w1_scale, w2, w2_scale):
        """the next launches use another layer's weights of the same shape"""
        self.w = (
            w1.view(torch.uint8),
            w1_scale.view(torch.uint8),
            w2.view(torch.uint8),
            w2_scale.view(torch.uint8),
        )

    def _ucap(self, T):
        return unit_cap(
            self.tp,
            max(self._split(T // self.tp)),
            self.K1,
            self.NE,
            UNIT_ROWS,
        )

    def _run(self, launch, args):
        if launch not in self._ready:
            # compile (or load) on every rank first: the kernel's waits on peers have a 2 s
            # watchdog, a launch must not start while a peer is still compiling
            _bigm_cached(launch, args)
            torch.cuda.synchronize()
            dist.barrier(group=self.sd.group)
            self._ready.add(launch)
        _run_compiled(launch, *args)

    def moe(
        self, x, topk_weights, topk_ids, out=None, tail=None, local=False, gather=False
    ):
        """The MoE of every rank's tokens (ag_rs): x [m, H] bf16 (this rank's rows, global
        rows rank * m + i), topk_weights f32 / topk_ids int32 [m, K1] (the shared expert E in
        any column; ids outside [0, experts] are no route) -> y [m, H] bf16 = this rank's rows of
        sum_r MoE_r(every row); tail = (res_in, res_out, norm_w): also res_out = y + res_in and
        the (Gemma)RMSNorm of res_out as FP8 rows (+ f32 scales) of every rank -> (y, q [T, H]
        fp8 (uint8), scales [T]); local: only this rank's q rows (rank * m ..) are written.
        gather (ygather, no tail): y of every rank's rows -> [tp * m, H] (rank r's at r * m; out,
        else a view of the arena valid until the next launch), gathered inside the kernel;
        copies out of the arena by shader kernels (_kcopy).
        Any m: the kernel runs on m rounded up to a multiple of 64 (PAD; the extra rows are zero,
        with no routes), its results cut back to m rows."""
        if gather and not (self.ygather and tail is None):
            raise ValueError("MegaMoeTP big-M: gather needs ygather and no tail")
        m = x.shape[0]
        if (
            x.dtype != torch.bfloat16
            or tuple(x.shape) != (m, self.H)
            or not x.is_contiguous()
            or tuple(topk_ids.shape) != (m, self.K1)
            or tuple(topk_weights.shape) != (m, self.K1)
            or topk_ids.dtype != torch.int32
            or topk_weights.dtype != torch.float32
            or not (topk_ids.is_contiguous() and topk_weights.is_contiguous())
        ):
            raise ValueError(
                f"MegaMoeTP big-M: need contiguous bf16 x [m, {self.H}], int32 topk_ids / "
                f"f32 topk_weights [m, {self.K1}]; got {x.dtype} {tuple(x.shape)}, "
                f"{topk_ids.dtype} {tuple(topk_ids.shape)}, {topk_weights.dtype} "
                f"{tuple(topk_weights.shape)}"
            )
        mp = -(-m // PAD) * PAD
        if mp * self.tp > self.Tmax:
            raise ValueError(
                f"MegaMoeTP big-M: {m} local tokens (padded {mp}) exceed "
                f"{self.Tmax // self.tp}"
            )
        if mp == m:
            if gather:
                ya = self._moe(x, topk_weights, topk_ids, None, None, local, gather)
                if out is not None:
                    _kcopy(out, ya)
                return ya if out is None else out
            return self._moe(x, topk_weights, topk_ids, out, tail, local)
        # one set of padded buffers (allocated at init, the largest padded size), sliced to mp:
        # rows m.. of x / tw / ids / res must be zero / no route; rows below _pad_dirty may hold
        # an earlier call's tokens
        xp, twp, idp, rp, qc, qsc = self._pads
        hi_ = min(self._pad_dirty, mp)
        if hi_ > m:  # rows m .. mp of this call: clear what an earlier call left there
            xp[m:hi_].zero_()
            twp[m:hi_].zero_()
            idp[m:hi_].fill_(-1)
            rp[m:hi_].zero_()
        # dirty from now on: this call's rows (< m), plus an earlier call's beyond mp
        self._pad_dirty = self._pad_dirty if self._pad_dirty > mp else m
        xp, twp, idp, rp = xp[:mp], twp[:mp], idp[:mp], rp[:mp]
        qc, qsc = qc[: self.tp * m], qsc[: self.tp * m]
        xp[:m].copy_(x)
        twp[:m].copy_(topk_weights)
        idp[:m].copy_(topk_ids)
        if tail is not None:
            res_in, res_out, norm_w = tail
            rp[:m].copy_(res_in)
            y, q, qs = self._moe(xp, twp, idp, None, (rp, rp, norm_w), local)
            res_out.copy_(rp[:m])
            # q / scales as [rank][m] rows (the kernel's are [rank][mp])
            src = [self.rank] if local else range(self.tp)
            for r in src:
                qc[r * m : (r + 1) * m].copy_(q[r * mp : r * mp + m])
                qsc[r * m : (r + 1) * m].copy_(qs[r * mp : r * mp + m])
            q, qs = qc, qsc
        elif gather:
            ya = self._moe(xp, twp, idp, None, None, local, gather)
            if out is None:
                out = torch.empty(
                    (self.tp * m, self.H), dtype=ya.dtype, device=ya.device
                )
            for r in range(self.tp):
                _kcopy(out[r * m : (r + 1) * m], ya[r * mp : r * mp + m])
            return out
        else:
            y = self._moe(xp, twp, idp, None, None, local)
        y = y[:m]
        if out is not None:
            y = out.copy_(y)
        return y if tail is None else (y, q, qs)

    @staticmethod
    def tailq(tail, local, gather=False):
        """the kernel variant (compile_bigm_kernel's tailq) of a moe() call"""
        return 3 if gather else 0 if not tail else 2 if local else 1

    def _io_launch(self, m, tailq):
        """the kernel of m (a multiple of PAD) local rows"""
        T = m * self.tp
        eps, gemma, lim = self.cfg
        w_, opts = self._plan(T)
        order = 4 if self.tp * max(self._split(m)) >= 8192 else 3
        return compile_bigm_kernel(
            self.H,
            self.I,
            self.E,
            self.K,
            self.tp,
            len(w_),
            eps,
            gemma,
            lim,
            NAB=self.nab,
            order=order,
            opts=opts,
            tailq=tailq,
            act=self.act,
        )

    def io_rows_per_kernel(self, lo, hi):
        """One local row count per distinct kernel (whatever the tail) among the batches of
        lo .. hi local rows (padded to PAD as moe() does): warming these up compiles every big-M
        kernel such batches can launch."""
        reps = {}
        for m in range(-(-max(lo, 1) // PAD) * PAD, -(-hi // PAD) * PAD + 1, PAD):
            reps.setdefault(self._io_launch(m, 0), m)
        return sorted(reps.values())

    def precompile(self, rows, tailq):
        """Collective: the kernels of these local row counts (tailq: 0 - 3 as moe()) into
        the disk cache, split over the ranks -- rank r compiles every tp-th -- so a cold cache
        costs about one compile, not one per kernel; the moe() warm-ups then load them. No-op
        without a disk cache."""
        if not _bigm_cache_dir():
            return
        mine = [m for i, m in enumerate(rows) if i % self.tp == self.rank]
        for m in mine:
            launch = self._io_launch(m, tailq)
            if launch in self._ready:
                continue
            # (compile only: the argument values do not matter, their types are the launch's)
            n = len(inspect.signature(launch.func).parameters)
            _bigm_cached(launch, (0,) * (n - 1) + (torch.cuda.current_stream(),))
        dist.barrier(group=self.sd.group)

    def _moe(self, x, topk_weights, topk_ids, out, tail, local, gather=False):
        """moe() of m (a multiple of PAD) rows"""
        m = x.shape[0]
        T = m * self.tp
        w1, w1s, w2, w2s = self.w
        y = self.normed[:m] if out is None else out
        if tail is not None:
            res_in, res_out, norm_w = tail
        else:
            res_in, res_out, norm_w = x, self.resout, x
        launch = self._io_launch(m, self.tailq(tail is not None, local, gather))
        chunks = self._chunks(T)  # (first: it may grow the unit table)
        self.ctrl[:BIGM_CTRL_INTS].zero_()
        self._run(
            launch,
            (
                self.sd.base,
                self.sd.bases.data_ptr(),
                self.sd.dev_comm.ptr,
                self._cons[self._cpar],
                self.offs.data_ptr(),
                x.data_ptr(),
                topk_ids.data_ptr(),
                topk_weights.data_ptr(),
                norm_w.data_ptr(),
                res_in.data_ptr(),
                y.data_ptr(),
                res_out.data_ptr(),
                self.ybuf.data_ptr(),
                self.routes.data_ptr(),
                self.lrix.data_ptr(),
                self.lwt.data_ptr(),
                self.utab.data_ptr(),
                w1.data_ptr(),
                w1s.data_ptr(),
                w2.data_ptr(),
                w2s.data_ptr(),
                self.ctrl.data_ptr(),
                chunks.data_ptr(),
                self.rank,
                m,
                self._ucap(T),
                self.cu,
                torch.cuda.current_stream(),
            ),
        )
        self._cpar ^= 1  # (the next launch reads what this one's GW wrote)
        if gather:
            return self.sd.view("q", (T, self.H), torch.bfloat16)
        if tail is None:
            return y
        q = self.sd.view("q", (T, self.H), torch.uint8)
        qs = self.sd.view("qs", (T,), torch.float32)
        return y, q, qs
