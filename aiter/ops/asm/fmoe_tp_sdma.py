# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Launcher and IPC/SDMA context for fused TP MoE + SDMA ``.co`` kernels."""

import contextlib
import ctypes
import functools
import hashlib
import math
import os
import threading
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import Tensor

import aiter
from aiter import dtypes
from aiter.jit.core import AITER_ASM_DIR, build_ctypes_module
from aiter.ops.asm import asm_utils as _au

_CSV_STR = frozenset({"a_dtype", "act", "epi_mode", "knl_name", "co_name", "sha256"})
_KFD = "/sys/class/kfd/kfd/topology/nodes"
_RING_B, _FLAG_B, _INT4_B = 256 * 1024, 4096, 1168
_CTRL_WORDS, _CTRL_TOTAL_WORDS = 40960, 81920
_CTRL_HRDY_BASE, _CTRL_MAX_HRDY_TILES, _CTRL_ERR = 5120, 35840, 63
_CTRL_LAYOUT_VERSION = 2
_EVEN_SDMA_ENG_MASK = 0x55555555
_MAX_MUBUF_P8_BYTES, _MAX_IPC_INBOX_BYTES = 0xFFFFF000, 0x7FFFF000
_WIN_CFGS = {
    8192: ((4096, 3072, 1024),),
    4096: ((2048, 1024, 1024),),
    3584: ((2048, 1024, 512),),
    2560: ((1280, 768, 512),),
}
_EPI_Q = ("add_rmsnorm_fp4", "add_rmsnorm_fp8", "rmsnorm_only_fp4", "rmsnorm_only_fp8")
EPI_MODES = ("add_rmsnorm", "rmsnorm_only", "raw_bf16", *_EPI_Q)
_ERR_NAMES = {1: "RS flag timeout", 2: "AG flag timeout", 4: "local counter timeout"}
_LOCK, _LOADED, _POOL = threading.RLock(), set(), {}
_U64_29, _U32_3 = ctypes.c_uint64 * 29, ctypes.c_uint32 * 3


class _FmoeTpKernarg(ctypes.Structure):
    _layout_, _pack_ = "ms", 1
    _fields_ = [("u64", _U64_29), ("u32", _U32_3)]


class _HipIpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * 64)]


@dataclass
class TpMoeCall:
    x: Tensor
    topk_ids: Tensor
    topk_weights: Tensor
    w1: Tensor
    w2: Tensor
    w1_scale: Tensor
    w2_scale: Tensor
    residual: Tensor | None = None
    norm_weight: Tensor | None = None
    eps: float = 1e-6
    gemma: bool = False
    x_scale: Tensor | None = None


@dataclass
class TpMoeOut:
    out: Tensor
    residual_out: Tensor | None = None
    out_scale: Tensor | None = None


def quantize_moe_input(x: Tensor, a_dtype: str = "fp4") -> tuple[Tensor, Tensor]:
    """Quantize ``[T, D]`` bf16/fp16 activations into row-major ``1x32`` MX format."""
    if a_dtype not in ("fp4", "fp8") or x.dim() != 2 or x.shape[1] % 32 != 0:
        raise ValueError(f"bad a_dtype={a_dtype!r} or shape={tuple(x.shape)}")
    t, d = int(x.shape[0]), int(x.shape[1])
    xc = x if x.is_contiguous() else x.contiguous()
    qdt, sdt = (dtypes.fp8 if a_dtype == "fp8" else dtypes.fp4x2), dtypes.fp8_e8m0
    xq, xs = aiter.per_1x32_mx_quant_hip(xc, quant_dtype=qdt, scale_type=sdt)
    return xq, xs.view(t, d // 32)


@functools.lru_cache(maxsize=64)
def _sort_ws_size(tokens: int, num_experts: int, topk: int, dev_idx: int) -> int:
    fn = aiter.moe_sorting_opus_get_workspace_size
    return int(fn(tokens, num_experts, topk, 0, dev_idx))


@functools.lru_cache(maxsize=1)
def _hip_ext():
    hip = _au.load_hip()
    vp, sz, u32, i32 = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_int
    hip.hipExtMallocWithFlags.argtypes = [ctypes.POINTER(vp), sz, u32]
    hip.hipFree.argtypes = hip.hipIpcCloseMemHandle.argtypes = [vp]
    hip.hipMemset.argtypes, hip.hipDeviceSynchronize.argtypes = [vp, i32, sz], []
    hip.hipMemcpy.argtypes = [vp, vp, sz, i32]
    hip.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_HipIpcHandle), vp]
    hip.hipIpcOpenMemHandle.argtypes = [ctypes.POINTER(vp), _HipIpcHandle, u32]
    hip.hipDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, i32, i32]
    return hip


@functools.lru_cache(maxsize=1)
def _hsakmt_lib():
    lib = ctypes.CDLL(build_ctypes_module("module_fmoe_tp_sdma_hsakmt"))
    u32, u64, i32 = ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int
    lib.xs_open.argtypes = lib.xs_close.argtypes = []
    lib.xs_node_for_gpu_id.argtypes = [u64, ctypes.POINTER(u32)]
    lib.xs_create.argtypes = [u32, u32, u64, i32, ctypes.POINTER(u64)]
    lib.xs_destroy.argtypes = [u64, u64, u64]
    return lib


def _alloc_uncached(nbytes: int) -> int:
    sz, hip, buf = -(-int(nbytes) // 4096) * 4096, _hip_ext(), ctypes.c_void_p()
    _au.hip_check(hip.hipExtMallocWithFlags(ctypes.byref(buf), sz, 3), "ExtMalloc")
    ptr = int(buf.value or 0)
    try:
        _au.hip_check(hip.hipMemset(buf, 0, sz), "hipMemset")
        _au.hip_check(hip.hipDeviceSynchronize(), "hipDeviceSynchronize")
    except Exception:
        if ptr:
            with contextlib.suppress(Exception):
                hip.hipFree(ctypes.c_void_p(ptr))
        raise
    return ptr


def _kfd_props(path: str) -> dict[str, int]:
    with contextlib.suppress(OSError), open(path) as f:
        return {p[0]: int(p[1], 0) for ln in f if len(p := ln.split()) == 2}
    return {}


def _kfd_node(device: int) -> tuple[int, int]:
    buf = ctypes.create_string_buffer(64)
    _au.hip_check(_hip_ext().hipDeviceGetPCIBusId(buf, 64, device), "GetPCIBusId")
    dom_s, bus_s, rest = buf.value.decode().split(":")
    dev_s, fn_s = rest.split(".")
    loc = (int(bus_s, 16) << 8) | (int(dev_s, 16) << 3) | int(fn_s, 16)
    for n in sorted(x for x in os.listdir(_KFD) if x.isdigit()):
        p = _kfd_props(f"{_KFD}/{n}/properties")
        if p.get("location_id") == loc and p.get("domain", 0) == int(dom_s, 16):
            with contextlib.suppress(OSError), open(f"{_KFD}/{n}/gpu_id") as f:
                if gid := int(f.read().strip() or 0):
                    return int(n), gid
    raise RuntimeError(f"no KFD node found for HIP device {device}")


def _max_sorted_tiles(tokens: int, num_experts: int, topk: int, tile_m: int) -> int:
    return int(num_experts) + (
        (max(int(tokens), 1) - 1) * int(topk) + int(tile_m) - 1
    ) // int(tile_m)


def _select_even_sdma_engines(mask: int) -> list[int]:
    even_mask = int(mask) & _EVEN_SDMA_ENG_MASK
    return [b for b in range(32) if (even_mask >> b) & 1]


def _select_tile_m(
    tokens: int,
    num_experts: int,
    model_dim: int,
    inter_dim: int,
    topk: int,
    reg: dict,
    mk,
    tm_req,
) -> int:
    nb32 = _max_sorted_tiles(tokens, num_experts, topk, 32)
    nb64 = _max_sorted_tiles(tokens, num_experts, topk, 64)
    if tm_req is None or tm_req == "auto":
        flash_next_sig = model_dim == 2560 and inter_dim == 256 and topk == 10
        bm32_pref = (
            tokens <= 4096 if flash_next_sig else tokens * topk <= 28 * num_experts
        )
        use32 = bm32_pref and nb32 <= _CTRL_MAX_HRDY_TILES and mk(32) in reg
        tm = 32 if use32 else (64 if mk(64) in reg or mk(32) not in reg else 32)
    elif (tm := int(tm_req)) not in (32, 64):
        raise ValueError(f"unsupported tile_m={tm_req!r}, expected 32 or 64")
    n_tiles_max = nb32 if tm == 32 else nb64
    if n_tiles_max > _CTRL_MAX_HRDY_TILES:
        raise ValueError(
            f"tile_m={tm} tiles ({n_tiles_max}) exceed {_CTRL_MAX_HRDY_TILES} HRDY limit"
        )
    return tm


def _open_sdma(st: SimpleNamespace, rank: int, src: int, gid: int, nodes: list[int]):
    lib = _hsakmt_lib()
    if rc := lib.xs_open():
        raise RuntimeError(f"hsaKmtOpenKFD failed: {rc}")
    st.sdma_open = True
    hk = ctypes.c_uint32(0)
    if rc := lib.xs_node_for_gpu_id(gid, ctypes.byref(hk)):
        raise RuntimeError(f"xs_node_for_gpu_id({gid}) failed: {rc}")
    nq = max(len(nodes) * 2, 16)
    st.quiet = _alloc_uncached(nq * 8 + 64)
    st.sdma_err = st.quiet + nq * 8
    rows = torch.zeros((nq, 16), dtype=torch.int64)
    for peer, dst in enumerate(nodes):
        if peer == rank:
            continue
        mask = 0
        for s in ("io_links", "p2p_links"):
            if not mask and os.path.isdir(base := f"{_KFD}/{src}/{s}"):
                for lk in sorted(os.listdir(base), key=int):
                    p = _kfd_props(f"{base}/{lk}/properties")
                    if p.get("node_to") == dst and (
                        m := p.get("recommended_sdma_engine_id_mask", 0)
                    ):
                        mask = m
                        break
        if not (engs := _select_even_sdma_engines(mask)):
            raise RuntimeError(
                f"no even gfx950 SDMA engine for KFD {src} -> {dst} (mask={mask:#x})"
            )
        for q in range(2):
            eng, out = engs[q % len(engs)], (ctypes.c_uint64 * 7)()
            if rc := lib.xs_create(hk.value, eng, _RING_B, 3, out):
                raise RuntimeError(f"SDMA create failed (peer={peer}): {rc}")
            r_va, qid, rp_va, wp_va, db_va, rp, wp = map(int, out)
            st.queues.append((qid, r_va, 0))
            sh = _alloc_uncached(_RING_B)
            st.queues[-1] = (qid, r_va, sh)
            i = peer * 2 + q
            hd = [r_va, _RING_B, rp_va, wp_va, db_va, wp, wp, rp, st.quiet + 8 * i]
            rows[i, :9] = torch.tensor(hd)
            rows[i, 10:15] = torch.tensor([eng, peer, sh, 1, st.sdma_err])
    st.handles = rows.to(st.dev)


def _free_state(st: SimpleNamespace, dev: torch.device) -> None:
    with contextlib.suppress(Exception):
        torch.cuda.synchronize(dev)
    hip = _hip_ext()
    if st.sdma_open:
        lib = _hsakmt_lib()
        for qid, ring, sh in st.queues:
            with contextlib.suppress(Exception):
                lib.xs_destroy(qid, ring, _RING_B)
            if sh:
                with contextlib.suppress(Exception):
                    hip.hipFree(ctypes.c_void_p(sh))
        st.queues.clear()
        with contextlib.suppress(Exception):
            lib.xs_close()
        st.sdma_open = False
    for b in filter(None, st.opened):
        with contextlib.suppress(Exception):
            hip.hipIpcCloseMemHandle(ctypes.c_void_p(b))
    for p in filter(None, (st.quiet, st.send_ptr, st.ipc_ptr)):
        with contextlib.suppress(Exception):
            hip.hipFree(ctypes.c_void_p(p))


@functools.lru_cache(maxsize=8)
def load_fmoe_tp_sdma_registry(co_dir=None):
    base = os.environ.get("AITER_ASM_DIR", AITER_ASM_DIR)
    raw = os.fspath(co_dir) if co_dir else os.getenv("AITER_ASM_TP_CO_DIR")
    rdir = os.path.abspath(raw or f"{base}/gfx950/fmoe_tp_sdma")
    rows = _au.load_asm_cfg_csv(f"{rdir}/fmoe_tp_sdma.csv", _CSV_STR)
    ks, ts = ("world", "model_dim", "inter_dim", "topk"), ("a_dtype", "act", "epi_mode")
    rk = lambda r: (*(r[c] for c in ks), r.get("tile_m", 64), *(r[c] for c in ts))
    reg = {}
    for r in rows:
        ver = r.get("layout_version")
        if ver != _CTRL_LAYOUT_VERSION:
            raise RuntimeError(
                f"fmoe_tp_sdma.csv row {r.get('co_name')!r} has layout_version={ver!r}, "
                f"expected {_CTRL_LAYOUT_VERSION}"
            )
        co_path = f"{rdir}/{r['co_name']}"
        if not os.path.isfile(co_path):
            raise FileNotFoundError(f"missing precompiled .co binary: {co_path}")
        if digest := r.get("sha256", ""):
            with open(co_path, "rb") as f:
                actual = hashlib.sha256(f.read()).hexdigest()
            if actual != digest:
                raise RuntimeError(
                    f"sha256 mismatch for {co_path}: expected {digest}, got {actual}"
                )
        reg[(*rk(r), r["gemma"])] = (co_path, r["knl_name"])
    return reg


class TpAsmContext:
    """Reference-counted per-TP-group IPC, SDMA queue, and workspace state."""

    def __init__(
        self, group, device, max_tokens, model_dim, topk, num_experts, inter_dim, **kw
    ):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("TpAsmContext constructed during CUDA Graph capture")
        dev = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        if (arch := _au.get_gfx_from_device(dev)) != "gfx950":
            raise NotImplementedError(f"fmoe_tp_sdma requires gfx950, got {arch!r}")
        self.device, self.device_idx = dev, int(dev.index)
        self.group = group
        self.co_dir = kw.get("co_dir")
        self.slot_id = int(kw.get("slot_id", 0))
        if self.slot_id < 0:
            raise ValueError(f"slot_id must be >= 0, got {self.slot_id}")
        self.rank = int(dist.get_rank(group=group))
        self.world = int(dist.get_world_size(group=group))
        if self.world not in (2, 4, 8) or int(model_dim) not in _WIN_CFGS:
            raise ValueError(f"unsupported world={self.world} or model_dim={model_dim}")
        self.max_tokens, self.model_dim = int(max_tokens), int(model_dim)
        self.topk, self.num_experts = int(topk), int(num_experts)
        self.inter_dim, win_cfgs = int(inter_dim), _WIN_CFGS[self.model_dim]
        sk = (self.max_tokens, self.model_dim, self.topk, self.num_experts)
        pkey = (id(group), self.device_idx, *sk, self.inter_dim, self.slot_id)
        self._closed, self._pkey = False, pkey
        if (st := _POOL.get(pkey)) is not None:
            st.refcount += 1
            self._st = st
            return

        bw = 2048 * self.world
        ru = bw // math.gcd(self.model_dim, bw)
        w_rus = [bw // math.gcd(w, bw) for w in {w for ws in win_cfgs for w in ws}]
        rus = (math.lcm(ru, *w_rus), *w_rus)
        pr = -(-self.max_tokens // 64) * 64 * self.topk
        p8_max_win = max(max(w[0], w[1], sum(w[2:])) for w in win_cfgs)
        if pr * p8_max_win > _MAX_MUBUF_P8_BYTES:
            raise ValueError(
                f"P8 buffer {pr * p8_max_win} B exceeds 32-bit MUBUF limit "
                f"{_MAX_MUBUF_P8_BYTES:#x}"
            )
        cmb_rows = max(-(-self.max_tokens // u) * u for u in rus)
        inbox_bytes = ((-(-cmb_rows // ru) * ru) * self.model_dim // 2048) * _INT4_B
        ipc_bytes = _FLAG_B + 4 * inbox_bytes
        if ipc_bytes >= 2**31 or inbox_bytes > _MAX_IPC_INBOX_BYTES:
            raise ValueError(f"IPC buffer {ipc_bytes} B exceeds 2 GiB int32 range")
        nb64_max = _max_sorted_tiles(self.max_tokens, self.num_experts, self.topk, 64)
        if nb64_max > _CTRL_MAX_HRDY_TILES:
            raise ValueError(
                f"max_tokens={self.max_tokens} requires {nb64_max} BM64 tiles "
                f"> {_CTRL_MAX_HRDY_TILES} HRDY limit"
            )

        st = SimpleNamespace(dev=dev, inbox_bytes=inbox_bytes, ipc_ptr=0, send_ptr=0)
        st.quiet, st.sdma_err, st.sdma_open, st.refcount = 0, 0, False, 1
        st.queues, st.opened = [], []
        try:
            with torch.cuda.device(dev):
                src, gid = _kfd_node(self.device_idx)
                st.ipc_ptr = _alloc_uncached(ipc_bytes)
                st.send_ptr = _alloc_uncached(inbox_bytes)
                hip, h, vp = _hip_ext(), _HipIpcHandle(), ctypes.c_void_p(st.ipc_ptr)
                _au.hip_check(hip.hipIpcGetMemHandle(ctypes.byref(h), vp), "ipcGet")
                me = (bytes(ctypes.string_at(ctypes.byref(h), 64)), src)
                objs = [[me if i == self.rank else None] for i in range(self.world)]
                for i, sr in enumerate(dist.get_process_group_ranks(group=group)):
                    dist.broadcast_object_list(objs[i], sr, group, device="cpu")
                bases = [st.ipc_ptr] * self.world
                for r in range(self.world):
                    if r != self.rank:
                        ph, out = _HipIpcHandle(), ctypes.c_void_p()
                        ctypes.memmove(ctypes.byref(ph), bytes(objs[r][0][0]), 64)
                        rc = hip.hipIpcOpenMemHandle(ctypes.byref(out), ph, 1)
                        _au.hip_check(rc, "ipcOpen")
                        bases[r] = int(out.value or 0)
                        st.opened.append(bases[r])
                st.peer_tab = torch.tensor(bases, dtype=torch.int64, device=dev)
                z32 = lambda n: torch.zeros(n, dtype=dtypes.i32, device=dev)
                eu8 = lambda *s: torch.empty(s, dtype=torch.uint8, device=dev)
                s8r = lambda w: -(-(w // 32) // 16) * 16
                pr = -(-self.max_tokens // 64) * 64 * self.topk
                p8_t = ((w[0], w[1], sum(w[2:])) for w in win_cfgs)
                s8_t = ((s8r(w[0]), s8r(w[1]), sum(map(s8r, w[2:]))) for w in win_cfgs)
                st.p8_wins = [eu8(pr, max(v)) for v in zip(*p8_t)]
                st.s8_wins = [eu8(pr, max(v)) for v in zip(*s8_t)]
                rt, ne = self.max_tokens * self.topk - self.topk, self.num_experts
                nb64, nb32 = -(-(rt + ne * 64) // 64), -(-(rt + ne * 32) // 32)
                max_nb, max_r = max(nb64, nb32, pr // 64), max(nb64 * 64, nb32 * 32)
                st.hq = eu8(max_nb * 64, self.inter_dim // 2)
                st.hss = eu8(max_nb, 2 * (((self.inter_dim // 32) + 7) // 8) * 256)
                st.ss_part = torch.empty(self.max_tokens, dtype=dtypes.fp32, device=dev)
                ei32 = lambda n: torch.empty(n, dtype=dtypes.i32, device=dev)
                st.s_ids, st.s_exp, st.n_val = ei32(max_r), ei32(max_nb), ei32(2)
                st.s_w = torch.empty(max_r, dtype=dtypes.fp32, device=dev)
                st.mbuf = torch.empty((0, 0), dtype=dtypes.bf16, device=dev)
                ps = {1, min(128, self.max_tokens), self.max_tokens}
                ws_m = max(_sort_ws_size(t, ne, self.topk, self.device_idx) for t in ps)
                st.sort_ws = eu8(ws_m)
                st.err, st.epoch = z32(4), z32(1)
                st.ctrl = z32(_CTRL_TOTAL_WORDS)
                torch.cuda.synchronize(dev)
                _open_sdma(st, self.rank, src, gid, [o[0][1] for o in objs])
        except Exception:
            _free_state(st, dev)
            raise
        dist.barrier(group=group)
        p8, s8 = st.p8_wins, st.s8_wins
        ws_ts = (st.hq, st.hss, p8[0], s8[0], p8[1], s8[1], p8[2], s8[2], st.peer_tab)
        tl = (st.send_ptr, st.ctrl.data_ptr(), st.err.data_ptr())
        eh = (st.epoch.data_ptr(), st.handles.data_ptr())
        st.mid_p = (*(t.data_ptr() for t in ws_ts), *tl, *eh)
        self._st = _POOL[pkey] = st

    def __getattr__(self, name: str):
        st = self.__dict__.get("_st")
        if st is not None and hasattr(st, name):
            return getattr(st, name)
        raise AttributeError(name)

    def _ensure_open(self) -> None:
        if getattr(self, "_closed", True):
            raise RuntimeError("TpAsmContext is closed")

    def acquire_arena(self, slot_id: int = 0) -> "TpAsmArenaLease":
        self._ensure_open()
        return TpAsmArenaLease(self, int(slot_id))

    def poison_ipc(self) -> None:
        self._ensure_open()
        vp = ctypes.c_void_p(self._st.ipc_ptr + _FLAG_B)
        rc = _hip_ext().hipMemset(vp, 0xFF, 4 * self._st.inbox_bytes)
        _au.hip_check(rc, "memset")

    def reset_epoch(
        self, val: int = 0, *, reset_ipc_flags: bool = True, group=None
    ) -> None:
        self._ensure_open()
        if not reset_ipc_flags:
            raise ValueError(
                "reset_epoch on live shared IPC state requires reset_ipc_flags=True"
            )
        grp = self.group if group is None else group
        has_dist = grp is not None and dist.is_initialized()
        torch.cuda.synchronize(self.device)
        if has_dist:
            dist.barrier(group=grp)
        rc = _hip_ext().hipMemset(ctypes.c_void_p(self._st.ipc_ptr), 0, _FLAG_B)
        _au.hip_check(rc, "hipMemset(ipc_flags)")
        self._st.ctrl.zero_()
        self._st.err.zero_()
        self._st.epoch.fill_(int(val))
        torch.cuda.synchronize(self.device)
        if has_dist:
            dist.barrier(group=grp)

    def reset(self, val: int = 0, *, group=None) -> None:
        self.reset_epoch(val=val, reset_ipc_flags=True, group=group)

    def check_error(self) -> None:
        self._ensure_open()
        torch.cuda.synchronize(self.device)
        st, v = self._st, ctypes.c_int32(0)
        rc = _hip_ext().hipMemcpy(ctypes.byref(v), ctypes.c_void_p(st.sdma_err), 4, 2)
        _au.hip_check(rc, "hipMemcpy(D2H)")
        c_err = int(st.ctrl[_CTRL_ERR].item()) | int(
            st.ctrl[_CTRL_WORDS + _CTRL_ERR].item()
        )
        code = int(st.err[0].item()) | c_err | int(v.value)
        if code:
            ns = ", ".join(nm for k, nm in _ERR_NAMES.items() if code & k) or "sdma"
            raise RuntimeError(f"rank {self.rank}: TP MoE error {code:#x} ({ns})")

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed, self._st.refcount = True, self._st.refcount - 1
        if self._st.refcount <= 0:
            _POOL.pop(self._pkey, None)
            _free_state(self._st, self.device)

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


class TpAsmArenaLease:
    """Explicit multi-slot arena lease for disjoint DBO microbatch execution."""

    def __init__(self, base_ctx: TpAsmContext, slot_id: int) -> None:
        base_ctx._ensure_open()
        self.slot_id = int(slot_id)
        if self.slot_id < 0:
            raise ValueError(f"slot_id must be >= 0, got {self.slot_id}")
        self._owns_ctx = self.slot_id != base_ctx.slot_id
        self.context = (
            TpAsmContext(
                base_ctx.group,
                base_ctx.device,
                base_ctx.max_tokens,
                base_ctx.model_dim,
                base_ctx.topk,
                base_ctx.num_experts,
                base_ctx.inter_dim,
                co_dir=base_ctx.co_dir,
                slot_id=self.slot_id,
            )
            if self._owns_ctx
            else base_ctx
        )

    def release(self) -> None:
        if self._owns_ctx:
            self.context.close()

    def __enter__(self) -> TpAsmContext:
        self.context._ensure_open()
        return self.context

    def __exit__(self, *_exc) -> None:
        self.release()


def _ranges_overlap(a: Tensor, b: Tensor) -> bool:
    a0, b0 = int(a.data_ptr()), int(b.data_ptr())
    return a0 < b0 + int(b.nbytes) and b0 < a0 + int(a.nbytes)


def tp_moe_layer_asm(
    call, ctx, epi_mode="add_rmsnorm", out=None, out_scale=None, **kw
) -> TpMoeOut:
    """Execute one TP rank's single-launch MoE + device-SDMA combine + epilogue."""
    if isinstance(ctx, TpAsmArenaLease):
        ctx = ctx.context
    ctx._ensure_open()
    res_o, co_dir = kw.get("residual_out"), kw.get("co_dir")
    a_dt, act_n = kw.get("a_dtype", "fp4"), kw.get("act", "silu")
    act_n = "situ" if act_n == "situv2" else act_n
    bad_m = epi_mode not in EPI_MODES or a_dt not in ("fp4", "fp8")
    if bad_m or act_n not in ("silu", "situ"):
        raise ValueError(f"unsupported epi={epi_mode!r}, a_dtype={a_dt!r}")
    ok_1b = lambda t, s: t.shape == s and t.is_contiguous() and t.element_size() == 1
    if call.x.device != ctx.device or not call.x.is_contiguous():
        raise ValueError(f"x must be contiguous on {ctx.device}")
    if call.x_scale is not None:
        if call.x.dim() != 2 or call.x_scale.dim() != 2:
            raise ValueError("pre-quantized x and x_scale must be 2D")
        if call.x_scale.device != ctx.device:
            raise ValueError(f"x_scale must be on {ctx.device}")
        tokens, d = int(call.x.shape[0]), int(call.x_scale.shape[1]) * 32
        xc = d // 2 if a_dt == "fp4" else d
        if not (ok_1b(call.x, (tokens, xc)) and ok_1b(call.x_scale, (tokens, d // 32))):
            raise ValueError(f"invalid pre-quantized x/x_scale for a_dtype={a_dt!r}")
        xq, xs = call.x, call.x_scale
    else:
        if call.x.dim() != 2 or call.x.dtype != dtypes.bf16:
            raise ValueError(
                f"expected 2D bf16 x, got {tuple(call.x.shape)} {call.x.dtype}"
            )
        tokens, d, xq, xs = int(call.x.shape[0]), int(call.x.shape[1]), None, None
    if call.topk_ids.dim() != 2 or call.topk_weights.dim() != 2:
        raise ValueError("topk_ids and topk_weights must be 2D")
    topk, num_experts = int(call.topk_ids.shape[1]), int(call.w2.shape[0])
    inter = int(call.w2.shape[2]) * 2
    ctx_s = (ctx.model_dim, ctx.topk, ctx.inter_dim, ctx.num_experts)
    if not (0 < tokens <= ctx.max_tokens) or (d, topk, inter, num_experts) != ctx_s:
        raise ValueError(
            f"call (T={tokens}, D={d}, k={topk}, I={inter}, E={num_experts}) != ctx"
        )
    if call.topk_ids.shape != (tokens, topk) or not call.topk_ids.is_contiguous():
        raise ValueError(f"topk_ids must be contiguous {(tokens, topk)}")
    if call.topk_ids.device != ctx.device or call.topk_ids.dtype != torch.int32:
        raise ValueError(f"topk_ids must be int32 on {ctx.device}")
    if (
        call.topk_weights.shape != (tokens, topk)
        or not call.topk_weights.is_contiguous()
        or call.topk_weights.device != ctx.device
        or call.topk_weights.dtype != torch.float32
    ):
        raise ValueError(
            f"topk_weights must be contiguous fp32 {(tokens, topk)} on {ctx.device}"
        )
    for nm, t in (
        ("w1", call.w1),
        ("w2", call.w2),
        ("w1_scale", call.w1_scale),
        ("w2_scale", call.w2_scale),
    ):
        if t.device != ctx.device or not t.is_contiguous():
            raise ValueError(f"{nm} must be contiguous on {ctx.device}")
    is_add, is_norm = epi_mode.startswith("add_rmsnorm"), epi_mode != "raw_bf16"
    is_fp4_o, is_fp8_o = epi_mode.endswith("_fp4"), epi_mode.endswith("_fp8")
    bp = (float(kw.get("beta1", 4.0)), float(kw.get("beta2", 25.0)))
    bad_eps = is_norm and abs(float(call.eps) - 1e-6) > 1e-12
    if bad_eps or (act_n == "situ" and bp != (4.0, 25.0)):
        raise NotImplementedError("unsupported eps or situ beta for precompiled .co")
    if is_add:
        if call.residual is None:
            raise ValueError(f"missing residual for {epi_mode!r}")
        if (
            call.residual.shape != (tokens, d)
            or call.residual.dtype != dtypes.bf16
            or not call.residual.is_contiguous()
            or call.residual.device != ctx.device
        ):
            raise ValueError(
                f"residual must be contiguous bf16 {(tokens, d)} on {ctx.device}"
            )
    if is_norm:
        if call.norm_weight is None:
            raise ValueError(f"missing norm_weight for {epi_mode!r}")
        if (
            call.norm_weight.shape != (d,)
            or call.norm_weight.dtype != dtypes.bf16
            or not call.norm_weight.is_contiguous()
            or call.norm_weight.device != ctx.device
        ):
            raise ValueError(
                f"norm_weight must be contiguous bf16 {(d,)} on {ctx.device}"
            )

    emp = lambda s, dt: torch.empty(s, dtype=dt, device=ctx.device)
    if is_fp4_o or is_fp8_o:
        oc, odt = (d // 2, dtypes.fp4x2) if is_fp4_o else (d, dtypes.fp8)
        out = emp((tokens, oc), odt) if out is None else out
        sc_shape = (tokens, d // 32)
        out_scale = emp(sc_shape, dtypes.fp8_e8m0) if out_scale is None else out_scale
        if (
            out.device != ctx.device
            or out_scale.device != ctx.device
            or not (ok_1b(out, (tokens, oc)) and ok_1b(out_scale, sc_shape))
            or out.nbytes < tokens * oc
            or out_scale.nbytes < tokens * (d // 32)
            or out.dtype not in (odt, torch.uint8)
            or out_scale.dtype not in (dtypes.fp8_e8m0, torch.uint8)
        ):
            raise ValueError(f"invalid out or out_scale for {epi_mode!r}")
        if _ranges_overlap(out, out_scale):
            raise ValueError("out and out_scale must not overlap")
    else:
        out = emp((tokens, d), dtypes.bf16) if out is None else out
        if (
            out.shape != (tokens, d)
            or out.dtype != dtypes.bf16
            or out.nbytes < tokens * d * 2
            or not out.is_contiguous()
            or out.device != ctx.device
        ):
            raise ValueError(
                f"invalid bf16 out {tuple(out.shape)} {out.dtype} for {epi_mode!r}"
            )
        out_scale = None
    res_u = res_o if is_norm else None
    res_u = emp((tokens, d), dtypes.bf16) if (is_add and res_u is None) else res_u
    if is_add:
        if (
            res_u.shape != (tokens, d)
            or res_u.dtype != dtypes.bf16
            or res_u.nbytes < tokens * d * 2
            or not res_u.is_contiguous()
            or res_u.device != ctx.device
        ):
            raise ValueError(f"invalid residual_out {tuple(res_u.shape)} {res_u.dtype}")
        if _ranges_overlap(out, res_u) or (
            out_scale is not None and _ranges_overlap(out_scale, res_u)
        ):
            raise ValueError("residual_out must not overlap out or out_scale")

    gm = 0 if not is_norm else int(bool(call.gemma))
    reg = load_fmoe_tp_sdma_registry(ctx.co_dir if co_dir is None else co_dir)
    mk = lambda tm: (ctx.world, d, inter, topk, tm, a_dt, act_n, epi_mode, gm)
    tm = _select_tile_m(tokens, num_experts, d, inter, topk, reg, mk, kw.get("tile_m"))
    if (entry := reg.get(key := mk(tm))) is None:
        raise NotImplementedError(f"no precompiled .co for {key}")
    co_path, symbol = entry
    if (ck := (ctx.device_idx, co_path, symbol)) not in _LOADED:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"Cannot load {symbol} during CUDA Graph capture")
        with _LOCK, _au.device_guard(ctx.device_idx):
            _au.current_raw_stream(ctx.device_idx)
            _au.get_function(co_path, symbol, device=ctx.device_idx)
            _LOADED.add(ck)
    fn = _au.get_function(co_path, symbol, device=ctx.device_idx)

    st = ctx._st
    with _au.device_guard(ctx.device_idx):
        xq, xs = quantize_moe_input(call.x, a_dt) if xq is None else (xq, xs)
        nb = -(-(tokens * topk + num_experts * tm - topk) // tm)
        s_ids, s_exp, s_w = st.s_ids[: nb * tm], st.s_exp[:nb], st.s_w[: nb * tm]
        sa = (call.topk_ids, call.topk_weights, s_ids, s_w, s_exp, st.n_val, st.mbuf)
        if os.environ.get("AITER_USE_CK_MOE_SORTING", "0") == "1":
            aiter.moe_sorting_fwd(*sa, num_experts, tm)
        else:
            ws_sz = _sort_ws_size(tokens, num_experts, topk, ctx.device_idx)
            ws = st.sort_ws[:ws_sz] if ws_sz else None
            aiter.moe_sorting_opus_fwd(*sa, num_experts, tm, workspace=ws)
        mt = (xq, xs, call.w1, call.w1_scale, call.w2, call.w2_scale)
        st_t = (*mt, s_ids, s_w, s_exp, st.n_val)
        sc_buf = out_scale if out_scale is not None else st.ss_part
        et = (call.residual if is_add else None, res_u, sc_buf, call.norm_weight, out)
        ep = (0 if t is None else t.data_ptr() for t in et)
        u64s = (*(t.data_ptr() for t in st_t), *st.mid_p, *ep)
        karg = _FmoeTpKernarg(_U64_29(*u64s), _U32_3(ctx.rank, tokens, st.inbox_bytes))
        stream = _au.current_raw_stream(ctx.device_idx)
        _au.launch_co(fn, (512, 1, 1), (256, 1, 1), karg, stream)
    return TpMoeOut(out, res_u if is_add else res_o, out_scale)


def prepare_downstream_mx_inputs(
    out_q: Tensor, out_scale: Tensor, a_dtype: str = "fp4"
) -> tuple[Tensor, Tensor]:
    """Prepare ``tp_moe_layer_asm`` quantized epilogue outputs for AITER dense MX GEMMs.

    - ``a_dtype == "fp4"``: ``gemm_a4w4`` on gfx950 consumes ``mfma_tile`` (32x8)
      preshuffled ``e8m0`` scales, so ``out_scale`` is shuffled via
      ``fp4_utils.e8m0_shuffle``.
    - ``a_dtype == "fp8"``: ``gemm_a8w8_mxfp8_bpreshuffle`` on gfx950 consumes
      plain row-major ``[T, D // 32]`` ``e8m0`` scales directly with zero copy.
    """
    if a_dtype == "fp4":
        from aiter.utility import fp4_utils

        return out_q, fp4_utils.e8m0_shuffle(out_scale)
    if a_dtype == "fp8":
        return out_q, out_scale
    raise ValueError(f"unsupported a_dtype={a_dtype!r}, expected 'fp4' or 'fp8'")


__all__ = [
    "EPI_MODES",
    "TpAsmArenaLease",
    "TpAsmContext",
    "TpMoeCall",
    "TpMoeOut",
    "prepare_downstream_mx_inputs",
    "quantize_moe_input",
    "tp_moe_layer_asm",
]
