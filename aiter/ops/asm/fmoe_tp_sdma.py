# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Launcher and IPC/SDMA context for fused TP MoE + SDMA ``.co`` kernels."""

import contextlib
import ctypes
import functools
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

_CSV_STR = frozenset({"a_dtype", "act", "epi_mode", "knl_name", "co_name"})
_KFD = "/sys/class/kfd/kfd/topology/nodes"
_RING_B, _FLAG_B, _INT4_B, _CTRL_ERR = 256 * 1024, 4096, 1168, 63
_WINS = {8192: (4096, 3072, 1024), 4096: (2048, 1024, 1024)}
_WINS.update({3584: (2048, 1024, 512), 2560: (1280, 768, 512)})
EPI_MODES = ("add_rmsnorm", "rmsnorm_only", "raw_bf16")
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


@dataclass
class TpMoeOut:
    out: Tensor
    residual_out: Tensor | None = None


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
    _au.hip_check(hip.hipMemset(buf, 0, sz), "hipMemset")
    _au.hip_check(hip.hipDeviceSynchronize(), "hipDeviceSynchronize")
    return int(buf.value or 0)


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
        if not (engs := [b for b in range(32) if (mask >> b) & 1]):
            raise RuntimeError(f"no SDMA engine for KFD {src} -> {dst}")
        for q in range(2):
            eng, out = engs[q % len(engs)], (ctypes.c_uint64 * 7)()
            if rc := lib.xs_create(hk.value, eng, _RING_B, 3, out):
                raise RuntimeError(f"SDMA create failed (peer={peer}): {rc}")
            sh = _alloc_uncached(_RING_B)
            r_va, qid, rp_va, wp_va, db_va, rp, wp = map(int, out)
            st.queues.append((qid, r_va, sh))
            i = peer * 2 + q
            hd = [r_va, _RING_B, rp_va, wp_va, db_va, wp, wp, rp, st.quiet + 8 * i]
            rows[i, :9] = torch.tensor(hd)
            rows[i, 10:15] = torch.tensor([eng, peer, sh, 1, st.sdma_err])
    st.handles = rows.to(st.dev)


def _free_state(st: SimpleNamespace, dev: torch.device) -> None:
    torch.cuda.synchronize(dev)
    hip = _hip_ext()
    if st.sdma_open:
        lib = _hsakmt_lib()
        for qid, ring, sh in st.queues:
            lib.xs_destroy(qid, ring, _RING_B)
            if sh:
                hip.hipFree(ctypes.c_void_p(sh))
        st.queues.clear()
        lib.xs_close()
        st.sdma_open = False
    for b in st.opened:
        if b:
            hip.hipIpcCloseMemHandle(ctypes.c_void_p(b))
    for p in (st.quiet, st.send_ptr, st.ipc_ptr):
        if p:
            hip.hipFree(ctypes.c_void_p(p))


@functools.lru_cache(maxsize=8)
def load_fmoe_tp_sdma_registry(co_dir=None):
    base = os.environ.get("AITER_ASM_DIR", AITER_ASM_DIR)
    raw = os.fspath(co_dir) if co_dir else os.getenv("AITER_ASM_TP_CO_DIR")
    rdir = os.path.abspath(raw or f"{base}/gfx950/fmoe_tp_sdma")
    rows = _au.load_asm_cfg_csv(f"{rdir}/fmoe_tp_sdma.csv", _CSV_STR)
    ks = ("world", "model_dim", "inter_dim", "topk", "a_dtype", "act", "epi_mode")
    out = {}
    for r in rows:
        k = (*(r[c] for c in ks), r["gemma"])
        out[k] = (f"{rdir}/{r['co_name']}", r["knl_name"])
    return out


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
        self.co_dir = kw.get("co_dir")
        self.rank = int(dist.get_rank(group=group))
        self.world = int(dist.get_world_size(group=group))
        if self.world not in (2, 4, 8) or int(model_dim) not in _WINS:
            raise ValueError(f"unsupported world={self.world} or model_dim={model_dim}")
        self.max_tokens, self.model_dim = int(max_tokens), int(model_dim)
        self.topk, self.num_experts = int(topk), int(num_experts)
        self.inter_dim, wins = int(inter_dim), _WINS[self.model_dim]
        sk = (self.max_tokens, self.model_dim, self.topk, self.num_experts)
        pkey = (id(group), self.device_idx, *sk, self.inter_dim)
        self._closed, self._pkey = False, pkey
        if (st := _POOL.get(pkey)) is not None:
            st.refcount += 1
            self._st = st
            return

        bw = 2048 * self.world
        ru = bw // math.gcd(self.model_dim, bw)
        w_rus = [bw // math.gcd(w, bw) for w in wins]
        rus = (math.lcm(ru, *w_rus), *w_rus)
        cmb_rows = max(-(-self.max_tokens // u) * u for u in rus)
        inbox_bytes = ((-(-cmb_rows // ru) * ru) * self.model_dim // 2048) * _INT4_B
        ipc_bytes = _FLAG_B + 4 * inbox_bytes
        if ipc_bytes >= 2**31:
            raise ValueError(f"IPC buffer {ipc_bytes} B exceeds 2 GiB int32 range")

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
                pr = -(-self.max_tokens // 64) * 64 * self.topk
                st.p8_wins = [eu8(pr, w) for w in wins]
                st.s8_wins = [eu8(pr, -(-(w // 32) // 16) * 16) for w in wins]
                raw_s = self.max_tokens * self.topk + self.num_experts * 64 - self.topk
                mt = max(pr, -(-raw_s // 64) * 64) // 64
                st.hq = eu8(mt * 64, self.inter_dim // 2)
                st.hss = eu8(mt, 2 * (((self.inter_dim // 32) + 7) // 8) * 256)
                st.ss_part = torch.empty(self.max_tokens, dtype=dtypes.fp32, device=dev)
                st.err, st.epoch, st.ctrl = z32(4), z32(1), z32(max(10240, 2048 + mt))
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

    def poison_ipc(self) -> None:
        vp = ctypes.c_void_p(self._st.ipc_ptr + _FLAG_B)
        rc = _hip_ext().hipMemset(vp, 0xFF, 4 * self._st.inbox_bytes)
        _au.hip_check(rc, "memset")

    def check_error(self) -> None:
        torch.cuda.synchronize(self.device)
        st, v = self._st, ctypes.c_int32(0)
        rc = _hip_ext().hipMemcpy(ctypes.byref(v), ctypes.c_void_p(st.sdma_err), 4, 2)
        _au.hip_check(rc, "hipMemcpy(D2H)")
        code = int(st.err[0].item()) | int(st.ctrl[_CTRL_ERR].item()) | int(v.value)
        if code:
            ns = ", ".join(nm for k, nm in _ERR_NAMES.items() if code & k) or "sdma"
            raise RuntimeError(f"rank {self.rank}: TP MoE error {code:#x} ({ns})")

    def close(self) -> None:
        if not getattr(self, "_closed", True):
            self._closed = True
            self._st.refcount -= 1
            if self._st.refcount <= 0:
                _POOL.pop(self._pkey, None)
                _free_state(self._st, self.device)

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


def tp_moe_layer_asm(call, ctx, epi_mode="add_rmsnorm", out=None, **kw) -> TpMoeOut:
    """Execute one TP rank's single-launch MoE + device-SDMA combine + epilogue."""
    from aiter.ops.quant import fused_dynamic_mx_quant_moe_sort

    res_o, co_dir = kw.get("residual_out"), kw.get("co_dir")
    a_dt, act_n = kw.get("a_dtype", "fp4"), kw.get("act", "silu")
    act_n = "situ" if act_n == "situv2" else act_n
    bad_m = epi_mode not in EPI_MODES or a_dt not in ("fp4", "fp8")
    if bad_m or act_n not in ("silu", "situ"):
        raise ValueError(f"unsupported epi={epi_mode!r}, a_dtype={a_dt!r}")
    tokens, d = int(call.x.shape[0]), int(call.x.shape[1])
    topk, num_experts = int(call.topk_ids.shape[1]), int(call.w2.shape[0])
    inter, ctx_s = int(call.w2.shape[2]) * 2, (ctx.model_dim, ctx.topk, ctx.inter_dim)
    if not (0 < tokens <= ctx.max_tokens) or (d, topk, inter) != ctx_s:
        raise ValueError(f"call (T={tokens}, D={d}, k={topk}, I={inter}) != ctx")
    bp = (float(kw.get("beta1", 4.0)), float(kw.get("beta2", 25.0)))
    bad_eps = epi_mode != "raw_bf16" and abs(float(call.eps) - 1e-6) > 1e-12
    if bad_eps or (act_n == "situ" and bp != (4.0, 25.0)):
        raise NotImplementedError("unsupported eps or situ beta for precompiled .co")
    need_r = epi_mode == "add_rmsnorm" and call.residual is None
    if need_r or (epi_mode != "raw_bf16" and call.norm_weight is None):
        raise ValueError(f"missing residual or norm_weight for {epi_mode!r}")

    if out is None:
        out = torch.empty((tokens, d), dtype=dtypes.bf16, device=ctx.device)
    is_add = epi_mode == "add_rmsnorm"
    res_u = res_o if epi_mode in ("rmsnorm_only", "add_rmsnorm") else None
    if is_add and res_u is None:
        res_u = torch.empty_like(call.residual)

    gm = 0 if epi_mode == "raw_bf16" else int(bool(call.gemma))
    reg = load_fmoe_tp_sdma_registry(ctx.co_dir if co_dir is None else co_dir)
    key = (ctx.world, d, inter, topk, a_dt, act_n, epi_mode, gm)
    if (entry := reg.get(key)) is None:
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

    st, dev = ctx._st, ctx.device
    with _au.device_guard(ctx.device_idx):
        nb = -(-(tokens * topk + num_experts * 64 - topk) // 64)
        ei32 = lambda n: torch.empty(n, dtype=dtypes.i32, device=dev)
        s_ids, s_exp, n_val = ei32(nb * 64), ei32(nb), ei32(2)
        s_w = torch.empty(nb * 64, dtype=dtypes.fp32, device=dev)
        mbuf = torch.empty((0, 0), dtype=dtypes.bf16, device=dev)
        ids, wts = call.topk_ids, call.topk_weights
        sa = (ids, wts, s_ids, s_w, s_exp, n_val, mbuf, num_experts, 64)
        if os.environ.get("AITER_USE_CK_MOE_SORTING", "0") == "1":
            aiter.moe_sorting_fwd(*sa)
        else:
            ws_fn = aiter.moe_sorting_opus_get_workspace_size
            ws_sz = ws_fn(tokens, num_experts, topk, 0, ctx.device_idx)
            ws = torch.empty(ws_sz, dtype=torch.uint8, device=dev) if ws_sz else None
            aiter.moe_sorting_opus_fwd(*sa, workspace=ws)
        qdt = dtypes.fp8 if a_dt == "fp8" else dtypes.fp4x2
        qa = (call.x, s_ids, n_val, tokens, topk, 64, qdt, None, 32, s_w, num_experts)
        xq, xs = fused_dynamic_mx_quant_moe_sort(*qa)
        st.ctrl.zero_()
        mt = (xq, xs, call.w1, call.w1_scale, call.w2, call.w2_scale)
        st_t = (*mt, s_ids, s_w, s_exp, n_val)
        et = (call.residual, res_u, st.ss_part, call.norm_weight, out)
        ep = (0 if t is None else t.data_ptr() for t in et)
        u64s = (*(t.data_ptr() for t in st_t), *st.mid_p, *ep)
        karg = _FmoeTpKernarg(_U64_29(*u64s), _U32_3(ctx.rank, tokens, st.inbox_bytes))
        stream = _au.current_raw_stream(ctx.device_idx)
        _au.launch_co(fn, (512, 1, 1), (256, 1, 1), karg, stream)
    return TpMoeOut(out, res_u if is_add else res_o)


__all__ = ["EPI_MODES", "TpAsmContext", "TpMoeCall", "TpMoeOut", "tp_moe_layer_asm"]
