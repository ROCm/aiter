# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host side of the single-kernel TP MegaMoE (launch configs, buffers, launches)."""

from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from ...quick_allreduce_int4_ipc import UncachedIpcHeap as _Ipc
from ..kernels_common import ceildiv
from ..tensor_shim import _preload_compiled, _run_compiled
from .mega_moe_tp_config import (
    CTRL_ERR,
    CTRL_INTS,
    DYN_MAX,
    ERR_CHUNK,
    ERR_COMM,
    ERR_FLAG,
    ERR_META,
    ERR_YAG,
    FLAG_INTS,
    MAX_TP,
    NCHLB_MAX,
    NCTA_MAX,
    NMETA_CAP,
    TN_MAX,
    gemm2_chunk_groups,
    gemm2_group_step,
    mega_moe_tp_consts,
    mega_moe_tp_shape_supported,
)
from .mega_moe_tp_kernel import compile_mega_moe_tp

__all__ = [
    "LaunchCfg",
    "MegaMoeTPEngine",
    "mega_moe_tp_shape_supported",
]

LDS_LIMIT = 160 * 1024
LB_PF_MAX = 1024
AG8_MIN = 512
COMM_MODES = ("ag_rs", "ar")
_ERR_NAMES = (
    (ERR_FLAG, "flag"),
    (ERR_META, "routing"),
    (ERR_CHUNK, "input chunk"),
    (ERR_COMM, "reduce"),
    (ERR_YAG, "output gather"),
)


@dataclass(frozen=True)
class LaunchCfg:
    """One kernel variant: row tile 16 * mt; ll / llr: LL reduce-scatter packets / LL
    route rows too; npp: GEMM1 column blocks per A pass; ag8: MXFP8 all-reduce second
    hop; lb: large-batch schedule (pf: GEMM2 prefetch, lp: forced GEMM1 inter pieces);
    tn: fused tail (1: FP8 rows, 2: + bf16 rows)."""

    mt: int
    ll: bool = False
    llr: bool = False
    npp: int = 1
    ag8: bool = False
    lb: bool = False
    tn: int = 0
    pf: bool = False
    lp: int = 0


# --- intra-node symmetric memory (hipIpc, via quick_allreduce_int4_ipc) ---------------

_ALIGN = 256
# a peer allocation maps once per process (arenas can share one): handle -> [ptr, refs]
_OPEN: dict[bytes, list] = {}


def _allocation_base(ptr: int) -> int:
    hip = _Ipc._load_hip()
    hip.hipMemGetAddressRange.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
    ]
    base, size = ctypes.c_void_p(), ctypes.c_size_t()
    err = hip.hipMemGetAddressRange(
        ctypes.byref(base), ctypes.byref(size), ctypes.c_void_p(ptr)
    )
    _Ipc._hip_check(err, what="hipMemGetAddressRange")
    return int(base.value or 0)


def _ipc_open(handle: bytes) -> int:
    ent = _OPEN.get(handle)
    if ent is None:
        ent = _OPEN[handle] = [_Ipc.open_mem_handle(handle), 0]
    ent[1] += 1
    return ent[0]


def _ipc_close(handle: bytes) -> None:
    ent = _OPEN[handle]
    ent[1] -= 1
    if ent[1] == 0:
        del _OPEN[handle]
        _Ipc.close_mem_handle(ent[0])


@contextlib.contextmanager
def _no_expandable_segments():
    # hipIpcGetMemHandle cannot export expandable-segment (VMM) memory
    conf = os.environ.get("PYTORCH_HIP_ALLOC_CONF") or os.environ.get(
        "PYTORCH_CUDA_ALLOC_CONF", ""
    )
    on = "expandable_segments:true" in conf.replace(" ", "").lower()
    if on:
        torch.cuda.memory._set_allocator_settings("expandable_segments:False")
    try:
        yield
    finally:
        if on:
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")


@dataclass
class _Slice:
    offset: int
    nbytes: int
    shape: tuple
    dtype: torch.dtype
    local: torch.Tensor | None = None


class SymmetricArena:
    """A same-layout device arena (torch caching-allocator memory) on every rank of
    ``group``, mapped into all peers with hipIpc. The exported handle covers the
    allocator segment holding the arena (peers address it by its offset in it).
    Destruction is not collective: synchronize every rank before dropping an instance.
    In containers on the host network ROCm 7.1 needs HSA_ENABLE_IPC_MODE_LEGACY=1."""

    def __init__(self, *, group=None, device: torch.device | None = None):
        self.group = group
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self._slices: dict[str, _Slice] = {}
        self._cursor = 0
        self._storage: torch.Tensor | None = None
        self._base_ptrs: tuple[int, ...] = ()
        self._opened: list[bytes] = []

    def reserve(self, name: str, shape, dtype: torch.dtype) -> _Slice:
        """Carve out a named region. Must run in the same order on every rank."""
        if self._storage is not None or name in self._slices:
            raise RuntimeError(f"cannot reserve {name!r}")
        shape = tuple(int(s) for s in shape)
        offset = ceildiv(self._cursor, _ALIGN) * _ALIGN
        nbytes = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
        self._cursor = offset + nbytes
        self._slices[name] = _Slice(offset, nbytes, shape, dtype)
        return self._slices[name]

    def commit(self) -> SymmetricArena:
        """Collective: allocate and map every rank's arena (raises on every rank if any
        rank fails)."""
        total = ceildiv(self._cursor, _ALIGN) * _ALIGN
        with _no_expandable_segments():
            self._storage = torch.zeros(total, dtype=torch.uint8, device=self.device)
        base_ptr = int(self._storage.data_ptr())
        for s in self._slices.values():
            s.local = self._storage[s.offset : s.offset + s.nbytes].view(s.dtype)
            s.local = s.local.view(s.shape)
        try:
            with torch.cuda.device(self.device):
                handle = _Ipc.get_mem_handle_bytes(base_ptr)
                payload = (handle, base_ptr - _allocation_base(base_ptr), total)
            err = ""
        except RuntimeError as exc:
            payload, err = None, str(exc)
        torch.cuda.synchronize(self.device)
        gathered: list = [None] * self.world_size
        dist.all_gather_object(gathered, (payload, err), group=self.group)
        bad = {r: e for r, (_, e) in enumerate(gathered) if e}
        if not bad and any(p[2] != total for p, _ in gathered):
            bad = {r: f"{p[2]} B, not {total} B" for r, (p, _) in enumerate(gathered)}
        ptrs, err = [], ""
        if not bad:
            try:
                with torch.cuda.device(self.device):
                    for r, ((handle, off, _), _) in enumerate(gathered):
                        if r == self.rank:
                            ptrs.append(base_ptr)
                        else:
                            ptrs.append(_ipc_open(handle) + off)
                            self._opened.append(handle)
            except RuntimeError as exc:
                err = str(exc)
            opened: list = [None] * self.world_size
            dist.all_gather_object(opened, err, group=self.group)
            bad = {r: e for r, e in enumerate(opened) if e}
        if bad:
            self.close()
            raise RuntimeError(f"SymmetricArena: mapping failed on ranks {bad}")
        self._base_ptrs = tuple(ptrs)
        dist.barrier(group=self.group)
        return self

    def close(self) -> None:
        """Unmap the peers' arenas (no kernel may use them any more)."""
        if self._opened:
            torch.cuda.synchronize(self.device)
            for handle in self._opened:
                _ipc_close(handle)
            self._opened = []
            self._base_ptrs = ()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110 (interpreter teardown)
            pass

    def barrier(self) -> None:
        """Collective; also waits for this device, so a barrier implemented as a GPU
        kernel (NCCL) never runs next to the next (persistent) launch."""
        dist.barrier(group=self.group)
        torch.cuda.synchronize(self.device)

    @property
    def storage(self) -> torch.Tensor:
        return self._storage

    @property
    def base_ptrs(self) -> tuple[int, ...]:
        return self._base_ptrs


class MegaMoeTPLDSError(ValueError):
    """max_local_tokens does not fit in LDS (retry with a smaller one)."""


def _lb_param(value, env: str, default):
    # explicit config value, else the AITER_MEGAMOE_TP_LB_* env var, else the default
    return value if value is not None else os.environ.get(env, default)


def _mt_table(spec) -> tuple[int, list[tuple[int, int]]]:
    *rows, last = [x.strip() for x in str(spec).split(",") if x.strip()]
    table = sorted(tuple(int(v) for v in r.split(":")) for r in rows)
    return int(last), table


class MegaMoeTPEngine:
    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        model_dim: int,
        inter_dim: int,
        experts: int,
        topk: int,
        max_local_tokens: int,
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        activation: str = "silu",
        situ_beta: float = 1.0,
        situ_linear_beta: float = 1.0,
        swiglu_limit: float | None = None,
        comm_mode: str = "ag_rs",
        comm_dtype: str = "fp8",
        ar_gather: str = "auto",
        schedule: str = "dynamic",
        act_dtype: str = "fp4",
        lb: dict | None = None,
        tail_eps: float = 1e-6,
        tail_gemma: bool = True,
        group=None,
        device: torch.device | None = None,
    ):
        if not mega_moe_tp_shape_supported(model_dim, inter_dim, world_size):
            raise ValueError(
                f"fused TP MegaMoE does not tile h{model_dim} i{inter_dim} tp{world_size}"
            )
        if comm_mode not in COMM_MODES:
            raise ValueError(f"unknown comm_mode {comm_mode!r}")
        if comm_dtype not in ("fp8", "bf16"):
            raise ValueError(f"unknown comm_dtype {comm_dtype!r}")
        self.comm_bf16 = comm_dtype == "bf16"
        if ar_gather not in ("bf16", "fp8", "auto"):
            raise ValueError(f"unknown ar_gather {ar_gather!r}")
        self.ag_fp8 = ar_gather == "fp8"
        self.ag_auto = AG8_MIN if ar_gather == "auto" else 0
        if not 1 <= topk <= experts or max_local_tokens < 1:
            raise ValueError(
                f"need 1 <= topk ({topk}) <= experts ({experts}), max_local_tokens >= 1"
            )
        self.mode = comm_mode
        self.check = os.environ.get("AITER_MEGAMOE_TP_CHECK", "0") == "1"
        if schedule != "dynamic":
            raise ValueError(f'unknown schedule {schedule!r} (only "dynamic")')
        if act_dtype not in ("fp4", "fp8"):
            raise ValueError(f"unknown act_dtype {act_dtype!r}")
        self.a8 = act_dtype == "fp8"
        lb = lb or {}
        # above DYN_MAX tokens always the LB schedule (fp8 activations: never)
        lb_min = int(_lb_param(lb.get("min"), "AITER_MEGAMOE_TP_LB_MIN", "256"))
        lb_min = lb_min if 0 < lb_min <= DYN_MAX else DYN_MAX + 1
        self.lb_min = 0 if self.a8 else lb_min
        self.lb_mt, self.lb_mt_table = _mt_table(
            _lb_param(lb.get("mt"), "AITER_MEGAMOE_TP_LB_MT", "3")
        )
        self.lb_mt_small = int(
            _lb_param(lb.get("mt_small"), "AITER_MEGAMOE_TP_LB_MT_SMALL", "3")
        )
        self.lb_small_max = int(
            _lb_param(lb.get("small_max"), "AITER_MEGAMOE_TP_LB_SMALL_MAX", "512")
        )
        self.lb_npp = int(_lb_param(lb.get("npp"), "AITER_MEGAMOE_TP_LB_NPP", "1"))
        lb_q = int(_lb_param(lb.get("q"), "AITER_MEGAMOE_TP_LB_Q", "1"))
        nck = mega_moe_tp_consts(model_dim, inter_dim, 1, 1)["NCK"]
        gpc = gemm2_chunk_groups(model_dim, inter_dim)
        step = gemm2_group_step(model_dim, inter_dim)
        valid = [d for d in range(1, nck + 1) if nck % d == 0 and d * gpc % step == 0]
        self.lb_q = max((d for d in valid if d <= lb_q), default=min(valid))
        self.ar = comm_mode == "ar"
        self.rank, self.tp = int(rank), int(world_size)
        self.H, self.I, self.E, self.K = model_dim, inter_dim, experts, topk
        self.mmax = int(max_local_tokens)
        device = torch.device(device if device is not None else "cuda")
        if device.index is None:
            device = torch.device(device.type, torch.cuda.current_device())
        self.device = device
        self.activation = activation
        self.situ = (float(situ_beta), float(situ_linear_beta))
        if swiglu_limit is None:
            swiglu_limit = 7.0 if activation == "swiglu" else float("inf")
        self.swiglu_limit = float(swiglu_limit)
        self._bind_weights(w1, w1_scale, w2, w2_scale)

        props = torch.cuda.get_device_properties(self.device)
        self.n_cta = int(props.multi_processor_count)
        if self.n_cta > NCTA_MAX:
            raise ValueError(f"at most {NCTA_MAX} CTAs (flag layout)")
        self.gfx = getattr(props, "gcnArchName", "").split(":")[0]
        if self.gfx != "gfx950":
            raise ValueError(f"fused TP MegaMoE needs gfx950, not {self.gfx}")
        self.agr = max(1, ceildiv(self.mmax, self.n_cta))
        self._check_limits()
        if not self._fits(self.mmax):
            ok = self.mmax
            while ok > 1 and not self._fits(ok):
                ok -= max(1, ok // 64)
            raise MegaMoeTPLDSError(
                f"max_local_tokens={self.mmax} does not fit in LDS for h{model_dim} "
                f"i{inter_dim} e{experts} k{topk} tp{world_size}: at most {ok}"
            )

        tot = self.mmax * self.tp
        H, K = model_dim, topk
        arena = SymmetricArena(group=group, device=self.device)
        self._x = arena.reserve("x", (tot, H if self.a8 else H // 2), torch.uint8)
        self._xs = arena.reserve("xs", (tot, H // 32), torch.uint8)
        self._ids = arena.reserve("ids", (2, tot, 4 * K), torch.int32)
        self._w = arena.reserve("w", (2, tot, K), torch.float32)
        self._recv = arena.reserve(
            "recv", (self.tp, tot if self.ar else self.mmax, H), torch.bfloat16
        )
        self._flag = arena.reserve("flag", (FLAG_INTS,), torch.int32)
        self._yall = arena.reserve(
            "yall", (tot, H) if self.ar else (1,), torch.bfloat16
        )
        tn_ok = comm_mode == "ag_rs"
        self._qall = arena.reserve(
            "qall", (tot * H * 3,) if tn_ok else (1,), torch.uint8
        )
        self._sall = arena.reserve("sall", (tot,) if tn_ok else (1,), torch.float32)
        # the fused tail's norm (compiled in; part of the launcher key)
        self.tn_eps = float(tail_eps)
        self.tn_gemma = bool(tail_gemma)
        arena.commit()
        self.arena = arena
        self.ctrl = torch.zeros(CTRL_INTS, dtype=torch.int32, device=self.device)
        self.routes = torch.zeros(
            (tot * topk + 1, H), dtype=torch.bfloat16, device=self.device
        )
        self.y = torch.empty(
            (tot if self.ar else self.mmax, H), dtype=torch.bfloat16, device=self.device
        )

        self.dyn_max = min(tot, DYN_MAX)
        # partial rows of the GEMM2 pieces past the first (dynamic schedule)
        extra = (inter_dim // 128 - 1) * self.dyn_max * topk
        if extra * H * 2 >= 1 << 31:
            raise ValueError("split-expert partial rows exceed 32-bit buffer offsets")
        self.proutes = torch.zeros(
            (extra + 1, H), dtype=torch.bfloat16, device=self.device
        )
        self.xg = torch.empty(
            tot * topk * (inter_dim // 2 + inter_dim // 32 + 8),
            dtype=torch.uint8,
            device=self.device,
        )
        self._cfgs: dict = {}
        self._launchers: dict = {}
        self._armed: set = set()
        self._args = (None, None)

    def _check_limits(self) -> None:
        """The kernel's fixed-size tables and 32-bit offsets, checked up front."""
        tot, H, K = self.mmax * self.tp, self.H, self.K
        bad = []
        if self.agr > (4 if self.a8 else 8):
            bad.append(f"max_local_tokens <= {(4 if self.a8 else 8) * self.n_cta}")
        if not self.ar and ceildiv(self.mmax * K, 256) > min(NMETA_CAP, self.n_cta):
            bad.append(f"max_local_tokens * topk <= {256 * min(NMETA_CAP, self.n_cta)}")
        if self.lb_min and self.lb_min <= tot and self.I // 128 >= 2:
            mts = [self.lb_mt, *(n for _, n in self.lb_mt_table)]
            if self.lb_small_max >= self.lb_min:
                mts.append(self.lb_mt_small)
            mt = min(self._fit_mt(t, lb=True) for t in mts)
            if self.E + ceildiv(tot * K, 16 * mt) + 1 > NCHLB_MAX:
                bad.append(f"LB row chunks <= {NCHLB_MAX} (larger LB row tiles)")
        for what, n in (
            ("route rows", tot * K * 2 * H),
            ("gathered rows", self.tp * tot * H * 2),
            ("tail rows", tot * H * 3),
            ("intermediate rows", tot * K * (self.I // 2 + self.I // 32 + 8)),
        ):
            if n >= 1 << 31:
                bad.append(f"{what} within 32-bit offsets")
        if bad:
            raise ValueError(
                f"MegaMoeTP h{H} i{self.I} e{self.E} k{K} tp{self.tp} "
                f"max_local_tokens={self.mmax}: need " + ", ".join(bad)
            )

    def _bind_weights(self, w1, w1_scale, w2, w2_scale) -> None:
        w = tuple(t.view(torch.uint8) for t in (w1, w1_scale, w2, w2_scale))
        H, I, E = self.H, self.I, self.E
        for name, t, n in (
            ("w1", w[0], E * 2 * I * H // 2),
            ("w1_scale", w[1], E * 2 * I * H // 32),
            ("w2", w[2], E * H * I // 2),
            ("w2_scale", w[3], E * H * I // 32),
        ):
            if t.device != self.device or not t.is_contiguous() or t.numel() < n:
                raise ValueError(
                    f"{name}: need a contiguous tensor of >= {n} bytes on {self.device}"
                )
        self.w1, self.w1s, self.w2, self.w2s = w

    def set_weights(self, w1, w1_scale, w2, w2_scale) -> None:
        self._bind_weights(w1, w1_scale, w2, w2_scale)
        self._args = (None, None)

    def _chunks(self, mt: int, mmax: int = 0, lb: bool = False) -> tuple[int, int]:
        rch = 16 * mt
        tot = (mmax or self.mmax) * self.tp
        if not lb:
            tot = min(tot, DYN_MAX)
        return rch, self.E + ceildiv(tot * self.K, rch) + 1

    def _consts(self, mt: int, nab: int = 4, mmax: int = 0, lb: bool = False) -> dict:
        mmax = mmax or self.mmax
        return mega_moe_tp_consts(
            self.H,
            self.I,
            mt,
            mmax * self.tp,
            max(1, ceildiv(mmax, self.n_cta)),
            self.E,
            nab,
            self._chunks(mt, mmax, lb)[1],
            self.a8,
            lb,
        )

    def _nab(self, mt: int, mmax: int = 0, lb: bool = False) -> int:
        if lb:
            return 3
        return 4 if self._consts(mt, 4, mmax)["LDS_BYTES"] <= LDS_LIMIT else 3

    def _lds(self, mt: int, mmax: int = 0, lb: bool = False) -> int:
        nab = self._nab(mt, mmax, lb)
        return self._consts(mt, nab, mmax, lb)["LDS_BYTES"]

    def _fits(self, mmax: int) -> bool:
        # the smallest row tile of each schedule these batches can take fits in LDS
        lb = bool(self.lb_min) and mmax * self.tp >= self.lb_min
        return self._lds(1, mmax) <= LDS_LIMIT and (
            not lb or self._lds(1, mmax, lb=True) <= LDS_LIMIT
        )

    def _fit_mt(self, mt: int, lb: bool = False) -> int:
        mt = max(1, min(6, int(mt)))
        while mt > 1 and self._lds(mt, lb=lb) > LDS_LIMIT:
            mt -= 1
        return mt

    def _fit_npp(self, npp: int) -> int:
        ok = npp >= 1 and 4 % (2 * npp) == 0 and (self.H // 128 * npp) % 4 == 0
        return npp if ok else 1

    def config(self, m: int) -> LaunchCfg:
        cfg = self._cfgs.get(m)
        if cfg is None:
            tot = m * self.tp
            lb = bool(self.lb_min) and tot >= self.lb_min
            fp8 = not self.comm_bf16
            if lb:
                mt = next((n for t, n in self.lb_mt_table if tot <= t), self.lb_mt)
                small = tot <= self.lb_small_max
                lp = 0 if tot <= LB_PF_MAX else 3
                cfg = LaunchCfg(
                    self._fit_mt(self.lb_mt_small if small else mt, lb=True),
                    npp=self._fit_npp(2 if small else self.lb_npp),
                    ag8=self.ar and bool(self.ag_auto) and tot >= self.ag_auto,
                    lb=True,
                    pf=tot <= LB_PF_MAX,
                    lp=lp if self.I // 128 % max(lp, 1) == 0 else 0,
                )
            elif tot <= self.dyn_max:
                cfg = LaunchCfg(
                    self._fit_mt(1 if tot <= 32 else 2 if tot <= 64 else 3),
                    ll=tot <= 256 and fp8,
                    llr=tot <= 128 and fp8,
                    ag8=self.ar and bool(self.ag_auto) and tot >= self.ag_auto,
                )
            else:
                raise ValueError(
                    f'act_dtype="fp8": {tot} tokens are beyond the dynamic schedule'
                )
            self._cfgs[m] = cfg
        return cfg

    def _arll(self, cfg: LaunchCfg) -> bool:
        return self.ar and cfg.ll and cfg.llr

    def _ag8(self, cfg: LaunchCfg) -> bool:
        return self.ar and (self.ag_fp8 or cfg.ag8) and not self._arll(cfg)

    def _key(self, cfg: LaunchCfg):
        return (cfg, self.tn_eps, self.tn_gemma) if cfg.tn else cfg

    def _launcher(self, cfg: LaunchCfg):
        fn = self._launchers.get(self._key(cfg))
        if fn is None:
            rch, nch = self._chunks(cfg.mt, lb=cfg.lb)
            fn = compile_mega_moe_tp(
                H=self.H,
                I=self.I,
                TOPK=self.K,
                MT=cfg.mt,
                TMAX=self.mmax * self.tp,
                E=self.E,
                act=self.activation,
                situ_beta=self.situ[0],
                situ_linear_beta=self.situ[1],
                swiglu_limit=self.swiglu_limit,
                route_fp8=cfg.ll and cfg.llr,
                agr=self.agr,
                tp=self.tp,
                ar=self.ar,
                nab=self._nab(cfg.mt, lb=cfg.lb),
                npp=cfg.npp,
                ll_rs=cfg.ll,
                ll_route=cfg.ll and cfg.llr,
                comm_bf16=self.comm_bf16,
                ag8=self._ag8(cfg),
                rch=rch,
                nch=nch,
                a8=self.a8,
                lb=cfg.lb,
                lbq=self.lb_q if cfg.lb else 1,
                tn=cfg.tn,
                tn_eps=self.tn_eps,
                tn_gemma=self.tn_gemma,
                lbpf=cfg.pf,
                lbp=cfg.lp,
            )
            self._launchers[self._key(cfg)] = fn
        return fn

    def _local_tokens(self, x, topk_weights, topk_ids) -> int:
        rows = int(x.shape[0]) if x.dim() == 2 else -1
        for name, t in (
            ("x", x),
            ("topk_weights", topk_weights),
            ("topk_ids", topk_ids),
        ):
            if t.device != self.device:
                raise ValueError(f"{name} is on {t.device}, the layer on {self.device}")
        if x.dim() != 2 or x.shape[1] != self.H or x.dtype != torch.bfloat16:
            raise ValueError(
                f"x: need bf16 [tokens, {self.H}], got {x.dtype} {tuple(x.shape)}"
            )
        want = (rows, self.K)
        if tuple(topk_ids.shape) != want or tuple(topk_weights.shape) != want:
            raise ValueError(
                f"topk_ids / topk_weights: need {list(want)}, got "
                f"{list(topk_ids.shape)} / {list(topk_weights.shape)}"
            )
        if topk_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"topk_ids: need int32 (or int64), got {topk_ids.dtype}")
        if not topk_weights.is_floating_point():
            raise ValueError(f"topk_weights: need float32, got {topk_weights.dtype}")
        m = rows
        if self.ar:
            if m % self.tp:
                raise ValueError(
                    f"{self.mode}: {m} tokens are not a multiple of tp={self.tp}"
                )
            m //= self.tp
        if m > self.mmax:
            raise ValueError(f"{m} local tokens exceed max_local_tokens {self.mmax}")
        if (
            self.check
            and dist.is_initialized()
            and not torch.cuda.is_current_stream_capturing()
        ):
            self._check_replicas(m, x, topk_weights, topk_ids)
        return m

    def _check_replicas(self, m, x, topk_weights, topk_ids) -> None:
        if self.ar:
            fp = [
                float(t.double().sum())
                for t in (x, topk_weights, topk_ids, (x.float() * x.float()).sum(1))
            ]
        else:
            fp = []
        got = [None] * self.tp
        dist.all_gather_object(got, (m, fp), group=self.arena.group)
        if len({g[0] for g in got}) > 1:
            raise ValueError(
                f"{self.mode}: local token counts differ across ranks: "
                f"{[g[0] for g in got]}"
            )
        if self.ar and any(g[1] != fp for g in got):
            raise ValueError(
                f"{self.mode}: x / topk_ids / topk_weights must be the same on every "
                "rank (replicated input)"
            )

    def prepare(
        self, local_tokens, tail: bool = False, tail_bf16: bool = False, warmup=True
    ) -> None:
        """Collective: compile and arm the launch configs of these local token counts
        (and their fused-tail variants), then (warmup) run each once eagerly and check
        every rank's watchdog: on a timeout reset() and retry once, else raise."""
        runs = []
        for m in sorted({int(m) for m in local_tokens if 0 < m <= self.mmax}):
            cfg = self.config(m)
            self._arm(cfg, m)
            runs.append((m, 0))
            if self.tail_ok(m):
                for on, tn in ((tail, 1), (tail_bf16, 2)):
                    if on:
                        self._arm(dataclasses.replace(cfg, tn=tn), m)
                        runs.append((m, tn))
        if not warmup:
            return
        for attempt in range(2):
            for m, tn in runs:
                self._warmup(m, tn)
            torch.cuda.synchronize(self.device)
            errs = [None] * self.tp
            dist.all_gather_object(errs, self.poll_errors(), group=self.arena.group)
            if not any(errs):
                return
            self.reset()
        raise RuntimeError(f"MegaMoeTP.prepare: warmup launches timed out ({errs})")

    def _warmup(self, m: int, tn: int = 0) -> None:
        rows = m * self.tp if self.ar else m
        dev = self.device
        x = torch.zeros((rows, self.H), dtype=torch.bfloat16, device=dev)
        ids = torch.arange(rows * self.K, dtype=torch.int32, device=dev) % self.E
        w = torch.full((rows, self.K), 1.0 / self.K, device=dev)
        tail = None
        if tn:
            res = torch.zeros((m, self.H), dtype=torch.bfloat16, device=dev)
            nw = torch.zeros(self.H, dtype=torch.bfloat16, device=dev)
            tail = (res, res, nw)
        self.forward(x, w, ids.view(rows, self.K), tail=tail, bf16=tn == 2)

    def tail_ok(self, m: int) -> bool:
        return (
            self.mode == "ag_rs"
            and 0 < m <= min(TN_MAX, self.mmax)
            and self.config(m).lb
        )

    def _arm(self, cfg: LaunchCfg, m: int = 0) -> None:
        if self._key(cfg) in self._armed:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"MegaMoeTP: launch config {cfg} first used inside CUDA graph "
                "capture; call prepare(local_tokens) (or run it eagerly) first"
            )
        fn = self._launcher(cfg)
        m = m or next(m for m, c in self._cfgs.items() if c == cfg)
        args = self._launch_args(m, cfg, self.y, None, None, None)
        _preload_compiled(fn, *args, torch.cuda.current_stream())
        torch.cuda.synchronize(self.device)
        self.arena.barrier()
        self._armed.add(self._key(cfg))

    def _launch_args(self, m, cfg, y, x, ids, tw, tail=None):
        peers = [int(b) for b in self.arena.base_ptrs] + [0] * (MAX_TP - self.tp)

        def ptr(t):
            return 0 if t is None else t.data_ptr()

        return (
            self.w1.data_ptr(),
            self.w1s.data_ptr(),
            self.w2.data_ptr(),
            self.w2s.data_ptr(),
            ptr(x),
            ptr(ids),
            ptr(tw),
            y.data_ptr(),
            self.routes.data_ptr(),
            self.proutes.data_ptr(),
            self.ctrl.data_ptr(),
            *peers,
            self._x.offset,
            self._xs.offset,
            self._ids.offset,
            self._w.offset,
            self._recv.offset,
            self._flag.offset,
            self._yall.offset,
            self.rank,
            self.tp,
            m,
            self.mmax,
            self.xg.data_ptr(),
            *(ptr(t) for t in (tail or (None, None, None))),
            self._qall.offset,
            self._sall.offset,
            self.n_cta,
        )

    def forward(
        self,
        x_local: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        out: torch.Tensor | None = None,
        tail=None,
        bf16: bool = False,
    ):
        m = self._local_tokens(x_local, topk_weights, topk_ids)
        rows = m * self.tp if self.ar else m
        if out is not None and (
            tuple(out.shape) != (rows, self.H)
            or out.dtype != torch.bfloat16
            or out.device != self.device
            or not out.is_contiguous()
        ):
            raise ValueError(f"out: need a contiguous bf16 [{rows}, {self.H}]")
        if m == 0:
            return x_local.new_empty((0, self.H)) if out is None else out
        x = x_local.contiguous()
        ids = topk_ids.to(torch.int32).contiguous()
        tw = topk_weights.to(torch.float32).contiguous()
        cfg = self.config(m)
        if tail is not None:
            if not self.tail_ok(m):
                raise ValueError(
                    "tail: ag_rs, the LB schedule, at most TN_MAX local tokens"
                )
            res_in, res_out, nw = tail
            for t, shp in (
                (res_in, (m, self.H)),
                (res_out, (m, self.H)),
                (nw, (self.H,)),
            ):
                if (
                    tuple(t.shape) != shp
                    or t.dtype != torch.bfloat16
                    or not t.is_contiguous()
                ):
                    raise ValueError(f"tail: need contiguous bf16 {shp}")
            cfg = dataclasses.replace(cfg, tn=2 if bf16 else 1)
        local_y = not self.ar or self._arll(cfg) or self._ag8(cfg)
        y = out if out is not None and local_y else self.y
        tptr = tuple(t.data_ptr() for t in tail) if tail is not None else ()
        key = (m, x.data_ptr(), ids.data_ptr(), tw.data_ptr(), y.data_ptr(), cfg, tptr)
        if self._args[0] != key:
            args = self._launch_args(m, cfg, y, x, ids, tw, tail)
            self._args = (key, (args, x, ids, tw))
        self._arm(cfg, m)
        _run_compiled(
            self._launcher(cfg), *self._args[1][0], torch.cuda.current_stream()
        )
        if self.check and not torch.cuda.is_current_stream_capturing():
            self.check_errors()
        if not local_y:
            y = self._yall.local[:rows]
            return y if out is None else out.copy_(y)
        y = y[:rows] if out is None else out
        if tail is not None:
            T_ = m * self.tp
            qall = self._qall.local
            q = qall[: T_ * self.H].view(T_, self.H)
            if bf16:
                b0 = self.mmax * self.tp * self.H
                b = (
                    qall[b0 : b0 + 2 * T_ * self.H]
                    .view(torch.bfloat16)
                    .view(T_, self.H)
                )
                return y, q, self._sall.local[:T_], b
            return y, q, self._sall.local[:T_]
        return y

    __call__ = forward

    def clear_errors(self) -> None:
        self.ctrl[CTRL_ERR] = 0

    def poll_errors(self) -> int:
        return int(self.ctrl[CTRL_ERR].item())

    def error_flag(self) -> torch.Tensor:
        """The sticky watchdog bits as a 1-element device tensor (no sync)."""
        return self.ctrl[CTRL_ERR : CTRL_ERR + 1]

    def check_errors(self) -> None:
        err = self.poll_errors()
        if err:
            self.clear_errors()
            what = [n for b, n in _ERR_NAMES if err & b]
            raise RuntimeError(
                f"MegaMoeTP rank {self.rank}: a wait inside the kernel timed out "
                f"({'|'.join(what)}); the output is invalid. Ranks out of step "
                "(different call counts or local token counts) need reset()."
            )

    def reset(self) -> None:
        torch.cuda.synchronize(self.device)
        self.arena.barrier()
        self.arena.storage.zero_()
        self.ctrl.zero_()
        self.routes.zero_()
        self.proutes.zero_()
        torch.cuda.synchronize(self.device)
        self.arena.barrier()
