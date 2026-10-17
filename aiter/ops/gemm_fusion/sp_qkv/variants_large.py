# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""sp_qkv.large.v1 = Phase-1 LG-QSDMA-r11 (GEMM + SDMA all-to-all; covers the sp_exchange prefix only).

plan.run(x, x_scale, w, w_scale, outs, meta, res, stream) contract (large):
  x fp8 [Ml,K] (Ml = local tokens, 256-multiple <= 8192), w fp8 preshuffled (16,16) [N,K], x_scale fp32 [Ml], w_scale fp32 [N] (raw order).
  Dense/reuse builds read only the first 9216 rows of w / w_scale (a prefix of the 9856-row indexer weight is contiguous: shuffle blocks are 16 rows).
  Result lands in the plan's receive window R (token-major, row = source*Ml + m): dense F=2304, indexer F=2560, reuse = dense build compact F=2304.
  plan.recv_view(res) -> torch bf16 view [T, F] (zero copy; stable pointer; valid until this rank launches the next epoch of the SAME plan).
  plan.reuse_view(res) -> [T, 2560] view with row stride 2304 (reuse only; the IQ|IK columns alias the next row and are never read).
  outs may contain "packed": if given, the view is copied into it (extra copy; prefer recv_view).
  Epoch state (credit protocol) is device-owned; all ranks must launch the same number of epochs per plan. After ANY poison on ANY rank:
  collective res.quiesce_and_reset() (re-inits the control image epoch=1/magic) before the next launch.
Requires res.finalize() to have run (the kernels take the process devcomm by value).
"""

from __future__ import annotations

import hashlib
from typing import Dict, List

from aiter.gemm_fusion.catalog import Contract, register_variant
from aiter.gemm_fusion.match import ContractError
from aiter.gemm_fusion.plan import Plan, WindowReq
from aiter.gemm_fusion.resources import round_up

from . import large_rt as lrt
from . import specs
from .jit import build_large, large_defs
from .small_rt import stream_handle

_F = {"dense": 2304, "indexer": 2560, "reuse": 2304}
LG_R_PAD = 4096
LGC_STAMP_OFF = 65536


def large_window_bytes(mode: str, M: int) -> int:
    """Mirror of lg_window_bytes_v1: roundup(max(P + pad, stamp area), 4 MiB), P = 8*M*F."""
    P = 8 * M * _F[mode]
    need = max(P + LG_R_PAD, LGC_STAMP_OFF + 64 * 4096, P)
    return round_up(need)


def preflight_attach(res, owner: str, M: int, wins, built):
    """Bounded checks of everything lg_attach_v1 needs, each raising a message that names the precondition (so a failure never
    reduces to 'attach failed'): finalized, queue count, devcomm facts, hsaco file + sha, window handles/pointers, rank/M ranges."""
    import hashlib
    import os

    if not 0 <= res.rank <= 3:
        raise RuntimeError(f"preflight {owner}: SP rank {res.rank} outside 0..3 (r11 kernels: rank == lsaRank; the HIP device index is separate)")
    if M % 256 or not 256 <= M <= 8192:
        raise RuntimeError(f"preflight {owner}: M={M} must be a multiple of 256 in [256, 8192]")
    info = getattr(res, "devcomm_info", None)
    if info:
        if info.get("lsaSize") != 4 or info.get("lsaRank") != res.rank or info.get("worldSize") != 4 or info.get("myNodeStart") != 0:
            raise RuntimeError(f"preflight {owner}: topology {info} (need worldSize=lsaSize=4, lsaRank=rank={res.rank}, myNodeStart=0)")
        if info.get("sdmaNumQueue") != 1:
            raise RuntimeError(f"preflight {owner}: devcomm sdmaNumQueue={info.get('sdmaNumQueue')} but the r11 kernels need 1 "
                               f"(requested {getattr(res, 'sdma_queues', '?')})")
    for k in ("/S", "/R", "/C"):
        w = wins.get(owner + k)
        if w is None or not w.handle or not w.local:
            raise RuntimeError(f"preflight {owner}: window {owner + k} missing/NULL (handle={getattr(w, 'handle', None)}, local={getattr(w, 'local', None)})")
        if w.bytes != wins[owner + "/S"].bytes:
            raise RuntimeError(f"preflight {owner}: windows must be equal size (S={wins[owner + '/S'].bytes}, {k[1:]}={w.bytes})")
    need = large_window_bytes(_mode_from_owner(owner), M)
    if wins[owner + "/S"].bytes < need:
        raise RuntimeError(f"preflight {owner}: window {wins[owner + '/S'].bytes} B < required {need} B")
    if not os.path.isfile(built.hsaco):
        raise RuntimeError(f"preflight {owner}: hsaco missing: {built.hsaco}")
    got = hashlib.sha256(open(built.hsaco, "rb").read()).hexdigest()
    if got != built.hsaco_sha256:
        raise RuntimeError(f"preflight {owner}: hsaco sha256 {got} != expected {built.hsaco_sha256}")


def _mode_from_owner(owner: str) -> str:  # owner = "sp_qkv.large/<mode>/M<M>"
    return owner.split("/")[1]


@register_variant(
    vid="sp_qkv.large.v1", family="gemm_ep.sp_qkv", stage_kinds=("sp_exchange",),
    contract=Contract(arch=("gfx950",), world=4, topology="single_node_xgmi_p2p", graph="capture_safe_epoch_state",
                      requires=("mori_cco", "sdma"), m_multiple=256, mutates=()),
    source_sha="r11=9874fe1ecb3b6a997523c96210a960434179c8eab84fffc00537170d49836643",
)
class SpQkvLarge(Plan):
    n_stages = 1

    def __init__(self, spec, params, ctx):
        self.spec, self.params = spec, dict(params)
        self.mode = specs.mode_of(spec)
        self.M = ctx.m
        large_defs(self.params, self.mode)  # validates params
        if self.M % 256 or not 256 <= self.M <= 8192:
            raise ContractError(f"sp_qkv.large.v1 needs local M multiple of 256 in [256, 8192], got {self.M}")
        self.n = int(self.params.get("n", 1))
        self.half_first = int(self.params.get("half_first", 0))
        chunks = self.M // 256
        if self.half_first == 2:
            if self.n != 3 or chunks != 8 or self.mode == "indexer":
                raise ContractError("quarter-first (n3q) needs n=3, M=2048 and a dense-build mode")
        else:
            if self.n not in (1, 2, 4) or chunks % self.n:
                raise ContractError(f"puts per peer n={self.n} invalid for {chunks} chunks")
            if self.half_first == 1 and self.n == 1:
                raise ContractError("half-first needs n >= 2")
        self.F = _F[self.mode]
        self.owner = f"sp_qkv.large/{self.mode}/M{self.M}"
        self.build_key = self.owner  # the key ATOM's plan_owner() uses for FusionResources.ensure / windows
        self._ctx = None
        self._epochs = 0

    def resources(self) -> List[WindowReq]:
        b = large_window_bytes(self.mode, self.M)
        return [WindowReq(self.owner + "/S", b, "p2p_window"), WindowReq(self.owner + "/R", b, "p2p_window"),
                WindowReq(self.owner + "/C", b, "control"), WindowReq(self.owner + "/q", 0, "sdma_queue")]

    def _context(self, res):
        sc = res.__dict__.setdefault("_sp_qkv_large", {})
        got = sc.get(self.owner)
        if got is None:
            if not res.finalized:
                raise RuntimeError("sp_qkv.large: res.finalize() must run before the first launch (devcomm)")
            if getattr(res, "sdma_queues", 1) != 1:
                raise RuntimeError(f"sp_qkv.large: the shared devcomm has sdma_queues={res.sdma_queues}; the r11 kernels require exactly 1 "
                                   f"(Phase-1 sdmaQueueCount=1)")
            wins = res.windows.get(self.owner) or res.ensure(self.owner, self.resources())
            built = build_large(self.params, self.mode)
            sha = hashlib.sha256(open(built.hsaco, "rb").read()).hexdigest()
            if sha != built.hsaco_sha256:
                raise RuntimeError(f"HSACO_HASH_MISMATCH {sha} != {built.hsaco_sha256}")
            lib = lrt.load_driver(built.driver_so)
            geo = lrt.info(lib)
            if geo["f"] != self.F or geo["indexer"] != (1 if self.mode == "indexer" else 0) or geo["fold"] != 0:
                raise RuntimeError(f"build/mode mismatch {geo} vs mode {self.mode}")
            if lib.lg_window_bytes_v1(self.M) != large_window_bytes(self.mode, self.M):
                raise RuntimeError("compiled window size != analytic window size")
            import ctypes as C

            S, R, Cw = wins[self.owner + "/S"], wins[self.owner + "/R"], wins[self.owner + "/C"]
            preflight_attach(res, self.owner, self.M, wins, built)  # bounded, pure-Python, names the failing precondition before the driver call
            diag = (C.c_uint64 * 8)()
            h = lib.lg_attach_v1(res.rank, self.M, C.c_void_p(res.comm), C.c_void_p(res.devcomm), C.c_void_p(S.handle), C.c_void_p(S.local),
                                 C.c_void_p(R.handle), C.c_void_p(R.local), C.c_void_p(Cw.handle), C.c_void_p(Cw.local), built.hsaco.encode(), built.hsaco_sha256.encode(), 1,
                                 diag)  # H-3: C++ exact-image gate; diag[] carries the failing stage + values (no stderr scraping)
            if not h:
                raise RuntimeError(f"lg_attach_v1 failed for {self.owner} rank={res.rank} M={self.M}: {lrt.decode_attach_diag(list(diag))}; "
                                   f"hsaco={built.hsaco} sha256={built.hsaco_sha256} devcomm_queues={getattr(res, 'sdma_queues', '?')} "
                                   f"devcomm_info={getattr(res, 'devcomm_info', None)}")
            got = sc[self.owner] = lrt.LargeContext(lib, C.c_void_p(h), self.M, self.F, res.rank)
            res.set_close(self.owner, got.close)  # L-5
            res.set_poison(self.owner, lambda c=got: c.poison(stream_handle(None)))  # cheap per-step check on the CURRENT torch stream (4-byte D2H + that stream sync)
            res.set_drain(self.owner, lambda c=got: c.drain_all(C.c_void_p(None)))  # A2-H: quietQueue on all peer queues before any reset
            res.set_init(self.owner, got.init_control)  # epoch=1 + magic; re-run after every reset
        self._ctx = got
        return got

    def prepare(self, *, res, **_):
        """Attach + init the control block. Call after res.finalize() and before graph capture; then barrier on all ranks."""
        return self._context(res)

    def run(self, *, x, x_scale, w, w_scale, outs=None, meta=None, res, stream=None):
        from .small_rt import stream_handle

        c = self._context(res)
        if x.shape[0] != self.M:
            raise ValueError(f"x rows {x.shape[0]} != plan local M {self.M}")
        c.launch(x.data_ptr(), w.data_ptr(), x_scale.data_ptr(), w_scale.data_ptr(), x.stride(0) * x.element_size(),
                 self.n, self.half_first, stream_handle(stream))
        self._epochs += 1
        packed = (outs or {}).get("packed")
        if packed is not None:
            packed.copy_(self.recv_view(res))

    def recv_view(self, res):
        import torch

        c = self._context(res)
        t = torch.as_tensor(lrt._Va(c.recv_ptr(), (4 * self.M, self.F)), device=f"cuda:{res.device}")
        return t.view(torch.bfloat16)

    def reuse_view(self, res):
        import torch

        if self.mode != "reuse":
            raise RuntimeError("reuse_view is only valid for the reuse mode")
        c = self._context(res)
        base = torch.as_tensor(lrt._Va(c.recv_ptr(), (4 * self.M * 2304 + 2048,)), device=f"cuda:{res.device}").view(torch.bfloat16)
        return torch.as_strided(base, (4 * self.M, 2560), (2304, 1))

    def poison(self, res) -> int:
        """Cheap host read of the device poison word (4-byte D2H on the current torch stream + that stream sync). Outside capture only; all-reduce MAX across ranks, then res.quiesce_and_reset()."""
        return self._context(res).poison(stream_handle(None))
