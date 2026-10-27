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
Under the (default) arena window policy every large plan uses ONE control block and ONE S/R pair: every rank must launch the same SEQUENCE of large plans (see the
resources module docstring: divergence is detected by the host-side digest, not by the device) and a recv_view is valid only until the next large launch.
"""

from __future__ import annotations

import logging

from aiter.gemm_fusion.catalog import Contract, register_variant
from aiter.gemm_fusion.errpage import GLOBAL_PAGE_BYTES, SHARE_GLOBAL
from aiter.gemm_fusion.match import ContractError
from aiter.gemm_fusion.plan import Plan, WindowReq
from aiter.gemm_fusion.resources import SHARED_KEY, round_up

from . import large_rt as lrt
from . import specs
from .jit import LARGE_SRC, _sha, build_large, large_defs, src_abi, src_pad_m
from .small_rt import stream_handle

log = logging.getLogger("aiter.gemm_fusion")

_F = {"dense": 2304, "indexer": 2560, "reuse": 2304}
LG_CONTROL_MIN_BYTES = 8192  # control window image (LGC_STATE_OFF + state words); the device code never addresses beyond it with LG_STAMPS=0
SHARE_S, SHARE_R, SHARE_C = (
    "sp_qkv.large/S",
    "sp_qkv.large/R",
    "sp_qkv.large/C",
)  # share tags (policy arena maps all large plans onto these)
LG_R_PAD = 4096
LGC_STAMP_OFF = 65536


def large_window_bytes(mode: str, M: int) -> int:
    """Mirror of lg_window_bytes_v1: roundup(max(P + pad, stamp area), 4 MiB), P = 8*M*F."""
    P = 8 * M * _F[mode]
    need = max(P + LG_R_PAD, LGC_STAMP_OFF + 64 * 4096, P)
    return round_up(need)


def preflight_attach(res, owner: str, M: int, wins, built):
    """Bounded checks of everything lg_attach_v1 needs, each raising a message that names the precondition (so a failure never
    reduces to 'attach failed'): finalized, queue count, devcomm facts, hsaco file + sha, window handles/pointers, rank/M ranges.
    """
    import os

    if not 0 <= res.rank <= 3:
        raise RuntimeError(
            f"preflight {owner}: SP rank {res.rank} outside 0..3 (r11 kernels: rank == lsaRank; the HIP device index is separate)"
        )
    if M % 256 or not 256 <= M <= 8192:
        raise RuntimeError(
            f"preflight {owner}: M={M} must be a multiple of 256 in [256, 8192]"
        )
    info = getattr(res, "devcomm_info", None)
    if info:
        if (
            info.get("lsaSize") != 4
            or info.get("lsaRank") != res.rank
            or info.get("worldSize") != 4
            or info.get("myNodeStart") != 0
        ):
            raise RuntimeError(
                f"preflight {owner}: topology {info} (need worldSize=lsaSize=4, lsaRank=rank={res.rank}, myNodeStart=0)"
            )
        if info.get("sdmaNumQueue") != 1:
            raise RuntimeError(
                f"preflight {owner}: devcomm sdmaNumQueue={info.get('sdmaNumQueue')} but the r11 kernels need 1 "
                f"(requested {getattr(res, 'sdma_queues', '?')})"
            )
    shared = all(
        getattr(wins.get(owner + k), "shared", False) for k in ("/S", "/R", "/C")
    )
    for k in ("/S", "/R", "/C"):
        w = wins.get(owner + k)
        if w is None or not w.handle or not w.local:
            raise RuntimeError(
                f"preflight {owner}: window {owner + k} missing/NULL (handle={getattr(w, 'handle', None)}, local={getattr(w, 'local', None)})"
            )
        if (
            not shared and w.bytes != wins[owner + "/S"].bytes
        ):  # per-owner policies keep the equal-size convention; a SHARED set is S=R=max plan, C=control
            raise RuntimeError(
                f"preflight {owner}: windows must be equal size (S={wins[owner + '/S'].bytes}, {k[1:]}={w.bytes})"
            )
    if shared:
        if wins[owner + "/S"].bytes != wins[owner + "/R"].bytes:
            raise RuntimeError(
                f"preflight {owner}: shared S and R must be equal (S={wins[owner + '/S'].bytes}, R={wins[owner + '/R'].bytes})"
            )
        if wins[owner + "/C"].bytes < LG_CONTROL_MIN_BYTES:
            raise RuntimeError(
                f"preflight {owner}: shared control window {wins[owner + '/C'].bytes} B < {LG_CONTROL_MIN_BYTES} B"
            )
    need = large_window_bytes(_mode_from_owner(owner), M)
    if wins[owner + "/S"].bytes < need:
        raise RuntimeError(
            f"preflight {owner}: window {wins[owner + '/S'].bytes} B < required {need} B"
        )
    if not os.path.isfile(built.hsaco):
        raise RuntimeError(f"preflight {owner}: hsaco missing: {built.hsaco}")
    got = _sha(built.hsaco)
    if got != built.hsaco_sha256:
        raise RuntimeError(
            f"preflight {owner}: hsaco sha256 {got} != expected {built.hsaco_sha256}"
        )


def decode_large_poison(word: int) -> str:
    """Which wait failed: the r11 sticky poison word is the reason code (lg_comm_abi_v2.h LGC_POISON_*)."""
    from aiter.gemm_fusion.resources import LG_POISON_NAMES

    return f"{LG_POISON_NAMES.get(int(word), 'UNKNOWN')} word={int(word)}"


def _mode_from_owner(owner: str) -> str:  # owner = "sp_qkv.large/<mode>/M<M>"
    return owner.split("/")[1]


# sha256 of large/<src>/SOURCE_SHA256.txt (pinned to the manifest files by a test)
LARGE_SRC_SHA = {
    "r11": "9874fe1ecb3b6a997523c96210a960434179c8eab84fffc00537170d49836643",
    "r12": "6b47263546d9189457b3d584fe8219e542a0cd32b19d1a32306c2fda015a7f38",  # R12 (abi 3) + LG_TAG_EARLY / LG_POLL_BATCH (r12b) + the NaN fill folded into quiet_ready, LG_QR_WGS (r12c) + qr_wgs in the launch args / lg_abi_info_v2 (r12d), derived from r11
    "r13": "03d6c758c75665d26f8f3928e6e31520c816ed701efb8b2e09cdc2ea3dfb2b3a",  # R16 (abi 4 = LG_COMM_ABI_VERSION 5): per-launch m_real (lg_launch_v3, tag with m_real), derived from r12; r13g = r13 with the item-epilogue data flow of r12 untouched (ISA clean, byte-identical to r12 at m == M)
}


@register_variant(
    vid="sp_qkv.large.v1",
    family="gemm_ep.sp_qkv",
    stage_kinds=("sp_exchange",),
    contract=Contract(
        arch=("gfx950",),
        world=4,
        topology="single_node_xgmi_p2p",
        graph="capture_safe_epoch_state",
        requires=("mori_cco", "sdma"),
        m_multiple=256,
        mutates=(),
    ),
    source_sha=";".join(f"{k}={v}" for k, v in LARGE_SRC_SHA.items()),
)
class SpQkvLarge(Plan):
    n_stages = 1
    # R16 bucket selection (match.bucket_for_m): a real local m is padded into a larger bucket only from this m on (profile field m_pad_min_large; default 896: native wins below m_real ~ 896 at the 1024 bucket (dense; ~768 for indexer / reuse), the large path's
    # first multiple: m 256..1023 stays native until measurements say otherwise)
    pad_m_min_field = "m_pad_min_large"
    pad_m_min_default = 896

    def __init__(self, spec, params, ctx):
        self.spec, self.params = spec, dict(params)
        self.mode = specs.mode_of(spec)
        self.M = ctx.m
        large_defs(self.params, self.mode)  # validates params
        if self.M % 256 or not 256 <= self.M <= 8192:
            raise ContractError(
                f"sp_qkv.large.v1 needs local M multiple of 256 in [256, 8192], got {self.M}"
            )
        self.n = int(self.params.get("n", 1))
        self.half_first = int(self.params.get("half_first", 0))
        chunks = self.M // 256
        if self.half_first == 2:
            if self.n != 3 or chunks != 8 or self.mode == "indexer":
                raise ContractError(
                    "quarter-first (n3q) needs n=3, M=2048 and a dense-build mode"
                )
        else:
            if self.n not in (1, 2, 4) or chunks % self.n:
                raise ContractError(
                    f"puts per peer n={self.n} invalid for {chunks} chunks"
                )
            if self.half_first == 1 and self.n == 1:
                raise ContractError("half-first needs n >= 2")
        self.src = self.params.get("src", LARGE_SRC)
        self.abi = src_abi(
            "large", self.src
        )  # 2 = r11 (per-plan poison word); 3 = r12 (per-comm ERR/STEP page, call_id tags; the best-effort NaN fill is folded into quiet_ready, no separate node)
        # the quiet_ready workgroup count the DEVICE image is built with (r12c / r12d: LG_QR_WGS, 64 by default); the host library reports its own through lg_abi_info_v2 and the
        # attach refuses a mismatch. r11 has no such knob.
        self.qr_wgs = (
            int(self.params.get("qr_wgs", 64)) if self.src in ("r12", "r13") else None
        )
        self.pad_m = src_pad_m(
            "large", self.src
        )  # R16: launches the real m_real <= the bucket M (self.M) of this plan
        self.last_m_real = (
            self.M
        )  # the rows of the latest launch: the default row count of recv_view / reuse_view
        self.F = _F[self.mode]
        self.owner = f"sp_qkv.large/{self.mode}/M{self.M}"
        self.build_key = (
            self.owner
        )  # the key ATOM's plan_owner() uses for FusionResources.ensure / windows
        self._ctx = None
        self._epochs = 0

    def resources(self) -> list[WindowReq]:
        b = large_window_bytes(self.mode, self.M)
        # bytes = the per-plan equal-size convention (the uniform policy); share + min_bytes serve the arena policy: ONE S and ONE R sized to the largest
        # large plan and ONE control window used by every large plan.
        reqs = [
            WindowReq(self.owner + "/S", b, "p2p_window", share=SHARE_S, min_bytes=b),
            WindowReq(self.owner + "/R", b, "p2p_window", share=SHARE_R, min_bytes=b),
            WindowReq(
                self.owner + "/C",
                b,
                "control",
                share=SHARE_C,
                min_bytes=LG_CONTROL_MIN_BYTES,
            ),
            WindowReq(self.owner + "/q", 0, "sdma_queue"),
        ]
        if (
            self.abi >= 3
        ):  # the per-comm ERR/STEP/DIAG page: ONE per comm, shared by every R12 plan (small and large)
            reqs.append(
                WindowReq(
                    self.owner + "/gpage",
                    GLOBAL_PAGE_BYTES,
                    "control",
                    share=SHARE_GLOBAL,
                    min_bytes=GLOBAL_PAGE_BYTES,
                )
            )
        return reqs

    def _context(self, res):
        sc = res.__dict__.setdefault("_sp_qkv_large", {})
        got = sc.get(self.owner)
        if got is None:
            if not res.finalized:
                raise RuntimeError(
                    "sp_qkv.large: res.finalize() must run before the first launch (devcomm)"
                )
            if getattr(res, "sdma_queues", 1) != 1:
                raise RuntimeError(
                    f"sp_qkv.large: the shared devcomm has sdma_queues={res.sdma_queues}; the r11 kernels require exactly 1 "
                    f"(Phase-1 sdmaQueueCount=1)"
                )
            wins = res.windows.get(self.owner) or res.ensure(
                self.owner, self.resources()
            )
            built = build_large(self.params, self.mode)
            sha = _sha(built.hsaco)
            if sha != built.hsaco_sha256:
                raise RuntimeError(f"HSACO_HASH_MISMATCH {sha} != {built.hsaco_sha256}")
            lib = lrt.load_driver(built.driver_so, abi=self.abi)
            if self.abi >= 3:
                lrt.check_abi_info(
                    lrt.abi_info(lib), self.qr_wgs, gen=self.abi
                )  # args size, page/tag offsets, and the host build's LG_QR_WGS vs the device image's: a mismatch names the field instead of corrupting a launch
            geo = lrt.info(lib)
            if (
                geo["f"] != self.F
                or geo["indexer"] != (1 if self.mode == "indexer" else 0)
                or geo["fold"] != 0
            ):
                raise RuntimeError(f"build/mode mismatch {geo} vs mode {self.mode}")
            if lib.lg_window_bytes_v1(self.M) != large_window_bytes(self.mode, self.M):
                raise RuntimeError("compiled window size != analytic window size")
            import ctypes as C

            S, R, Cw = (
                wins[self.owner + "/S"],
                wins[self.owner + "/R"],
                wins[self.owner + "/C"],
            )
            preflight_attach(
                res, self.owner, self.M, wins, built
            )  # bounded, pure-Python, names the failing precondition before the driver call
            diag = (C.c_uint64 * 8)()
            h = lib.lg_attach_v1(
                res.rank,
                self.M,
                C.c_void_p(res.comm),
                C.c_void_p(res.devcomm),
                C.c_void_p(S.handle),
                C.c_void_p(S.local),
                C.c_void_p(R.handle),
                C.c_void_p(R.local),
                C.c_void_p(Cw.handle),
                C.c_void_p(Cw.local),
                built.hsaco.encode(),
                built.hsaco_sha256.encode(),
                1,
                diag,
            )  # H-3: C++ exact-image gate; diag[] carries the failing stage + values (no stderr scraping)
            if not h:
                raise RuntimeError(
                    f"lg_attach_v1 failed for {self.owner} rank={res.rank} M={self.M}: {lrt.decode_attach_diag(list(diag))}; "
                    f"hsaco={built.hsaco} sha256={built.hsaco_sha256} devcomm_queues={getattr(res, 'sdma_queues', '?')} "
                    f"devcomm_info={getattr(res, 'devcomm_info', None)}"
                )
            got = sc[self.owner] = lrt.LargeContext(
                lib, C.c_void_p(h), self.M, self.F, res.rank
            )
            if (
                self.abi >= 3
            ):  # attach the per-comm ERR/STEP page (+ the device-resident peer-ERR table) once per context, before the first launch
                page = res.global_page()
                got.gpage_attach(
                    page.err_addr, page.step_addr, res.global_peers().data_ptr()
                )
            got.budget_us = int(
                getattr(res, "wait_budget_us", 0) or 0
            )  # profile wait_budget_us / env / 5 s default; baked into captured graphs at capture time
            log.info(
                "event=binding large_wait_budget rank=%d owner=%s budget_us=%d source=%s (kernel argument, baked at graph capture)",
                res.rank,
                self.owner,
                got.budget_us,
                getattr(res, "wait_budget_source", "?"),
            )
            res.set_close(
                self.owner, got.close
            )  # L-5 (per context: each frees its own module/ctx)
            hooks = {
                "poison": (
                    None
                    if self.abi >= 3
                    else lambda c=got: c.poison(
                        stream_handle(None)
                    )  # cheap per-step check on the CURRENT torch stream (4-byte D2H + that stream sync)
                ),  # R12: no per-plan poison: the error is the per-comm ERR word
                "drain": lambda c=got: c.drain_all(
                    C.c_void_p(None)
                ),  # A2-H: quietQueue on all peer queues before any reset
                "init": got.init_control,
            }  # epoch=1 + magic; re-run after every reset
            if getattr(Cw, "shared", False):
                # ONE control block for every large plan: its init/drain/poison are registered ONCE (by the first context that attaches). A later
                # context attaching to the already LIVE block must not run init_control (that would reset the epoch on this rank only).
                first = res.set_shared_hooks(
                    init=hooks["init"],
                    drain=hooks["drain"],
                    poison=hooks["poison"],
                    poison_report=got.report,
                    poison_decode=decode_large_poison,
                )
                if self.abi >= 3 and first:
                    res.set_report_hook(
                        SHARED_KEY, got.report
                    )  # logged by reset() next to the page's ERR/DIAG
            else:
                if hooks["poison"] is not None:
                    res.set_poison(
                        self.owner, hooks["poison"], got.report, decode_large_poison
                    )
                else:
                    res.set_report_hook(self.owner, got.report)
                res.set_drain(self.owner, hooks["drain"])
                res.set_init(self.owner, hooks["init"])
        self._ctx = got
        return got

    @staticmethod
    def accepts_launch_m(params, bucket_m: int, m: int) -> bool:
        """R16 pad-M, a pure function: can a plan built for the bucket M `bucket_m` with these params launch the real local m? m == bucket always; below it only r13 (lg_launch_v3
        with m_real), for 1 <= m < bucket_m (the profile decides from which m on a bucket is used: m_pad_min_large).
        """
        src = dict(params).get("src", LARGE_SRC)
        if m == bucket_m:
            return True
        return bool(src_pad_m("large", src) and 1 <= m < bucket_m)

    def _check_inputs(self, x, x_scale, w_scale, w=None, M=None):
        """Reject, with ContractError, tensors the kernels cannot handle (see inputs.py). Dense/reuse builds read the first 9216 rows of w / w_scale, indexer 9856. M = the REAL rows of
        the launch (R16: x / x_scale carry m_real rows, not the bucket's)."""
        from . import inputs

        N = 9856 if self.mode == "indexer" else 9216
        inputs.check_activation(self.owner, x, x_scale, M=self.M if M is None else M)
        inputs.check_weight_scale(self.owner, w_scale, N=N)
        if w is not None:
            inputs.check_weight(self.owner, w, N=N)

    def prepare(self, *, res, x=None, x_scale=None, w_scale=None, **_):
        """Attach + init the control block. Call after res.finalize() and before graph capture; then barrier on all ranks. When the activation tensors are
        passed (ATOM does) their contract is checked here, once, outside capture."""
        if x is not None and x_scale is not None and w_scale is not None:
            self._check_inputs(x, x_scale, w_scale)
        return self._context(res)

    def run(
        self,
        *,
        x,
        x_scale,
        w,
        w_scale,
        outs=None,
        meta=None,
        res,
        stream=None,
        call_id=None,
    ):
        from .small_rt import stream_handle
        from .variants import _check_launch_gate, _require_zeroed

        _check_launch_gate(res, self.owner)
        _require_zeroed(res, self.owner)
        c = self._context(res)
        m_real = int(x.shape[0])
        if (
            self.pad_m
        ):  # R16: the kernel gets the valid rows only; the bucket M only selected this plan (geometry)
            if not 1 <= m_real <= self.M:
                raise ValueError(
                    f"x rows {m_real} not in 1..{self.M} (the bucket M of plan {self.owner})"
                )
        elif m_real != self.M:
            raise ValueError(f"x rows {x.shape[0]} != plan local M {self.M}")
        self._check_inputs(
            x, x_scale, w_scale, w, M=m_real
        )  # at capture once, in eager every call; a graph replay runs no Python
        if (
            self.abi >= 3
        ):  # R12: the tag E = (STEP << 32) | call_id; the launch is GEMM + quiet_ready (which includes the best-effort NaN fill under ERR)
            if call_id is None:
                raise ContractError(
                    f"{self.owner}: an R12 (abi {self.abi}) plan needs run(call_id=...): the 31-bit decision key of (site, tokens, vid, n_fused)"
                )
            c.launch_v3(
                x.data_ptr(),
                w.data_ptr(),
                x_scale.data_ptr(),
                w_scale.data_ptr(),
                x.stride(0) * x.element_size(),
                self.n,
                self.half_first,
                int(call_id),
                stream_handle(stream),
                m_real=m_real if self.pad_m else None,
            )
        else:
            c.launch(
                x.data_ptr(),
                w.data_ptr(),
                x_scale.data_ptr(),
                w_scale.data_ptr(),
                x.stride(0) * x.element_size(),
                self.n,
                self.half_first,
                stream_handle(stream),
            )
        self._epochs += 1
        self.last_m_real = m_real
        packed = (outs or {}).get("packed")
        if packed is not None:
            packed.copy_(self.recv_view(res, m_real))

    def _view_rows(self, m_real):
        """Rows of the receive view: 4 * m_real (R16: R is contiguous [:4 * m_real]); default = the rows of this plan's latest launch (the bucket M until a pad-M launch).
        A graph replay runs no Python, so the caller (ATOM) passes the m_real it captured with.
        """
        m = self.last_m_real if m_real is None else int(m_real)
        if not 1 <= m <= self.M:
            raise ValueError(
                f"m_real {m} not in 1..{self.M} (the bucket M of plan {self.owner})"
            )
        return 4 * m

    def recv_view(self, res, m_real=None):
        import torch

        rows = self._view_rows(m_real)
        c = self._context(res)
        t = torch.as_tensor(
            lrt._Va(c.recv_ptr(), (rows, self.F)), device=f"cuda:{res.device}"
        )
        return t.view(torch.bfloat16)

    def reuse_view(self, res, m_real=None):
        import torch

        if self.mode != "reuse":
            raise RuntimeError("reuse_view is only valid for the reuse mode")
        rows = self._view_rows(m_real)
        c = self._context(res)
        base = torch.as_tensor(
            lrt._Va(c.recv_ptr(), (rows * 2304 + 2048,)),
            device=f"cuda:{res.device}",
        ).view(torch.bfloat16)
        return torch.as_strided(base, (rows, 2560), (2304, 1))

    def poison(self, res) -> int:
        """Cheap host read of the device poison word (4-byte D2H on the current torch stream + that stream sync). Outside capture only; all-reduce MAX across ranks, then res.quiesce_and_reset()."""
        if self.abi >= 3:
            raise RuntimeError(
                "R12 plans have no per-plan poison word: use FusionResources.error_flag() / readback_enqueue() / readback_consume() and the collective reset()"
            )
        return self._context(res).poison(stream_handle(None))

    @classmethod
    def prebuild_units(cls, rows, *, recipe=None, tool=None):
        """Hook of aiter.gemm_fusion.prebuild: the build units (and skipped-row reasons) of this variant's profile rows."""
        from . import prebuild

        return prebuild.units_for_rows(rows, recipe=recipe, tool=tool, large=True)
