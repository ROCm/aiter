# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""sp_qkv variants. sp_qkv.small.v1 = Phase-1 SM-T1-r17b / r18b kernels (algorithm unchanged), per-cell flags from
the profile row params. (sp_qkv.large.v1 is added in S2-2b-iii.)

plan.run(x, x_scale, w, w_scale, outs, meta, res, stream) contract (small):
  x        fp8 [Ml, K] row-major (Ml = local tokens), w fp8 preshuffled (16,16) [N, K], x_scale fp32 [Ml,1], w_scale fp32 [N]
  outs     {"q": bf16 [T,16,128]; dense/reuse: "k","v": bf16 [T,1,128]; indexer: "index_q": bf16 [T,1,128]}
  meta     {"q_norm_w","k_norm_w" (fp32 [128]); "cos_cache","sin_cache" (bf16 [max_pos,32]); "positions","slot_mapping" (int64 [T]);
            "k_cache","v_cache","k_scale","v_scale" (golden SHUFFLE fp8 caches + fp32 scales; page = 128 dense, 16 indexer/reuse, per case_config.cache_page_size);
            indexer only: "index_q_norm_w","index_k_norm_w" (fp32 [128]), "index_cache" (fp8 flat page128)}
  Binding (one device-resident SmDev block) is created lazily on the first run() with a given set of pointers; this MUST
  happen outside graph capture (call plan.prepare(...) from before_graph_capture/bind_kv_cache). Pointers (scales, outs,
  meta tensors incl. caches) must not move after binding; a different pointer set creates a new binding, which is an error
  inside capture. x / w are passed per launch (kernargs) and may vary.
  T = 4*Ml; all ranks must run identical launch counts (epochs are device-owned).
"""

from __future__ import annotations

import logging
import re

from aiter.gemm_fusion.catalog import Contract, register_variant
from aiter.gemm_fusion.errpage import GLOBAL_PAGE_BYTES, SHARE_GLOBAL
from aiter.gemm_fusion.match import ContractError
from aiter.gemm_fusion.plan import Plan, WindowReq
from aiter.gemm_fusion.resources import round_up

log = logging.getLogger("aiter.gemm_fusion")

from . import (
    inputs,
    specs,
)
from . import small_rt as rt
from .jit import build_small, small_defs, src_abi, src_mp

SRC_SHA = {
    "r17": "3f597cdb6141c39dd4450a1ea635785ffe44867006434157478f7822357d8ee3",
    "r18": "0e48e3f01e917c9c5209c71efd90748e83eacf59d0e92cd10e5549955a450c29",
    "r19": "1b01853ce0441478002297bca075b3e0b1a14a1013fd7b3a04a5fd157fb1b02d",  # R12 (abi 3), derived from r18; sha256 of small/r19/SOURCE_SHA256.txt
    "r20": "a63f387ecff31e2fad83d68c105a521db2525c4476213bb075587061f1fab92e",  # R13 MP64 (abi 3, SM_MP 64), derived from r19
    "r21": "b83f266da5b51405ca039f35bf5ce2ae25b95e72809d09665107b986c4a2ec05",  # R13 MP128 (abi 3, SM_MP 128, 8 MiB Rx view), derived from r20
}
# analytic geometry (mirrors sm_abi_v2.h; cross-checked against sm_info_v1 after build)
_F = {"dense": 2304, "indexer": 2560, "reuse": 2560}
_NA = {"dense": 72, "indexer": 77, "reuse": 77}
_MAX_BINDINGS = 64


def small_window_bytes(mode: str, mp: int | None = None) -> dict[str, int]:
    """Analytic window bytes of a small owner: Rx [2][4][MP][F] bf16 and the control window. mp = the source's SM_MP (32 for r17-r19, 64 for r20)."""
    mp = rt.SM_MP if mp is None else mp
    rx = 2 * 4 * mp * _F[mode] * 2
    ctl = 256 + 8 * 2 * 4 * _NA[mode]
    return {"rx": rx, "ctl": ctl}


def small_window_caps(mode: str, mp: int | None = None) -> tuple[int, int]:
    """(Rx, control) window capacities a small owner requests, each rounded up to the 4 MiB granule and sized SEPARATELY: 4 MiB each for MP <= 64 (Rx 1.1 / 2.4 / 2.6 MiB), but
    MP128's Rx (4.5 MiB dense, 5.0 MiB indexer / reuse) needs an 8 MiB arena view while its control window stays one granule.
    """
    b = small_window_bytes(mode, mp)
    return round_up(max(b["rx"], 8192)), round_up(max(b["ctl"], 8192))


def small_p_bytes(mode: str, mp: int | None = None, sk: int = 3) -> int:
    """Analytic bytes of the driver-owned split-K partial buffer P [SK][MP][N] fp32 (hipMalloc'd per context, NOT a comm window): dense 7,077,888 B and
    indexer / reuse 7,569,408 B at MP64 / SK3. Reported so the HBM cost of an MP64 owner is visible next to its windows.
    """
    mp = rt.SM_MP if mp is None else mp
    return sk * mp * _NA[mode] * 128 * 4


def decode_small_poison(word: int) -> str:
    """The small-M poison word is the `why` code of the failing site; bit 0x100 = the wait found the flag AHEAD of the expected epoch (overshoot / epoch
    mismatch), otherwise a bounded-poll TIMEOUT (poll_iterations = 20,000,000 s_sleep(2) iterations). The per-site `why` values are kernel-internal (ran_kernel.cu).
    """
    w = int(word)
    kind = (
        "OVERSHOOT(flag ahead of expected epoch)"
        if w & 0x100
        else "TIMEOUT-or-other(bounded poll / contract check)"
    )
    return f"{kind} why={w & 0xFF:#x} word={w:#x}"


def _require_zeroed(res, owner: str):
    """With the default zero mode (h2d_deferred) the windows are zeroed in FusionResources.finalize(); a launch before that would run on garbage control state."""
    if getattr(res, "pending_zero", 0):
        raise RuntimeError(
            f"{owner}: {res.pending_zero} CCO window(s) are registered but not yet zeroed; call res.finalize() (collective) before the first launch"
        )


def _diff_parts(old: dict, new: dict) -> list[str]:
    return [
        f"{k}: {old.get(k)} -> {new.get(k)}"
        for k in sorted(set(old) | set(new))
        if old.get(k) != new.get(k)
    ]


def check_cache_page(k_cache, v_cache, page: int):
    """Contract: the golden SHUFFLE caches must have the page the spec was matched with.
    K [blocks, kvh, head_dim/x(8), page, x(16)] -> shape[3]; V [blocks, kvh, page/x, head_dim, x(16)] -> shape[2]*16.
    """
    kp = k_cache.shape[3] if len(k_cache.shape) == 5 else None
    vp = v_cache.shape[2] * 16 if len(v_cache.shape) == 5 else None
    if kp != page or vp != page:
        raise ContractError(
            f"sp_qkv.small.v1: cache page mismatch: spec page={page}, k_cache page={kp}, v_cache page={vp} "
            f"(k {tuple(k_cache.shape)}, v {tuple(v_cache.shape)})"
        )


@register_variant(
    vid="sp_qkv.small.v1",
    family="gemm_ep.sp_qkv",
    stage_kinds=("sp_exchange", "head_norm_rope_cache"),
    contract=Contract(
        arch=("gfx950",),
        world=4,
        topology="single_node_xgmi_p2p",
        graph="capture_safe_epoch_state",
        requires=("mori_cco",),
        m_multiple=1,
        mutates=("k_cache", "v_cache", "k_scale", "v_scale", "index_cache"),
    ),
    source_sha="r17="
    + SRC_SHA["r17"]
    + ";r18="
    + SRC_SHA["r18"]
    + ";r19="
    + SRC_SHA["r19"]
    + ";r20="
    + SRC_SHA["r20"]
    + ";r21="
    + SRC_SHA["r21"],
)
class SpQkvSmall(Plan):
    n_stages = 2

    def __init__(self, spec, params, ctx):
        self.spec, self.params = spec, dict(params)
        self.mode = specs.mode_of(spec)
        self.ctx_m = ctx.m
        small_defs(self.params, self.mode)  # validates params
        self.src = self.params.get("src", "r17")
        self.abi = src_abi(
            "small", self.src
        )  # 2 = r17/r18; 3 = r19 (R12: per-launch kernargs, per-comm ERR/STEP page, call_id tags)
        if self.params.get("rx_bypass") and "gfx950_sc_bypass" not in ctx.available:
            raise ContractError(
                "sp_qkv.small.v1 rx_bypass=1 requires capability gfx950_sc_bypass (R2 is gfx950-only)"
            )
        self.mp = src_mp(
            self.src
        )  # padded rows of this source: r17-r19 serve local m 1..32, r20 (MP64) 1..64
        if not 1 <= ctx.m <= self.mp:
            raise ContractError(
                f"sp_qkv.small.v1 (src {self.src}) supports local M 1..{self.mp}, got {ctx.m}"
            )
        if self.params.get("push8") and ctx.m > 16:
            raise ContractError("push8 build supports M<=16")
        if self.params.get("b_rowsplit") and ctx.m <= 16:
            raise ContractError("b_rowsplit build supports M>16")
        hn = spec.stages[1]
        self.eps, self.block = hn.eps, int(
            re.search(r"page(\d+)", hn.kv_cache).group(1)
        )
        # owner key: plans with identical build + mode share windows/context/epoch domain (runtime M may differ only if flags allow)
        self.owner = (
            "sp_qkv.small/"
            + self.mode
            + "/"
            + ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        )
        self.build_key = (
            self.owner
        )  # the key ATOM's plan_owner() uses for FusionResources.ensure / windows
        self._bindings: dict[tuple, object] = {}
        self._parts: dict[tuple, dict] = (
            {}
        )  # key -> named components (for diffing a miss)
        self._ctx = None

    def resources(self) -> list[WindowReq]:
        rx_cap, ctl_cap = small_window_caps(self.mode, self.mp)
        reqs = [
            WindowReq(self.owner + "/rx", rx_cap, "p2p_window"),
            WindowReq(self.owner + "/ctl", ctl_cap, "control"),
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

    # ---- runtime ----
    def _context(self, res):
        sc = res.__dict__.setdefault("_sp_qkv_small", {})
        got = sc.get(self.owner)
        if got is None:
            wins = res.ensure(self.owner, self.resources())
            built = build_small(self.params, self.mode)
            lib = rt.load_driver(built.driver_so, abi=self.abi)
            geo = rt.info(lib)
            exp = small_window_bytes(self.mode, self.mp)
            if geo["rx_bytes"] != exp["rx"] or geo["ctl_bytes"] != exp["ctl"]:
                raise RuntimeError(f"compiled geometry {geo} != analytic {exp}")
            import ctypes as C

            h = C.c_void_p()
            rx, ct = wins[self.owner + "/rx"], wins[self.owner + "/ctl"]
            if self.abi >= 3:
                # R12 attach (card D2.6): + the global page's local address; the peers table is a device pointer (0 = the driver resolves the peers itself)
                page = res.global_page()
                rc = lib.sm_attach_v3(
                    res.rank,
                    res.comm,
                    rx.handle,
                    rx.local,
                    ct.handle,
                    ct.local,
                    C.c_void_p(page.local),
                    C.c_void_p(res.global_peers().data_ptr()),
                    built.gemm_hsaco.encode(),
                    built.ran_hsaco.encode(),
                    C.byref(h),
                )
            else:
                rc = lib.sm_attach_v1(
                    res.rank,
                    res.comm,
                    rx.handle,
                    rx.local,
                    ct.handle,
                    ct.local,
                    built.gemm_hsaco.encode(),
                    built.ran_hsaco.encode(),
                    C.byref(h),
                )
            if rc != 0:
                if (
                    h.value
                ):  # driver keeps a quarantined ctx on failure (Phase-1 convention); free it, never use it
                    rt.SmallContext(lib, h).close()
                raise RuntimeError(f"sm_attach rc={rc}")
            got = sc[self.owner] = rt.SmallContext(lib, h)
            res.set_close(
                self.owner, got.close
            )  # L-5: frees Blk, P, modules, ctx before comm teardown
            if (
                self.abi < 3
            ):  # R12 plans have no per-plan poison word: the error is the per-comm ERR word (FusionResources.error_flag / readback)
                res.set_poison(
                    self.owner,
                    lambda c=got: c.poison_fast(rt.stream_handle(None)),
                    None,
                    decode_small_poison,
                )  # cheap per-step check on the CURRENT torch stream (4-byte D2H + that stream sync)
        self._ctx = got
        return got

    def _binding(self, x, x_scale, w_scale, outs, meta, res, in_capture=False):
        import torch

        m = self.mode
        caches = {n: meta[n] for n in ("k_cache", "v_cache", "k_scale", "v_scale")}
        iqw = meta.get("index_q_norm_w") if m == "indexer" else None
        ikw = meta.get("index_k_norm_w") if m == "indexer" else None
        zeros = None
        if (
            iqw is None
        ):  # dense/reuse: kernel never reads these; Phase-1 harness passes zeroed 16B-aligned buffers
            zeros = res.__dict__.get(
                "_sp_qkv_zero128"
            )  # NOT setdefault(..., torch.zeros()): that allocated on EVERY call, even inside capture
            if zeros is None:
                zeros = res.__dict__["_sp_qkv_zero128"] = torch.zeros(
                    128, device=x.device, dtype=torch.float32
                )
            iqw = ikw = zeros
        named = {
            "x_scale": x_scale,
            "w_scale": w_scale,
            "q_norm_w": meta["q_norm_w"],
            "k_norm_w": meta["k_norm_w"],
            "iqw": iqw,
            "ikw": ikw,
            "cos_cache": meta["cos_cache"],
            "sin_cache": meta["sin_cache"],
            "positions": meta["positions"],
            "slot_mapping": meta["slot_mapping"],
            "q_out": outs["q"],
            **caches,
        }
        if (
            self.abi >= 3
        ):  # R12: x_scale / positions / slot_mapping are PER-LAUNCH kernargs, not part of the binding: eager calls with fresh tensors never create one
            for n in ("x_scale", "positions", "slot_mapping"):
                named.pop(n)
        for n in ("k", "v", "index_q"):
            if n in outs:
                named[n + "_out"] = outs[n]
        if m == "indexer":
            named["index_cache"] = meta["index_cache"]
        parts = {n: t.data_ptr() for n, t in named.items()}
        # everything else the device block bakes in from these tensors (strides) or from x (rows): a change must be a NEW binding, not stale reuse
        parts["x.rows"] = x.shape[0]
        parts["cos.stride0"] = meta["cos_cache"].stride(0)
        parts["k_cache.strides"] = tuple(caches["k_cache"].stride())
        parts["v_cache.strides"] = tuple(caches["v_cache"].stride())
        parts["k_scale.strides"] = (
            caches["k_scale"].stride(0),
            caches["k_scale"].stride(1),
        )
        parts["v_scale.strides"] = (
            caches["v_scale"].stride(0),
            caches["v_scale"].stride(1),
        )
        key = tuple(parts.items())
        blk = self._bindings.get(key)
        if blk is not None:
            return blk
        # a NEW key: say exactly which components differ from the closest existing binding (this is the hidden eager second-binding too)
        if self._bindings:
            best = min(self._parts.values(), key=lambda p: len(_diff_parts(p, parts)))
            why = _diff_parts(best, parts)
            msg = (
                f"{self.owner}: binding #{len(self._bindings) + 1} needed; differs from the closest of {len(self._bindings)} existing "
                f"binding(s) in: {why}"
            )
        else:
            why, msg = [], f"{self.owner}: first binding"
        if in_capture:
            raise RuntimeError(
                "sp_qkv.small: new binding required during graph capture (prepare() did not bind these exact tensors): "
                + msg
            )
        if self._bindings:
            log.warning("event=binding new %s", msg)
        else:
            log.info("event=binding first owner=%s", self.owner)
        if len(self._bindings) >= _MAX_BINDINGS:
            raise RuntimeError(
                "sp_qkv.small: too many distinct bindings (pointers must be stable): "
                + msg
            )
        check_cache_page(
            caches["k_cache"], caches["v_cache"], self.block
        )  # spec page vs actual cache layout
        sb = rt.SmBindingV1()
        sb.Ascale, sb.Wscale = (
            None if self.abi >= 3 else x_scale.data_ptr(),
            w_scale.data_ptr(),
        )
        sb.qw, sb.kw = meta["q_norm_w"].data_ptr(), meta["k_norm_w"].data_ptr()
        sb.cos_cache, sb.sin_cache = (
            meta["cos_cache"].data_ptr(),
            meta["sin_cache"].data_ptr(),
        )
        sb.cos_stride_pos = meta["cos_cache"].stride(0)
        sb.positions, sb.slot_mapping = (
            (None, None)
            if self.abi >= 3
            else (meta["positions"].data_ptr(), meta["slot_mapping"].data_ptr())
        )
        for n, t in caches.items():
            setattr(sb, n, t.data_ptr())
        for pre, t, nms in (
            ("kc", caches["k_cache"], ("sb", "sh", "sdx", "ss", "sx")),
            ("vc", caches["v_cache"], ("sb", "sh", "ssc", "sd", "sx")),
        ):
            for n, st in zip(nms, t.stride()):
                setattr(sb, pre + "_" + n, st)
        for pre, t in (("ks", caches["k_scale"]), ("vs", caches["v_scale"])):
            setattr(sb, pre + "_sb", t.stride(0))
            setattr(sb, pre + "_sh", t.stride(1))
        sb.q_out = outs["q"].data_ptr()
        if m == "indexer":  # golden returns no k/v in indexer mode
            sb.k_out = sb.v_out = None
            sb.index_q_out = outs["index_q"].data_ptr()
            sb.index_cache = meta["index_cache"].data_ptr()
        else:
            sb.k_out, sb.v_out = outs["k"].data_ptr(), outs["v"].data_ptr()
            sb.index_q_out = zeros.data_ptr() if zeros is not None else None
        sb.iqw, sb.ikw = iqw.data_ptr(), ikw.data_ptr()
        sb.index_slot_mapping = (
            None if self.abi >= 3 else meta["slot_mapping"].data_ptr()
        )
        sb.M, sb.T, sb.block_size, sb.x_size = (
            x.shape[0],
            4 * x.shape[0],
            self.block,
            16,
        )
        sb.strideA = x.stride(0) * x.element_size()
        sb.eps = self.eps
        sb.fp8_amax = torch.finfo(caches["k_cache"].dtype).max
        sb.poll_iterations = 20000000
        blk = self._context(res).bind(sb)
        self._bindings[key] = blk
        self._parts[key] = parts
        return blk

    def _check_inputs(self, x, x_scale, w_scale, outs, meta, w=None):
        """Reject, with ContractError, tensors the kernels cannot handle (see inputs.py) BEFORE any binding or launch. prepare() has no weight, run() passes it."""
        M = int(x.shape[0])
        if not 1 <= M <= self.mp:
            raise ContractError(
                f"{self.owner}: x has {M} rows, a src {self.src} small plan takes local M 1..{self.mp}"
            )
        inputs.check_activation(self.owner, x, x_scale, M=M)
        inputs.check_weight_scale(self.owner, w_scale, N=_NA[self.mode] * 128)
        if w is not None:
            inputs.check_weight(self.owner, w, N=_NA[self.mode] * 128)
        if (
            self.abi >= 3
        ):  # R12: short arrays are legal (the kernel pads rows t >= t_valid as position 0 / slot -1); returns t_valid
            return inputs.check_token_arrays_v3(
                self.owner,
                T=4 * M,
                positions=meta["positions"],
                slot_mapping=meta["slot_mapping"],
                outs=outs,
            )
        inputs.check_token_arrays(
            self.owner,
            T=4 * M,
            positions=meta["positions"],
            slot_mapping=meta["slot_mapping"],
            outs=outs,
        )
        return 4 * M

    def prepare(self, *, x, x_scale, w_scale, outs, meta, res):
        """Create the binding (and attach the context) outside graph capture. Idempotent. Checks the input contract (once, outside capture)."""
        self._check_inputs(x, x_scale, w_scale, outs, meta)
        self._context(res)
        return self._binding(x, x_scale, w_scale, outs, meta, res)

    def run(
        self,
        *,
        x,
        x_scale,
        w,
        w_scale,
        outs,
        meta,
        res,
        stream=None,
        call_id=None,
    ):
        import torch

        _require_zeroed(res, self.owner)
        t_valid = self._check_inputs(
            x, x_scale, w_scale, outs, meta, w
        )  # at capture once, in eager every call; a graph replay runs no Python
        if self.abi >= 3:
            if call_id is None:
                raise ContractError(
                    f"{self.owner}: an R12 (abi {self.abi}) plan needs run(call_id=...): the 31-bit decision key of (site, tokens, vid, n_fused)"
                )
            inputs.check_call_id(self.owner, call_id)
        sc = self._context(res)
        blk = self._binding(
            x,
            x_scale,
            w_scale,
            outs,
            meta,
            res,
            in_capture=torch.cuda.is_current_stream_capturing(),
        )
        if (
            self.abi >= 3
        ):  # x_scale / positions / slot_mapping / t_valid / call_id are per-launch kernargs; err and step come from the attached global page
            sc.launch_v3(
                blk,
                x.data_ptr(),
                w.data_ptr(),
                x.stride(0) * x.element_size(),
                x_scale.data_ptr(),
                meta["positions"].data_ptr(),
                meta["slot_mapping"].data_ptr(),
                t_valid,
                call_id,
                rt.stream_handle(stream),
            )
            return
        sc.launch(
            blk,
            x.data_ptr(),
            w.data_ptr(),
            x.stride(0) * x.element_size(),
            rt.stream_handle(stream),
        )


from . import variants_large  # noqa: F401  (registers sp_qkv.large.v1)


def _small_poison(self, res) -> int:
    """Cheap host read of the device poison word (4-byte D2H on the current torch stream + that stream sync). Outside capture only; all-reduce MAX across ranks, then res.quiesce_and_reset()."""
    if self.abi >= 3:
        raise RuntimeError(
            "R12 plans have no per-plan poison word: use FusionResources.error_flag() / readback_enqueue() / readback_consume() and the collective reset()"
        )
    return self._context(res).poison_fast(rt.stream_handle(None))


SpQkvSmall.poison = _small_poison


def _small_prebuild_units(cls, rows, *, recipe=None, tool=None):
    """Hook of aiter.gemm_fusion.prebuild: the build units (and skipped-row reasons) of this variant's profile rows."""
    from . import prebuild

    return prebuild.units_for_rows(rows, recipe=recipe, tool=tool, large=False)


SpQkvSmall.prebuild_units = classmethod(_small_prebuild_units)
