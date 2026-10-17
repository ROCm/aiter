# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""sp_qkv variants. sp_qkv.small.v1 = Phase-1 SM-T1-r17b / r18b kernels (algorithm unchanged), per-cell flags from
the profile row params. (sp_qkv.large.v1 is added in S2-2b-iii.)

plan.run(x, x_scale, w, w_scale, outs, meta, res, stream) contract (small):
  x        fp8 [Ml, K] row-major (Ml = local tokens), w fp8 preshuffled (16,16) [N, K], x_scale fp32 [Ml,1], w_scale fp32 [N]
  outs     {"q": bf16 [T,16,128]; dense/reuse: "k","v": bf16 [T,1,128]; indexer: "index_q": bf16 [T,1,128]}
  meta     {"q_norm_w","k_norm_w" (fp32 [128]); "cos_cache","sin_cache" (bf16 [max_pos,32]); "positions","slot_mapping" (int64 [T]);
            "k_cache","v_cache","k_scale","v_scale" (golden SHUFFLE page16 fp8 caches + fp32 scales);
            indexer only: "index_q_norm_w","index_k_norm_w" (fp32 [128]), "index_cache" (fp8 flat page128)}
  Binding (one device-resident SmDev block) is created lazily on the first run() with a given set of pointers; this MUST
  happen outside graph capture (call plan.prepare(...) from before_graph_capture/bind_kv_cache). Pointers (scales, outs,
  meta tensors incl. caches) must not move after binding; a different pointer set creates a new binding, which is an error
  inside capture. x / w are passed per launch (kernargs) and may vary.
  T = 4*Ml; all ranks must run identical launch counts (epochs are device-owned).
"""

from __future__ import annotations

import re
from typing import Dict, List

from aiter.gemm_fusion.catalog import Contract, register_variant
from aiter.gemm_fusion.match import ContractError
from aiter.gemm_fusion.plan import Plan, WindowReq
from aiter.gemm_fusion.resources import round_up

from . import specs
from .jit import build_small, small_defs
from . import small_rt as rt

SRC_SHA = {
    "r17": "3f597cdb6141c39dd4450a1ea635785ffe44867006434157478f7822357d8ee3",
    "r18": "0e48e3f01e917c9c5209c71efd90748e83eacf59d0e92cd10e5549955a450c29",
}
# analytic geometry (mirrors sm_abi_v2.h; cross-checked against sm_info_v1 after build)
_F = {"dense": 2304, "indexer": 2560, "reuse": 2560}
_NA = {"dense": 72, "indexer": 77, "reuse": 77}
_MAX_BINDINGS = 64


def small_window_bytes(mode: str) -> Dict[str, int]:
    rx = 2 * 4 * rt.SM_MP * _F[mode] * 2
    ctl = 256 + 8 * 2 * 4 * _NA[mode]
    return {"rx": rx, "ctl": ctl}


@register_variant(
    vid="sp_qkv.small.v1", family="gemm_ep.sp_qkv", stage_kinds=("sp_exchange", "head_norm_rope_cache"),
    contract=Contract(arch=("gfx950",), world=4, topology="single_node_xgmi_p2p", graph="capture_safe_epoch_state",
                      requires=("mori_cco",), m_multiple=1,
                      mutates=("k_cache", "v_cache", "k_scale", "v_scale", "index_cache")),
    source_sha="r17=" + SRC_SHA["r17"] + ";r18=" + SRC_SHA["r18"],
)
class SpQkvSmall(Plan):
    n_stages = 2

    def __init__(self, spec, params, ctx):
        self.spec, self.params = spec, dict(params)
        self.mode = specs.mode_of(spec)
        self.ctx_m = ctx.m
        small_defs(self.params, self.mode)  # validates params
        if self.params.get("rx_bypass") and "gfx950_sc_bypass" not in ctx.available:
            raise ContractError("sp_qkv.small.v1 rx_bypass=1 requires capability gfx950_sc_bypass (R2 is gfx950-only)")
        if not 1 <= ctx.m <= rt.SM_MP:
            raise ContractError(f"sp_qkv.small.v1 supports local M 1..{rt.SM_MP}, got {ctx.m}")
        if self.params.get("push8") and ctx.m > 16:
            raise ContractError("push8 build supports M<=16")
        if self.params.get("b_rowsplit") and ctx.m <= 16:
            raise ContractError("b_rowsplit build supports M>16")
        hn = spec.stages[1]
        self.eps, self.block = hn.eps, int(re.search(r"page(\d+)", hn.kv_cache).group(1))
        # owner key: plans with identical build + mode share windows/context/epoch domain (runtime M may differ only if flags allow)
        self.owner = "sp_qkv.small/" + self.mode + "/" + ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        self._bindings: Dict[tuple, object] = {}
        self._ctx = None

    def resources(self) -> List[WindowReq]:
        b = small_window_bytes(self.mode)
        cap = round_up(max(b["rx"], b["ctl"], 8192))
        return [WindowReq(self.owner + "/rx", cap, "p2p_window"), WindowReq(self.owner + "/ctl", cap, "control")]

    # ---- runtime ----
    def _context(self, res):
        sc = res.__dict__.setdefault("_sp_qkv_small", {})
        got = sc.get(self.owner)
        if got is None:
            wins = res.ensure(self.owner, self.resources())
            built = build_small(self.params, self.mode)
            lib = rt.load_driver(built.driver_so)
            geo = rt.info(lib)
            exp = small_window_bytes(self.mode)
            if geo["rx_bytes"] != exp["rx"] or geo["ctl_bytes"] != exp["ctl"]:
                raise RuntimeError(f"compiled geometry {geo} != analytic {exp}")
            import ctypes as C

            h = C.c_void_p()
            rx, ct = wins[self.owner + "/rx"], wins[self.owner + "/ctl"]
            rc = lib.sm_attach_v1(res.rank, res.comm, rx.handle, rx.local, ct.handle, ct.local,
                                  built.gemm_hsaco.encode(), built.ran_hsaco.encode(), C.byref(h))
            if rc != 0:
                raise RuntimeError(f"sm_attach rc={rc}")
            got = sc[self.owner] = rt.SmallContext(lib, h)
        self._ctx = got
        return got

    def _binding(self, x, x_scale, w_scale, outs, meta, res, in_capture=False):
        import torch

        m = self.mode
        caches = {n: meta[n] for n in ("k_cache", "v_cache", "k_scale", "v_scale")}
        iqw = meta.get("index_q_norm_w") if m == "indexer" else None
        ikw = meta.get("index_k_norm_w") if m == "indexer" else None
        zeros = None
        if iqw is None:  # dense/reuse: kernel never reads these; Phase-1 harness passes zeroed 16B-aligned buffers
            zeros = res.__dict__.setdefault("_sp_qkv_zero128", torch.zeros(128, device=x.device, dtype=torch.float32))
            iqw = ikw = zeros
        tens = [x_scale, w_scale, meta["q_norm_w"], meta["k_norm_w"], iqw, ikw, meta["cos_cache"], meta["sin_cache"],
                meta["positions"], meta["slot_mapping"], outs["q"], *caches.values()]
        for n in ("k", "v", "index_q"):
            if n in outs:
                tens.append(outs[n])
        if m == "indexer":
            tens.append(meta["index_cache"])
        key = tuple(t.data_ptr() for t in tens) + (x.shape[0],)
        blk = self._bindings.get(key)
        if blk is not None:
            return blk
        if in_capture:
            raise RuntimeError("sp_qkv.small: new binding required during graph capture; call plan.prepare() before capture")
        if len(self._bindings) >= _MAX_BINDINGS:
            raise RuntimeError("sp_qkv.small: too many distinct bindings (pointers must be stable)")
        sb = rt.SmBindingV1()
        sb.Ascale, sb.Wscale = x_scale.data_ptr(), w_scale.data_ptr()
        sb.qw, sb.kw = meta["q_norm_w"].data_ptr(), meta["k_norm_w"].data_ptr()
        sb.cos_cache, sb.sin_cache = meta["cos_cache"].data_ptr(), meta["sin_cache"].data_ptr()
        sb.cos_stride_pos = meta["cos_cache"].stride(0)
        sb.positions, sb.slot_mapping = meta["positions"].data_ptr(), meta["slot_mapping"].data_ptr()
        for n, t in caches.items():
            setattr(sb, n, t.data_ptr())
        for pre, t, nms in (("kc", caches["k_cache"], ("sb", "sh", "sdx", "ss", "sx")),
                            ("vc", caches["v_cache"], ("sb", "sh", "ssc", "sd", "sx"))):
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
        sb.index_slot_mapping = meta["slot_mapping"].data_ptr()
        sb.M, sb.T, sb.block_size, sb.x_size = x.shape[0], 4 * x.shape[0], self.block, 16
        sb.strideA = x.stride(0) * x.element_size()
        sb.eps = self.eps
        sb.fp8_amax = torch.finfo(caches["k_cache"].dtype).max
        sb.poll_iterations = 20000000
        blk = self._context(res).bind(sb)
        self._bindings[key] = blk
        return blk

    def prepare(self, *, x, x_scale, w_scale, outs, meta, res):
        """Create the binding (and attach the context) outside graph capture. Idempotent."""
        self._context(res)
        return self._binding(x, x_scale, w_scale, outs, meta, res)

    def run(self, *, x, x_scale, w, w_scale, outs, meta, res, stream=None):
        import torch

        sc = self._context(res)
        blk = self._binding(x, x_scale, w_scale, outs, meta, res, in_capture=torch.cuda.is_current_stream_capturing())
        sc.launch(blk, x.data_ptr(), w.data_ptr(), rt.stream_handle(stream))
