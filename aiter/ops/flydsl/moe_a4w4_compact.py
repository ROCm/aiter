# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL A4W4 compact MoE: MXFP4 prefill MoE for gfx950.

MXFP4 weights and activations (e8m0 scale per 1x32 group), SwiGLU-OAI, prefill
batches. Rows are compact (exactly ``tokens * topk``, no ``block_m`` padding) and
any ``inter_dim % 128 == 0`` runs at its true width. One call runs K0 (on-device
activation quant + routing plan), stage 1 (gate/up GEMM, SwiGLU, fp4 requant of
h), stage 2 (down GEMM, router weight) and the top-k combine.

``fused_moe`` dispatches here for tuned rows named by :mod:`.a4w4c_kname`
(``kernelName1 = flydsl_a4w4c_g1_*``, ``kernelName2 = flydsl_a4w4c_g2_*``).
Weights are the usual per_1x32 fp4 layout (``shuffle_weight(w, (16, 16))``,
``e8m0_shuffle`` scales, gate/up SEPARATED); fp4 bytes are read in place and only
the scales are repacked, once per weight tensor. Padded calls fall back.

Candidate configs live in ``kernels/moe_a4w4_compact/configs/I{inter_dim}.json``
(one cell per power-of-two token bucket); ``tune_space`` lists them.

The fused combine (``_fc_`` in kernelName2) is decided on the device per batch:
it runs when expert E-1 is routed exactly once per token (vLLM fused shared
experts); any other routing takes the separate combine.

Environment:
    AITER_MOE_A4W4_COMPACT_SCALE_RULE: e8m0 rule of the runtime activation quant,
        ``ceil`` (default, AITER's MX quant) or ``even`` (Quark "even" checkpoints).
"""

import functools
import json
import os
import weakref

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe_registry import FusedMoeRequest
from aiter.jit.utils.chip_info import get_gfx

from . import a4w4c_kname
from .kernels.moe_a4w4_compact import combine, gemm, layout, prologue

TABLE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "kernels", "moe_a4w4_compact", "configs"
)
MIN_TOKENS = 512
_SWIGLU_LIMIT = 7.0
_SCALE_RULE = os.environ.get("AITER_MOE_A4W4_COMPACT_SCALE_RULE", "ceil")


@functools.cache
def load_table(inter_dim: int) -> dict:
    """Token bucket -> {"s1": ..., "s2": ..., "global": ...} for ``inter_dim``."""
    path = os.path.join(TABLE_DIR, f"I{inter_dim}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {int(t): c for t, c in json.load(f).items()}


def cfg_kwargs(cell: dict) -> dict:
    """MoERun keyword arguments for one table cell."""
    kwargs = {f"{k}1": v for k, v in cell["s1"].items()}
    kwargs.update({f"{k}2": v for k, v in cell["s2"].items()})
    kwargs.update(cell.get("global", {}))
    return kwargs


def tune_space(inter_dim: int) -> list[tuple[str, str]]:
    """Distinct (kernelName1, kernelName2) prefill candidates for ``inter_dim``."""
    seen = []
    for token, cell in sorted(load_table(inter_dim).items()):
        names = a4w4c_kname.kernel_names(cell)
        if token >= MIN_TOKENS and names not in seen:
            seen.append(names)
    return seen


class MoEWeights:
    """Expert weights of one TP shard (E experts incl. shared).

    w13 [E, 2I, H/2] ([gate; up]) and w2 [E, H, I/2]: fp4x2 in
    ``shuffle_weight(w, layout=(16, 16))`` layout, read in place.
    bs1 / bs2: ``layout.pack_scales`` packed e8m0 scales."""

    def __init__(self, w13, w2, bs1, bs2):
        self.E, n13, hh = w13.shape
        self.I, self.H = n13 // 2, hh * 2
        assert w13.is_contiguous() and w2.is_contiguous()
        assert w2.shape == (self.E, self.H, self.I // 2), (w13.shape, w2.shape)
        assert bs1.numel() == self.E * n13 * (self.H // 32)
        assert bs2.numel() == self.E * self.H * (self.I // 32)
        self.b1, self.b2, self.bs1, self.bs2 = w13, w2, bs1, bs2

    @classmethod
    def build(cls, w13, s13, w2, s2):
        """s13 [E, 2I, H/32], s2 [E, H, I/32]: natural-row e8m0 scales."""
        return cls(
            w13,
            w2,
            layout.pack_scales(1, s13.view(torch.uint8)),
            layout.pack_scales(2, s2.view(torch.uint8)),
        )


class MoERun:
    """Buffers + launches for one token count T. Call forward() per step.

    Per-stage settings (suffix 1 / 2): BM, D (pipeline depth), pipe, NW (waves),
    WM (waves along M), EF (fast SwiGLU epilogue), MV (no AGPRs), PERS
    (persistent), diag (``+``-joined kernel options).
    HT: stage 1 writes h K-step-major so stage-2 A loads are contiguous 1 KB DMAs.
    FC: fused combine; shared-expert (E-1) tiles add the token's routed rows and
        write ``out``. ``*F`` override the shared-tile launch. The plan checks on the
        device that E-1 is exactly once per token; otherwise E-1 runs as a routed
        expert and the gated combine writes ``out``.
    QP: the activation quant runs as extra CTAs of the plan's first launch.
    SR: e8m0 scale rule of the activation / h quant, "even" or "ceil".
    """

    def __init__(
        self,
        x,
        topk_ids,
        topk_w,
        W: MoEWeights,
        BM1=128,
        BM2=128,
        D1=3,
        D2=2,
        pipe1="async",
        pipe2="hybrid2",
        NW1=4,
        NW2=4,
        diag1="",
        diag2="",
        WM1=1,
        WM2=1,
        EF1=False,
        EF2=False,
        MV1=0,
        MV2=0,
        PERS1=0,
        PERS2=0,
        HT=False,
        FC=False,
        QP=False,
        SR="ceil",
        validate=True,
        arena=None,
        BMF=None,
        NWF=None,
        DF=None,
        pipeF=None,
        diagF=None,
        WMF=None,
    ):
        T, H = x.shape
        k = topk_ids.shape[1]
        assert (
            x.dtype == torch.bfloat16 and x.is_contiguous()
        ), "x must be contiguous bf16"
        assert H == W.H, f"x has H={H}, weights have H={W.H}"
        assert topk_ids.shape == topk_w.shape == (T, k)
        assert SR in ("ceil", "even")
        self.SE = SR == "even"
        if validate:
            # Host sync. Out-of-range ids (e.g. expert-parallel -1) are clamped, not dropped.
            assert bool(
                ((topk_ids >= 0) & (topk_ids < W.E)).all()
            ), "expert ids must be in [0, E)"
        R = T * k
        I = W.I
        dev = x.device
        self.x, self.W = x, W
        self.T, self.H, self.I, self.k, self.R = T, H, I, k, R
        self.cfg1 = {
            "BM": BM1,
            "D": D1,
            "pipe": pipe1,
            "NW": NW1,
            "diag": diag1,
            "WM": WM1,
            "EF": EF1,
            "MV": MV1,
            "PERS": PERS1,
        }
        self.cfg2 = {
            "BM": BM2,
            "D": D2,
            "pipe": pipe2,
            "NW": NW2,
            "diag": diag2,
            "WM": WM2,
            "EF": EF2,
            "MV": MV2,
            "PERS": PERS2,
        }
        # Step-major A scales only pay off for large stage-1 tiles.
        self.AST = BM1 >= 128
        self.HT = bool(HT)
        if arena is not None:
            alloc = arena.take
        else:

            def alloc(shape, dtype):
                return torch.empty(*shape, dtype=dtype, device=dev)

        self.ids = alloc((R,), torch.int32)
        self.w = alloc((R,), torch.float32)
        self.a_q = alloc((T, H // 2), torch.uint8)
        self.a_s = alloc((T, H // 32), torch.uint8)
        self.row_tok = alloc((R,), torch.int32)
        self.row_w = alloc((R,), torch.float32)
        self.inv = alloc((R,), torch.int32)
        self.FC = bool(FC)
        if self.FC:
            over = {
                "BM": BMF,
                "NW": NWF,
                "D": DF,
                "pipe": pipeF,
                "diag": diagF,
                "WM": WMF,
            }
            cf = dict(
                self.cfg2, PERS=0, **{k_: v for k_, v in over.items() if v is not None}
            )
            self.launches2 = [
                ((BM2, -2), self.cfg2, "rows"),
                ((cf["BM"], -3), cf, "fused"),
            ]
        else:
            self.launches2 = [(BM2, self.cfg2, "rows")]
        specs = [BM1]
        for s, _, _ in self.launches2:
            if s not in specs:
                specs.append(s)
        self.launches1 = [(0, self.cfg1)]
        self.launches2 = [(specs.index(s), c, e) for s, c, e in self.launches2]
        self.bms = tuple(specs)
        self.spec_mt = [prologue.spec_max_tiles(b, R, W.E) for b in specs]
        self.MAXT = max(self.spec_mt)
        self.QP = bool(QP)
        self.tiles = alloc((len(specs) * self.MAXT, 4), torch.int32)
        self.ntiles = alloc((len(specs),), torch.int32)
        self.plan_scratch = (
            alloc((W.E + 1,), torch.int32),
            alloc((W.E + 1,), torch.int32),
            alloc(((R + prologue.CHUNK - 1) // prologue.CHUNK * W.E,), torch.int32),
        )
        self.h_q = alloc((R, I // 2), torch.uint8)
        self.h_s = alloc((R, I // 32), torch.uint8)
        # +64 B: a 16 B/lane scale DMA may read up to 3 rows past R.
        self.a_s_t = alloc((R * (H // 32) + 64,), torch.uint8) if self.AST else None
        self.y_rows = alloc((R, H), torch.bfloat16)
        self.out = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
        self.ids.copy_(topk_ids.reshape(-1))
        self.w.copy_(topk_w.reshape(-1))
        self.tiles.zero_()
        self.ntiles.zero_()
        self.plan_scratch[0].zero_()
        self.plan_scratch[1].zero_()

    def _tl(self, b):
        return (
            self.tiles.data_ptr() + b * self.MAXT * 16,
            self.ntiles.data_ptr() + b * 4,
        )

    def prologue(self):
        prologue.run_plan(
            self.ids,
            self.w,
            self.row_tok,
            self.row_w,
            self.inv,
            self.tiles,
            self.ntiles,
            self.W.E,
            self.k,
            self.bms,
            self.MAXT,
            scratch=self.plan_scratch,
            shared_last=self.FC,
            quant=(self.x, self.a_q, self.a_s, self.SE) if self.QP else None,
        )
        if not self.QP:
            prologue.run_quant(self.x, self.a_q, self.a_s, even=self.SE)
        if self.AST:
            prologue.run_scale_t(self.a_s, self.row_tok, self.a_s_t)

    def stage1(self):
        W = self.W
        a_s = (self.a_s_t if self.AST else self.a_s).data_ptr()
        for b, c in self.launches1:
            tp, ntp = self._tl(b)
            args = (
                self.a_q.data_ptr(),
                a_s,
                W.b1.data_ptr(),
                W.bs1.data_ptr(),
                tp,
                ntp,
                self.row_tok.data_ptr(),
                0,
                self.h_q.data_ptr(),
                self.h_s.data_ptr(),
                0,
                self.T,
                self.R,
                self.T,
            )
            diag = "+".join(t for t in (c["diag"], "se" if self.SE else "") if t)
            gemm.run_gemm(
                1,
                self.H,
                2 * self.I,
                c["BM"],
                args,
                self.spec_mt[b],
                D=c["D"],
                pipe=c["pipe"],
                NW=c["NW"],
                diag=diag,
                WM=c["WM"],
                EF=c["EF"],
                MV=c["MV"],
                AST=self.AST,
                HT=self.HT,
                PERS=c["PERS"],
            )

    def stage2(self):
        W = self.W
        for b, c, epi in self.launches2:
            tp, ntp = self._tl(b)
            args = (
                self.h_q.data_ptr(),
                self.h_s.data_ptr(),
                W.b2.data_ptr(),
                W.bs2.data_ptr(),
                tp,
                ntp,
                self.row_tok.data_ptr(),
                self.row_w.data_ptr(),
                self.y_rows.data_ptr(),
                self.out.data_ptr(),
                self.inv.data_ptr(),
                self.R,
                self.R,
                self.T,
            )
            gemm.run_gemm(
                2,
                self.I,
                self.H,
                c["BM"],
                args,
                self.spec_mt[b],
                D=c["D"],
                epi=epi,
                KTOP=self.k,
                pipe=c["pipe"],
                NW=c["NW"],
                diag=c["diag"],
                WM=c["WM"],
                EF=c["EF"],
                MV=c["MV"],
                AST=self.HT,
                HT=self.HT,
                PERS=c["PERS"],
            )

    def combine(self):
        # FC: offs[E] (set by the plan) is 1 when the fused epilogue wrote out.
        skip = self.plan_scratch[1][self.W.E :] if self.FC else None
        combine.run_combine(self.y_rows, self.inv, self.out, self.k, skip=skip)

    def forward(self, x=None, topk_ids=None, topk_w=None):
        """Optional new inputs of the construction shapes: x is re-bound (no copy), routing is
        copied into the run's int32 / fp32 buffers (graph-safe, no host sync)."""
        if x is not None:
            assert (
                x.shape == self.x.shape
                and x.dtype == torch.bfloat16
                and x.is_contiguous()
            )
            self.x = x
        if topk_ids is not None:
            assert topk_ids.shape == (self.T, self.k)
            self.ids.copy_(topk_ids.reshape(-1))
        if topk_w is not None:
            assert topk_w.shape == (self.T, self.k)
            self.w.copy_(topk_w.reshape(-1))
        self.prologue()
        self.stage1()
        self.stage2()
        self.combine()
        return self.out


def _unshuffle_scale(s: torch.Tensor, e: int, n: int, groups: int) -> torch.Tensor:
    """Inverse of ``e8m0_shuffle`` for an [e * n, groups] e8m0 scale."""
    s = s.view(torch.uint8).reshape(-1)
    cols = (groups + 7) // 8 * 8
    rows = s.numel() // cols
    s = s.view(rows // 32, cols // 8, 4, 16, 2, 2).permute(0, 5, 3, 1, 4, 2)
    return s.reshape(rows, cols)[: e * n, :groups].reshape(e, n, groups)


_packed: dict[tuple, tuple] = {}


class _Grow(Exception):
    def __init__(self, nbytes):
        self.nbytes = nbytes


class _Arena:
    """Grow-only device pool for MoERun workspaces. Every batch size carves the
    same block, so serving never re-allocates once the largest batch has run."""

    def __init__(self):
        self.buf = None
        self.off = 0

    def take(self, shape, dtype):
        n = dtype.itemsize
        for d in shape:
            n *= d
        start = (self.off + 255) // 256 * 256
        self.off = start + n
        if self.buf is None or self.buf.numel() < self.off:
            raise _Grow(self.off)
        return self.buf[start : self.off].view(dtype).view(*shape)

    def build(self, make, device):
        """Run make() against the arena, growing it until the workspace fits."""
        while True:
            self.off = 0
            try:
                return make()
            except _Grow as grow:
                self.buf = None
                self.buf = torch.empty(grow.nbytes, dtype=torch.uint8, device=device)


_arena = _Arena()
# One workspace shared by all layers; released before another shape allocates.
_runs: dict[tuple, MoERun] = {}


def _weights(w1, w2, w1_scale, w2_scale) -> MoEWeights:
    """The preshuffled weights in place plus their repacked scales, cached while
    the source tensors live."""
    sources = (w1, w2, w1_scale, w2_scale)
    # _version: in-place updates (weight reload) must invalidate the repacked scales.
    key = tuple((t.data_ptr(), t.shape, t._version) for t in sources)
    hit = _packed.get(key)
    if hit is None or not all(r() is t for r, t in zip(hit[0], sources)):
        stale = [k for k in _packed if [p for p, *_ in k] == [p for p, *_ in key]]
        for k in stale:
            del _packed[k]
        e, n13, kh = w1.shape
        inter, hidden = n13 // 2, kh * 2
        bs1 = layout.pack_scales(1, _unshuffle_scale(w1_scale, e, n13, hidden // 32))
        bs2 = layout.pack_scales(2, _unshuffle_scale(w2_scale, e, hidden, inter // 32))
        hit = (tuple(weakref.ref(t) for t in sources), bs1, bs2)
        _packed[key] = hit
        weakref.finalize(w1, _packed.pop, key, None)
    return MoEWeights(w1, w2, hit[1], hit[2])


def unsupported_reason(request: FusedMoeRequest) -> str | None:
    w1, w2 = request.w1, request.w2
    if get_gfx() != "gfx950":
        return f"gfx {get_gfx()!r}"
    if not (getattr(w1, "is_shuffled", False) and getattr(w2, "is_shuffled", False)):
        return "weights not preshuffled"
    if w1.dtype != dtypes.fp4x2 or w2.dtype != dtypes.fp4x2:
        return f"weight dtype {w1.dtype}"
    if request.q_dtype_a not in (None, dtypes.fp4x2):
        return f"activation dtype {request.q_dtype_a}"
    if request.quant_type != QuantType.per_1x32:
        return f"quant_type {request.quant_type}"
    if request.activation != ActivationType.Swiglu:
        return f"activation {request.activation}"
    if request.hidden_states.dtype != dtypes.bf16:
        return f"hidden dtype {request.hidden_states.dtype}"
    if request.dtype not in (None, dtypes.bf16):
        return f"output dtype {request.dtype}"
    if request.expert_mask is not None or request.num_local_tokens is not None:
        return "expert parallelism"
    if request.bias1 is not None or request.bias2 is not None:
        return "per-expert bias"
    if request.doweight_stage1:
        return "doweight_stage1"
    if request.a1_scale is not None or request.a2_scale is not None:
        return "prequantized activations"
    if request.hidden_pad or request.intermediate_pad:
        return "hidden/intermediate padding"
    if request.swiglu_limit not in (None, _SWIGLU_LIMIT):
        return f"swiglu_limit {request.swiglu_limit}"
    gate_mode = getattr(request.gate_mode, "value", request.gate_mode)
    if gate_mode not in (None, "separated"):
        return f"gate_mode {gate_mode!r}"
    _, n13, kh = w1.shape
    inter_dim, model_dim = n13 // 2, kh * 2
    if inter_dim % 128 != 0 or model_dim % 256 != 0:
        return f"shape inter_dim={inter_dim} model_dim={model_dim}"
    if request.hidden_states.shape[0] * model_dim * 2 >= 2**31:
        return "T * model_dim too large for 32-bit offsets"
    return None


@functools.lru_cache(maxsize=256)
def _run_kwargs(kname1: str, kname2: str) -> tuple:
    return tuple(sorted(a4w4c_kname.parse_knames(kname1, kname2).items()))


def _flat(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Flat routing buffer for the plan kernel; no copy when already in place."""
    return (
        t.reshape(-1)
        if t.dtype == dtype and t.is_contiguous()
        else t.reshape(-1).to(dtype)
    )


def _bind(request, kname1, kname2) -> MoERun:
    reason = unsupported_reason(request)
    if reason is not None:
        raise NotImplementedError(f"FlyDSL A4W4 compact MoE does not support {reason}")
    x = request.hidden_states.contiguous()
    weights = _weights(request.w1, request.w2, request.w1_scale, request.w2_scale)
    kwargs = _run_kwargs(kname1, kname2)
    key = (x.shape, request.topk_ids.shape, x.device, weights.E, weights.I, kwargs)
    run = _runs.get(key)
    if run is None:
        _runs.clear()
        run = _arena.build(
            lambda: MoERun(
                x,
                request.topk_ids,
                request.topk_weight,
                weights,
                SR=_SCALE_RULE,
                validate=False,
                arena=_arena,
                **dict(kwargs),
            ),
            x.device,
        )
        _runs[key] = run
    run.W = weights
    run.x = x
    run.ids = _flat(request.topk_ids, torch.int32)
    run.w = _flat(request.topk_weight, torch.float32)
    run.out = torch.empty_like(x)
    return run


def moe_a4w4_compact(
    request: FusedMoeRequest, kname1: str, kname2: str
) -> torch.Tensor:
    return _bind(request, kname1, kname2).forward()


def run_moe_a4w4_compact(request: FusedMoeRequest, config: str) -> torch.Tensor:
    """Registry entry point; ``config`` is :func:`a4w4c_kname.impl_config`."""
    return moe_a4w4_compact(request, *a4w4c_kname.split_impl_config(config))
