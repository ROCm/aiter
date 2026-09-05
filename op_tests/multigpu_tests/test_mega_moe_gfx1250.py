# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Multi-layer EP MoE end-to-end perf + accuracy on the mori v2 cco/FlyDSL op-layer.

N (default 61, DeepSeek-V4-Pro) MoE layers are chained. The ``base`` mode uses
Mori v2 dispatch -> AITER fused_moe -> Mori v2 combine. ``fused`` calls only
``MegaMoEGfx1250``, which owns AITER's dispatch -> fused_moe -> fused-combine
pipeline. The combined output plus residual feeds the next layer.

Two isolated paths (never touch each other's intermediates; they only share the
config, the bf16 weights and the per-layer routings):

  * ``RefModel``  -- pure-torch fp32 reference (mxfp4-dequant weights, per-token
    routed FFN + residual, chained over N layers). Uses NO mori/cco/fused_moe
    kernel. This is the ground truth (mirrors test_moe_ep.py's torch_moe idea).
  * ``DeviceMoEPipeline`` -- the device path: cco Communicator + EpDispatchCombineOp
    + a8w4 fused_moe. The whole N-layer dispatch->gemm->combine chain is captured
    into a SINGLE CUDA graph; perf is measured with torch.profiler over graph
    replays (not cuda.Event). Contains no fp32-reference logic.

Launcher: torchrun (one process per rank / GPU), mirroring test_moe_layer_ep.py.

Launch (4x gfx1250; every env knob below is already the script's default):
    cd <dir not under /app>   # avoid the /app/triton namespace shadow
    torchrun --standalone --nproc_per_node=4 test_mega_moe_gfx1250.py \
      -q a4w4_mxfp4 -e 384 -k 6 -hd 7168 -id 3072 --layers 61 --combine base
    # Set MORI_CCO_BC to a prebuilt libmori_cco_device.bc to skip CCO JIT.

Env / CLI: --layers --logits_tol --acc_verify --dispatch_wire --combine
           -tpr -hd -id -e -k --shared_E -q
           --data-init --seed --warmup --iters --prof_replays

``--data-init`` / ``--scale-init`` / ``--seed`` are the shared ubench knobs from
``aiter.test_common.add_data_init_args``. ``--scale-init`` is accepted for CLI
compatibility but unused here: every scale is derived by quantizing the
generated weights, never drawn independently.
"""

import argparse
import math
import os
import time

import torch
import torch.distributed as dist
import torch.profiler as tprof

import aiter
from aiter import (
    ActivationType,
    QuantType,
    dtypes,
    get_gfx,
    get_torch_quant,
    pertoken_quant,
)
from aiter.fused_moe import fused_moe
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import moe_shuffle_scale, shuffle_weight
from aiter.test_common import add_data_init_args, fill, make_generator, print_json_table
from aiter.utility import fp4_utils

try:
    from aiter.test_common import get_trace_perf
except Exception:  # noqa: BLE001 # pragma: no cover
    get_trace_perf = None

# gfx1250 grouped mxfp4 kernel knobs, all overridable from the environment.
# AITER_FORCE_A8W4 picks the ACTIVATION dtype of the grouped kernel: 0 -> fp4
# (a4w4), 1 -> fp8 (a8w4). The weights are mxfp4 either way; -q only decides how
# they are laid out, so a4w4 is the default here and `AITER_FORCE_A8W4=1
# -q a8w4_mxfp4` gets the fp8-activation variant back.
os.environ.setdefault("ENABLE_CK", "0")
os.environ.setdefault("AITER_FORCE_A8W4", "0")
os.environ.setdefault("AITER_USE_GROUPED_GEMM", "1")
os.environ.setdefault("AITER_BF16_FP8_MOE_BOUND", "0")
# Both EP paths go through mori's HIP/JIT dispatch: MORI_V2_KERNEL_BACKEND picks
# it for the `base` path's EpDispatchCombineOp, MEGA_DISPATCH for the dispatch
# inside MegaMoEGfx1250. Same dispatch on both sides -> the kernel tables differ
# only in the combine.
os.environ.setdefault("MORI_V2_KERNEL_BACKEND", "hip")
os.environ.setdefault("MEGA_DISPATCH", "mori")

os.environ.setdefault("FLYDSL_GPU_ARCH", get_gfx())

_FP8_DTYPE = dtypes.fp8
QUANT_KEYS = ["No", "per_Token", "per_128x128", "a8w4_mxfp4", "a4w4_mxfp4"]
_MXFP4_KEYS = ("a8w4_mxfp4", "a4w4_mxfp4")
# add_data_init_args' --scale-init default. Kept here so main() can tell whether
# the caller asked for a scale distribution this test cannot honour.
_DEFAULT_SCALE_INIT = "constant"


def _import_mori_comm():
    """Import Mori's communicator (needed by every combine mode)."""
    from mori.cco import Communicator

    return Communicator


def _import_mori_v2():
    """Import Mori's non-fused dispatch/combine v2 path.

    Deferred: the fused mode runs entirely on aiter's own mega_moe kernels and
    only needs the communicator, so importing this eagerly would make the fused
    path fail whenever Mori's copy lags the installed flydsl API.
    """
    from mori.ops.dispatch_combine_v2 import (
        EpDispatchCombineConfig,
        EpDispatchCombineOp,
    )

    return EpDispatchCombineConfig, EpDispatchCombineOp


# Config / quant-path spec
def resolve_spec(quant_key):
    """How to prepare weights / quantize activations / call fused_moe for a quant
    key."""
    is_mxfp4 = quant_key in _MXFP4_KEYS

    if quant_key == "No":
        aiter_qtype = QuantType.No
    elif quant_key == "per_Token":
        aiter_qtype = QuantType.per_Token
    elif quant_key == "per_128x128":
        aiter_qtype = QuantType.per_128x128
    else:  # a8w4_mxfp4 / a4w4_mxfp4
        aiter_qtype = QuantType.per_1x32

    # The gfx1250 grouped MoE GEMM reads GUGU (gate/up row-interleaved) w1 only,
    # so both mxfp4 keys -- a8w4 and a4w4 -- have to ask for INTERLEAVE; a
    # SEPARATED layout silently falls through to the generic 2-stage MoE.
    gate_mode = GateMode.INTERLEAVE if is_mxfp4 else GateMode.SEPARATED

    return {
        "key": quant_key,
        "aiter_qtype": aiter_qtype,
        "gate_mode": gate_mode,
        "activation": ActivationType.Silu,
        "is_mxfp4": is_mxfp4,
    }


# The MegaMoE (--combine fused) dispatch wire.
_DISPATCH_WIRE_FOR_QUANT = {"a8w4_mxfp4": "fp8", "a4w4_mxfp4": "fp4"}


def resolve_dispatch_wire(wire, quant_key):
    """What MegaMoE's dispatch puts on the wire: bf16 | fp8 | fp4.

    A quantizing wire is not a free choice -- the receiver hands the payload to
    the grouped GEMM as its A operand, so it has to be the width that GEMM wants
    (a8w4 -> fp8, a4w4 -> fp4), which is what ``auto`` resolves to. The other
    pairing is a width error, not a slow path, so it is rejected here rather
    than deep inside the gather.
    """
    if wire == "auto":
        return _DISPATCH_WIRE_FOR_QUANT.get(quant_key, "bf16")
    if wire == "bf16":
        return "bf16"
    want = _DISPATCH_WIRE_FOR_QUANT.get(quant_key)
    if want is None:
        raise ValueError(
            f"--dispatch_wire={wire} needs an MX quant key "
            f"({'/'.join(_DISPATCH_WIRE_FOR_QUANT)}), got -q {quant_key}"
        )
    if wire != want:
        raise ValueError(
            f"-q {quant_key} wants a {want} A operand, so --dispatch_wire={wire} "
            "would hand the GEMM the wrong payload width"
        )
    return wire


def resolve_data_init(data_init):
    """``--data-init`` is the shared ubench list form, meant to sweep several
    distributions in one invocation. Here the weights are quantized and the whole
    N-layer chain is captured into a CUDA graph once per process, so a run takes
    exactly one distribution -- sweep by launching the script per distribution."""
    dists = list(data_init) if isinstance(data_init, (list, tuple)) else [data_init]
    if len(dists) != 1:
        raise ValueError(
            f"--data-init takes a single distribution here, got {dists}: the "
            "weight quant and the graph capture are both per-run"
        )
    return dists[0]


# Weight quantization + shuffle (device path) / dequant (reference)
def weight_per_128x128_quant(weight, quant_dtype):
    E, dim1, dim2 = weight.shape
    wb = weight.view(E, dim1 // 128, 128, dim2 // 128, 128)
    wb = wb.permute(0, 1, 3, 2, 4).contiguous().view(E, -1, 128 * 128)
    w_qt, w_s = aiter.pertoken_quant(wb, quant_dtype=quant_dtype)
    w_qt = w_qt.view(E, dim1 // 128, dim2 // 128, 128, 128)
    w_qt = w_qt.permute(0, 1, 3, 2, 4).contiguous().view(E, dim1, dim2)
    return w_qt, w_s.view(E, dim1 // 128, dim2 // 128)


def _mxfp4_quant(w):
    """per_1x32 mxfp4 quant: packed fp4x2 weight [E, d1, d2//2] + e8m0 scale."""
    tq = get_torch_quant(QuantType.per_1x32)
    w_qt, w_scale = tq(w, quant_dtype=dtypes.fp4x2)
    w_qt = w_qt.view(w.shape[0], w.shape[1], w.shape[2] // 2)
    return w_qt, w_scale


def _mxfp4_dequant(w_qt, w_scale, orig_shape):
    """Inverse of _mxfp4_quant to fp32 (matches the kernel's mxfp4_to_f32 x e8m0),
    used by the reference so both sides see the same lossy weights."""
    wf = fp4_utils.mxfp4_to_f32(w_qt).view(*orig_shape)
    sf = fp4_utils.e8m0_to_f32(w_scale).view(orig_shape[0], orig_shape[1], -1)
    sf = sf.unsqueeze(-1).expand(-1, -1, -1, 32).reshape(*orig_shape)
    return (wf * sf).to(torch.float32)


def _gguu_to_gugu_rows(t):
    """`(E, 2*I, ...)` GGUU [g..,u..] -> GUGU [g0,u0,g1,u1,...]."""
    _E, two_inter = t.shape[:2]
    inter = two_inter // 2
    g, u = t[:, :inter], t[:, inter:]
    return torch.stack([g, u], dim=2).flatten(1, 2).contiguous()


def raw_quant_weights(w1, w2, spec):
    """Quantize (unshuffled) a group of routed-expert weights."""
    key = spec["key"]
    if key == "No":
        tq = get_torch_quant(QuantType.No)
        w1_qt, _ = tq(w1, quant_dtype=None)
        w2_qt, _ = tq(w2, quant_dtype=None)
        return w1_qt.view(w1.shape), None, w2_qt.view(w2.shape), None
    if key == "per_Token":
        w1_qt, w1_s = pertoken_quant(w1, quant_dtype=_FP8_DTYPE)
        w2_qt, w2_s = pertoken_quant(w2, quant_dtype=_FP8_DTYPE)
        return w1_qt, w1_s, w2_qt, w2_s
    if key == "per_128x128":
        w1_qt, w1_s = weight_per_128x128_quant(w1, quant_dtype=_FP8_DTYPE)
        w2_qt, w2_s = weight_per_128x128_quant(w2, quant_dtype=_FP8_DTYPE)
        return w1_qt, w1_s, w2_qt, w2_s
    w1_qt, w1_s = _mxfp4_quant(w1)
    w2_qt, w2_s = _mxfp4_quant(w2)
    return w1_qt, w1_s, w2_qt, w2_s


def shuffle_group(w1_qt, w1_s, w2_qt, w2_s, spec, n_experts):
    """Layout-shuffle a group of `n_experts` quantized experts for the kernel.

    Both mxfp4 keys share the grouped gfx1250 layout (GUGU-interleaved w1 + the
    n32k4 e8m0 B-scale). The only difference is the weight DTYPE handed to
    ``fused_moe``: it keys off ``w1.dtype`` to pick the activation dtype, so
    uint8 selects the fp8-activation (a8w4) kernel and fp4x2 the fp4-activation
    (a4w4) one. See ``grouped_moe_gfx1250._grouped_a8w4_tdm_moe``.
    """
    key = spec["key"]
    if key in ("No", "per_Token", "per_128x128"):
        return shuffle_weight(w1_qt), shuffle_weight(w2_qt), w1_s, w2_s
    w1_phys = _gguu_to_gugu_rows(w1_qt.view(torch.uint8))
    w1_a = shuffle_weight(w1_phys, layout=(16, 16))
    w2_a = shuffle_weight(w2_qt.view(torch.uint8), layout=(16, 16))
    w1_ss = moe_shuffle_scale(
        w1_s.contiguous(),
        experts_cnt=n_experts,
        is_guinterleave=True,
        gate_up=True,
    )
    w2_ss = moe_shuffle_scale(w2_s.contiguous(), experts_cnt=n_experts)
    if key == "a4w4_mxfp4":
        w1_a = w1_a.view(dtypes.fp4x2)
        w2_a = w2_a.view(dtypes.fp4x2)
    return w1_a, w2_a, w1_ss, w2_ss


def moe_forward(
    hidden,
    w1_a,
    w2_a,
    w1_s,
    w2_s,
    topk_weights,
    topk_ids,
    expert_mask,
    spec,
    a1_scale=None,
    num_local_tokens=None,
):
    """Single fused_moe call (device path). ``num_local_tokens`` (device int32
    scalar == total_recv) lets the caller feed the FULL, un-truncated dispatch
    buffer: routes past total_recv*topk are dropped in the grouped route kernel,
    so no host .item()/slice/clone is needed and the call stays graph-capturable."""
    if num_local_tokens is None:
        num_local_tokens = torch.tensor(
            [hidden.shape[0]], dtype=dtypes.i32, device=hidden.device
        )
    if spec["is_mxfp4"]:
        return fused_moe(
            hidden,
            w1_a,
            w2_a,
            topk_weights,
            topk_ids,
            expert_mask=expert_mask,
            activation=spec["activation"],
            gate_mode=spec["gate_mode"].value,
            quant_type=spec["aiter_qtype"],
            w1_scale=w1_s,
            w2_scale=w2_s,
            dtype=dtypes.bf16,
            num_local_tokens=num_local_tokens,
        )
    return fused_moe(
        hidden,
        w1_a,
        w2_a,
        topk_weights,
        topk_ids,
        expert_mask,
        num_local_tokens=num_local_tokens,
        w1_scale=w1_s,
        w2_scale=w2_s,
        quant_type=spec["aiter_qtype"],
        a1_scale=a1_scale,
        dtype=dtypes.bf16,
    )


# Shared setup (fed to BOTH reference and device path)
_WEIGHT_SEED = 70000  # identical on every rank so the global expert set agrees
_WEIGHT_AMPL = 0.1  # weight amplitude, was the literal `/ 10` below


def make_shared_weights(
    E, hdim, idim, dtype, dev, shared_E=0, seed=_WEIGHT_SEED, data_dist="norm"
):
    """One weight set reused by every layer. Same seed on all ranks so the global
    expert partition is consistent. Returns bf16 (w1[E,2I,H], w2[E,H,I], sw1, sw2).

    ``data_dist`` is a ``--data-init`` distribution. Every mode is scaled down by
    _WEIGHT_AMPL: at unit amplitude the narrow fp4/fp8 activation quant saturates
    and the N-layer residual chain diverges to NaN after a few layers."""
    gen = make_generator(seed, device=dev)

    def _w(experts, rows, cols):
        # fill() only row-chunks its fp32 staging for 2-D shapes, so ask for the
        # flat [experts*rows, cols] and view it back: at E=384 that caps the
        # staging near 1 GiB instead of materializing the set in fp32.
        w = fill((experts * rows, cols), data_dist, gen, dtype=dtype, device=dev)
        return w.mul_(_WEIGHT_AMPL).view(experts, rows, cols)

    w1 = _w(E, 2 * idim, hdim)
    w2 = _w(E, hdim, idim)
    sw1 = sw2 = None
    if shared_E > 0:
        sw1 = _w(shared_E, 2 * idim, hdim)
        sw2 = _w(shared_E, hdim, idim)
    return w1, w2, sw1, sw2


def make_routings(n_layers, ct, E, topk, dev, seed):
    """Per-layer random routing, RETAINED so device + reference replay the same.
    topk_ids are distinct experts per token (top-k over a random score); weights
    are random and renormalized. Returns list[(ids[ct,topk] i32, wts[ct,topk] f32)]."""
    routings = []
    for layer_idx in range(n_layers):
        gen = torch.Generator(device=dev).manual_seed(seed + layer_idx)
        score = torch.rand(ct, E, generator=gen, device=dev, dtype=torch.float32)
        _, ids = score.topk(topk, dim=-1)  # distinct experts per token
        wts = torch.rand(ct, topk, generator=gen, device=dev, dtype=torch.float32)
        wts = wts / wts.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        routings.append((ids.to(dtypes.i32), wts))
    return routings


def _rmsnorm(x, eps=1e-6):
    """RMSNorm (no learnable gain) on the last dim. Applied to each layer's MoE
    input so activations stay unit-scale across the 61-layer residual chain --
    without it the narrow activation quant (fp4/fp8) saturates and the chain
    diverges to NaN after a few layers. Both device and reference use the SAME
    normalization."""
    xf = x.float()
    n = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return n.to(x.dtype)


def _calc_diff(x, y):
    """1 - cosine similarity (fp64), mirrors test_moe_ep.py::_calc_diff."""
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    if denom == 0:
        return 0.0
    return float(1 - 2 * (x * y).sum() / denom)


# Accuracy budget, measured on gfx1250 (2 ranks, 1024 tok/rank, 7168x3072, E=384,
# topk=6, --combine fused -- the worst of the quant x combine scenarios).
#
# _calc_diff is ||x-y||^2 / (||x||^2 + ||y||^2), a SQUARED error, and the per-layer
# errors accumulate as a random walk: r ~ sqrt(L) makes r^2 ~ L, so the metric grows
# about linearly in the layer count, then saturates as it approaches the bound.
# Measured (mxfp8 wire on; it costs a flat +32% on a8w4 and +1.6% on a4w4):
#
#             L=1       L=2       L=4       L=8
#   a4w4   0.021877  0.042683  0.080897  0.144881
#   a8w4   0.001433  0.002871  0.005742  0.011369
#
# slope * L / (1 + sat * L) reproduces both rows within 1%, so scaling that curve
# keeps the SAME headroom at every layer count. A flat tol cannot: 0.1 rejects a
# healthy 8-layer a4w4 run (0.145) yet passes anything at all on a8w4.
_ACC_TOL = {  # quant key -> (per-layer slope, saturation)
    "a4w4_mxfp4": (0.0225, 0.031),
    "a8w4_mxfp4": (0.00143, 0.0012),
}
_ACC_TOL_FALLBACK = _ACC_TOL["a4w4_mxfp4"]  # unknown key: assume the fp4 budget
_ACC_TOL_SAFETY = 1.5


def default_logits_tol(quant_key, n_layers):
    # Per-quant tol for an n_layers chain; see _ACC_TOL for the calibration.
    slope, sat = _ACC_TOL.get(quant_key, _ACC_TOL_FALLBACK)
    return _ACC_TOL_SAFETY * slope * n_layers / (1.0 + sat * n_layers)


# torchrun rendezvous helper
class Dist:
    def __init__(self):
        self.rank = int(os.environ["RANK"])
        self.world = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo")
        torch.cuda.set_device(self.local_rank)

    def bcast_uid(self, uid):
        objs = [uid if self.rank == 0 else None]
        dist.broadcast_object_list(objs, src=0)
        return objs[0]

    def allreduce_sum(self, value):
        t = torch.tensor([value], dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return int(t.item())

    def allreduce_avg_float(self, value):
        """Average a scalar float across all ranks (collective; call on every rank)."""
        t = torch.tensor([float(value)], dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t.item()) / self.world

    def gather_objects(self, obj):
        """All-gather a python object from every rank (collective). Returns a list
        indexed by rank."""
        out = [None] * self.world
        dist.all_gather_object(out, obj)
        return out

    def shutdown(self):
        if dist.is_initialized():
            dist.destroy_process_group()


# Reference: pure-torch fp32 multi-layer chained MoE (ground truth, ISOLATED)
class RefModel:
    """fp32 reference. mxfp4-dequant weights (shared, lazily per expert), per-token
    routed FFN summed over topk, dense shared expert, chained N layers with a
    residual. Uses only torch + fp4_utils -- NO mori/cco/fused_moe. Runs in fp32
    on `dev`; for tractable memory/time use a modest token count for --check."""

    def __init__(self, w1_bf, w2_bf, sw1, sw2, spec, dev):
        self.w1_bf, self.w2_bf = w1_bf, w2_bf
        self.sw1, self.sw2 = sw1, sw2
        self.spec = spec
        self.dev = dev
        self._cache = {}

    def _expert(self, g):
        wd = self._cache.get(g)
        if wd is None:
            w1_g = self.w1_bf[g : g + 1]
            w2_g = self.w2_bf[g : g + 1]
            if self.spec["is_mxfp4"]:
                w1_qt, w1_s = _mxfp4_quant(w1_g)
                w2_qt, w2_s = _mxfp4_quant(w2_g)
                w1d = _mxfp4_dequant(w1_qt, w1_s, (1, *w1_g.shape[1:]))[0]
                w2d = _mxfp4_dequant(w2_qt, w2_s, (1, *w2_g.shape[1:]))[0]
            else:
                # No / fp8 paths: use the bf16 weights directly (approximate ref).
                w1d = w1_g[0].float()
                w2d = w2_g[0].float()
            wd = self._cache[g] = (w1d, w2d)
        return wd

    @staticmethod
    def _ffn(x, w1d, w2d):
        gate, up = (x @ w1d.t()).chunk(2, dim=-1)
        return (torch.nn.functional.silu(gate) * up) @ w2d.t()

    def _shared(self, x):
        if self.sw1 is None:
            return torch.zeros_like(x)
        acc = torch.zeros_like(x)
        for e in range(self.sw1.shape[0]):
            acc = acc + self._ffn(x, self.sw1[e].float(), self.sw2[e].float())
        return acc

    def layer(self, x, ids, wts):
        """x [ct,H] fp32; ids/wts [ct,topk]. RMSNorm the input, then routed+shared
        FFN. Returns the block output [ct,H] fp32 (caller adds the residual)."""
        xn = _rmsnorm(x)
        out = torch.zeros_like(xn)
        ids_l = ids.long()
        for g in torch.unique(ids_l).tolist():
            sel = ids_l == g
            rows = sel.any(dim=1)
            w = (wts * sel).sum(dim=1)
            w1d, w2d = self._expert(int(g))
            out[rows] += w[rows, None] * self._ffn(xn[rows], w1d, w2d)
        return out + self._shared(xn)

    def run(self, x0, routings):
        """Chain N layers with residual: x = x + layer(x). Returns bf16 [ct,H]."""
        x = x0.float()
        for ids, wts in routings:
            x = x + self.layer(x, ids, wts)
        return x.to(dtypes.bf16)


# Device pipeline: N-layer dispatch->gemm->combine, one CUDA graph (ISOLATED)
class DeviceMoEPipeline:
    """Owns the cco Communicator + EpDispatchCombineOp + a8w4 shuffled weights.
    Each layer recomputes its own routing inside dispatch (e2e-faithful), and the
    whole N-layer chain is captured into ONE CUDA graph and timed with
    torch.profiler. No fp32-reference logic here."""

    def __init__(
        self,
        dist_ctx,
        E,
        hdim,
        idim,
        topk,
        spec,
        n_layers,
        w1_bf,
        w2_bf,
        sw1,
        sw2,
        routings,
        ct,
        combine_mode="base",
    ):
        self.dist_ctx = dist_ctx
        self.E, self.hdim, self.idim, self.topk = E, hdim, idim, topk
        self.spec = spec
        self.n_layers = n_layers
        self.w1_bf, self.w2_bf = w1_bf, w2_bf
        self.sw1, self.sw2 = sw1, sw2
        self.routings = routings
        self.ct = ct
        self.combine_mode = combine_mode
        self.EPR = E // dist_ctx.world
        self.dev = torch.device("cuda", dist_ctx.local_rank)
        self.comm = None
        self.op = None
        self.mega = None
        self.graph = None
        self.x0_static = None
        self.out_static = None

    # ---- initialization (grouped together) ---- #
    def setup(self, x0):
        Communicator = _import_mori_comm()
        # torch.cuda.set_device sets the process HIP current device (== driver
        # hipSetDevice) that cco keys off; Dist already set it, repeat for safety.
        torch.cuda.set_device(self.dist_ctx.local_rank)
        dev, r = self.dev, self.dist_ctx.rank

        # this rank's LOCAL expert weights (quant + layout shuffle), a8w4.
        w1_g = self.w1_bf[r * self.EPR : (r + 1) * self.EPR].contiguous()
        w2_g = self.w2_bf[r * self.EPR : (r + 1) * self.EPR].contiguous()
        q1, gs1, q2, gs2 = raw_quant_weights(w1_g, w2_g, self.spec)
        self.w1_a, self.w2_a, self.w1_s, self.w2_s = shuffle_group(
            q1, gs1, q2, gs2, self.spec, self.EPR
        )
        self.expert_mask = torch.zeros((self.E,), dtype=dtypes.i32, device=dev)
        self.expert_mask[self.EPR * r : self.EPR * (r + 1)] = 1

        self.transport_dtype = torch.bfloat16  # bf16 transport (mxfp4 path)

        # cco rendezvous + op (ONE op, reused by every layer; config is per-layer
        # identical). max_num_inp_token_per_rank = ct.
        uid = Communicator.get_unique_id() if r == 0 else None
        uid = self.dist_ctx.bcast_uid(uid)
        self.comm = Communicator.init(
            self.dist_ctx.world, r, uid, per_rank_vmm=16 * 1024**3
        )
        if self.combine_mode == "fused":
            if not self.spec["is_mxfp4"]:
                raise NotImplementedError(
                    "the fused combine is available only for the mxfp4 quant keys"
                )
            from aiter.ops.flydsl.kernels.mega_moe_gfx1250 import MegaMoEGfx1250

            # Geometry + the expert-GEMM recipe are per-model, so they are fixed
            # here; the weights are per-layer and go to each forward() call.
            self.mega = MegaMoEGfx1250(
                communicator=self.comm,
                rank=r,
                world_size=self.dist_ctx.world,
                model_dim=self.hdim,
                inter_dim=self.idim,
                experts=self.E,
                topk=self.topk,
                max_tokens_per_rank=self.ct,
                activation=self.spec["activation"],
                gate_mode=self.spec["gate_mode"].value,
                quant_type=self.spec["aiter_qtype"],
                # Explicit so a stale $MEGA_DISPATCH_WIRE cannot change what is measured.
                dispatch_wire=self.spec["dispatch_wire"],
            )
        else:
            EpDispatchCombineConfig, EpDispatchCombineOp = _import_mori_v2()
            cfg = EpDispatchCombineConfig(
                rank=r,
                world_size=self.dist_ctx.world,
                hidden_dim=self.hdim,
                max_num_inp_token_per_rank=self.ct,
                num_experts_per_rank=self.EPR,
                num_experts_per_token=self.topk,
                data_type=self.transport_dtype,
                combine_mode="gather",  # mori's name for the `base` combine
            )
            self.op = EpDispatchCombineOp(cfg, self.comm)
        self.comm.barrier()

    # ---- one graph-capturable layer + full chain (calls grouped together) ---- #
    def _layer_step(self, x, layer_idx):
        ids, wts = self.routings[layer_idx]
        xn = _rmsnorm(x)  # keep the quantized activations in range across 61 layers
        if self.mega is not None:
            y = self.mega(
                xn,
                wts,
                ids,
                w1=self.w1_a,
                w2=self.w2_a,
                w1_scale=self.w1_s,
                w2_scale=self.w2_s,
            )
            if self.sw1 is not None:
                y = y + _device_shared_ffn(xn, self.sw1, self.sw2)
            return x + y

        # Recompute routing every layer (mode A: atomic routing inside dispatch)
        # instead of replaying a precomputed handle. return_routing=True hands
        # back this layer's forward dest-slot map, which combine then consumes.
        recv_x, recv_w, _rs, recv_idx, total_recv_t, handle = self.op.dispatch(
            xn, wts, None, ids, return_routing=True
        )
        out = moe_forward(
            recv_x,
            self.w1_a,
            self.w2_a,
            self.w1_s,
            self.w2_s,
            recv_w,
            recv_idx.to(dtypes.i32),
            self.expert_mask,
            self.spec,
            num_local_tokens=total_recv_t,
        )
        combine_out, _ = self.op.combine(out.to(self.transport_dtype), routing=handle)
        y = combine_out[: self.ct].to(dtypes.bf16)
        if self.sw1 is not None:
            y = y + _device_shared_ffn(xn, self.sw1, self.sw2)
        return x + y  # residual

    def _pipeline(self, x0):
        x = x0
        for layer_idx in range(self.n_layers):
            x = self._layer_step(x, layer_idx)
        return x

    # ---- CUDA graph capture (all N layers in ONE graph) ---- #
    # Eager passes before the capture. NOT a measurement knob, hence not on the
    # CLI: they prime the fused_moe lru_cache, the JIT and the allocator, and
    # without them that work would land inside the capture and fail it. 3 is what
    # torch's own CUDA-graph guidance warms up with.
    _CAPTURE_WARMUP = 3

    def capture(self, x0):
        self.x0_static = x0.clone()
        # warmup on a side stream: primes fused_moe lru_cache + allocator.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self._CAPTURE_WARMUP):
                self._pipeline(self.x0_static)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.comm.barrier()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out_static = self._pipeline(self.x0_static)
        torch.cuda.synchronize()
        self.comm.barrier()

    # ---- perf: torch.profiler breakdown + graph-replay wall-clock ---- #
    def bench(self, warmup=5, iters=10, prof_replays=3):
        """Time the ONE-graph N-layer dispatch->gemm->combine chain. The graph
        already contains all N layers, so ONE replay is one full chain; `iters` of
        them are timed individually after `warmup` untimed ones.
        Returns (stats, prof_us) with stats = {min, median, mean, max} in us.

        - a sample is the host wall-clock of one graph replay (one sync after; not
          cuda.Event). For a GPU-bound MoE chain this ~= GPU time.
        - min is the closest to an undisturbed replay, median to the steady state
          of a GPU held at full load; the two drifting apart is what a straggler
          rank or a throttled clock looks like, hence both are reported.
        - torch.profiler over `prof_replays` replays for the per-op breakdown.

        NOTE: this ROCm torch build reports self_device_time_total == 0 for every
        event (verified even for a plain matmul), so torch.profiler cannot give a
        device-time number here; it is kept for the per-op (CPU-side) breakdown.
        If a future build populates device time, prof_us below becomes > 0."""
        import time

        assert iters >= 1, f"--iters must time at least one replay, got {iters}"

        for _ in range(warmup):
            self.graph.replay()
        torch.cuda.synchronize()
        self.comm.barrier()

        # each full N-layer graph replay is one measurement.
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            self.graph.replay()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1e6)
        self.comm.barrier()

        samples.sort()
        stats = {
            "min": samples[0],
            "median": samples[len(samples) // 2],
            "mean": sum(samples) / len(samples),
            "max": samples[-1],
        }

        # torch.profiler breakdown over CUDA-graph replays: roctracer/kineto does
        # surface the per-kernel timeline inside the graph on this build, so we
        # profile the actual graph (matches the measured per-layer wall) instead of
        # a separate eager pass.
        with tprof.profile(
            activities=[tprof.ProfilerActivity.CPU, tprof.ProfilerActivity.CUDA]
        ) as prof:
            for _ in range(prof_replays):
                self.graph.replay()
            torch.cuda.synchronize()
        self.comm.barrier()
        self._prof = prof
        prof_us = sum(_event_device_us(e) for e in prof.key_averages())
        return stats, prof_us

    def final_output(self):
        self.graph.replay()
        torch.cuda.synchronize()
        return self.out_static.detach().clone()

    def teardown(self):
        self.graph = None
        if self.mega is not None:
            self.mega.close()
        if self.comm is not None:
            self.comm.destroy()


def _event_device_us(e):
    """GPU-side self time (us) of a profiler key_averages event, across torch
    versions (self_device_time_total on newer, self_cuda_time_total on older)."""
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(e, attr, None)
        if v:
            return float(v)
    return 0.0


def _run_distributed_smi_replay(pipe, dist_ctx, median_us, n_layers):
    """Replay the Mega graph while every rank monitors its local GPU."""
    if os.environ.get("AITER_SMI_MONITOR", "0") != "1":
        return

    from op_tests.smi_monitor import GpuMonitor, emit_smi_result

    interval_s = float(os.environ.get("AITER_SMI_INTERVAL", "0.05"))
    duration_s = float(os.environ.get("AITER_SMI_DURATION", "1.0"))
    if interval_s <= 0 or duration_s <= 0 or median_us <= 0:
        raise ValueError(
            "Mega MoE SMI interval, duration and measured median must be positive"
        )
    replay_count = max(1, math.ceil(duration_s * 1e6 / median_us))

    monitor = None
    monitor_error = None
    try:
        monitor = GpuMonitor(
            device_index=torch.cuda.current_device(), interval_s=interval_s
        )
        monitor.start()
    except Exception as error:  # noqa: BLE001 - propagate to every rank
        monitor_error = f"rank {dist_ctx.rank}: {type(error).__name__}: {error}"

    monitor_errors = dist_ctx.gather_objects(monitor_error)
    failed = [error for error in monitor_errors if error is not None]
    if failed:
        if monitor is not None:
            monitor.stop()
        raise RuntimeError("Mega MoE SMI monitor failed to start: " + "; ".join(failed))

    # Gloo barriers align CPU submission without adding a GPU collective to the
    # measured Mega graph window. Every rank executes exactly the same replay
    # count; a local duration loop would diverge and deadlock the collectives.
    dist.barrier()
    window_start = time.perf_counter()
    for _ in range(replay_count):
        pipe.graph.replay()
        torch.cuda.synchronize()
    window_end = time.perf_counter()
    monitor.stop()
    dist.barrier()

    samples = [
        sample
        for sample in monitor.samples
        if window_start <= sample["timestamp_s"] <= window_end
    ]
    expected_samples = max(1, int(duration_s / interval_s))
    base_label = os.environ.get("AITER_SMI_LABEL", "mega_moe")
    local_result = {
        "label": f"{base_label}/mega_graph_{n_layers}_layers",
        "device": dist_ctx.local_rank,
        "rank": dist_ctx.rank,
        "interval_s": interval_s,
        "duration_s": window_end - window_start,
        "launches": replay_count,
        "samples": len(samples),
        "sample_status": (
            "ok" if len(samples) >= max(2, expected_samples // 2) else "insufficient"
        ),
        "metrics": monitor.summary(start_s=window_start, end_s=window_end),
    }
    results = dist_ctx.gather_objects(local_result)
    if dist_ctx.rank == 0:
        for result in results:
            emit_smi_result(result)


def _aggregate_prof_table(prof, dist_ctx, per_layer_denom=1.0, row_limit=200):
    """Collect the torch.profiler per-kernel table ACROSS ranks (collective; call
    on every rank). Each rank contributes {name: (self_device_us_total, count)};
    rank 0 returns rows with each kernel's per-call self device time with ONE
    COLUMN PER RANK plus the cross-rank mean, so a straggler (a throttled GPU, an
    unbalanced expert distribution) shows up as a row that disagrees across
    columns instead of being averaged away. `-` means the kernel never ran there.

    Rows are ordered by total self device time (mean over ranks), so the kernels
    that dominate the budget come first regardless of their per-call cost. Also
    prints the TOTAL over ALL kernels and the implied device time per layer
    (total / per_layer_denom, where per_layer_denom = replays * n_layers) so it
    can be compared against the measured per_layer wall -- if they match, the GPU
    has no idle bubble. Non-zero ranks return None."""
    local = {}
    for e in prof.key_averages():
        local[e.key] = (_event_device_us(e), int(e.count))
    per_rank = dist_ctx.gather_objects(local)
    if dist_ctx.rank != 0:
        return None
    world = len(per_rank)
    rows = []
    total_self = 0.0
    for name in {n for d in per_rank for n in d}:
        present = [d[name] for d in per_rank if name in d]
        avg_self = sum(s for s, _ in present) / len(present)
        avg_count = sum(c for _, c in present) / len(present)
        total_self += avg_self
        per_call = [
            (d[name][0] / d[name][1] if d.get(name) and d[name][1] else None)
            for d in per_rank
        ]
        seen = [v for v in per_call if v is not None]
        pc_avg = sum(seen) / len(seen) if seen else 0.0
        rows.append((avg_self, name, per_call, pc_avg, avg_count))
    rows.sort(key=lambda r: (-r[0], r[1]))
    dev_per_layer = total_self / per_layer_denom if per_layer_denom else 0.0
    table_rows = []
    for _avg_self, name, per_call, pc_avg, avg_count in rows[:row_limit]:
        row = {
            "kernel": name,
            "avg_us": pc_avg,
            "calls": avg_count,
        }
        row.update({f"rank{rank}_us": value for rank, value in enumerate(per_call)})
        table_rows.append(row)
    return {
        "rows": table_rows,
        "summary": [{
            "world_size": world,
            "profiled_kernels": len(rows),
            "total_self_device_us": total_self,
            "device_us_per_layer": dev_per_layer,
        }],
    }


def _device_shared_ffn(tokens, sw1, sw2):
    """Dense shared-expert FFN (SwiGLU), graph-capturable (all on-device)."""
    x = tokens.float()
    acc = torch.zeros(
        tokens.shape[0], sw2.shape[1], device=tokens.device, dtype=torch.float32
    )
    for e in range(sw1.shape[0]):
        gate, up = (x @ sw1[e].float().t()).chunk(2, dim=-1)
        acc = acc + (torch.nn.functional.silu(gate) * up) @ sw2[e].float().t()
    return acc.to(tokens.dtype)


# Driver
def main():
    args = _parse_args()
    dist_ctx = Dist()
    dev = torch.device("cuda", dist_ctx.local_rank)
    # Set, not setdefault: the wire has to match this, so a stale environment
    # value would otherwise make -q a4w4_mxfp4 silently measure a8w4.
    os.environ["AITER_FORCE_A8W4"] = "0" if args.quant_type == "a4w4_mxfp4" else "1"
    spec = resolve_spec(args.quant_type)
    spec["dispatch_wire"] = resolve_dispatch_wire(args.dispatch_wire, args.quant_type)

    if spec["is_mxfp4"] and get_gfx() not in ("gfx950", "gfx1250"):
        if dist_ctx.rank == 0:
            print(
                f"skip {args.quant_type}: mxfp4 requires gfx950/gfx1250, got {get_gfx()}"
            )
        dist_ctx.shutdown()
        return

    E, hdim, idim, topk = args.expert, args.hidden, args.inter, args.topk
    ct, n_layers = args.token_per_rank, args.layers
    assert (
        E % dist_ctx.world == 0
    ), f"E={E} must be divisible by world_size={dist_ctx.world}"

    data_dist = resolve_data_init(args.data_init)

    if dist_ctx.rank == 0:
        print(
            f"[cfg] world={dist_ctx.world} layers={n_layers} tokens/rank={ct} hidden={hdim} "
            f"inter={idim} E={E} topk={topk} EPR={E // dist_ctx.world} quant={args.quant_type} "
            f"combine={args.combine} dispatch_wire={spec['dispatch_wire']} "
            f"force_a8w4={os.environ['AITER_FORCE_A8W4']} "
            f"gate={spec['gate_mode'].name} shared_E={args.shared_experts} "
            f"data_init={data_dist} seed={args.seed} gfx={get_gfx()}",
            flush=True,
        )
        if list(args.scale_init) != [_DEFAULT_SCALE_INIT]:
            print(
                f"# note: --scale-init {' '.join(args.scale_init)} is ignored -- "
                "every scale here comes from quantizing the generated weights",
                flush=True,
            )

    # ---- shared inputs: weights (same on all ranks) + this rank's tokens/routing.
    # args.seed shifts all RNG; weights stay rank-independent (identical global
    # experts), tokens/routing vary per rank. Default keeps runs reproducible.
    # --data-init picks how the weights and the layer-0 tokens are filled; the
    # routing stays random in every mode, since a zero/constant routing would
    # collapse every token onto expert 0 and stop measuring dispatch/combine.
    w1_bf, w2_bf, sw1, sw2 = make_shared_weights(
        E,
        hdim,
        idim,
        dtypes.bf16,
        dev,
        shared_E=args.shared_experts,
        seed=_WEIGHT_SEED + args.seed,
        data_dist=data_dist,
    )
    x0 = fill(
        (ct, hdim),
        data_dist,
        make_generator(1000 + dist_ctx.rank + args.seed, device=dev),
        dtype=dtypes.bf16,
        device=dev,
    )
    routings = make_routings(
        n_layers, ct, E, topk, dev, seed=4242 + 100 * dist_ctx.rank + args.seed
    )

    # ---- device path (isolated): setup -> capture 61 layers in one graph -> bench.
    pipe = DeviceMoEPipeline(
        dist_ctx,
        E,
        hdim,
        idim,
        topk,
        spec,
        n_layers,
        w1_bf,
        w2_bf,
        sw1,
        sw2,
        routings,
        ct,
        combine_mode=args.combine,
    )
    pipe.setup(x0)
    pipe.capture(x0)
    stats, prof_us = pipe.bench(
        warmup=args.warmup, iters=args.iters, prof_replays=args.prof_replays
    )
    # Aggregate perf across ranks (collective calls -> run on every rank, and the
    # dict is built in the same order everywhere so the allreduces stay in step).
    stats = {k: dist_ctx.allreduce_avg_float(v) for k, v in stats.items()}
    per_layer_us = stats["median"] / n_layers
    prof_us = dist_ctx.allreduce_avg_float(prof_us)
    _run_distributed_smi_replay(pipe, dist_ctx, stats["median"], n_layers)
    tbl = None
    if args.profile_table:
        tbl = _aggregate_prof_table(
            pipe._prof,
            dist_ctx,
            per_layer_denom=args.prof_replays * n_layers,
        )
        # Save a chrome/perfetto timeline per rank so the actual kernel timeline
        # (and any gaps) can be inspected directly. Opt-in (--save_trace): the
        # export can stall multi-rank graph-profile runs, so it is off by default.
        if args.save_trace:
            _trace_path = f"/tmp/mega_trace_{args.combine}_rank{dist_ctx.rank}.json"
            try:
                pipe._prof.export_chrome_trace(_trace_path)
                if dist_ctx.rank == 0:
                    print(
                        f"# trace saved: /tmp/mega_trace_{args.combine}_rank*.json",
                        flush=True,
                    )
            except Exception as _e:  # noqa: BLE001
                if dist_ctx.rank == 0:
                    print(f"# trace export failed: {_e}", flush=True)
    if dist_ctx.rank == 0:
        print_json_table(
            "mega_moe summary",
            [{
                "quant_type": args.quant_type,
                "combine": args.combine,
                "data_init": data_dist,
                "seed": args.seed,
                "world_size": dist_ctx.world,
                "tokens_per_rank": ct,
                "experts": E,
                "topk": topk,
                "hidden": hdim,
                "intermediate": idim,
                "layers": n_layers,
                "warmup": args.warmup,
                "iters": args.iters,
                "min_us": stats["min"],
                "mean_us": stats["mean"],
                "median_us": stats["median"],
                "max_us": stats["max"],
                "per_layer_us": per_layer_us,
                "prof_device_us": prof_us if prof_us > 0 else None,
            }],
        )
        if tbl is not None:
            print_json_table("mega_moe kernel profile", tbl["rows"])
            print_json_table("mega_moe kernel profile summary", tbl["summary"])

    # ---- accuracy (isolated CPU/fp32 reference): end-to-end accumulated compare.
    accuracy_failure = None
    if args.acc_verify:
        auto_tol = args.logits_tol is None
        tol = (
            default_logits_tol(args.quant_type, n_layers)
            if auto_tol
            else args.logits_tol
        )
        tol_desc = f"{tol:.6f}{' auto' if auto_tol else ''}"
        out_dev = pipe.final_output().float()
        ref = RefModel(w1_bf, w2_bf, sw1, sw2, spec, dev)
        ref_out = ref.run(x0, routings).float()
        logits_diff = _calc_diff(ref_out, out_dev)
        errs = dist_ctx.allreduce_sum(0 if logits_diff < tol else 1)
        avg_diff = dist_ctx.allreduce_avg_float(logits_diff)
        if dist_ctx.rank == 0:
            print(
                f"# MEGA-CHECK layers={n_layers}: {'PASS' if errs == 0 else 'FAIL'} "
                f"(avg logits_diff={avg_diff:.6f} over {dist_ctx.world} ranks, "
                f"tol={tol_desc})",
                flush=True,
            )
        if errs != 0:
            accuracy_failure = (
                f"MegaMoE accuracy check failed on {errs}/{dist_ctx.world} ranks: "
                f"average logits_diff={avg_diff:.6f}, tolerance={tol_desc}"
            )

    pipe.teardown()
    dist_ctx.shutdown()
    if accuracy_failure is not None:
        raise AssertionError(accuracy_failure)


def _parse_args():
    # Imported here, not at module scope: pulling in the mega package before
    # FLYDSL_GPU_ARCH is set below would hand flydsl the wrong arch.
    from aiter.ops.flydsl.kernels.mega_moe_gfx1250 import read_dispatch_wire_env

    p = argparse.ArgumentParser(description="multi-layer EP MoE perf + accuracy")
    p.add_argument(
        "-q",
        "--quant_type",
        type=str,
        choices=QUANT_KEYS,
        default="a4w4_mxfp4",
        help="quantization type",
    )
    p.add_argument(
        "-tpr", "--token_per_rank", type=int, default=128, help="tokens per rank"
    )
    p.add_argument("-hd", "--hidden", type=int, default=7168, help="model/hidden dim")
    p.add_argument("-id", "--inter", type=int, default=3072, help="intermediate dim")
    p.add_argument(
        "-e", "--expert", type=int, default=384, help="routed experts (global)"
    )
    p.add_argument("-k", "--topk", type=int, default=6, help="top-k")
    p.add_argument("--shared_experts", type=int, default=0, help="dense shared experts")
    p.add_argument("--layers", type=int, default=61, help="number of MoE layers")
    # The shared ubench data-init knobs: --data-init, --scale-init and --seed.
    # default_dist=norm reproduces the historical N(0,1)*0.1 weights, which is
    # what the _ACC_TOL budget was calibrated against.
    add_data_init_args(p, default_dist="norm", default_scale=_DEFAULT_SCALE_INIT)
    p.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="untimed graph replays before the timed ones",
    )
    p.add_argument(
        "--iters",
        type=int,
        default=1,
        help="timed graph replays; each one is a full --layers chain, reported as "
        "min/median/mean/max",
    )
    p.add_argument(
        "--prof_replays",
        type=int,
        default=3,
        help="graph replays profiled for the --profile_table breakdown",
    )
    p.add_argument(
        "--logits_tol",
        type=float,
        default=None,
        help="end-to-end accuracy tol; default: the per-quant budget for --layers "
        "(see _ACC_TOL)",
    )
    p.add_argument(
        "--acc_verify", type=int, default=1, help="run fp32 reference accuracy check"
    )
    p.add_argument(
        "--profile_table", type=int, default=0, help="print per-kernel table"
    )
    p.add_argument(
        "--save_trace",
        type=int,
        default=0,
        help="export a chrome/perfetto timeline per rank to /tmp (opt-in; "
        "can stall multi-rank graph-profile runs)",
    )
    p.add_argument(
        "--dispatch_wire",
        type=str,
        choices=["auto", "bf16", "fp8", "fp4"],
        default=read_dispatch_wire_env(),
        help="what dispatch puts on the wire (--combine fused only): bf16 sends "
        "activations and the receiver quantizes each copy; fp8/fp4 quantize once "
        "on the sender and forward the e8m0 row. 'auto' picks what the quant "
        "key's GEMM wants.",
    )
    p.add_argument(
        "--combine",
        type=str,
        choices=["base", "fused"],
        default=os.environ.get("COMBINE", "base"),
        help="EP combine mode: base (mori v2 dispatch/combine around fused_moe) "
        "| fused (gemm2-fused P2P scatter; mxfp4 only). Falls back to $COMBINE.",
    )
    return p.parse_args()


if __name__ == "__main__":
    main()
