# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Multi-layer EP MoE end-to-end perf + accuracy on the mori v2 cco/FlyDSL op-layer.

N (default 61, DeepSeek-V4-Pro) MoE layers are chained: each layer runs
mori ``dispatch`` (cross-rank all2all) -> aiter ``fused_moe`` (a8w4 mxfp4 grouped
GEMM) -> mori ``combine``, and the combined output (plus residual) feeds the next
layer. All layers SHARE one weight set; only the per-layer routing (topk ids +
weights) is regenerated randomly per layer and retained, so the reference can
replay the exact same routing.

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

Launch (4x gfx1250, must build CK-free on gfx1250 -> ENABLE_CK=0):
    cd <dir not under /app>   # avoid the /app/triton namespace shadow
    ENABLE_CK=0 AITER_FORCE_A8W4=1 AITER_USE_GROUPED_GEMM=1 AITER_BF16_FP8_MOE_BOUND=0 \
    MORI_ROOT=/app/mori \
    torchrun --standalone --nproc_per_node=4 test_mega_moe.py \
      -q a8w4_mxfp4 -e 384 -k 6 -hd 7168 -id 3072 --layers 61

Env / CLI: --layers --logits_tol --acc_verify --dispatch_commu_dtype -m -hd -id -e -k --shared_E -q
"""
import os
import sys
import argparse

import torch
import torch.distributed as dist
import torch.profiler as tprof

import aiter
from aiter import dtypes
from aiter import ActivationType, QuantType, get_gfx
from aiter.fused_moe import fused_moe
from aiter.ops.shuffle import shuffle_weight, moe_shuffle_scale
from aiter.ops.flydsl.moe_common import GateMode
from aiter.utility import fp4_utils
from aiter import get_hip_quant, get_torch_quant, pertoken_quant

try:
    from aiter.test_common import get_trace_perf
except Exception:  # pragma: no cover
    get_trace_perf = None

# a8w4 (fp8 activation + mxfp4 weight) grouped kernel knobs. Force the real
# fp8/mxfp4 grouped path regardless of token count (mirrors test_moe_ep.py).
os.environ.setdefault("ENABLE_CK", "0")
os.environ.setdefault("AITER_FORCE_A8W4", "1")
os.environ.setdefault("AITER_USE_GROUPED_GEMM", "1")
os.environ.setdefault("AITER_BF16_FP8_MOE_BOUND", "0")

os.environ.setdefault("FLYDSL_GPU_ARCH", get_gfx())

# mori v2 (dispatch_combine_v2) lives outside the aiter tree; import by top-level
# module name after inserting its dir on sys.path. MORI_ROOT is the mori checkout.
MORI_ROOT = os.environ.get("MORI_ROOT", "/home/yashao/mori")

_FP8_DTYPE = dtypes.fp8
QUANT_KEYS = ["No", "per_Token", "per_128x128", "a8w4_mxfp4", "a4w4_mxfp4"]
_MXFP4_KEYS = ("a8w4_mxfp4", "a4w4_mxfp4")
_FP8_KEYS = ("per_Token", "per_128x128")


def _import_mori_v2():
    """Import the mori v2 op-layer. The dispatch_combine_v2 op module is not yet
    exported as a package API, so its dir is put on sys.path and imported by
    top-level name. cco itself comes from the installed ``mori`` package."""
    sys.path.insert(
        0, os.path.join(MORI_ROOT, "python", "mori", "ops", "dispatch_combine_v2")
    )
    from mori.cco import Communicator
    from dispatch_combine_op import EpDispatchCombineConfig, EpDispatchCombineOp

    return Communicator, EpDispatchCombineConfig, EpDispatchCombineOp


# --------------------------------------------------------------------------- #
# Config / quant-path spec
# --------------------------------------------------------------------------- #
def resolve_spec(quant_key, transport):
    """How to prepare weights / quantize activations / call fused_moe for a quant
    key, plus the dispatch transport dtype. transport: auto|bf16|fp8."""
    is_mxfp4 = quant_key in _MXFP4_KEYS
    is_fp8 = quant_key in _FP8_KEYS

    if transport == "auto":
        transport = "fp8" if is_fp8 else "bf16"
    if transport == "fp8" and not is_fp8:
        transport = "bf16"

    if quant_key == "No":
        aiter_qtype = QuantType.No
    elif quant_key == "per_Token":
        aiter_qtype = QuantType.per_Token
    elif quant_key == "per_128x128":
        aiter_qtype = QuantType.per_128x128
    else:  # a8w4_mxfp4 / a4w4_mxfp4
        aiter_qtype = QuantType.per_1x32

    gate_mode = GateMode.INTERLEAVE if quant_key == "a8w4_mxfp4" else GateMode.SEPARATED

    return {
        "key": quant_key,
        "aiter_qtype": aiter_qtype,
        "gate_mode": gate_mode,
        "activation": ActivationType.Silu,
        "is_mxfp4": is_mxfp4,
        "is_fp8": is_fp8,
        "transport": transport,
        "prequant": transport == "fp8",
        "fp8_dtype": _FP8_DTYPE,
    }


# --------------------------------------------------------------------------- #
# Weight quantization + shuffle (device path) / dequant (reference)
# --------------------------------------------------------------------------- #
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
    E, two_inter = t.shape[:2]
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
    """Layout-shuffle a group of `n_experts` quantized experts for the kernel."""
    key = spec["key"]
    if key in ("No", "per_Token", "per_128x128"):
        return shuffle_weight(w1_qt), shuffle_weight(w2_qt), w1_s, w2_s
    if key == "a8w4_mxfp4":
        if spec["gate_mode"] == GateMode.INTERLEAVE:
            w1_phys = _gguu_to_gugu_rows(w1_qt.view(torch.uint8))
            w1_a = shuffle_weight(w1_phys, layout=(16, 16))
            w1_ss = moe_shuffle_scale(
                w1_s.contiguous(), experts_cnt=n_experts,
                is_guinterleave=True, gate_up=True,
            )
        else:
            w1_a = shuffle_weight(w1_qt.view(torch.uint8), layout=(16, 16))
            w1_ss = moe_shuffle_scale(w1_s.contiguous(), experts_cnt=n_experts)
        w2_a = shuffle_weight(w2_qt.view(torch.uint8), layout=(16, 16))
        w2_ss = moe_shuffle_scale(w2_s.contiguous(), experts_cnt=n_experts)
        return w1_a, w2_a, w1_ss, w2_ss
    # a4w4_mxfp4
    w1_a = shuffle_weight(w1_qt, layout=(16, 16))
    w2_a = shuffle_weight(w2_qt, layout=(16, 16))
    w1_ss = fp4_utils.e8m0_shuffle(w1_s)
    w2_ss = fp4_utils.e8m0_shuffle(w2_s)
    w1_a.is_shuffled = True
    w2_a.is_shuffled = True
    return w1_a, w2_a, w1_ss, w2_ss


def quant_tokens_fp8(tokens, spec):
    """Per-token / per-block fp8 quant of the activations (fp8 pre-quant transport)."""
    qt = spec["aiter_qtype"]
    quant_func = get_hip_quant(
        qt if qt != QuantType.per_128x128 else QuantType.per_1x128
    )
    return quant_func(tokens, quant_dtype=spec["fp8_dtype"])


def moe_forward(hidden, w1_a, w2_a, w1_s, w2_s, topk_weights, topk_ids,
                expert_mask, spec, a1_scale=None, num_local_tokens=None):
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
            hidden, w1_a, w2_a, topk_weights, topk_ids,
            expert_mask=expert_mask,
            activation=spec["activation"],
            gate_mode=spec["gate_mode"].value,
            quant_type=spec["aiter_qtype"],
            w1_scale=w1_s, w2_scale=w2_s,
            dtype=dtypes.bf16,
            num_local_tokens=num_local_tokens,
        )
    return fused_moe(
        hidden, w1_a, w2_a, topk_weights, topk_ids, expert_mask,
        num_local_tokens=num_local_tokens,
        w1_scale=w1_s, w2_scale=w2_s,
        quant_type=spec["aiter_qtype"],
        a1_scale=a1_scale,
        dtype=dtypes.bf16,
    )


# --------------------------------------------------------------------------- #
# Shared setup (fed to BOTH reference and device path)
# --------------------------------------------------------------------------- #
_WEIGHT_SEED = 70000  # identical on every rank so the global expert set agrees


def make_shared_weights(E, hdim, idim, dtype, dev, shared_E=0, seed=_WEIGHT_SEED):
    """One weight set reused by every layer. Same seed on all ranks so the global
    expert partition is consistent. Returns bf16 (w1[E,2I,H], w2[E,H,I], sw1, sw2)."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    w1 = (torch.randn((E, 2 * idim, hdim), generator=gen, device=dev, dtype=torch.float32) / 10).to(dtype)
    w2 = (torch.randn((E, hdim, idim), generator=gen, device=dev, dtype=torch.float32) / 10).to(dtype)
    sw1 = sw2 = None
    if shared_E > 0:
        sw1 = (torch.randn((shared_E, 2 * idim, hdim), generator=gen, device=dev, dtype=torch.float32) / 10).to(dtype)
        sw2 = (torch.randn((shared_E, hdim, idim), generator=gen, device=dev, dtype=torch.float32) / 10).to(dtype)
    return w1, w2, sw1, sw2


def make_routings(n_layers, ct, E, topk, dev, seed):
    """Per-layer random routing, RETAINED so device + reference replay the same.
    topk_ids are distinct experts per token (top-k over a random score); weights
    are random and renormalized. Returns list[(ids[ct,topk] i32, wts[ct,topk] f32)]."""
    routings = []
    for l in range(n_layers):
        gen = torch.Generator(device=dev).manual_seed(seed + l)
        score = torch.rand(ct, E, generator=gen, device=dev, dtype=torch.float32)
        _, ids = score.topk(topk, dim=-1)  # distinct experts per token
        wts = torch.rand(ct, topk, generator=gen, device=dev, dtype=torch.float32)
        wts = wts / wts.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        routings.append((ids.to(dtypes.i32), wts))
    return routings


def _rmsnorm(x, eps=1e-6):
    """RMSNorm (no learnable gain) on the last dim. Applied to each layer's MoE
    input so activations stay unit-scale across the 61-layer residual chain --
    without it the a8w4 fp8 activation quant (max ~448) overflows to NaN after a
    few layers. Both device and reference use the SAME normalization."""
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


# --------------------------------------------------------------------------- #
# torchrun rendezvous helper
# --------------------------------------------------------------------------- #
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

    def shutdown(self):
        if dist.is_initialized():
            dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# Reference: pure-torch fp32 multi-layer chained MoE (ground truth, ISOLATED)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Device pipeline: N-layer dispatch->gemm->combine, one CUDA graph (ISOLATED)
# --------------------------------------------------------------------------- #
class DeviceMoEPipeline:
    """Owns the cco Communicator + EpDispatchCombineOp + a8w4 shuffled weights.
    Each layer recomputes its own routing inside dispatch (e2e-faithful), and the
    whole N-layer chain is captured into ONE CUDA graph and timed with
    torch.profiler. No fp32-reference logic here."""

    def __init__(self, dist_ctx, E, hdim, idim, topk, spec, n_layers,
                 w1_bf, w2_bf, sw1, sw2, routings, ct):
        self.dist_ctx = dist_ctx
        self.E, self.hdim, self.idim, self.topk = E, hdim, idim, topk
        self.spec = spec
        self.n_layers = n_layers
        self.w1_bf, self.w2_bf = w1_bf, w2_bf
        self.sw1, self.sw2 = sw1, sw2
        self.routings = routings
        self.ct = ct
        self.EPR = E // dist_ctx.world
        self.dev = torch.device("cuda", dist_ctx.local_rank)
        self.comm = None
        self.op = None
        self.graph = None
        self.x0_static = None
        self.out_static = None

    # ---- initialization (grouped together) ---- #
    def setup(self, x0):
        (Communicator,
         EpDispatchCombineConfig, EpDispatchCombineOp) = _import_mori_v2()
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
        win_bytes = (
            self.dist_ctx.world * self.ct * self.hdim * self.transport_dtype.itemsize * 2
            + (1 << 24)
        )
        self.comm = Communicator.init(
            self.dist_ctx.world, r, uid
        )
        cfg = EpDispatchCombineConfig(
            rank=r,
            world_size=self.dist_ctx.world,
            hidden_dim=self.hdim,
            max_num_inp_token_per_rank=self.ct,
            num_experts_per_rank=self.EPR,
            num_experts_per_token=self.topk,
            data_type=self.transport_dtype,
        )
        self.op = EpDispatchCombineOp(cfg, self.comm)
        self.comm.barrier()

    # ---- one graph-capturable layer + full chain (calls grouped together) ---- #
    def _layer_step(self, x, l):
        ids, wts = self.routings[l]
        xn = _rmsnorm(x)  # keep a8w4 fp8 activations in range across 61 layers
        # Recompute routing every layer (mode A: atomic routing inside dispatch)
        # instead of replaying a precomputed handle. return_routing=True hands
        # back this layer's forward dest-slot map, which combine then consumes.
        recv_x, recv_w, _rs, recv_idx, total_recv_t, handle = self.op.dispatch(
            xn, wts, None, ids, return_routing=True
        )
        out = moe_forward(
            recv_x, self.w1_a, self.w2_a, self.w1_s, self.w2_s,
            recv_w, recv_idx.to(dtypes.i32), self.expert_mask, self.spec,
            num_local_tokens=total_recv_t,
        )
        combine_out, _ = self.op.combine(out.to(self.transport_dtype), routing=handle)
        y = combine_out[: self.ct].to(dtypes.bf16)
        if self.sw1 is not None:
            y = y + _device_shared_ffn(xn, self.sw1, self.sw2)
        return x + y  # residual

    def _pipeline(self, x0):
        x = x0
        for l in range(self.n_layers):
            x = self._layer_step(x, l)
        return x

    # ---- CUDA graph capture (all N layers in ONE graph) ---- #
    def capture(self, x0):
        self.x0_static = x0.clone()
        # warmup on a side stream: primes fused_moe lru_cache + allocator.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
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
    _N_WARMUP = 5

    def bench(self):
        """Time the ONE-graph N-layer dispatch->gemm->combine chain. The graph
        already contains all N layers, so a single replay IS the per-chain
        measurement -- no separate replay-count knob. 5 warmup replays first.
        Returns (total_us for all N layers, per_layer_us, prof_us).

        - total_us = host wall-clock of one graph replay (one sync after; not
          cuda.Event). For a GPU-bound MoE chain this ~= GPU time.
        - torch.profiler over one EAGER pipeline pass for the per-op breakdown.

        NOTE: this ROCm torch build reports self_device_time_total == 0 for every
        event (verified even for a plain matmul), so torch.profiler cannot give a
        device-time number here; it is kept for the per-op (CPU-side) breakdown.
        If a future build populates device time, prof_us below becomes > 0."""
        import time

        for _ in range(self._N_WARMUP):
            self.graph.replay()
        torch.cuda.synchronize()
        self.comm.barrier()

        # one full N-layer graph replay == the performance measurement.
        t0 = time.perf_counter()
        self.graph.replay()
        torch.cuda.synchronize()
        total_us = (time.perf_counter() - t0) * 1e6
        self.comm.barrier()

        # torch.profiler breakdown (one eager pipeline pass; a graph replay would
        # collapse to a single hipGraphLaunch).
        with tprof.profile(
            activities=[tprof.ProfilerActivity.CPU, tprof.ProfilerActivity.CUDA]
        ) as prof:
            self._pipeline(self.x0_static)
            torch.cuda.synchronize()
        self.comm.barrier()
        self._prof = prof
        prof_us = sum(_event_device_us(e) for e in prof.key_averages())
        return total_us, total_us / self.n_layers, prof_us

    def final_output(self):
        self.graph.replay()
        torch.cuda.synchronize()
        return self.out_static.detach().clone()

    def teardown(self):
        self.graph = None
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


def _device_shared_ffn(tokens, sw1, sw2):
    """Dense shared-expert FFN (SwiGLU), graph-capturable (all on-device)."""
    x = tokens.float()
    acc = torch.zeros(tokens.shape[0], sw2.shape[1], device=tokens.device, dtype=torch.float32)
    for e in range(sw1.shape[0]):
        gate, up = (x @ sw1[e].float().t()).chunk(2, dim=-1)
        acc = acc + (torch.nn.functional.silu(gate) * up) @ sw2[e].float().t()
    return acc.to(tokens.dtype)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    args = _parse_args()
    dist_ctx = Dist()
    dev = torch.device("cuda", dist_ctx.local_rank)
    spec = resolve_spec(args.quant_type, args.dispatch_commu_dtype)

    if spec["is_mxfp4"] and get_gfx() not in ("gfx950", "gfx1250"):
        if dist_ctx.rank == 0:
            print(f"skip {args.quant_type}: mxfp4 requires gfx950/gfx1250, got {get_gfx()}")
        dist_ctx.shutdown()
        return

    E, hdim, idim, topk = args.expert, args.hidden, args.inter, args.topk
    ct, n_layers = args.tokens, args.layers
    assert E % dist_ctx.world == 0, f"E={E} must be divisible by world_size={dist_ctx.world}"

    if dist_ctx.rank == 0:
        print(
            f"[cfg] world={dist_ctx.world} layers={n_layers} tokens/rank={ct} hidden={hdim} "
            f"inter={idim} E={E} topk={topk} EPR={E // dist_ctx.world} quant={args.quant_type} "
            f"gate={spec['gate_mode'].name} shared_E={args.shared_experts} gfx={get_gfx()}",
            flush=True,
        )

    # ---- shared inputs: weights (same on all ranks) + this rank's tokens/routing.
    # args.seed shifts all RNG; weights stay rank-independent (identical global
    # experts), tokens/routing vary per rank. Default keeps runs reproducible.
    w1_bf, w2_bf, sw1, sw2 = make_shared_weights(
        E, hdim, idim, dtypes.bf16, dev, shared_E=args.shared_experts,
        seed=_WEIGHT_SEED + args.seed,
    )
    x0 = torch.randn(
        ct, hdim,
        generator=torch.Generator(device=dev).manual_seed(1000 + dist_ctx.rank + args.seed),
        device=dev, dtype=torch.float32,
    ).to(dtypes.bf16)
    routings = make_routings(
        n_layers, ct, E, topk, dev, seed=4242 + 100 * dist_ctx.rank + args.seed
    )

    # ---- device path (isolated): setup -> capture 61 layers in one graph -> bench.
    pipe = DeviceMoEPipeline(
        dist_ctx, E, hdim, idim, topk, spec, n_layers, w1_bf, w2_bf, sw1, sw2, routings, ct
    )
    pipe.setup(x0)
    pipe.capture(x0)
    total_us, per_layer_us, prof_us = pipe.bench()
    if dist_ctx.rank == 0:
        prof_note = (
            f"prof_device={prof_us:.1f}us"
            if prof_us > 0
            else "prof_device=n/a (this ROCm torch.profiler emits no device time)"
        )
        print(
            f"# MEGA-MOE layers={n_layers} tokens/rank={ct}: "
            f"total={total_us:.1f} us per_layer={per_layer_us:.1f} us "
            f"(dispatch+gemm+combine, 1 graph replay) {prof_note}",
            flush=True,
        )
        if args.profile_table:
            try:
                tbl = pipe._prof.key_averages().table(
                    sort_by="self_device_time_total", row_limit=25)
            except Exception:
                tbl = pipe._prof.key_averages().table(
                    sort_by="self_cuda_time_total", row_limit=25)
            print(tbl, flush=True)

    # ---- accuracy (isolated CPU/fp32 reference): end-to-end accumulated compare.
    if args.acc_verify:
        out_dev = pipe.final_output().float()
        ref = RefModel(w1_bf, w2_bf, sw1, sw2, spec, dev)
        ref_out = ref.run(x0, routings).float()
        logits_diff = _calc_diff(ref_out, out_dev)
        errs = dist_ctx.allreduce_sum(0 if logits_diff < args.logits_tol else 1)
        if dist_ctx.rank == 0:
            print(
                f"# MEGA-CHECK layers={n_layers}: {'PASS' if errs == 0 else 'FAIL'} "
                f"(rank0 logits_diff={logits_diff:.6f} tol={args.logits_tol})",
                flush=True,
            )

    pipe.teardown()
    dist_ctx.shutdown()


def _parse_args():
    p = argparse.ArgumentParser(description="multi-layer EP MoE perf + accuracy")
    p.add_argument("-q", "--quant_type", type=str, choices=QUANT_KEYS,
                   default="a8w4_mxfp4", help="quantization type")
    p.add_argument("-bs", "--tokens", type=int, default=128, help="tokens per rank")
    p.add_argument("-hd", "--hidden", type=int, default=7168, help="model/hidden dim")
    p.add_argument("-id", "--inter", type=int, default=3072, help="intermediate dim")
    p.add_argument("-e", "--expert", type=int, default=384, help="routed experts (global)")
    p.add_argument("-k", "--topk", type=int, default=6, help="top-k")
    p.add_argument("--shared_experts", type=int, default=0, help="dense shared experts")
    p.add_argument("--layers", type=int, default=61, help="number of MoE layers")
    p.add_argument("--seed", type=int, default=0,
                   help="base RNG seed for weights/tokens/routing (optional; default 0)")
    p.add_argument("--logits_tol", type=float, default=0.1, help="end-to-end 1-cosine tol")
    p.add_argument("--acc_verify", type=int, default=1, help="run fp32 reference accuracy check")
    p.add_argument("--profile_table", type=int, default=0, help="print per-kernel table")
    p.add_argument("--dispatch_commu_dtype", type=str, choices=["auto", "bf16", "fp8"],
                   default="auto", help="dispatch transport (communication) dtype")
    return p.parse_args()


if __name__ == "__main__":
    main()
