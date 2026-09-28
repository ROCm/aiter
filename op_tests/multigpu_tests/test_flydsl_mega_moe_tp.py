# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
import ctypes
import gc
import itertools
import os
import struct
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.isdir(os.path.join(_REPO_ROOT, "aiter")) and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.environ.setdefault("AITER_USE_SYSTEM_TRITON", "1")
# split baseline: SiTUv2 on the a4w4 (fp4 activation) aiter MoE, and a4w4 below
# M=256 too (the default bound would switch to bf16 activations)
os.environ.setdefault("AITER_SITUV2_A4W4", "1")
os.environ.setdefault("AITER_BF16_FP8_MOE_BOUND", "0")

import pandas as pd
import torch
import torch.distributed as dist

import aiter
from aiter.fused_moe import fused_moe
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose

from aiter.ops.flydsl.mega_moe_tp_glm import (
    NUM_MOE_WEIGHTS,
    MegaMoeTpW8A8Glm,
    UncachedSymmetricBuffer,
    mega_moe_tp_w8a8_glm_supported,
)
from aiter.ops.flydsl.mega_moe_tp_a4w4_glm import MegaMoeTpA4W4Glm
from aiter.ops.flydsl.mega_moe_tp_kimi3 import MegaMoeTpKimi3
from aiter.ops.flydsl.kernels.mega_moe_tp.mega_moe_tp_kimi3 import (
    HIDDEN as K3_HIDDEN,
    INTER as K3_INTER,
    NUM_EXPERTS as K3_EXPERTS,
    ROUTER_HIDDEN as K3_ROUTER_HIDDEN,
    SITU_BETA as K3_SITU_BETA,
    SITU_LINEAR_BETA as K3_SITU_LINEAR_BETA,
    SUPPORTED_SAMPLES as K3_SUPPORTED_SAMPLES,
    TOP_K as K3_TOP_K,
)
from aiter.utility.fp4_utils import _f32_to_floatx_unpacked, mxfp4_to_f32, pack_uint4
from aiter.ops.flydsl.kernels.mega_moe_tp.mega_moe_tp_w8a8_glm import (
    HIDDEN,
    INTER,
    NUM_EXPERTS,
    SLOTS,
    TOP_K,
)

SUPPORTED_GFX = ["gfx950"]
ROUTE_SCALE = 2.5
RMS_EPS = 1e-5
FP8_MAX = 448.0
AMAX_EPS = 1e-4
MX_BLOCK = 32
INV_FP4_MAX = torch.tensor(0x3E2AAAAB, dtype=torch.int32).view(torch.float32).item()
MODULES = {4: MegaMoeTpA4W4Glm, 8: MegaMoeTpW8A8Glm}
QUANT_NAME = {4: "a4w4", 8: "a8w8"}
_TILERT_REL = ("tile_rt", "fused_moe_allreduce_w8a8_v4", "kernel", "fused_moe_allreduce_w8a8_v4.gfx950.hsaco")
DEFAULT_TILERT_HSACO = next(
    (
        p
        for p in (os.path.join(_REPO_ROOT, *up, *_TILERT_REL) for up in (("..",), ("..", ".."), ("..", "..", "..")))
        if os.path.isfile(p)
    ),
    os.path.join(_REPO_ROOT, "..", *_TILERT_REL),
)


def setup_dist():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world > 1:
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29555")
        dist.init_process_group("gloo", rank=rank, world_size=world)
    return rank, world, local_rank


def barrier():
    torch.cuda.synchronize()
    dist.barrier()


def proc_max(value: float) -> float:
    t = torch.tensor(float(value), dtype=torch.float64)
    if dist.get_world_size() > 1:
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def proc_min(value: float) -> float:
    return -proc_max(-value)


def sync_all(ranks):
    for r in ranks:
        torch.cuda.synchronize(r["device"])


def quantize_fp8_block(w: torch.Tensor):
    *b, rows, k = w.shape
    blocks = w.float().reshape(*b, rows // 128, 128, k // 128, 128)
    amax = blocks.abs().amax(dim=(-3, -1), keepdim=True).clamp(min=1e-12)
    scales = amax / FP8_MAX
    q = (blocks / scales).to(torch.float8_e4m3fn)
    return q.reshape(*b, rows, k), scales.reshape(*b, rows // 128, k // 128)


def mxfp4_quant(x: torch.Tensor):
    *b, k = x.shape
    xb = x.float().reshape(*b, k // MX_BLOCK, MX_BLOCK)
    amax = xb.abs().amax(-1)
    w = (amax * INV_FP4_MAX).view(torch.int32)
    e = (((w >> 23) & 0xFF) + ((w & 0x7FFFFF) != 0).int()).clamp(max=254)
    inv = ((254 - e) << 23).view(torch.float32)
    codes = _f32_to_floatx_unpacked((xb * inv[..., None]).reshape(*b, k), 2, 1)
    return pack_uint4(codes), e.to(torch.uint8)


def mxfp4_dequant(q: torch.Tensor, e: torch.Tensor):
    v = mxfp4_to_f32(q.view(torch.uint8))
    *b, k = v.shape
    sc = (e.to(torch.int32) << 23).view(torch.float32)
    return (v.reshape(*b, k // MX_BLOCK, MX_BLOCK) * sc[..., None]).reshape(*b, k)


def make_weights(rank: int, seed: int, device, quants):
    g_rep = torch.Generator(device=device).manual_seed(seed)
    g_rank = torch.Generator(device=device).manual_seed(seed + 1000 * (rank + 1))

    def rn(gen, *shape, std=1.0):
        return torch.randn(*shape, generator=gen, device=device) * std

    shared = {
        "router_w": rn(g_rep, NUM_EXPERTS, HIDDEN, std=0.02).to(torch.bfloat16),
        "gamma": 1.0 + 0.1 * rn(g_rep, HIDDEN),
        "bias": 0.01 * rn(g_rep, NUM_EXPERTS),
    }
    E = NUM_MOE_WEIGHTS
    u8 = {"dtype": torch.uint8, "device": device}
    fp8 = {"dtype": torch.float8_e4m3fn, "device": device}
    bf16 = {"dtype": torch.bfloat16, "device": device}
    ws = {0: dict(shared, ug_w=torch.empty(E, 2 * INTER, HIDDEN, **bf16), down_w=torch.empty(E, HIDDEN, INTER, **bf16))}
    if 8 in quants:
        ws[8] = dict(
            shared,
            ug_w=torch.empty(E, 2 * INTER, HIDDEN, **fp8),
            ug_scales=torch.empty(E, 2 * INTER // 128, HIDDEN // 128, device=device),
            down_w=torch.empty(E, HIDDEN, INTER, **fp8),
            down_scales=torch.empty(E, HIDDEN // 128, INTER // 128, device=device),
        )
    if 4 in quants:
        ws[4] = dict(
            shared,
            ug_w=torch.empty(E, 2 * INTER, HIDDEN // 2, **u8),
            ug_scales=torch.empty(E, 2 * INTER, HIDDEN // MX_BLOCK, **u8),
            down_w=torch.empty(E, HIDDEN, INTER // 2, **u8),
            down_scales=torch.empty(E, HIDDEN, INTER // MX_BLOCK, **u8),
        )
    quantize = {8: quantize_fp8_block, 4: mxfp4_quant}
    for e0 in range(0, E, 32):
        e1 = min(E, e0 + 32)
        ug = rn(g_rank, e1 - e0, 2 * INTER, HIDDEN, std=0.02)
        dn = rn(g_rank, e1 - e0, HIDDEN, INTER, std=0.02)
        ws[0]["ug_w"][e0:e1], ws[0]["down_w"][e0:e1] = ug, dn
        for q in quants:
            ws[q]["ug_w"][e0:e1], ws[q]["ug_scales"][e0:e1] = quantize[q](ug)
            ws[q]["down_w"][e0:e1], ws[q]["down_scales"][e0:e1] = quantize[q](dn)
    return ws


def make_inputs(samples: int, seed: int, device):
    g = torch.Generator(device=device).manual_seed(seed + 7)
    hidden = torch.randn(samples, HIDDEN, generator=g, device=device).to(torch.bfloat16)
    residual = torch.randn(samples, HIDDEN, generator=g, device=device).to(torch.bfloat16)
    return hidden, residual


def quant_std_blocks(x_bf16: torch.Tensor):
    shape = x_bf16.shape
    xb = x_bf16.float().reshape(*shape[:-1], shape[-1] // 128, 128)
    amax = xb.abs().amax(dim=-1)
    scale = amax.clamp_min(AMAX_EPS) * torch.tensor(1.0 / FP8_MAX, device=xb.device)
    inv = torch.ones_like(scale) / scale
    q = (xb * inv.unsqueeze(-1)).to(torch.float8_e4m3fn).reshape(shape)
    return q, scale


def _route(hidden, w):
    x = hidden.float()
    rms = torch.rsqrt((x * x).sum(-1, keepdim=True) / HIDDEN + RMS_EPS)
    norm = (w["gamma"].float()[None] * x * rms).to(torch.bfloat16)
    logits = norm.float() @ w["router_w"].float().T
    scores = torch.sigmoid(logits)
    idx = torch.topk(scores + w["bias"].float()[None], TOP_K, dim=-1, sorted=True).indices
    vals = torch.gather(scores, 1, idx)
    total = vals[:, 0].clone()
    for k in range(1, TOP_K):
        total = total + vals[:, k]
    probs = vals * (ROUTE_SCALE / total)[:, None]
    experts = torch.cat([torch.zeros_like(idx[:, :1]), idx + 1], 1)
    w_slot = torch.cat([torch.ones_like(probs[:, :1]), probs], 1)
    return norm, logits, probs, idx.to(torch.int32), experts, w_slot


def run_torch_a8w8(hidden, w):
    s_n = hidden.shape[0]
    norm, logits, probs, idx, experts, w_slot = _route(hidden, w)
    a8, a_sc = quant_std_blocks(norm)
    wq = w["ug_w"][experts].float().view(s_n, SLOTS, 2 * INTER, HIDDEN // 128, 128)
    part = torch.einsum("sjrck,sck->sjrc", wq, a8.float().view(s_n, HIDDEN // 128, 128))
    fac = w["ug_scales"][experts].repeat_interleave(128, dim=2) * a_sc[:, None, None, :]
    acc = (part * fac).sum(-1)
    gate, up = acc[..., :INTER], acc[..., INTER:]
    mid = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)
    q, m_sc = quant_std_blocks(mid)
    wd = w["down_w"][experts].float().view(s_n, SLOTS, HIDDEN, 2, 128)
    partd = torch.einsum("sjrck,sjck->sjrc", wd, q.float().view(s_n, SLOTS, 2, 128))
    facd = (
        w["down_scales"][experts].repeat_interleave(128, dim=2)
        * w_slot[:, :, None, None]
        * m_sc[:, :, None, :]
    )
    partial = (partd * facd).sum(-1).sum(1).to(torch.bfloat16)
    return norm, logits, probs, idx, mid, partial


def run_torch_a4w4(hidden, w):
    norm, logits, probs, idx, experts, w_slot = _route(hidden, w)
    act = mxfp4_dequant(*mxfp4_quant(norm.float()))
    wug = mxfp4_dequant(w["ug_w"][experts], w["ug_scales"][experts])
    acc = torch.einsum("sjrk,sk->sjr", wug, act)
    gate, up = acc[..., :INTER], acc[..., INTER:]
    mid = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)
    midq = mxfp4_dequant(*mxfp4_quant(mid.float()))
    wd = mxfp4_dequant(w["down_w"][experts], w["down_scales"][experts])
    part = torch.einsum("sjrk,sjk->sjr", wd, midq)
    partial = (part * w_slot[:, :, None]).sum(1).to(torch.bfloat16)
    return norm, logits, probs, idx, mid, partial


def run_torch_bf16(hidden, w):
    norm, logits, probs, idx, experts, w_slot = _route(hidden, w)
    acc = torch.einsum("sjrk,sk->sjr", w["ug_w"][experts].float(), norm.float())
    gate, up = acc[..., :INTER], acc[..., INTER:]
    mid = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)
    part = torch.einsum("sjrk,sjk->sjr", w["down_w"][experts].float(), mid.float())
    partial = (part * w_slot[:, :, None]).sum(1).to(torch.bfloat16)
    return norm, logits, probs, idx, mid, partial


GOLDEN = {4: run_torch_a4w4, 8: run_torch_a8w8}
MID_TOL = {4: 2e-2, 8: 5e-3}


def all_partials(ranks, partials):
    cpu = torch.stack([p.cpu() for p in partials])
    if dist.get_world_size() == 1:
        return list(cpu)
    gathered = [torch.empty_like(cpu) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, cpu)
    return [p for g in gathered for p in g]


def reference_out(parts, residual):
    total = torch.zeros(parts[0].shape, dtype=torch.float32)
    for p in parts:
        total = total + p.float()
    return (total + residual.float().cpu()).to(torch.bfloat16)


class TileRtKernel:
    SYM_BYTES = 8 << 20

    def __init__(self, path: str, device, rank: int, world: int, sym: torch.Tensor):
        self.hip = ctypes.CDLL("libamdhip64.so")
        self.device, self.rank, self.world, self.sym = device, rank, world, sym
        data = open(path, "rb").read()
        self._image = ctypes.create_string_buffer(data, len(data))
        mod = ctypes.c_void_p()
        with torch.cuda.device(device):
            torch.cuda.current_stream()
            err = self.hip.hipModuleLoadData(ctypes.byref(mod), self._image)
        if err:
            raise RuntimeError(f"hipModuleLoadData failed: {err}")
        self.mod = mod
        self.funcs = {}
        self.tag = 0
        i32 = {"dtype": torch.int32, "device": device}
        self.score_lines = torch.zeros(4, 32, 32, **i32)
        self.flags = torch.zeros(512, **i32)
        self.mid_pairs = torch.zeros(4, SLOTS, INTER, **i32)

    @classmethod
    def build(cls, path, ranks, world, multi_process):
        if multi_process:
            r = ranks[0]
            buf = UncachedSymmetricBuffer(cls.SYM_BYTES, rank=r["rank"], world_size=world, device=r["device"])
            ks = [cls(path, r["device"], r["rank"], world, buf.table)]
            ks[0]._buf = buf
            return ks
        bufs = [torch.zeros(cls.SYM_BYTES, dtype=torch.uint8, device=r["device"]) for r in ranks]
        ptrs = [b.data_ptr() for b in bufs]
        ks = []
        for r in ranks:
            k = cls(path, r["device"], r["rank"], world, torch.tensor(ptrs, dtype=torch.int64, device=r["device"]))
            k._bufs = bufs
            ks.append(k)
        return ks

    def close(self):
        if getattr(self, "_buf", None) is not None:
            self._buf.close()

    def func(self, proto, s, ki):
        key = (proto, s, ki)
        if key not in self.funcs:
            name = (
                "_ZN6tilert3ops7glm_5_227fused_moe_allreduce_w8a8_v434fused_moe_allreduce_w8a8_v4_kernel"
                f"ILNS_4cell4xfer5ProtoE{proto}ELi{s}ELi{ki}ELb0EEEvPK14__hip_bfloat16PKfPKhSD_SB_SB_"
                "SD_SB_S9_PKPviijPS7_PfPjSJ_SI_PiSH_SH_SJ_jPm"
            )
            f = ctypes.c_void_p()
            err = self.hip.hipModuleGetFunction(ctypes.byref(f), self.mod, name.encode())
            if err:
                raise RuntimeError(f"no TileRT kernel for proto={proto} S={s} KI={ki}")
            self.funcs[key] = f
        return self.funcs[key]

    def launch(self, moe, hidden, residual, outs, proto, ki):
        self.tag += 1
        norm, scores, mid, probs, indices, out = outs
        a = [
            hidden, moe.gamma, moe.router_w, moe.ug_w, moe.ug_scales, moe.bias,
            moe.down_w, moe.down_scales, residual, self.sym,
        ]
        buf = bytearray(184)
        for i, t in enumerate(a):
            struct.pack_into("<Q", buf, 8 * i, 0 if t is None else t.data_ptr())
        struct.pack_into("<iiI", buf, 80, self.rank, self.world, self.tag)
        b = [norm, scores, self.score_lines, self.flags, probs, indices, mid, out, self.mid_pairs]
        for i, t in enumerate(b):
            struct.pack_into("<Q", buf, 96 + 8 * i, t.data_ptr())
        struct.pack_into("<IQ", buf, 168, self.tag, 0)
        args = ctypes.create_string_buffer(bytes(buf), len(buf))
        size = ctypes.c_size_t(len(buf))
        extra = (ctypes.c_void_p * 5)(
            1, ctypes.cast(args, ctypes.c_void_p), 2, ctypes.cast(ctypes.pointer(size), ctypes.c_void_p), 3
        )
        f = self.func(proto, hidden.shape[0], ki)
        stream = torch.cuda.current_stream(self.device).cuda_stream
        err = self.hip.hipModuleLaunchKernel(f, 256, 1, 1, 512, 1, 1, 0, ctypes.c_void_p(stream), None, extra)
        if err:
            raise RuntimeError(f"hipModuleLaunchKernel failed: {err}")
        self._keep = (args, size, extra)


def _capture(ranks, launch, reps):
    graphs = []
    for r in ranks:
        with torch.cuda.device(r["device"]):
            s = torch.cuda.Stream(r["device"])
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(s), torch.cuda.graph(graph, stream=s):
                for _ in range(reps):
                    launch(r)
            graphs.append((graph, s))
    sync_all(ranks)
    return graphs


def _replay_us(ranks, graphs, reps, iters) -> float:
    barrier()
    evs = []
    for r, (_, s) in zip(ranks, graphs):
        with torch.cuda.device(r["device"]):
            e = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            e[0].record(s)
            evs.append(e)
    for _ in range(iters):
        for r, (g, s) in zip(ranks, graphs):
            with torch.cuda.device(r["device"]), torch.cuda.stream(s):
                g.replay()
    for r, (_, s), e in zip(ranks, graphs, evs):
        with torch.cuda.device(r["device"]):
            e[1].record(s)
    sync_all(ranks)
    local = max(e[0].elapsed_time(e[1]) for e in evs) * 1000.0 / (iters * reps)
    return proc_max(local)


def time_fly(ranks, samples, quant, args) -> float:
    def launch(r):
        h, res = r["inputs"][samples]
        r["moe"][quant](h, res, out=r["fly_out"][quant][samples])

    graphs = _capture(ranks, launch, args.reps)
    _replay_us(ranks, graphs, args.reps, 2)
    best = min(_replay_us(ranks, graphs, args.reps, args.iters) for _ in range(args.rounds))
    del graphs
    gc.collect()
    return best


def time_tilert(ranks, samples, proto, ki, args) -> float:
    def launch(r):
        h, res = r["inputs"][samples]
        r["tilert"].launch(r["moe"][8], h, res, r["tilert_outs"][samples], proto, ki)

    best = float("inf")
    for _ in range(args.rounds + 1):
        n = args.reps * args.iters
        graphs = _capture(ranks, launch, n)
        best = min(best, _replay_us(ranks, graphs, n, 1))
        del graphs
        gc.collect()
    return best


def rel_l2(a, b) -> float:
    a, b = a.float().cpu(), b.float().cpu()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def _bytes_moved(samples: int, quant: int) -> int:
    if quant == 8:
        experts = 2 * INTER * HIDDEN + HIDDEN * INTER
    else:
        experts = (2 * INTER * HIDDEN + HIDDEN * INTER) * (MX_BLOCK // 2 + 1) // MX_BLOCK
    return samples * (NUM_EXPERTS * HIDDEN * 2 + SLOTS * experts)


def _flops(samples: int) -> int:
    per_token = 2 * NUM_EXPERTS * HIDDEN + SLOTS * 2 * (2 * INTER * HIDDEN + HIDDEN * INTER)
    return samples * per_token


def moe_vs_bf16(ranks, got, samples) -> float:
    refs = []
    for r in ranks:
        with torch.cuda.device(r["device"]):
            refs.append(run_torch_bf16(r["inputs"][samples][0], r["w"][0])[5])
    res = ranks[0]["inputs"][samples][1].float().cpu()
    moe_ref = reference_out(all_partials(ranks, refs), res).float() - res
    return proc_max(max(rel_l2(g[5].float().cpu() - res, moe_ref) for g in got))


_CTX: dict = {}


@benchmark()
def test_mega_moe_tp_glm(samples, tp, proto, quant):
    ranks, args = _CTX["ranks"], _CTX["args"]
    names = ("norm", "scores", "mid", "probs", "indices", "out")
    ret = {"gfx": get_gfx()}
    got = []
    for r in ranks:
        moe = r["moe"][quant]
        moe.proto = proto
        with torch.cuda.device(r["device"]):
            h, res = r["inputs"][samples]
            got.append(moe(h, res, out=r["fly_out"][quant][samples]))
    sync_all(ranks)

    refs = []
    for r in ranks:
        with torch.cuda.device(r["device"]):
            refs.append(GOLDEN[quant](r["inputs"][samples][0], r["w"][quant]))
    parts = all_partials(ranks, [ref[5] for ref in refs])
    out_ref = reference_out(parts, ranks[0]["inputs"][samples][1])
    ok, worst, mid_err = True, 0.0, 0.0
    for r, g, ref in zip(ranks, got, refs):
        norm, scores, mid, probs, indices, out = g
        ok &= torch.equal(norm, ref[0]) and torch.equal(indices, ref[3])
        ok &= rel_l2(scores, ref[1]) < 1e-5 and rel_l2(probs, ref[2]) < 1e-5
        mid_err = max(mid_err, rel_l2(mid, ref[4]))
        worst = max(worst, rel_l2(out, out_ref))
        stuck = r["moe"][quant].poll_errors()
        if stuck:
            aiter.logger.error("[rank %d] poll watchdog fired: %s", r["rank"], stuck)
        ok &= not stuck
    ok &= mid_err < MID_TOL[quant]
    err = checkAllclose(
        out_ref.float(), got[0][5].float().cpu(), rtol=2e-2, atol=2e-2,
        msg=f"fly {QUANT_NAME[quant]} out S={samples} proto={proto}", printLog=_CTX["rank0"],
    )
    ret["fly err"] = proc_max(err)
    ret["fly out rel_l2"] = proc_max(worst)
    ret["fly mid rel_l2"] = proc_max(mid_err)
    ret["fly ok"] = bool(proc_min(float(ok and err < 0.05 and worst < 1e-2)) > 0)
    ret["moe rel_l2 vs bf16"] = moe_vs_bf16(ranks, got, samples)

    if not args.no_perf:
        us = time_fly(ranks, samples, quant, args)
        if any(r["moe"][quant].poll_errors() for r in ranks):
            aiter.logger.error("poll watchdog fired while timing")
            ret["fly ok"] = False
        ret["fly us"] = us
        ret["fly TB/s"] = _bytes_moved(samples, quant) / us / 1e6
        ret["fly TFLOPS"] = _flops(samples) / us / 1e6

    if quant == 8 and ranks[0].get("tilert") is not None:
        best_ki, best_us = None, float("inf")
        for ki in args.tilert_ki:
            for r in ranks:
                with torch.cuda.device(r["device"]):
                    h, res = r["inputs"][samples]
                    r["tilert"].launch(r["moe"][8], h, res, r["tilert_outs"][samples], proto, ki)
            sync_all(ranks)
            same = {
                n: min(float((a[i] == b[i]).float().mean()) for a, b in zip(got, [r["tilert_outs"][samples] for r in ranks]))
                for i, n in enumerate(names)
            }
            ret[f"tilert KI{ki} out bitexact"] = proc_min(same["out"])
            ret[f"tilert KI{ki} out rel_l2"] = proc_max(
                max(rel_l2(r["tilert_outs"][samples][5], out_ref) for r in ranks)
            )
            if not args.no_perf:
                us = time_tilert(ranks, samples, proto, ki, args)
                ret[f"tilert KI{ki} us"] = us
                if us < best_us:
                    best_ki, best_us = ki, us
        if best_ki is not None:
            ret["tilert us"] = best_us
            ret["tilert TB/s"] = _bytes_moved(samples, 8) / best_us / 1e6
            ret["fly/tilert"] = ret["fly us"] / best_us
    return ret


# ---------------------------------------------------------------------------
# Kimi-K3 (A4W4) and the batch sweep against the split MoE, both models
# ---------------------------------------------------------------------------
K3 = dict(
    RH=K3_ROUTER_HIDDEN, H=K3_HIDDEN, I=K3_INTER, E=K3_EXPERTS, TOPK=K3_TOP_K,
    beta=K3_SITU_BETA, lbeta=K3_SITU_LINEAR_BETA,
)


def make_kimi3_weights(rank: int, seed: int, device):
    """router [896, 7168] bf16 and bias replicated; each rank's experts are its
    own intermediate shard (MXFP4 codes + E8M0 scales, as the kernel packs)."""
    g_rep = torch.Generator(device=device).manual_seed(seed)
    g_rank = torch.Generator(device=device).manual_seed(seed + 1000 * (rank + 1))
    E, H, I = K3["E"], K3["H"], K3["I"]
    u8 = {"dtype": torch.uint8, "device": device}
    w = {
        "router_w": (torch.randn(E, K3["RH"], generator=g_rep, device=device) * 0.02).to(torch.bfloat16),
        "bias": 0.01 * torch.randn(E, generator=g_rep, device=device),
        "ug_w": torch.empty(E, 2 * I, H // 2, **u8),
        "ug_scales": torch.empty(E, 2 * I, H // MX_BLOCK, **u8),
        "down_w": torch.empty(E, H, I // 2, **u8),
        "down_scales": torch.empty(E, H, I // MX_BLOCK, **u8),
    }
    for e0 in range(0, E, 64):
        e1 = min(E, e0 + 64)
        ug = torch.randn(e1 - e0, 2 * I, H, generator=g_rank, device=device) * 0.02
        w["ug_w"][e0:e1], w["ug_scales"][e0:e1] = mxfp4_quant(ug)
        del ug
        dn = torch.randn(e1 - e0, H, I, generator=g_rank, device=device) * 0.02
        w["down_w"][e0:e1], w["down_scales"][e0:e1] = mxfp4_quant(dn)
        del dn
    return w


def make_kimi3_inputs(tokens: int, seed: int, device):
    g = torch.Generator(device=device).manual_seed(seed + 7)
    x = torch.randn(tokens, K3["RH"], generator=g, device=device).to(torch.bfloat16)
    latent = torch.randn(tokens, K3["H"], generator=g, device=device).to(torch.bfloat16)
    return x, latent


def situ_ref(gate, up):
    gate = K3["beta"] * torch.tanh(gate / K3["beta"]) * torch.sigmoid(gate)
    up = K3["lbeta"] * torch.tanh(up / K3["lbeta"])
    return gate * up


def run_torch_kimi3(x, latent, w, idx=None):
    """Golden of the A4W4 kernel: fp32 router, top-16 of sigmoid + bias,
    MXFP4 activation / weights / mid, SiTUv2, bf16 mid and partial.
    ``idx`` pins the routing (to check the experts on the kernel's own)."""
    logits = x.float() @ w["router_w"].float().T
    scores = torch.sigmoid(logits)
    if idx is None:
        idx = torch.topk(scores + w["bias"].float()[None], K3["TOPK"], dim=-1, sorted=True).indices
    idx = idx.long()
    vals = torch.gather(scores, 1, idx)
    probs = vals / vals.sum(-1, keepdim=True)
    act = mxfp4_dequant(*mxfp4_quant(latent.float()))
    wug = mxfp4_dequant(w["ug_w"][idx], w["ug_scales"][idx])
    acc = torch.einsum("sjrk,sk->sjr", wug, act)
    mid = situ_ref(acc[..., : K3["I"]], acc[..., K3["I"]:]).to(torch.bfloat16)
    midq = mxfp4_dequant(*mxfp4_quant(mid.float()))
    wd = mxfp4_dequant(w["down_w"][idx], w["down_scales"][idx])
    part = (torch.einsum("sjrk,sjk->sjr", wd, midq) * probs[..., None]).sum(1).to(torch.bfloat16)
    return logits, probs, idx.to(torch.int32), mid, part


@benchmark()
def test_mega_moe_tp_kimi3(samples, tp):
    """Kimi-K3 accuracy: router logits, top-16, probs and mids per rank; the
    all-reduced output against the rank-ordered sum of every rank's golden."""
    ranks = _CTX["ranks"]
    ret = {"gfx": get_gfx(), "model": "kimi3", "quant": 4}
    got = []
    for r in ranks:
        with torch.cuda.device(r["device"]):
            x, lat = r["k3_inputs"][samples]
            got.append(r["k3"](x, lat))
    sync_all(ranks)
    refs, ok, mid_err, route_ok = [], True, 0.0, True
    for r, g in zip(ranks, got):
        with torch.cuda.device(r["device"]):
            x, lat = r["k3_inputs"][samples]
            scores, mid, probs, idx, _ = g
            free = run_torch_kimi3(x, lat, r["k3_w"])
            route_ok &= torch.equal(idx, free[2])
            ref = run_torch_kimi3(x, lat, r["k3_w"], idx)
            refs.append(ref)
            ok &= rel_l2(scores, ref[0]) < 1e-5 and rel_l2(probs, ref[1]) < 1e-5
            mid_err = max(mid_err, rel_l2(mid, ref[3]))
            stuck = r["k3"].poll_errors()
            if stuck:
                aiter.logger.error("[rank %d] poll watchdog fired: %s", r["rank"], stuck)
            ok &= not stuck
    zero = torch.zeros(samples, K3["H"], dtype=torch.bfloat16)
    out_ref = reference_out(all_partials(ranks, [ref[4] for ref in refs]), zero)
    worst = max(rel_l2(g[4], out_ref) for g in got)
    ok &= route_ok and mid_err < MID_TOL[4] and worst < 1e-2
    ret["routing == golden"] = proc_min(float(route_ok)) > 0
    ret["fly mid rel_l2"] = proc_max(mid_err)
    ret["fly out rel_l2"] = proc_max(worst)
    ret["fly ok"] = bool(proc_min(float(ok)) > 0)
    return ret


E2E_TOL = 0.1


def batch_launches(tokens: int):
    """(offset, S) launches covering ``tokens`` with S in {4, 2, 1}."""
    out, off = [], 0
    while off < tokens:
        s = 4 if tokens - off >= 4 else (2 if tokens - off >= 2 else 1)
        out.append((off, s))
        off += s
    return out


class SplitGlm5A8W8:
    """RMSNorm -> router GEMM -> biased top-8 -> aiter fused_moe (FP8 block,
    shared expert as slot 0 of 257) -> all-reduce -> + residual."""

    def __init__(self, w, comm, device):
        from aiter.ops.shuffle import shuffle_weight

        self.gamma = w["gamma"].to(device, torch.bfloat16)
        self.router_w = w["router_w"].to(device)
        self.bias = w["bias"].to(device, torch.float32)
        self.w1 = shuffle_weight(w["ug_w"].to(device), layout=(16, 16))
        self.w2 = shuffle_weight(w["down_w"].to(device), layout=(16, 16))
        self.s1, self.s2 = w["ug_scales"].to(device), w["down_scales"].to(device)
        self.comm, self.device, self.bufs = comm, device, {}

    def __call__(self, x, residual, local=False):
        t = x.shape[0]
        b = self.bufs.get(t)
        if b is None:
            tp = self.comm.tp_size
            pad = (t + tp - 1) // tp * tp
            b = self.bufs[t] = dict(
                tw=torch.empty(t, TOP_K, dtype=torch.float32, device=self.device),
                ids=torch.empty(t, TOP_K, dtype=torch.int32, device=self.device),
                one=torch.ones(t, 1, dtype=torch.float32, device=self.device),
                zero=torch.zeros(t, 1, dtype=torch.int32, device=self.device),
                ypad=torch.zeros(pad, HIDDEN, dtype=torch.bfloat16, device=self.device),
                red=torch.empty(pad, HIDDEN, dtype=torch.bfloat16, device=self.device),
            )
        norm = aiter.rms_norm(x, self.gamma, RMS_EPS)
        logits = torch.mm(norm, self.router_w.t()).float()
        aiter.biased_grouped_topk(logits, self.bias, b["tw"], b["ids"], 1, 1, True, ROUTE_SCALE)
        ids = torch.cat([b["zero"], b["ids"] + 1], 1)
        tw = torch.cat([b["one"], b["tw"]], 1)
        y = fused_moe(
            norm, self.w1, self.w2, tw, ids, activation=aiter.ActivationType.Silu,
            quant_type=aiter.QuantType.per_1x128, w1_scale=self.s1, w2_scale=self.s2,
        )
        b["ypad"][:t].copy_(y)
        if not local:
            self.comm.all_reduce(b["ypad"], b["red"])
        return b["red"][:t] + residual


class SplitKimi3A4W4:
    """Router GEMM -> biased top-16 -> aiter fused_moe (A4W4 MXFP4, SiTUv2)
    on the latent -> all-reduce."""

    def __init__(self, w, comm, device):
        from aiter.ops.shuffle import shuffle_weight
        from aiter.utility import fp4_utils

        E, H, I = K3["E"], K3["H"], K3["I"]
        self.router_w = w["router_w"].to(device)
        self.bias = w["bias"].to(device, torch.float32)
        fp4 = aiter.dtypes.fp4x2
        e8 = aiter.dtypes.fp8_e8m0
        self.w1 = shuffle_weight(w["ug_w"].to(device).view(fp4), layout=(16, 16))
        self.w2 = shuffle_weight(w["down_w"].to(device).view(fp4), layout=(16, 16))
        self.s1 = fp4_utils.e8m0_shuffle(w["ug_scales"].to(device).reshape(E * 2 * I, H // MX_BLOCK).view(e8))
        self.s2 = fp4_utils.e8m0_shuffle(w["down_scales"].to(device).reshape(E * H, I // MX_BLOCK).view(e8))
        self.comm, self.device, self.bufs = comm, device, {}

    def __call__(self, x, latent, local=False):
        t = x.shape[0]
        b = self.bufs.get(t)
        if b is None:
            tp = self.comm.tp_size
            pad = (t + tp - 1) // tp * tp
            b = self.bufs[t] = dict(
                tw=torch.empty(t, K3["TOPK"], dtype=torch.float32, device=self.device),
                ids=torch.empty(t, K3["TOPK"], dtype=torch.int32, device=self.device),
                ypad=torch.zeros(pad, K3["H"], dtype=torch.bfloat16, device=self.device),
                red=torch.empty(pad, K3["H"], dtype=torch.bfloat16, device=self.device),
            )
        logits = torch.mm(x, self.router_w.t()).float()
        aiter.biased_grouped_topk(logits, self.bias, b["tw"], b["ids"], 1, 1, True, 1.0)
        y = fused_moe(
            latent, self.w1, self.w2, b["tw"], b["ids"], activation=aiter.ActivationType.Situv2,
            quant_type=aiter.QuantType.per_1x32, w1_scale=self.s1, w2_scale=self.s2,
            beta=K3["beta"], linear_beta=K3["lbeta"],
        )
        b["ypad"][:t].copy_(y)
        if not local:
            self.comm.all_reduce(b["ypad"], b["red"])
        return b["red"][:t]


def _replay_threaded_us(ranks, graphs, reps, iters) -> float:
    """Replay every rank's graph from its own host thread, started together:
    one thread replaying rank after rank serializes eight graph launches
    (~20 us each) and would time the host, not the GPUs."""
    import threading

    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in ranks]
    gate = threading.Barrier(len(ranks))

    def run(i):
        r, (g, s) = ranks[i], graphs[i]
        torch.cuda.set_device(r["device"])
        with torch.cuda.stream(s):
            gate.wait()
            evs[i][0].record(s)
            for _ in range(iters):
                g.replay()
            evs[i][1].record(s)

    ts = [threading.Thread(target=run, args=(i,)) for i in range(len(ranks))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    sync_all(ranks)
    return max(e[0].elapsed_time(e[1]) for e in evs) * 1000.0 / (iters * reps)


def _time_graph(ranks, launch, args, reps) -> float:
    """Per-forward time: ``reps`` forwards per captured graph (amortizes the
    graph launch for small batches), best of --rounds, slowest rank."""
    graphs = _capture(ranks, launch, reps)
    _replay_threaded_us(ranks, graphs, reps, 2)
    best = min(_replay_threaded_us(ranks, graphs, reps, args.iters) for _ in range(args.rounds))
    del graphs
    gc.collect()
    return best


def _run_threaded(ranks, fn):
    """Eager call of fn(rank) on every rank, one host thread each: a host sync
    inside one rank's call (first-call JIT, lazy buffers) must not hold back
    the launches its peers' collectives wait for."""
    import threading

    errs = []

    def run(r):
        try:
            torch.cuda.set_device(r["device"])
            fn(r)
            torch.cuda.synchronize(r["device"])
        except Exception as e:  # noqa: BLE001 - re-raised below
            errs.append(e)

    ts = [threading.Thread(target=run, args=(r,)) for r in ranks]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errs:
        raise errs[0]


def _topk_sets(ids):
    return torch.sort(ids.long().cpu(), dim=-1).values


def sweep_batches(model, ranks, args):
    """Fused (ceil(T / 4) decode launches of 4, 2, 1 tokens) against the split
    MoE for every batch size: per-batch time over one CUDA graph, slowest rank,
    and the relative L2 between the two outputs (same weights, same tokens)."""
    rows = []
    for tokens in args.batch:
        for r in ranks:
            with torch.cuda.device(r["device"]):
                if model == "kimi3":
                    x, lat = make_kimi3_inputs(tokens, args.seed + tokens, r["device"])
                    r["sw"] = dict(a=x, b=lat, fused=torch.empty(tokens, K3["H"], dtype=torch.bfloat16, device=r["device"]))
                else:
                    h, res = make_inputs(tokens, args.seed + tokens, r["device"])
                    r["sw"] = dict(a=h, b=res, fused=torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device=r["device"]))
        plan = batch_launches(tokens)

        def fused(r):
            sw = r["sw"]
            ids = []
            for off, s in plan:
                sl = slice(off, off + s)
                if model == "kimi3":
                    ret = r["k3"](sw["a"][sl], sw["b"][sl], out=sw["fused"][sl])
                else:
                    ret = r["moe"][8](sw["a"][sl], sw["b"][sl], out=sw["fused"][sl])
                ids.append(ret[-2])
            sw["fused_ids"] = ids

        def split(r):
            sw = r["sw"]
            sw["split"] = r["split"](sw["a"], sw["b"])

        if not args.no_split:
            # first use of a batch size JIT-loads its tuned kernels: do that
            # outside the collective, or a slow rank times its peers out
            _run_threaded(ranks, lambda r: r["split"](r["sw"]["a"], r["sw"]["b"], local=True))
        fns = (fused,) if args.no_split else (fused, split)
        for fn in fns:
            _run_threaded(ranks, fn)
        # routing: the split router rounds its logits to bf16 (the fused one
        # keeps f32), so near-tied experts can differ; compare the outputs of
        # the tokens both routed to the same expert set
        same_frac, diff = float("nan"), float("nan")
        if not args.no_split:
            r0 = ranks[0]
            fi = _topk_sets(torch.cat([t for t in r0["sw"]["fused_ids"]], 0))
            si = _topk_sets(r0["split"].bufs[tokens]["ids"])
            same = (fi == si).all(-1)
            same_frac = float(same.float().mean())
            if bool(same.any()):
                diff = proc_max(max(rel_l2(r["sw"]["fused"][same.to(r["device"])], r["sw"]["split"][same.to(r["device"])]) for r in ranks))
        stuck = [r["rank"] for r in ranks if (r["k3"] if model == "kimi3" else r["moe"][8]).poll_errors()]
        row = {"model": model, "quant": "a4w4" if model == "kimi3" else "a8w8", "batch": tokens, "launches": len(plan)}
        if not args.no_perf:
            reps = max(1, 32 // len(plan))
            row["fused us"] = _time_graph(ranks, fused, args, reps)
            if not args.no_split:
                row["split us"] = _time_graph(ranks, split, args, reps)
                row["speedup (split/fused)"] = row["split us"] / row["fused us"]
        row["same top-k tokens"] = same_frac
        row["fused vs split rel_l2 (same top-k)"] = diff
        row["watchdog"] = "ok" if not stuck else f"ranks {stuck}"
        if _CTX["rank0"]:
            aiter.logger.info("%s", row)
        rows.append(row)
    return rows


def _flydsl_multi_device() -> None:
    """Let every FlyDSL kernel launch on whichever GPU is current.

    A compiled FlyDSL artifact loads its code object into the device that is
    current when its execution engine is first built, and every cache above it
    (call states, ``flyc.compile`` results kept by aiter) holds that engine's
    entry point. One process driving several GPUs needs one engine per device,
    so the entry point becomes a dispatcher that builds (a pickled copy of the
    artifact, i.e. the same binary) and caches one engine per device."""
    import pickle
    import threading

    from flydsl.compiler import jit_executor, jit_function

    art_cls = jit_executor.CompiledArtifact
    lock = threading.RLock()
    jf_call = jit_function.JitFunction.__call__

    def _locked_call(self, *a, **k):
        # compiles and first launches from several rank threads at once
        with lock:
            return jf_call(self, *a, **k)

    jit_function.JitFunction.__call__ = _locked_call
    if getattr(art_cls, "_mega_moe_tp_multi_device", False):
        return
    orig = art_cls._get_func_exe

    class _PerDevice:
        def __init__(self, art):
            self.art = art
            self.home = torch.cuda.current_device()
            self.fns = {}
            self.keep = []

        def __call__(self, packed):
            dev = torch.cuda.current_device()
            fn = self.fns.get(dev)
            if fn is None:
                with lock:
                    fn = self._build(dev)
            return fn(packed)

        def _build(self, dev):
            fn = self.fns.get(dev)
            if fn is None:
                if dev == self.home:
                    fn = orig(self.art)
                else:
                    twin = pickle.loads(pickle.dumps(self.art))
                    twin._post_load_processors = list(self.art._post_load_processors)
                    fn = orig(twin)
                    self.keep.append(twin)
                self.fns[dev] = fn
            return fn

    def _get_func_exe(self):
        disp = getattr(self, "_per_device_exe", None)
        if disp is None:
            disp = _PerDevice(self)
            self._per_device_exe = disp
        return disp

    art_cls._get_func_exe = _get_func_exe
    art_cls._mega_moe_tp_multi_device = True


def main_models(args, rank, world, devices):
    """--model kimi3 accuracy (samples) and --batch sweeps for every model."""
    from p2p_collectives import P2PGroup

    _flydsl_multi_device()

    models = list(dict.fromkeys(args.model))
    moes8 = MegaMoeTpW8A8Glm.peer_group(devices) if "glm5" in models else None
    k3s = MegaMoeTpKimi3.peer_group(devices) if "kimi3" in models else None
    mmax = max([max(args.batch, default=8)] + [8])
    p2p = P2PGroup(devices, (mmax + world - 1) // world) if args.batch and not args.no_split else None
    ranks = []
    for i, dev in enumerate(devices):
        r = {"rank": i, "device": dev, "moe": {}}
        with torch.cuda.device(dev):
            if moes8 is not None:
                w = make_weights(i, args.seed, dev, [8])
                moes8[i].load_weights(**w[8])
                r["moe"][8] = moes8[i]
                r["glm5_w"] = w[8]
            if k3s is not None:
                w = make_kimi3_weights(i, args.seed, dev)
                k3s[i].load_weights(**w)
                r["k3"], r["k3_w"] = k3s[i], w
                r["k3_inputs"] = {s: make_kimi3_inputs(s, args.seed, dev) for s in args.samples}
        ranks.append(r)
    for r in ranks:
        if 8 in r["moe"]:
            r["moe"][8].warmup(K3_SUPPORTED_SAMPLES, (0,))
        if "k3" in r:
            r["k3"].warmup()
    _CTX.update(ranks=ranks, args=args, rank0=rank == 0)
    rows, failed = [], []
    if "kimi3" in models:
        for s in args.samples:
            row = test_mega_moe_tp_kimi3(s, world)
            rows.append(row)
            failed += [] if row["fly ok"] else [f"kimi3 S={s}"]
        if rank == 0:
            aiter.logger.info("kimi3 accuracy (markdown):\n%s", pd.DataFrame(rows).to_markdown(index=False))
    sweep = []
    if args.batch:
        for model in models:
            for r in ranks:
                if args.no_split:
                    continue
                with torch.cuda.device(r["device"]):
                    comm = p2p.comm(r["rank"])
                    if model == "kimi3":
                        r["split"] = SplitKimi3A4W4(r["k3_w"], comm, r["device"])
                    else:
                        r["split"] = SplitGlm5A8W8(r["glm5_w"], comm, r["device"])
            sweep += sweep_batches(model, ranks, args)
            for r in ranks:
                r.pop("split", None)
            gc.collect()
            torch.cuda.empty_cache()
        df = pd.DataFrame(sweep)
        if rank == 0:
            aiter.logger.info("mega_moe_tp vs split MoE e2e (markdown):\n%s", df.to_markdown(index=False))
            if args.csv:
                df.to_csv(args.csv, index=False)
        # fused and split quantize activations differently (kernel vs aiter
        # MXFP4 / FP8 rounding): ~0.03-0.05 apart on tokens routed alike
        failed += [
            f"{r['model']} T={r['batch']}"
            for r in sweep
            if r["watchdog"] != "ok"
            or (not args.no_split and not r["fused vs split rel_l2 (same top-k)"] < E2E_TOL)
        ]
    for r in ranks:
        if 8 in r["moe"]:
            r["moe"][8].close()
        if "k3" in r:
            r["k3"].close()
    if failed:
        raise SystemExit(f"failed: {failed}")


def perf_table(df):
    out = []
    for (samples, proto), g in df.groupby(["samples", "proto"], sort=False):
        row = {"samples": samples, "proto": proto}
        a8 = g[g["quant"] == 8]
        a4 = g[g["quant"] == 4]
        if len(a8) and "tilert us" in a8 and pd.notna(a8["tilert us"].iloc[0]):
            row["tilert a8w8"] = a8["tilert us"].iloc[0]
        if len(a8):
            row["flydsl a8w8"] = a8["fly us"].iloc[0]
        if len(a4):
            row["flydsl a4w4"] = a4["fly us"].iloc[0]
        if "tilert a8w8" in row and "flydsl a8w8" in row:
            row["tilert/flydsl a8w8"] = row["tilert a8w8"] / row["flydsl a8w8"]
        if "flydsl a8w8" in row and "flydsl a4w4" in row:
            row["flydsl a8w8/a4w4"] = row["flydsl a8w8"] / row["flydsl a4w4"]
        out.append(row)
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL fused MoE + all-reduce (GLM-5 TileRT op, Kimi-K3): accuracy and perf",
    )
    parser.add_argument("--model", nargs="*", choices=["glm5", "kimi3"], default=["glm5"],
                        help="glm5: W8A8/A4W4 vs TileRT (default); kimi3: A4W4 Kimi-K3 routed MoE.\n"
                             "With kimi3 or --batch the run checks kimi3 accuracy at --samples and\n"
                             "sweeps --batch sizes against the split MoE (router + top-k + aiter\n"
                             "fused_moe + all-reduce); glm5 there is the W8A8 kernel")
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=[],
                        help="tokens per forward for the fused vs split e2e sweep (e.g. 1 2 4 ... 2048)")
    parser.add_argument("--no-split", action="store_true", help="skip the split baseline in the sweep")
    parser.add_argument("--csv", default="", help="write the sweep table here")
    parser.add_argument("--tp", type=int, default=None,
                        help="ranks driven by this process when not under torchrun (default: all GPUs, max 8)")
    parser.add_argument("-s", "--samples", type=int, nargs="*", default=[1, 2, 4],
                        help="tokens per launch (1, 2, 4)")
    parser.add_argument("-q", "--quant", type=int, nargs="*", choices=[4, 8], default=[8],
                        help="quantization: 4 = a4w4 (MXFP4), 8 = a8w8 (FP8 block)")
    parser.add_argument("--tilert-ki", type=int, nargs="*", default=[8],
                        help="TileRT down-prefetch template variants to time (4, 6, 8, 12)")
    parser.add_argument("--tilert-hsaco", default=DEFAULT_TILERT_HSACO,
                        help="TileRT fused_moe_allreduce_w8a8_v4 code object ('' disables)")
    parser.add_argument("--proto", type=int, nargs="*", default=[0, 1],
                        help="all-reduce wire protocol: 0 = 16 B flag packets, 1 = 64 B lines")
    parser.add_argument("--reps", type=int, default=50, help="launches per captured graph")
    parser.add_argument("--iters", type=int, default=10, help="graph replays per round")
    parser.add_argument("--rounds", type=int, default=3, help="timing rounds (best is kept)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-perf", action="store_true", help="accuracy only")
    args = parser.parse_args()

    rank, nproc, local_rank = setup_dist()
    try:
        if get_gfx() not in SUPPORTED_GFX or not mega_moe_tp_w8a8_glm_supported():
            if rank == 0:
                aiter.logger.warning("mega_moe_tp unsupported on %s; skipping", get_gfx())
            return
        if "kimi3" in args.model or args.batch:
            if nproc > 1:
                raise SystemExit("--model kimi3 / --batch run in one process driving every GPU")
            world = args.tp or min(torch.cuda.device_count(), 8)
            main_models(args, rank, world, [torch.device("cuda", i) for i in range(world)])
            return
        quants = sorted(set(args.quant), reverse=True)
        multi_process = nproc > 1
        if multi_process:
            world = nproc
            devices = [torch.device("cuda", local_rank)]
            moes = {q: [MODULES[q](rank=rank, world_size=world, device=devices[0])] for q in quants}
        else:
            world = args.tp or min(torch.cuda.device_count(), 8)
            devices = [torch.device("cuda", i) for i in range(world)]
            moes = {q: MODULES[q].peer_group(devices) for q in quants}
        ranks = []
        for i, dev in enumerate(devices):
            moe = {q: moes[q][i] for q in quants}
            rk = moe[quants[0]].rank
            with torch.cuda.device(dev):
                w = make_weights(rk, args.seed, dev, quants)
                for q in quants:
                    moe[q].load_weights(**w[q])
                inputs = {s: make_inputs(s, args.seed, dev) for s in args.samples}
                fly_out = {
                    q: {s: torch.empty(s, HIDDEN, dtype=torch.bfloat16, device=dev) for s in args.samples}
                    for q in quants
                }
            ranks.append({"rank": rk, "device": dev, "w": w, "moe": moe, "inputs": inputs, "fly_out": fly_out})
        if 8 in quants and args.tilert_hsaco and os.path.isfile(args.tilert_hsaco):
            for r, k in zip(ranks, TileRtKernel.build(args.tilert_hsaco, ranks, world, multi_process)):
                r["tilert"] = k
                with torch.cuda.device(r["device"]):
                    r["tilert_outs"] = {
                        s: [
                            torch.empty(s, HIDDEN, dtype=torch.bfloat16, device=r["device"]),
                            torch.empty(s, NUM_EXPERTS, device=r["device"]),
                            torch.empty(s, SLOTS, INTER, dtype=torch.bfloat16, device=r["device"]),
                            torch.empty(s, TOP_K, device=r["device"]),
                            torch.empty(s, TOP_K, dtype=torch.int32, device=r["device"]),
                            torch.empty(s, HIDDEN, dtype=torch.bfloat16, device=r["device"]),
                        ]
                        for s in args.samples
                    }
        elif 8 in quants and rank == 0:
            aiter.logger.warning("TileRT code object not found (%s): no baseline", args.tilert_hsaco)
        for r in ranks:
            for q in quants:
                r["moe"][q].warmup(args.samples, args.proto)
        _CTX.update(ranks=ranks, args=args, rank0=rank == 0)
        rows = []
        for samples, proto, quant in itertools.product(args.samples, args.proto, quants):
            barrier()
            rows.append(test_mega_moe_tp_glm(samples, world, proto, quant))
        if rank == 0:
            df = pd.DataFrame(rows)
            aiter.logger.info("mega_moe_tp summary (markdown):\n%s", df.to_markdown(index=False))
            if not args.no_perf:
                aiter.logger.info("perf (us):\n%s", perf_table(df).to_markdown(index=False))
        failed = [r for r in rows if not r["fly ok"]]
        barrier()
        for r in ranks:
            for q in quants:
                r["moe"][q].close()
            if r.get("tilert") is not None:
                r["tilert"].close()
        if failed:
            raise SystemExit(f"{len(failed)} case(s) failed accuracy")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
