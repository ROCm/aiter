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

import pandas as pd
import torch
import torch.distributed as dist

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose

from aiter.ops.flydsl.fused_moe_allreduce import (
    NUM_MOE_WEIGHTS,
    FusedMoeAllreduceW8A8,
    UncachedSymmetricBuffer,
    fused_moe_allreduce_w8a8_supported,
)
from aiter.ops.flydsl.kernels.fused_moe_allreduce.fused_moe_allreduce_w8a8 import (
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
DEFAULT_TILERT_HSACO = os.path.join(
    _REPO_ROOT,
    "..",
    "tile_rt",
    "fused_moe_allreduce_w8a8_v4",
    "kernel",
    "fused_moe_allreduce_w8a8_v4.gfx950.hsaco",
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


def make_weights(rank: int, seed: int, device):
    g_rep = torch.Generator(device=device).manual_seed(seed)
    g_rank = torch.Generator(device=device).manual_seed(seed + 1000 * (rank + 1))

    def rn(gen, *shape, std=1.0):
        return torch.randn(*shape, generator=gen, device=device) * std

    w = {
        "router_w": rn(g_rep, NUM_EXPERTS, HIDDEN, std=0.02).to(torch.bfloat16),
        "gamma": 1.0 + 0.1 * rn(g_rep, HIDDEN),
        "bias": 0.01 * rn(g_rep, NUM_EXPERTS),
    }
    ug_w = torch.empty(NUM_MOE_WEIGHTS, 2 * INTER, HIDDEN, dtype=torch.float8_e4m3fn, device=device)
    ug_s = torch.empty(NUM_MOE_WEIGHTS, 2 * INTER // 128, HIDDEN // 128, device=device)
    dn_w = torch.empty(NUM_MOE_WEIGHTS, HIDDEN, INTER, dtype=torch.float8_e4m3fn, device=device)
    dn_s = torch.empty(NUM_MOE_WEIGHTS, HIDDEN // 128, INTER // 128, device=device)
    for e0 in range(0, NUM_MOE_WEIGHTS, 32):
        e1 = min(NUM_MOE_WEIGHTS, e0 + 32)
        ug_w[e0:e1], ug_s[e0:e1] = quantize_fp8_block(rn(g_rank, e1 - e0, 2 * INTER, HIDDEN, std=0.02))
        dn_w[e0:e1], dn_s[e0:e1] = quantize_fp8_block(rn(g_rank, e1 - e0, HIDDEN, INTER, std=0.02))
    w.update(ug_w=ug_w, ug_scales=ug_s, down_w=dn_w, down_scales=dn_s)
    return w


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


def run_torch(hidden, w):
    s_n = hidden.shape[0]
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
    w_slot = torch.cat([torch.ones_like(probs[:, :1]), probs], 1)
    facd = (
        w["down_scales"][experts].repeat_interleave(128, dim=2)
        * w_slot[:, :, None, None]
        * m_sc[:, :, None, :]
    )
    partial = (partd * facd).sum(-1).sum(1).to(torch.bfloat16)
    return norm, logits, probs, idx.to(torch.int32), mid, partial


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


def time_fly(ranks, samples, args) -> float:
    hidden = {id(r): r["inputs"][samples] for r in ranks}

    def launch(r):
        h, res = hidden[id(r)]
        r["moe"](h, res, out=r["fly_out"][samples])

    graphs = _capture(ranks, launch, args.reps)
    _replay_us(ranks, graphs, args.reps, 2)
    best = min(_replay_us(ranks, graphs, args.reps, args.iters) for _ in range(args.rounds))
    del graphs
    gc.collect()
    return best


def time_tilert(ranks, samples, proto, ki, args) -> float:
    def launch(r):
        h, res = r["inputs"][samples]
        r["tilert"].launch(r["moe"], h, res, r["tilert_outs"][samples], proto, ki)

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


def _bytes_moved(samples: int) -> int:
    per_token = NUM_EXPERTS * HIDDEN * 2 + SLOTS * (2 * INTER * HIDDEN + HIDDEN * INTER)
    return samples * per_token


def _flops(samples: int) -> int:
    per_token = 2 * NUM_EXPERTS * HIDDEN + SLOTS * 2 * (2 * INTER * HIDDEN + HIDDEN * INTER)
    return samples * per_token


_CTX: dict = {}


@benchmark()
def test_fused_moe_allreduce(samples, tp, proto):
    ranks, args = _CTX["ranks"], _CTX["args"]
    names = ("norm", "scores", "mid", "probs", "indices", "out")
    ret = {"gfx": get_gfx()}
    for r in ranks:
        r["moe"].proto = proto
        with torch.cuda.device(r["device"]):
            h, res = r["inputs"][samples]
            r["got"] = r["moe"](h, res, out=r["fly_out"][samples])
    sync_all(ranks)

    refs = []
    for r in ranks:
        with torch.cuda.device(r["device"]):
            refs.append(run_torch(r["inputs"][samples][0], r["w"]))
    parts = all_partials(ranks, [ref[5] for ref in refs])
    out_ref = reference_out(parts, ranks[0]["inputs"][samples][1])
    ok, worst = True, 0.0
    for r, ref in zip(ranks, refs):
        norm, scores, mid, probs, indices, out = r["got"]
        ok &= torch.equal(norm, ref[0]) and torch.equal(indices, ref[3])
        ok &= rel_l2(scores, ref[1]) < 1e-5 and rel_l2(probs, ref[2]) < 1e-5
        ok &= rel_l2(mid, ref[4]) < 5e-3
        worst = max(worst, rel_l2(out, out_ref))
        stuck = r["moe"].poll_errors()
        if stuck:
            aiter.logger.error("[rank %d] poll watchdog fired: %s", r["rank"], stuck)
        ok &= not stuck
    err = checkAllclose(
        out_ref.float(), ranks[0]["got"][5].float().cpu(), rtol=2e-2, atol=2e-2,
        msg=f"fly out S={samples} proto={proto}", printLog=_CTX["rank0"],
    )
    ret["fly err"] = proc_max(err)
    ret["fly out rel_l2"] = proc_max(worst)
    ret["fly ok"] = bool(proc_min(float(ok and err < 0.05 and worst < 1e-2)) > 0)

    if not args.no_perf:
        us = time_fly(ranks, samples, args)
        if any(r["moe"].poll_errors() for r in ranks):
            aiter.logger.error("poll watchdog fired while timing")
            ret["fly ok"] = False
        ret["fly us"] = us
        ret["fly TB/s"] = _bytes_moved(samples) / us / 1e6
        ret["fly TFLOPS"] = _flops(samples) / us / 1e6

    if ranks[0].get("tilert") is not None:
        best_ki, best_us = None, float("inf")
        for ki in args.tilert_ki:
            for r in ranks:
                with torch.cuda.device(r["device"]):
                    h, res = r["inputs"][samples]
                    r["tilert"].launch(r["moe"], h, res, r["tilert_outs"][samples], proto, ki)
            sync_all(ranks)
            same = {
                n: min(float((a == b).float().mean()) for r in ranks for a, b in [(r["got"][i], r["tilert_outs"][samples][i])])
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
            ret["tilert TB/s"] = _bytes_moved(samples) / best_us / 1e6
            ret["fly/tilert"] = ret["fly us"] / best_us
    return ret


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL fused MoE + all-reduce (GLM-5 W8A8, TileRT op): accuracy and perf",
    )
    parser.add_argument("--tp", type=int, default=None,
                        help="ranks driven by this process when not under torchrun (default: all GPUs, max 8)")
    parser.add_argument("-s", "--samples", type=int, nargs="*", default=[1, 2, 4],
                        help="tokens per launch (1, 2, 4)")
    parser.add_argument("--proto", type=int, nargs="*", default=[0, 1],
                        help="all-reduce wire protocol: 0 = 16 B flag packets, 1 = 64 B lines")
    parser.add_argument("--tilert-ki", type=int, nargs="*", default=[8],
                        help="TileRT down-prefetch template variants to time (4, 6, 8, 12)")
    parser.add_argument("--tilert-hsaco", default=DEFAULT_TILERT_HSACO,
                        help="TileRT fused_moe_allreduce_w8a8_v4 code object ('' disables)")
    parser.add_argument("--reps", type=int, default=50, help="launches per captured graph")
    parser.add_argument("--iters", type=int, default=10, help="graph replays per round")
    parser.add_argument("--rounds", type=int, default=3, help="timing rounds (best is kept)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-perf", action="store_true", help="accuracy only")
    args = parser.parse_args()

    rank, nproc, local_rank = setup_dist()
    try:
        if get_gfx() not in SUPPORTED_GFX or not fused_moe_allreduce_w8a8_supported():
            if rank == 0:
                aiter.logger.warning("fused_moe_allreduce_w8a8 unsupported on %s; skipping", get_gfx())
            return
        multi_process = nproc > 1
        if multi_process:
            world = nproc
            devices = [torch.device("cuda", local_rank)]
            moes = [FusedMoeAllreduceW8A8(rank=rank, world_size=world, device=devices[0])]
        else:
            world = args.tp or min(torch.cuda.device_count(), 8)
            devices = [torch.device("cuda", i) for i in range(world)]
            moes = FusedMoeAllreduceW8A8.peer_group(devices)
        ranks = []
        for moe, dev in zip(moes, devices):
            rk = moe.rank
            with torch.cuda.device(dev):
                w = make_weights(rk, args.seed, dev)
                moe.load_weights(**w)
                inputs = {s: make_inputs(s, args.seed, dev) for s in args.samples}
                fly_out = {s: torch.empty(s, HIDDEN, dtype=torch.bfloat16, device=dev) for s in args.samples}
            ranks.append({"rank": rk, "device": dev, "w": w, "moe": moe, "inputs": inputs, "fly_out": fly_out})
        if args.tilert_hsaco and os.path.isfile(args.tilert_hsaco):
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
        elif rank == 0:
            aiter.logger.warning("TileRT code object not found (%s): no baseline", args.tilert_hsaco)
        for r in ranks:
            r["moe"].warmup(args.samples, args.proto)
        _CTX.update(ranks=ranks, args=args, rank0=rank == 0)
        rows = []
        for samples, proto in itertools.product(args.samples, args.proto):
            barrier()
            rows.append(test_fused_moe_allreduce(samples, world, proto))
        if rank == 0:
            df = pd.DataFrame(rows)
            aiter.logger.info("fused_moe_allreduce_w8a8 summary (markdown):\n%s", df.to_markdown(index=False))
        failed = [r for r in rows if not r["fly ok"]]
        barrier()
        for r in ranks:
            r["moe"].close()
            if r.get("tilert") is not None:
                r["tilert"].close()
        if failed:
            raise SystemExit(f"{len(failed)} case(s) failed accuracy")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
