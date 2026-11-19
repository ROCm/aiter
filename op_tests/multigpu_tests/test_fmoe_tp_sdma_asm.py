# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Multi-GPU correctness, epoch-wrap, poisoned-buffer, and CUDA-Graph tests."""

import os
import socket
import subprocess
import sys

import pytest
import torch
import torch.multiprocessing as mp

_REPO_ROOT = os.path.abspath(f"{__file__}/../../..")
sys.path.insert(0, _REPO_ROOT)
sys.modules["flydsl"] = None
EPS = 1e-6
EPI_ADD = (("add_rmsnorm", False), ("add_rmsnorm", True))
EPI_K3 = (("rmsnorm_only", False), ("rmsnorm_only", True), *EPI_ADD)
EPI_RAW = (("raw_bf16", False), *EPI_ADD)
SHAPES = (
    ("q38", 64, 10, 8192, 256, 256, "fp4", "silu", EPI_ADD),
    ("k3", 128, 16, 3584, 384, 384, "fp8", "situ", EPI_K3),
    ("flash_next", 64, 10, 2560, 384, 256, "fp4", "silu", EPI_RAW),
    ("v4_flash", 64, 6, 4096, 256, 256, "fp4", "silu", EPI_RAW),
)


@torch.no_grad()
def _setup(E, D, I, a_dt, act, tp, rank, dev):
    import aiter as ai
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility import fp4_utils

    def dq(q, s, fp8=False):
        u8 = q.view(torch.uint8)
        v = q.view(torch.float8_e4m3fn).float() if fp8 else fp4_utils.mxfp4_to_f32(u8)
        sc = fp4_utils.e8m0_to_f32(s.view(torch.uint8)).view(*v.shape[:-1], -1)
        return (v.view(*v.shape[:-1], -1, 32) * sc.unsqueeze(-1)).view(v.shape)

    gen = torch.Generator(device=dev).manual_seed(2000 + 37 * rank + D + E)
    rn = lambda *s: torch.randn(*s, generator=gen, device=dev)
    hot = torch.ones(D, device=dev)
    hot[torch.randint(0, D, (8,), generator=gen, device=dev)] = 6.0
    w1_q, w1_s = ai.per_1x32_f4_quant((rn(E, 2 * I, D) * (D**-0.5)).bfloat16())
    r2 = rn(E, D, I) * ((I * tp) ** -0.5) * hot.view(1, D, 1)
    w2_q, w2_s = ai.per_1x32_f4_quant(r2.bfloat16())
    w1_dq = dq(w1_q, w1_s.view(E, 2 * I, D // 32))
    w2_dq = dq(w2_q, w2_s.view(E, D, I // 32))

    def oracle(x_b, ids, wts):
        q8 = ai.per_1x32_f8_scale_f8_quant
        qa = (x_b, None, ai.dtypes.fp8, ai.dtypes.fp8_e8m0, False)
        xq, xs = q8(*qa) if a_dt == "fp8" else ai.per_1x32_f4_quant(x_b)
        xf = dq(xq, xs, fp8=(a_dt == "fp8"))
        out = torch.zeros((x_b.shape[0], D), dtype=torch.float32, device=dev)
        for e in range(E):
            ti, si = torch.nonzero(ids.long() == e, as_tuple=True)
            if ti.numel() == 0:
                continue
            gu = xf.index_select(0, ti) @ w1_dq[e].t()
            g, u = gu[:, :I], gu[:, I:]
            gp = 4.0 * torch.tanh(g / 4.0) * torch.sigmoid(g)
            h = gp * (25.0 * torch.tanh(u / 25.0))
            if act != "situ":
                h = torch.nn.functional.silu(g) * u
            hq, hs = ai.per_1x32_f4_quant(h.contiguous())
            ye = (dq(hq, hs) * wts[ti, si].unsqueeze(-1)) @ w2_dq[e].t()
            out.index_add_(0, ti, ye)
        return out

    w1_ss = fp4_utils.e8m0_shuffle(w1_s.view(E * 2 * I, D // 32))
    w2_ss = fp4_utils.e8m0_shuffle(w2_s.view(E * D, I // 32))
    w1_sh, w2_sh = shuffle_weight(w1_q, (16, 16)), shuffle_weight(w2_q, (16, 16))
    return w1_sh, w2_sh, w1_ss, w2_ss, oracle


def _worker(rank: int, tp: int, init_m: str, full: bool) -> None:
    sys.modules["flydsl"] = None
    import torch.distributed as dist

    from aiter.ops.asm import TpAsmContext, TpMoeCall, tp_moe_layer_asm

    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", init_m, world_size=tp, rank=rank, device_id=dev)
    gloo = dist.new_group(backend="gloo")
    sync = lambda: (torch.cuda.synchronize(dev), dist.barrier(group=gloo))

    def rrms(a, b) -> float:
        d, n = (a.float() - b.float()).pow(2).sum(), b.float().pow(2).sum()
        return float((d / n.clamp_min(1e-12)).sqrt().item())

    def check(out, ro, moe, res, nw, epi, gm) -> None:
        assert bool(torch.isfinite(out).all())
        sc = nw.float() + float(gm)
        norm = lambda z: z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + EPS) * sc
        if epi == "add_rmsnorm":
            rr = moe + res.float()
            assert ro is not None and bool(torch.isfinite(ro).all())
            assert rrms(ro, rr) <= 0.08
            ref_o, tol = norm(rr), 0.08
        else:
            ref_o, tol = (moe if epi == "raw_bf16" else norm(moe)), 0.14
        assert rrms(out, ref_o) <= tol
        oi = out.contiguous().view(torch.int32)
        peers = [torch.empty_like(oi) for _ in range(tp)]
        dist.all_gather(peers, oi)
        assert all(torch.equal(oi, p) for p in peers)

    try:
        for sname, E, k, D, i2, i4, a_dt, act, epi_cases in SHAPES:
            I = i2 if tp == 2 else i4
            odd = (16, 517) if sname in ("q38", "flash_next") else (256, 1021)
            toks = odd if full else (256, 517)
            w1, w2, w1_ss, w2_ss, oracle = _setup(E, D, I, a_dt, act, tp, rank, dev)
            ctx = TpAsmContext(gloo, dev, max(toks), D, k, E, I)
            try:
                for T in toks:
                    gen = torch.Generator(device=dev).manual_seed(998 + T + D)
                    rn = lambda *s, g=gen: torch.randn(*s, generator=g, device=dev)
                    xa, res = rn(T, D).bfloat16(), (rn(T, D) * 0.5).bfloat16()
                    res[:, :2] = (res[:, :2].float() * 12.0).bfloat16()
                    vals, idx = torch.topk(rn(T, E), k, -1)
                    ids = idx.to(torch.int32).contiguous()
                    wts = torch.softmax(vals, -1).float().contiguous()
                    nw = (1.0 + 0.2 * rn(D)).bfloat16()
                    xb = (xa.float() * -0.7).bfloat16()
                    la, lb = oracle(xa, ids, wts), oracle(xb, ids, wts)
                    for t in (la, lb):
                        dist.all_reduce(t)
                    for epi, gm in epi_cases:
                        is_add = epi == "add_rmsnorm"
                        ob = torch.full_like(xa, float("nan"))
                        rb = torch.full_like(res, float("nan")) if is_add else None
                        for t in (*ctx.p8_wins, *ctx.s8_wins, ctx.hq):
                            t.fill_(0xFF)
                        ctx.poison_ipc()
                        sync()
                        kw = {"residual_out": rb, "a_dtype": a_dt, "act": act}
                        ri = res if is_add else None
                        wi = nw if epi != "raw_bf16" else None
                        for xi, li in ((xa, la), (xb, lb)):
                            ca = (xi, ids, wts, w1, w2, w1_ss, w2_ss, ri, wi, EPS, gm)
                            r = tp_moe_layer_asm(TpMoeCall(*ca), ctx, epi, ob, **kw)
                            ctx.check_error()
                            check(r.out, r.residual_out, li, res, nw, epi, gm)
                            sync()
                epi, gm = epi_cases[0]
                ob = torch.empty_like(xa)
                rb = torch.empty_like(res) if epi == "add_rmsnorm" else None
                kw = {"residual_out": rb, "a_dtype": a_dt, "act": act}
                cg = TpMoeCall(xa, ids, wts, w1, w2, w1_ss, w2_ss, res, nw, EPS, gm)
                if sname == "q38":
                    for ep0 in (0x7FFFFFFE, -2):
                        sync()
                        ctx.epoch.fill_(ep0)
                        sync()
                        for _ in range(4):
                            r = tp_moe_layer_asm(cg, ctx, epi, ob, **kw)
                            ctx.check_error()
                            check(r.out, r.residual_out, la, res, nw, epi, gm)
                tp_moe_layer_asm(cg, ctx, epi, ob, **kw)
                sync()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    tp_moe_layer_asm(cg, ctx, epi, ob, **kw)
                xa.copy_((xa.float() * -0.6).bfloat16())
                res.copy_((res.float() * 0.8).bfloat16())
                lg2 = oracle(xa, ids, wts)
                dist.all_reduce(lg2)
                for buf in (b for b in (ob, rb) if b is not None):
                    buf.fill_(float("nan"))
                sync()
                for _ in range(3):
                    g.replay()
                    ctx.check_error()
                    check(ob, rb, lg2, res, nw, epi, gm)
                    sync()
                del g
            finally:
                ctx.close()
                sync()
    finally:
        dist.destroy_process_group()
    assert sys.modules["flydsl"] is None


def test_zero_flydsl_import() -> None:
    c1 = "import sys; sys.modules['flydsl']=None; from aiter.ops.asm import tp_moe_layer_asm;"
    c2 = "from aiter.ops.asm.fmoe_tp_sdma import load_fmoe_tp_sdma_registry as r; assert len(r())==36"
    env = {**os.environ, "PYTHONPATH": f"{_REPO_ROOT}:{os.getenv('PYTHONPATH', '')}"}
    subprocess.run([sys.executable, "-c", f"{c1}{c2}"], check=True, env=env)


@pytest.mark.parametrize(("tp", "full"), [(2, True), (4, False)])
def test_fmoe_tp_sdma_asm(tp: int, full: bool) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < tp:
        pytest.skip(f"requires >= {tp} GPUs")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0].lower()
    if arch != "gfx950":
        pytest.skip("fmoe_tp_sdma requires gfx950")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        init = f"tcp://127.0.0.1:{sock.getsockname()[1]}"
    mp.spawn(_worker, args=(tp, init, full), nprocs=tp, join=True)
