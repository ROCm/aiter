# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Multi-GPU correctness, pre-quant, bm32, quant-epi, 2buf-ctrl, and graph tests."""

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
EPI_Q38 = (*EPI_ADD, ("add_rmsnorm_fp4", False), ("add_rmsnorm_fp4", True))
EPI_K3 = (
    ("rmsnorm_only", False),
    ("rmsnorm_only", True),
    *EPI_ADD,
    ("rmsnorm_only_fp8", False),
    ("rmsnorm_only_fp8", True),
    ("add_rmsnorm_fp8", False),
    ("add_rmsnorm_fp8", True),
)
EPI_RAW = (("raw_bf16", False), *EPI_ADD, ("add_rmsnorm_fp4", False))
SHAPES = (
    ("q38", 64, 10, 8192, 256, 256, "fp4", "silu", EPI_Q38),
    ("k3", 128, 16, 3584, 384, 384, "fp8", "situ", EPI_K3),
    ("flash_next", 64, 10, 2560, 384, 256, "fp4", "silu", EPI_RAW),
    ("v4_flash", 64, 6, 4096, 256, 256, "fp4", "silu", EPI_RAW),
)


def _dq_mx(q: torch.Tensor, s: torch.Tensor, fp8: bool = False) -> torch.Tensor:
    from aiter.utility import fp4_utils

    u8 = q.view(torch.uint8)
    v = q.view(torch.float8_e4m3fn).float() if fp8 else fp4_utils.mxfp4_to_f32(u8)
    sc = fp4_utils.e8m0_to_f32(s.view(torch.uint8)).view(*v.shape[:-1], -1)
    return (v.view(*v.shape[:-1], -1, 32) * sc.unsqueeze(-1)).view(v.shape)


@torch.no_grad()
def _setup(E, D, I, a_dt, act, tp, rank, dev):
    import aiter as ai
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility import fp4_utils

    gen = torch.Generator(device=dev).manual_seed(2000 + 37 * rank + D + E)
    rn = lambda *s: torch.randn(*s, generator=gen, device=dev)
    hot = torch.ones(D, device=dev)
    hot[torch.randint(0, D, (8,), generator=gen, device=dev)] = 6.0
    w1_q, w1_s = ai.per_1x32_f4_quant((rn(E, 2 * I, D) * (D**-0.5)).bfloat16())
    r2 = rn(E, D, I) * ((I * tp) ** -0.5) * hot.view(1, D, 1)
    w2_q, w2_s = ai.per_1x32_f4_quant(r2.bfloat16())
    w1_dq = _dq_mx(w1_q, w1_s.view(E, 2 * I, D // 32))
    w2_dq = _dq_mx(w2_q, w2_s.view(E, D, I // 32))

    def oracle(x_b, ids, wts):
        q8 = ai.per_1x32_f8_scale_f8_quant
        qa = (x_b, None, ai.dtypes.fp8, ai.dtypes.fp8_e8m0, False)
        xq, xs = q8(*qa) if a_dt == "fp8" else ai.per_1x32_f4_quant(x_b)
        xf = _dq_mx(xq, xs, fp8=(a_dt == "fp8"))
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
            ye = (_dq_mx(hq, hs) * wts[ti, si].unsqueeze(-1)) @ w2_dq[e].t()
            out.index_add_(0, ti, ye)
        return out

    w1_ss = fp4_utils.e8m0_shuffle(w1_s.view(E * 2 * I, D // 32))
    w2_ss = fp4_utils.e8m0_shuffle(w2_s.view(E * D, I // 32))
    w1_sh, w2_sh = shuffle_weight(w1_q, (16, 16)), shuffle_weight(w2_q, (16, 16))
    return w1_sh, w2_sh, w1_ss, w2_ss, oracle


def _worker(rank: int, tp: int, init_m: str, full: bool) -> None:
    sys.modules["flydsl"] = None
    import torch.distributed as dist

    from aiter import dtypes
    from aiter.ops.asm import (
        TpAsmContext,
        TpMoeCall,
        quantize_moe_input,
        tp_moe_layer_asm,
    )

    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", init_m, world_size=tp, rank=rank, device_id=dev)
    gloo = dist.new_group(backend="gloo")
    sync = lambda: (torch.cuda.synchronize(dev), dist.barrier(group=gloo))

    def rrms(a, b) -> float:
        d, n = (a.float() - b.float()).pow(2).sum(), b.float().pow(2).sum()
        return float((d / n.clamp_min(1e-12)).sqrt().item())

    def alloc_epi(t_cnt: int, d_cnt: int, epi: str):
        is_add = epi.startswith("add_rmsnorm")
        rb = (
            torch.full((t_cnt, d_cnt), float("nan"), dtype=dtypes.bf16, device=dev)
            if is_add
            else None
        )
        if epi.endswith("_fp4"):
            ob = torch.full((t_cnt, d_cnt // 2), 0xFF, dtype=torch.uint8, device=dev)
            ob = ob.view(dtypes.fp4x2)
            sb = torch.full((t_cnt, d_cnt // 32), 0xFF, dtype=torch.uint8, device=dev)
            sb = sb.view(dtypes.fp8_e8m0)
        elif epi.endswith("_fp8"):
            ob = torch.full((t_cnt, d_cnt), 0x7F, dtype=torch.uint8, device=dev)
            ob = ob.view(dtypes.fp8)
            sb = torch.full((t_cnt, d_cnt // 32), 0xFF, dtype=torch.uint8, device=dev)
            sb = sb.view(dtypes.fp8_e8m0)
        else:
            ob = torch.full((t_cnt, d_cnt), float("nan"), dtype=dtypes.bf16, device=dev)
            sb = None
        return ob, rb, sb

    def check(out, ro, os_, moe, res, nw, epi, gm) -> None:
        is_add = epi.startswith("add_rmsnorm")
        is_fp4_o, is_fp8_o = epi.endswith("_fp4"), epi.endswith("_fp8")
        if is_fp4_o or is_fp8_o:
            assert os_ is not None and os_.shape == (moe.shape[0], moe.shape[1] // 32)
            assert bool((os_.view(torch.uint8) != 0xFF).all())
            out_f = _dq_mx(out, os_, fp8=is_fp8_o)
            assert bool(torch.isfinite(out_f).all())
        else:
            assert os_ is None and bool(torch.isfinite(out).all())
            out_f = out.float()
        sc = nw.float() + float(gm)
        norm = lambda z: z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + EPS) * sc
        if is_add:
            rr = moe + res.float()
            assert ro is not None and ro.dtype == dtypes.bf16
            assert bool(torch.isfinite(ro).all()) and rrms(ro, rr) <= 0.08
            ref_o, tol = norm(rr), (0.14 if is_fp4_o else (0.10 if is_fp8_o else 0.08))
        else:
            ref_o = moe if epi == "raw_bf16" else norm(moe)
            tol = 0.16 if is_fp4_o else 0.14
        assert rrms(out_f, ref_o) <= tol
        for buf in (out, *(() if os_ is None else (os_,))):
            bi = buf.contiguous().view(torch.uint8)
            peers = [torch.empty_like(bi) for _ in range(tp)]
            dist.all_gather(peers, bi)
            assert all(torch.equal(bi, p) for p in peers)

    try:
        for sname, E, k, D, i2, i4, a_dt, act, epi_cases in SHAPES:
            I = i2 if tp == 2 else i4
            odd = (16, 517) if sname in ("q38", "flash_next") else (256, 1021)
            med_t = 2048 if full else 1024
            toks = odd if full else (256, 517)
            w1, w2, w1_ss, w2_ss, oracle = _setup(E, D, I, a_dt, act, tp, rank, dev)
            ctx = TpAsmContext(gloo, dev, max(*toks, med_t), D, k, E, I)
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
                        is_add = epi.startswith("add_rmsnorm")
                        is_norm = not epi.startswith("raw_bf16")
                        ob, rb, sb = alloc_epi(T, D, epi)
                        for t in (*ctx.p8_wins, *ctx.s8_wins, ctx.hq):
                            t.fill_(0xFF)
                        ctx.poison_ipc()
                        sync()
                        kw = {"residual_out": rb, "a_dtype": a_dt, "act": act}
                        ri, wi = (res if is_add else None), (nw if is_norm else None)
                        for xi, li in ((xa, la), (xb, lb)):
                            ca = (xi, ids, wts, w1, w2, w1_ss, w2_ss, ri, wi, EPS, gm)
                            r = tp_moe_layer_asm(TpMoeCall(*ca), ctx, epi, ob, sb, **kw)
                            ctx.check_error()
                            check(
                                r.out, r.residual_out, r.out_scale, li, res, nw, epi, gm
                            )
                            sync()

                # 1. Explicit pre-quantized (x_q, x_scale) + tile_m=32 medium-batch test
                gen_m = torch.Generator(device=dev).manual_seed(4099 + med_t + D)
                rnm = lambda *s, g=gen_m: torch.randn(*s, generator=g, device=dev)
                xm, resm = rnm(med_t, D).bfloat16(), (rnm(med_t, D) * 0.5).bfloat16()
                vm, im = torch.topk(rnm(med_t, E), k, -1)
                idsm = im.to(torch.int32).contiguous()
                wtsm = torch.softmax(vm, -1).float().contiguous()
                nwm = (1.0 + 0.2 * rnm(D)).bfloat16()
                lm = oracle(xm, idsm, wtsm)
                dist.all_reduce(lm)
                xmq, xms = quantize_moe_input(xm, a_dt)
                assert xmq.shape == (med_t, D // 2 if a_dt == "fp4" else D)
                assert xms.shape == (med_t, D // 32) and xms.is_contiguous()
                epi_m, gm_m = epi_cases[0]
                obm, rbm, sbm = alloc_epi(med_t, D, epi_m)
                rim = resm if epi_m.startswith("add_rmsnorm") else None
                wim = nwm if epi_m != "raw_bf16" else None
                pq_call = TpMoeCall(
                    xmq,
                    idsm,
                    wtsm,
                    w1,
                    w2,
                    w1_ss,
                    w2_ss,
                    rim,
                    wim,
                    EPS,
                    gm_m,
                    x_scale=xms,
                )
                from aiter.ops.asm.fmoe_tp_sdma import load_fmoe_tp_sdma_registry

                reg = load_fmoe_tp_sdma_registry()
                tms = [
                    tm_c
                    for tm_c in (32, 64)
                    if (tp, D, I, k, tm_c, a_dt, act, epi_m, int(gm_m)) in reg
                ]
                assert 64 in tms
                for tm_try in tms:
                    rm = tp_moe_layer_asm(
                        pq_call,
                        ctx,
                        epi_m,
                        obm,
                        sbm,
                        residual_out=rbm,
                        a_dtype=a_dt,
                        act=act,
                        tile_m=tm_try,
                    )
                    ctx.check_error()
                    check(
                        rm.out,
                        rm.residual_out,
                        rm.out_scale,
                        lm,
                        resm,
                        nwm,
                        epi_m,
                        gm_m,
                    )
                    sync()

                # 2. Epoch-wrap + >= 4 back-to-back unsynced replays (2-buffer ctrl without zero_)
                epi, gm = epi_cases[0]
                ob, rb, sb = alloc_epi(xa.shape[0], D, epi)
                kw = {"residual_out": rb, "a_dtype": a_dt, "act": act}
                ri = res if epi.startswith("add_rmsnorm") else None
                wi = nw if epi != "raw_bf16" else None
                cg = TpMoeCall(xa, ids, wts, w1, w2, w1_ss, w2_ss, ri, wi, EPS, gm)
                if sname == "q38":
                    for ep0 in (0x7FFFFFFE, -2):
                        sync()
                        ctx.reset_epoch(ep0)
                        sync()
                        for _ in range(4):
                            r = tp_moe_layer_asm(cg, ctx, epi, ob, sb, **kw)
                            ctx.check_error()
                            check(
                                r.out, r.residual_out, r.out_scale, la, res, nw, epi, gm
                            )
                tp_moe_layer_asm(cg, ctx, epi, ob, sb, **kw)
                sync()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    tp_moe_layer_asm(cg, ctx, epi, ob, sb, **kw)
                xa.copy_((xa.float() * -0.6).bfloat16())
                res.copy_((res.float() * 0.8).bfloat16())
                lg2 = oracle(xa, ids, wts)
                dist.all_reduce(lg2)
                for buf in (b for b in (ob, rb) if b is not None):
                    buf.fill_(float("nan"))
                sync()
                for _ in range(5):
                    g.replay()
                ctx.check_error()
                check(ob, rb, sb, lg2, res, nw, epi, gm)
                sync()
                del g
            finally:
                ctx.close()
                sync()
            if sname == "q38":
                _check_s4_disjoint_slots(gloo, dev, 64, D, k, E, I)
                sync()
    finally:
        dist.destroy_process_group()
    assert sys.modules["flydsl"] is None


def test_zero_flydsl_import() -> None:
    c1 = (
        "import sys; sys.modules['flydsl']=None; "
        "from aiter.ops.asm import EPI_MODES, TpAsmArenaLease, TpMoeCall, TpMoeOut, quantize_moe_input, tp_moe_layer_asm; "
    )
    c2 = (
        "from aiter.ops.asm.fmoe_tp_sdma import load_fmoe_tp_sdma_registry as r; "
        "assert len(EPI_MODES)==7 and len(r())==108"
    )
    env = {**os.environ, "PYTHONPATH": f"{_REPO_ROOT}:{os.getenv('PYTHONPATH', '')}"}
    subprocess.run([sys.executable, "-c", f"{c1}{c2}"], check=True, env=env)


def test_gate0_host_bounds_and_policies() -> None:
    from types import SimpleNamespace
    from unittest.mock import patch

    import aiter.ops.asm.fmoe_tp_sdma as mod

    assert mod._CTRL_LAYOUT_VERSION >= 2
    assert mod._CTRL_HRDY_BASE == 5120
    assert mod._CTRL_MAX_HRDY_TILES == 35840
    assert mod._CTRL_WORDS == 40960
    assert mod._CTRL_TOTAL_WORDS == 81920

    # Odd-engine filtering (ROCM-32598): 0xfffc keeps even bits >= 2; 0x4 -> [2]; 0x8 -> []
    assert mod._select_even_sdma_engines(0xFFFC)[0] == 2
    assert all(b % 2 == 0 for b in mod._select_even_sdma_engines(0xFFFC))
    assert mod._select_even_sdma_engines(0x4) == [2]
    assert mod._select_even_sdma_engines(0x8) == []

    # _open_sdma must fail closed when only odd SDMA engines exist
    fake_lib = SimpleNamespace(
        xs_open=lambda: 0,
        xs_node_for_gpu_id=lambda gid, hk: 0,
    )
    st = SimpleNamespace(queues=[], sdma_open=False, dev=torch.device("cpu"))
    with (
        patch.object(mod, "_hsakmt_lib", return_value=fake_lib),
        patch.object(mod, "_alloc_uncached", return_value=4096),
        patch.object(mod.os.path, "isdir", return_value=True),
        patch.object(mod.os, "listdir", return_value=["0"]),
        patch.object(
            mod,
            "_kfd_props",
            return_value={"node_to": 2, "recommended_sdma_engine_id_mask": 0x8},
        ),
        pytest.raises(RuntimeError, match="no even gfx950 SDMA engine"),
    ):
        mod._open_sdma(st, 0, 1, 100, [1, 2])

    # BM=32 auto-dispatch policy: 28*E default, flash_next (2560,256,10) up to 4096
    mk = lambda tm: (4, 3584, 384, 16, tm, "fp8", "situ", "add_rmsnorm", 0)
    reg = {mk(32): ("a.co", "a"), mk(64): ("b.co", "b")}
    assert mod._select_tile_m(224, 128, 3584, 384, 16, reg, mk, "auto") == 32
    assert mod._select_tile_m(225, 128, 3584, 384, 16, reg, mk, "auto") == 64
    mk_fn = lambda tm: (4, 2560, 256, 10, tm, "fp4", "silu", "add_rmsnorm", 0)
    reg_fn = {mk_fn(32): ("c.co", "c"), mk_fn(64): ("d.co", "d")}
    assert mod._select_tile_m(4096, 64, 2560, 256, 10, reg_fn, mk_fn, "auto") == 32
    assert mod._select_tile_m(4097, 64, 2560, 256, 10, reg_fn, mk_fn, "auto") == 64

    # HRDY capacity refusal
    with pytest.raises(ValueError, match="HRDY limit"):
        mod._select_tile_m(2, mod._CTRL_MAX_HRDY_TILES, 3584, 384, 16, reg, mk, 32)

    # 4 GiB P8 SRD overflow refusal on q38 at T=131072
    with (
        patch.object(mod.torch.cuda, "is_current_stream_capturing", return_value=False),
        patch.object(mod._au, "get_gfx_from_device", return_value="gfx950"),
        patch.object(mod.dist, "get_rank", return_value=0),
        patch.object(mod.dist, "get_world_size", return_value=8),
        pytest.raises(ValueError, match="32-bit MUBUF limit"),
    ):
        mod.TpAsmContext(None, 0, 131072, 8192, 10, 64, 256)

    # P8 window trim: q38 uses 3 windows summing to 8192 (not 9216)
    assert (
        sum(
            max(w)
            for w in zip(*((w[0], w[1], sum(w[2:])) for w in mod._WIN_CFGS[8192]))
        )
        == 8192
    )


def test_hardened_v2_validation_and_manifest(tmp_path) -> None:
    from types import SimpleNamespace
    from unittest.mock import patch

    import aiter.ops.asm.fmoe_tp_sdma as mod
    from aiter import dtypes
    from aiter.ops.asm import TpMoeCall, tp_moe_layer_asm

    # 1. Manifest layout_version mismatch & missing .co refusal
    mod.load_fmoe_tp_sdma_registry.cache_clear()
    d_v1 = tmp_path / "v1"
    d_v1.mkdir()
    (d_v1 / "fmoe_tp_sdma.csv").write_text(
        "world,model_dim,inter_dim,topk,tile_m,a_dtype,act,epi_mode,gemma,layout_version,sha256,knl_name,co_name\n"
        "2,8192,256,10,64,fp4,silu,add_rmsnorm,0,1,deadbeef,k,missing.co\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="layout_version=1"):
        mod.load_fmoe_tp_sdma_registry(str(d_v1))

    mod.load_fmoe_tp_sdma_registry.cache_clear()
    d_miss = tmp_path / "missing"
    d_miss.mkdir()
    (d_miss / "fmoe_tp_sdma.csv").write_text(
        "world,model_dim,inter_dim,topk,tile_m,a_dtype,act,epi_mode,gemma,layout_version,sha256,knl_name,co_name\n"
        "2,8192,256,10,64,fp4,silu,add_rmsnorm,0,2,deadbeef,k,missing.co\n",
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="missing.co"):
        mod.load_fmoe_tp_sdma_registry(str(d_miss))
    mod.load_fmoe_tp_sdma_registry.cache_clear()

    # 2. Safe reset_epoch: refuses reset_ipc_flags=False; zeroes ipc_ptr[:_FLAG_B] when True
    memsets = []
    fake_hip = SimpleNamespace(
        hipMemset=lambda vp, val, sz: (memsets.append((vp.value, val, sz)), 0)[1]
    )
    ctx_fake = mod.TpAsmContext.__new__(mod.TpAsmContext)
    ctx_fake._closed = False
    ctx_fake._pkey = ("fake",)
    ctx_fake.device = torch.device("cpu")
    ctx_fake.device_idx = 0
    ctx_fake.group = None
    ctx_fake.slot_id = 0
    ctx_fake.model_dim, ctx_fake.topk, ctx_fake.inter_dim, ctx_fake.num_experts = (
        8192,
        10,
        256,
        64,
    )
    ctx_fake.max_tokens = 16
    ctx_fake._st = SimpleNamespace(
        ipc_ptr=0x10000,
        ctrl=torch.ones(16, dtype=torch.int32),
        err=torch.ones(4, dtype=torch.int32),
        epoch=torch.tensor([9], dtype=torch.int32),
        refcount=1,
        sdma_open=False,
        queues=[],
        opened=[],
        quiet=0,
        send_ptr=0,
    )
    with pytest.raises(ValueError, match="reset_ipc_flags=True"):
        ctx_fake.reset_epoch(0, reset_ipc_flags=False)
    with (
        patch.object(mod, "_hip_ext", return_value=fake_hip),
        patch.object(mod.torch.cuda, "synchronize", return_value=None),
    ):
        ctx_fake.reset_epoch(0)
    assert memsets == [(0x10000, 0, mod._FLAG_B)]
    assert int(ctx_fake._st.ctrl.sum().item()) == 0
    assert int(ctx_fake._st.err.sum().item()) == 0
    assert int(ctx_fake._st.epoch.item()) == 0

    # 3. Caller validation: bad topk_ids shape, bad out dtype, overlapping out/out_scale
    t, d, k, e, i = 4, 8192, 10, 64, 256
    x = torch.empty((t, d), dtype=dtypes.bf16)
    ids_ok = torch.zeros((t, k), dtype=torch.int32)
    ids_bad = torch.zeros((t - 1, k), dtype=torch.int32)
    wts = torch.ones((t, k), dtype=torch.float32)
    w1 = torch.empty((e, 2 * i, d // 2), dtype=torch.uint8)
    w2 = torch.empty((e, d, i // 2), dtype=torch.uint8)
    w1s = torch.empty((e, 2 * i, d // 32), dtype=torch.uint8)
    w2s = torch.empty((e, d, i // 32), dtype=torch.uint8)
    nw = torch.ones((d,), dtype=dtypes.bf16)

    with pytest.raises(ValueError, match="topk_ids must be contiguous"):
        tp_moe_layer_asm(
            TpMoeCall(x, ids_bad, wts, w1, w2, w1s, w2s, norm_weight=nw),
            ctx_fake,
            "raw_bf16",
        )
    out_u8 = torch.empty((t, d), dtype=torch.uint8)
    with pytest.raises(ValueError, match="invalid bf16 out"):
        tp_moe_layer_asm(
            TpMoeCall(x, ids_ok, wts, w1, w2, w1s, w2s, norm_weight=nw),
            ctx_fake,
            "raw_bf16",
            out=out_u8,
        )
    backing = torch.empty((t, d // 2), dtype=torch.uint8)
    out_fp4 = backing.view(dtypes.fp4x2)
    sc_overlap = (
        backing.view(-1)[: t * (d // 32)].view(t, d // 32).view(dtypes.fp8_e8m0)
    )
    with pytest.raises(ValueError, match="must not overlap"):
        tp_moe_layer_asm(
            TpMoeCall(x, ids_ok, wts, w1, w2, w1s, w2s, residual=x, norm_weight=nw),
            ctx_fake,
            "add_rmsnorm_fp4",
            out=out_fp4,
            out_scale=sc_overlap,
        )

    # 4. Use-after-close refusal
    ctx_fake._closed = True
    with pytest.raises(RuntimeError, match="closed"):
        tp_moe_layer_asm(
            TpMoeCall(x, ids_ok, wts, w1, w2, w1s, w2s), ctx_fake, "raw_bf16"
        )
    with pytest.raises(RuntimeError, match="closed"):
        ctx_fake.check_error()
    with pytest.raises(RuntimeError, match="closed"):
        ctx_fake.acquire_arena(1)


def _check_s4_disjoint_slots(gloo, dev, T, D, k, E, I) -> None:
    from aiter.ops.asm import TpAsmContext

    ctx0 = TpAsmContext(gloo, dev, T, D, k, E, I, slot_id=0)
    try:
        with ctx0.acquire_arena(1) as ctx1:
            assert ctx0.slot_id == 0 and ctx1.slot_id == 1
            ptrs0 = [
                ctx0.ipc_ptr,
                ctx0.send_ptr,
                ctx0.ctrl.data_ptr(),
                ctx0.epoch.data_ptr(),
                ctx0.err.data_ptr(),
                ctx0.hq.data_ptr(),
                ctx0.hss.data_ptr(),
                ctx0.s_ids.data_ptr(),
                ctx0.s_exp.data_ptr(),
                ctx0.s_w.data_ptr(),
                ctx0.n_val.data_ptr(),
                ctx0.handles.data_ptr(),
                *(t.data_ptr() for t in ctx0.p8_wins),
                *(t.data_ptr() for t in ctx0.s8_wins),
                *((ctx0.sort_ws.data_ptr(),) if ctx0.sort_ws.numel() else ()),
            ]
            ptrs1 = [
                ctx1.ipc_ptr,
                ctx1.send_ptr,
                ctx1.ctrl.data_ptr(),
                ctx1.epoch.data_ptr(),
                ctx1.err.data_ptr(),
                ctx1.hq.data_ptr(),
                ctx1.hss.data_ptr(),
                ctx1.s_ids.data_ptr(),
                ctx1.s_exp.data_ptr(),
                ctx1.s_w.data_ptr(),
                ctx1.n_val.data_ptr(),
                ctx1.handles.data_ptr(),
                *(t.data_ptr() for t in ctx1.p8_wins),
                *(t.data_ptr() for t in ctx1.s8_wins),
                *((ctx1.sort_ws.data_ptr(),) if ctx1.sort_ws.numel() else ()),
            ]
            assert all(ptrs0) and all(ptrs1)
            assert set(ptrs0).isdisjoint(set(ptrs1))
    finally:
        ctx0.close()


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
