# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Multi-GPU tests for S3 (FlyQuickAllReduceRMSNorm MX) and S1+S3 end-to-end seams."""

import os
import socket
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_REPO_ROOT = os.path.abspath(f"{__file__}/../../..")
sys.path.insert(0, _REPO_ROOT)

EPS = 1e-6


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _rrms(a: torch.Tensor, b: torch.Tensor) -> float:
    d = (a.float() - b.float()).pow(2).sum()
    n = b.float().pow(2).sum().clamp_min(1e-12)
    return float((d / n).sqrt().item())


def _dq_mx(q: torch.Tensor, s: torch.Tensor, fp8: bool = False) -> torch.Tensor:
    from aiter.utility import fp4_utils

    v = (
        q.view(torch.float8_e4m3fn).float()
        if fp8
        else fp4_utils.mxfp4_to_f32(q.view(torch.uint8))
    )
    sc = fp4_utils.e8m0_to_f32(s.view(torch.uint8)).view(*v.shape[:-1], -1)
    return (v.view(*v.shape[:-1], -1, 32) * sc.unsqueeze(-1)).view(v.shape)


def _s3_worker(rank: int, tp: int, init_m: str) -> None:
    import aiter as ai
    from aiter import dtypes
    from aiter.ops.flydsl.quick_allreduce import FlyQuickAllReduceRMSNorm

    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", init_m, world_size=tp, rank=rank, device_id=dev)
    gloo = dist.new_group(backend="gloo")
    sync = lambda: (torch.cuda.synchronize(dev), dist.barrier(group=gloo))

    qr = FlyQuickAllReduceRMSNorm(
        group=gloo, device=dev, rank=rank, world_size=tp, algorithm="mesh"
    )
    widths = (8192, 7168, 4096, 3584, 2560)
    for d in widths:
        for qdt in ("fp4", "fp8"):
            qr.preload(d, quant_dtype=qdt, scale_shuffle=False, return_norm_bf16=True)
    qr.preload(8192, quant_dtype="fp4", scale_shuffle=False, return_norm_bf16=False)
    sync()

    cases = [
        (8192, "fp4", False, 512),
        (8192, "fp4", True, 1021),
        (8192, "fp4", False, 4096),
        (8192, "fp4", False, 16384),
        (7168, "fp4", False, 512),
        (4096, "fp4", False, 1021),
        (4096, "fp4", True, 4096),
        (3584, "fp8", False, 512),
        (3584, "fp8", True, 1021),
        (3584, "fp8", False, 4096),
        (3584, "fp8", False, 16384),
        (2560, "fp4", False, 512),
        (2560, "fp4", True, 4096),
    ]
    for d, qdt, gemma, t_cnt in cases:
        gen = torch.Generator(device=dev).manual_seed(1000 + 31 * rank + d + t_cnt)
        inp = torch.randn(t_cnt, d, generator=gen, device=dev, dtype=dtypes.bf16) * 0.5
        gen_s = torch.Generator(device=dev).manual_seed(555 + d + t_cnt)
        res = (
            torch.randn(t_cnt, d, generator=gen_s, device=dev, dtype=dtypes.bf16) * 0.5
        )
        nw = (0.2 * torch.randn(d, generator=gen_s, device=dev, dtype=dtypes.bf16)) + (
            0.0 if gemma else 1.0
        )

        ref_ar = inp.float()
        dist.all_reduce(ref_ar, op=dist.ReduceOp.SUM)
        ref_res = ref_ar.bfloat16().float() + res.float()
        sc_w = nw.float() + (1.0 if gemma else 0.0)
        ref_norm = (
            ref_res * torch.rsqrt(ref_res.pow(2).mean(-1, keepdim=True) + EPS) * sc_w
        ).bfloat16()
        if qdt == "fp4":
            ref_q, ref_s = ai.per_1x32_f4_quant_hip(ref_norm, shuffle=False)
        else:
            ref_q, ref_s = ai.per_1x32_mx_quant_hip(
                ref_norm, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
            )
        ref_qf = _dq_mx(ref_q, ref_s.view(t_cnt, d // 32), fp8=(qdt == "fp8"))

        q_cols = d // 2 if qdt == "fp4" else d
        q_dt = dtypes.fp4x2 if qdt == "fp4" else dtypes.fp8
        out_q = torch.empty((t_cnt, q_cols), dtype=q_dt, device=dev)
        out_s = torch.empty((t_cnt, d // 32), dtype=dtypes.fp8_e8m0, device=dev)
        out_bf16 = torch.empty((t_cnt, d), dtype=dtypes.bf16, device=dev)
        res_out = torch.empty((t_cnt, d), dtype=dtypes.bf16, device=dev)

        ret_bf16, ret_res = qr.allreduce_rmsnorm_mx(
            inp,
            res,
            nw,
            EPS,
            out_q,
            out_s,
            quant_dtype=qdt,
            scale_shuffle=False,
            return_norm_bf16=True,
            out=out_bf16,
            residual_out=res_out,
            gemma_norm=gemma,
        )
        sync()
        if qdt == "fp4":
            bf16_q, bf16_s = ai.per_1x32_f4_quant_hip(out_bf16, shuffle=False)
        else:
            bf16_q, bf16_s = ai.per_1x32_mx_quant_hip(
                out_bf16, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
            )
        bf16_qf = _dq_mx(bf16_q, bf16_s.view(t_cnt, d // 32), fp8=(qdt == "fp8"))
        assert ret_bf16 is out_bf16 and ret_res is res_out
        assert (
            _rrms(res_out, ref_res) <= 0.10
        ), f"res rrms={_rrms(res_out, ref_res)} d={d} T={t_cnt}"
        assert (
            _rrms(out_bf16, ref_norm) <= 0.10
        ), f"norm rrms={_rrms(out_bf16, ref_norm)} d={d} T={t_cnt}"
        got_qf = _dq_mx(out_q, out_s, fp8=(qdt == "fp8"))
        assert bool(
            torch.isfinite(got_qf).all()
        ), f"non-finite got_qf d={d} qdt={qdt} T={t_cnt}"
        assert (
            _rrms(got_qf, bf16_qf) <= 0.02
        ), f"d={d} qdt={qdt} T={t_cnt}: vs_bf16_q={_rrms(got_qf, bf16_qf)}"
        tol = 0.18 if qdt == "fp4" else 0.12
        assert (
            _rrms(got_qf, ref_qf) <= tol
        ), f"d={d} qdt={qdt} T={t_cnt}: rrms={_rrms(got_qf, ref_qf)}"

        for buf in (out_q, out_s, out_bf16, res_out):
            bi = buf.contiguous().view(torch.uint8)
            peers = [torch.empty_like(bi) for _ in range(tp)]
            dist.all_gather(peers, bi, group=gloo)
            assert all(
                torch.equal(bi, p) for p in peers
            ), f"rank mismatch d={d} qdt={qdt}"

    # Also test return_norm_bf16=False on D=8192, T=4096
    t_cnt, d = 4096, 8192
    gen = torch.Generator(device=dev).manual_seed(42 + rank)
    inp = torch.randn(t_cnt, d, generator=gen, device=dev, dtype=dtypes.bf16) * 0.5
    res = torch.randn(t_cnt, d, generator=gen, device=dev, dtype=dtypes.bf16) * 0.5
    nw = torch.ones(d, device=dev, dtype=dtypes.bf16)
    out_q = torch.empty((t_cnt, d // 2), dtype=dtypes.fp4x2, device=dev)
    out_s = torch.empty((t_cnt, d // 32), dtype=dtypes.fp8_e8m0, device=dev)
    ret_bf16, ret_res = qr.allreduce_rmsnorm_mx(
        inp,
        res,
        nw,
        EPS,
        out_q,
        out_s,
        quant_dtype="fp4",
        scale_shuffle=False,
        return_norm_bf16=False,
    )
    sync()
    assert ret_bf16 is None and ret_res is not None
    assert bool(torch.isfinite(_dq_mx(out_q, out_s, fp8=False)).all())

    qr.close()
    sync()
    dist.destroy_process_group()


def _run_one_e2e_cfg(
    rank: int, tp: int, dev: torch.device, gloo, sync, qr, cfg: tuple
) -> None:
    import aiter as ai
    import aiter.fused_moe as fm
    from aiter import dtypes
    from aiter.ops.asm import (
        TpAsmContext,
        TpMoeCall,
        prepare_downstream_mx_inputs,
        tp_moe_layer_asm,
    )
    from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight, shuffle_weight

    name, E, k, D, I, a_dt, act, epi, N_dn = cfg
    qr.preload(D, quant_dtype=a_dt, scale_shuffle=False, return_norm_bf16=True)
    T_ = 512
    ctx = TpAsmContext(gloo, dev, T_, D, k, E, I)
    gen_r = torch.Generator(device=dev).manual_seed(77 + 37 * rank + D)
    gen_s = torch.Generator(device=dev).manual_seed(888 + D)
    inp = torch.randn(T_, D, generator=gen_r, device=dev, dtype=dtypes.bf16) * 0.4
    res0 = torch.randn(T_, D, generator=gen_s, device=dev, dtype=dtypes.bf16) * 0.4
    res1 = torch.randn(T_, D, generator=gen_s, device=dev, dtype=dtypes.bf16) * 0.4
    nw0 = 1.0 + 0.2 * torch.randn(D, generator=gen_s, device=dev, dtype=dtypes.bf16)
    nw1 = 1.0 + 0.2 * torch.randn(D, generator=gen_s, device=dev, dtype=dtypes.bf16)
    rw = torch.randn(E, D, generator=gen_s, device=dev, dtype=dtypes.bf16) * (D**-0.5)

    def q4(r: int, c: int, sc: int, g):
        q, s = ai.per_1x32_f4_quant(
            torch.randn(E, r, c, generator=g, device=dev, dtype=dtypes.bf16)
            * (sc**-0.5)
        )
        return q, s.view(E * r, c // 32).view(torch.float8_e8m0fnu)

    w1_q, w1_s = q4(2 * I, D, D, gen_r)
    w2_q, w2_s = q4(D, I, I * tp, gen_r)
    w1 = moe_shuffle_weight(w1_q)
    w2 = moe_shuffle_weight(w2_q)
    w1_ss = moe_shuffle_scale(w1_s, E)
    w2_ss = moe_shuffle_scale(w2_s, E)

    w_dn = torch.randn(N_dn, D, generator=gen_s, device=dev, dtype=dtypes.bf16) * (
        D**-0.5
    )
    if a_dt == "fp4":
        wd_q, wd_s = ai.per_1x32_f4_quant_hip(w_dn, shuffle=True)
    else:
        wd_q, wd_s = ai.per_1x32_mx_quant_hip(
            w_dn, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
        wd_s = wd_s.view(N_dn, D // 32)
    wd_sh = shuffle_weight(wd_q, (16, 16))

    xq_buf = torch.empty(
        (T_, D // 2 if a_dt == "fp4" else D),
        dtype=dtypes.fp4x2 if a_dt == "fp4" else dtypes.fp8,
        device=dev,
    )
    xs_buf = torch.empty((T_, D // 32), dtype=dtypes.fp8_e8m0, device=dev)
    x_bf16 = torch.empty((T_, D), dtype=dtypes.bf16, device=dev)
    res0_out = torch.empty((T_, D), dtype=dtypes.bf16, device=dev)
    out_q = torch.empty_like(xq_buf)
    out_s = torch.empty_like(xs_buf)
    res1_out = (
        torch.empty((T_, D), dtype=dtypes.bf16, device=dev)
        if epi.startswith("add")
        else None
    )

    def run_chain():
        qr.allreduce_rmsnorm_mx(
            inp,
            res0,
            nw0,
            EPS,
            xq_buf,
            xs_buf,
            quant_dtype=a_dt,
            scale_shuffle=False,
            return_norm_bf16=True,
            out=x_bf16,
            residual_out=res0_out,
        )
        logits = x_bf16 @ rw.t()
        topk_w, topk_ids = fm.fused_topk(x_bf16, logits, k, True)
        call = TpMoeCall(
            xq_buf,
            topk_ids,
            topk_w,
            w1,
            w2,
            w1_ss,
            w2_ss,
            res1 if epi.startswith("add") else None,
            nw1,
            EPS,
            x_scale=xs_buf,
        )
        moe_out = tp_moe_layer_asm(
            call,
            ctx,
            epi,
            out_q,
            out_s,
            residual_out=res1_out,
            a_dtype=a_dt,
            act=act,
        )
        dn_q, dn_s = prepare_downstream_mx_inputs(moe_out.out, moe_out.out_scale, a_dt)
        if a_dt == "fp4":
            return ai.gemm_a4w4(dn_q, wd_sh, dn_s, wd_s, dtype=dtypes.bf16)
        return ai.gemm_a8w8_mxfp8_bpreshuffle(
            dn_q, wd_sh, dn_s, wd_s, dtype=dtypes.bf16
        )

    y = run_chain()
    sync()
    ctx.check_error()
    assert y.shape == (T_, N_dn) and bool(torch.isfinite(y).all()), (
        f"{name}: out_q finite={bool(torch.isfinite(out_q.float()).all())}, "
        f"out_s finite={bool((out_s.view(torch.uint8) != 0xFF).all())}, "
        f"y finite={bool(torch.isfinite(y).all())}"
    )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        yg = run_chain()
    g.replay()
    sync()
    ctx.check_error()
    assert torch.allclose(y, yg, atol=1e-2, rtol=1e-2)
    ctx.close()
    sync()


def _e2e_worker(rank: int, tp: int, init_m: str) -> None:
    from aiter.ops.flydsl.quick_allreduce import FlyQuickAllReduceRMSNorm

    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", init_m, world_size=tp, rank=rank, device_id=dev)
    gloo = dist.new_group(backend="gloo")
    sync = lambda: (torch.cuda.synchronize(dev), dist.barrier(group=gloo))

    qr = FlyQuickAllReduceRMSNorm(
        group=gloo, device=dev, rank=rank, world_size=tp, algorithm="mesh"
    )
    configs = [
        ("q38", 64, 10, 8192, 256, "fp4", "silu", "add_rmsnorm_fp4", 2048),
        ("k3", 128, 16, 3584, 384, "fp8", "situ", "rmsnorm_only_fp8", 1792),
        ("v4_flash", 64, 6, 4096, 256, "fp4", "silu", "add_rmsnorm_fp4", 2048),
        (
            "flash_next",
            64,
            10,
            2560,
            256 if tp == 4 else 384,
            "fp4",
            "silu",
            "add_rmsnorm_fp4",
            1536,
        ),
    ]
    for cfg in configs:
        _run_one_e2e_cfg(rank, tp, dev, gloo, sync, qr, cfg)

    qr.close()
    sync()
    dist.destroy_process_group()


def _run_mp(fn, tp: int) -> None:
    if torch.cuda.device_count() < tp:
        pytest.skip(f"need {tp} GPUs, have {torch.cuda.device_count()}")
    init_m = f"tcp://127.0.0.1:{_free_port()}"
    mp.spawn(fn, args=(tp, init_m), nprocs=tp, join=True)


def test_s3_mx_plain_tp2() -> None:
    _run_mp(_s3_worker, 2)


def test_s3_mx_plain_tp4() -> None:
    _run_mp(_s3_worker, 4)


def test_e2e_s1_s3_downstream_tp2() -> None:
    _run_mp(_e2e_worker, 2)


def test_e2e_s1_s3_downstream_tp4() -> None:
    _run_mp(_e2e_worker, 4)
