# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FusedEPMoE (aiter.ops.flydsl.fused_ep_moe) on 8 ranks against a torch reference.

    torchrun --nproc-per-node 8 op_tests/multigpu_tests/test_fused_ep_moe.py
    torchrun --nproc-per-node 8 op_tests/multigpu_tests/test_fused_ep_moe.py --experts-per-rank 48

or ``pytest op_tests/multigpu_tests/test_fused_ep_moe.py`` (launches torchrun; skipped without 8 GPUs).

Per experts-per-rank (48, 56, 63): uniform, skewed and all-to-rank-0 routing, bf16 and MXFP8 rows, unequal and
empty ranks, ids that are not routed (-1 and past the last expert), up to 1024 tokens per rank. The output of every
rank's first 24 tokens matches the reference (cosine >= 0.99999 over the output, >= 0.9999 per row, and
checkAllclose at rtol = atol = 2e-2 on >= 98 % of the elements); two runs, a run after other tokens on the same
routes, and CUDA-graph replays with new inputs are bitwise identical to the eager run; nothing compiles after
``prepare``.

Reference numerics, per (token, slot) route of expert e: ``gate, up = x @ w13[e].T`` in fp32 (x the MXFP8 rows),
``act = silu(min(gate, 10)) * clamp(up, -10, 10)``, ``act`` quantized to MXFP8 per 32 columns (e8m0 = exponent
of ceil_pow2(amax / 448), at least 1), ``slot = bf16(w * (act_q @ w2[e].T))``, ``out = bf16(sum of the 6 slots
in slot order, fp32)``. The kernel accumulates in another order, so a slot may differ in one bf16 ulp near a
rounding tie; hence the cosine bounds.
"""

import argparse
import os
import shutil
import subprocess
import sys

import torch
import torch.distributed as dist

H, I, TOPK, LIMIT = 5120, 2304, 6, 10.0
WORLD = 8
CHECKED = 24  # tokens per rank compared with the reference (rows are independent)
_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def dequant_mxfp4(codes, scales):
    lut = torch.cat([_GRID, -_GRID]).to(codes.device)
    v = torch.stack([lut[(codes & 0xF).long()], lut[(codes >> 4).long()]], -1)
    v = v.flatten(-2)
    return v * torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=-1)


def dequant_mxfp8(codes, scales):
    m, k = codes.shape
    v = codes.float().reshape(m, k // 32, 32)
    return (v * torch.exp2(scales.float() - 127.0).unsqueeze(-1)).reshape(m, k)


def quantize_mxfp8(x):
    """-> e4m3 codes [M, K], e8m0 bytes [M, K/32]."""
    m, k = x.shape
    blocks = x.float().reshape(m, k // 32, 32)
    u = (blocks.abs().amax(-1) * (1.0 / 448.0)).view(torch.int32)
    e = (u >> 23) & 255
    e = torch.where(((u & 0x7FFFFF) != 0) & (e < 255), e + 1, e).clamp_min(1)
    codes = blocks * torch.exp2(127.0 - e.float()).unsqueeze(-1)
    codes = codes.clamp(-448.0, 448.0).reshape(m, k)
    return codes.to(torch.float8_e4m3fn), e.to(torch.uint8)


def experts_slots(xq, xs, w13, w13s, w2, w2s, topk_weights, local_ids):
    """bf16 [M, 6, 5120] slots of one rank's experts (row-major checkpoint weights)."""
    x = dequant_mxfp8(xq, xs)
    slots = torch.zeros((x.shape[0], TOPK, H), dtype=torch.bfloat16, device=x.device)
    for e in local_ids.unique().tolist():
        if not 0 <= e < w13.shape[0]:
            continue
        t, s = torch.nonzero(local_ids == e, as_tuple=True)
        gu = x[t] @ dequant_mxfp4(w13[e], w13s[e]).t()
        g, u = gu[:, :I].clamp(max=LIMIT), gu[:, I:].clamp(-LIMIT, LIMIT)
        act = dequant_mxfp8(*quantize_mxfp8(torch.nn.functional.silu(g) * u))
        y = act @ dequant_mxfp4(w2[e], w2s[e]).t()
        slots[t, s] = (y * topk_weights[t, s].float().unsqueeze(1)).bfloat16()
    return slots


def run_torch(xq, xs, weights, topk_weights, topk_ids):
    """bf16 [M, 5120]; ``weights``: one row-major (w13, w13_scale, w2, w2_scale) per rank."""
    slots = None
    for owner, w in enumerate(weights):
        local = topk_ids - owner * w[0].shape[0]
        part = experts_slots(xq, xs, *w, topk_weights, local)
        slots = part if slots is None else slots + part
    acc = slots[:, 0].float()
    for s in range(1, TOPK):
        acc = acc + slots[:, s].float()
    return acc.bfloat16()


def make_experts(seed, device, epr):
    g = torch.Generator(device=device).manual_seed(seed)

    def rnd(lo, hi, shape):
        return torch.randint(
            lo, hi, shape, dtype=torch.uint8, device=device, generator=g
        )

    return (
        rnd(0, 256, (epr, 2 * I, H // 2)),
        rnd(118, 124, (epr, 2 * I, H // 32)),
        rnd(0, 256, (epr, H, I // 2)),
        rnd(118, 124, (epr, H, I // 32)),
    )


def make_routes(n, kind, seed, device, epr):
    """6 distinct experts per token. uniform; skew: odds ~ 1 / rank over a fixed order; hot: all on rank 0."""
    g = torch.Generator(device=device).manual_seed(seed)
    ne = epr * WORLD
    logits = torch.zeros(ne, device=device)
    if kind == "skew":
        gp = torch.Generator(device=device).manual_seed(99)
        order = torch.randperm(ne, device=device, generator=gp)
        logits[order] = -torch.log(torch.arange(1, ne + 1, device=device).float())
    elif kind == "hot":
        logits[epr:] = -1e4
    u = torch.rand((n, ne), device=device, generator=g).clamp_min(1e-20)
    ids = (logits - torch.log(-torch.log(u))).topk(TOPK, dim=1).indices.int()
    tw = torch.softmax(torch.randn((n, TOPK), device=device, generator=g), -1)
    return ids, tw


def make_tokens(n, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((n, H), device=device, generator=g).bfloat16()
    return (x, *quantize_mxfp8(x))


def _cos(a, b):
    a, b = a.double(), b.double()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    row = torch.nn.functional.cosine_similarity(a, b, dim=1).min()
    return cos.item(), row.item()


CASES = {  # (routing, tokens per rank, bf16 rows, unrouted ids)
    48: [
        ("uniform", [16] * WORLD, True, False),
        ("uniform", [64] * WORLD, False, True),
        ("skew", [192] * WORLD, True, False),
        ("hot", [0, 5, 300, 64, 1, 0, 128, 33], True, False),
        ("uniform", [768] * WORLD, True, False),
        ("skew", [1024] * WORLD, False, False),
    ],
    56: [("uniform", [64] * WORLD, True, False), ("skew", [160] * WORLD, False, False)],
    63: [("uniform", [48, 0, 17, 130, 1, 64, 300, 5], True, True)],
}
GRAPHS = {
    48: [[64] * WORLD, [768] * WORLD, [768, 1, 0, 33, 200, 64, 5, 128]],
    56: [[32] * WORLD],
    63: [[24] * WORLD],
}


def run(epr, rank, dev, errors):
    from flydsl.compiler.jit_function import MlirCompiler

    from aiter.ops.flydsl.fused_ep_moe import FusedEPMoE, FusedEPMoEWeights
    from aiter.test_common import checkAllclose

    raws = [make_experts(100 + r, dev, epr) for r in range(WORLD)]
    w = FusedEPMoEWeights.from_row_major(*raws[rank])
    moe = FusedEPMoE(
        rank=rank,
        world_size=WORLD,
        group=dist.group.WORLD,
        device=dev,
        experts_per_rank=epr,
    )
    moe.prepare(w)
    orig = MlirCompiler.compile

    def unexpected(*a, **kw):
        raise AssertionError("FlyDSL compile after prepare()")

    MlirCompiler.compile = unexpected
    try:
        for ci, (kind, counts, bf16, bad_ids) in enumerate(CASES[epr]):
            m = counts[rank]
            ids, tw = make_routes(max(m, 1), kind, 31 * ci + rank, dev, epr)
            ids, tw = ids[:m].contiguous(), tw[:m].contiguous()
            if bad_ids and m > 2:
                ids[1, 2] = -1
                ids[2, 0] = epr * WORLD + 5
            x, codes, scales = (
                t[:m] for t in make_tokens(max(m, 1), 77 * ci + rank, dev)
            )
            args = (x, None) if bf16 else (codes, scales)
            out = moe(*args, w, tw, ids).clone()
            again = moe(*args, w, tw, ids)
            torch.cuda.synchronize()
            tag = f"epr {epr} rank {rank} case {ci} {kind} {counts} {'bf16' if bf16 else 'mxfp8'}"
            if not torch.equal(out, again):
                errors.append(f"{tag}: two runs differ")
            x2, c2, s2 = (
                t[:m] for t in make_tokens(max(m, 1), 1000 + 77 * ci + rank, dev)
            )
            moe(*((x2, None) if bf16 else (c2, s2)), w, tw, ids)
            third = moe(*args, w, tw, ids)
            torch.cuda.synchronize()
            if not torch.equal(out, third):
                errors.append(
                    f"{tag}: the run after other tokens on the same routes differs"
                )
            k = min(m, CHECKED)
            if k:
                q, s = quantize_mxfp8(x[:k]) if bf16 else (codes[:k], scales[:k])
                ref = run_torch(q, s, raws, tw[:k], ids[:k])
                cos, row = _cos(out[:k], ref)
                err = checkAllclose(
                    ref.float(),
                    out[:k].float(),
                    rtol=2e-2,
                    atol=2e-2,
                    msg=tag,
                    printLog=False,
                )
                if not (cos >= 0.99999 and row >= 0.9999) or err > 0.02:
                    errors.append(
                        f"{tag}: cosine {cos:.7f}, worst row {row:.6f}, allclose miss {err:.2%}"
                    )
                if torch.isnan(out.float()).any():
                    errors.append(f"{tag}: NaN")
        for counts in GRAPHS[epr]:
            m = counts[rank]
            sx = make_tokens(max(m, 1), 5 + rank, dev)[0][:m].contiguous()
            sids, stw = (
                t[:m].contiguous()
                for t in make_routes(max(m, 1), "uniform", 9 + rank, dev, epr)
            )
            dist.barrier()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                gout = moe(sx, None, w, stw, sids)
            for seed in (1, 2, 3):
                nx = make_tokens(max(m, 1), seed * 10 + rank, dev)[0][:m]
                nids, ntw = make_routes(
                    max(m, 1), "uniform", seed * 10 + rank, dev, epr
                )
                sx.copy_(nx)
                sids.copy_(nids[:m])
                stw.copy_(ntw[:m])
                g.replay()
                torch.cuda.synchronize()
                eager = moe(sx, None, w, stw, sids)
                torch.cuda.synchronize()
                if not torch.equal(gout, eager):
                    errors.append(
                        f"epr {epr} rank {rank}: graph {counts} replay {seed} differs from eager"
                    )
            del g
    finally:
        MlirCompiler.compile = orig
    moe.close()


def main():
    from aiter.jit.utils.chip_info import get_gfx_runtime

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--experts-per-rank",
        type=int,
        nargs="*",
        default=[48, 56, 63],
        choices=sorted(CASES),
    )
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert world == WORLD, f"run with torchrun --nproc-per-node {WORLD}"
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("gloo")
    from aiter.ops.flydsl.fused_ep_moe import fused_ep_moe_supported

    if not fused_ep_moe_supported(
        world_size=WORLD, model_dim=H, inter_dim=I, experts=48 * WORLD, topk=TOPK
    ):
        if rank == 0:
            print(f"FusedEPMoE unsupported on {get_gfx_runtime()}; skipping")
        dist.destroy_process_group()
        return
    errors = []
    for epr in args.experts_per_rank:
        run(epr, rank, dev, errors)
        dist.barrier()
    gathered = [None] * WORLD
    dist.all_gather_object(gathered, errors)
    failed = [e for errs in gathered for e in errs]
    if rank == 0:
        print(
            "\n".join(failed)
            if failed
            else f"PASS: experts per rank {args.experts_per_rank}"
        )
    dist.destroy_process_group()
    if failed:
        sys.exit(1)


def test_fused_ep_moe_8_ranks():
    import pytest

    if torch.cuda.device_count() < WORLD or shutil.which("torchrun") is None:
        pytest.skip(f"needs {WORLD} GPUs and torchrun")
    cmd = ["torchrun", "--nproc-per-node", str(WORLD), os.path.abspath(__file__)]
    subprocess.run(cmd, check=True, timeout=3600)


if __name__ == "__main__":
    main()
