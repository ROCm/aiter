# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""TPMoEStage1 correctness tests. Run with:

    torchrun --standalone --nproc_per_node=8 \
        op_tests/multigpu_tests/test_tp_moe_stage1.py --case <name>

Single-rank cases (construct/capacity) also run as plain `python3 <file>`.
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

from aiter.ops.flydsl.kernels.mega_moe.tp_moe_stage1 import (
    TPMoEStage1,
    TPMoEStage1Output,
)
from aiter.ops.flydsl.kernels.mxmoe_dispatcher import mxfp4_moe_gemm2
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

# torchrun rewrites sys.path[0] to the launcher's dir, so make the
# same-directory reference module importable by name.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tp_moe_stage1_ref import (  # noqa: E402
    build_global_weights,
    build_mxfp4_w1,
    dequant_w1_expert,
    per_1x32_fp8_quant_dequant,
    read_shuffled_scale,
    reference_full_moe,
    reference_inter_row,
    shard_w1,
    shard_w2,
)

NETWORK = dict(
    model_dim=7168,
    experts=384,
    topk=6,
    swiglu_limit=10.0,
)
STAGE1_KERNEL = "flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w4_gui_xcd4_kw4_fp8"


def _fake_w1(experts, inter_dim, model_dim, device):
    """Byte-shaped stand-in for a preshuffled MXFP4 W1 (values are irrelevant here)."""
    w1 = torch.zeros(
        (experts, 2 * inter_dim, model_dim // 2), dtype=torch.uint8, device=device
    )
    w1_scale = torch.full(
        (experts, 2 * inter_dim, model_dim // 32),
        0x7F,
        dtype=torch.uint8,
        device=device,
    )
    return w1, w1_scale


def case_construct_validates():
    device = torch.device("cuda", 0)
    inter_dim = 384
    w1, w1_scale = _fake_w1(NETWORK["experts"], inter_dim, NETWORK["model_dim"], device)

    # tp_size must be 4 or 8
    try:
        TPMoEStage1(
            model_dim=NETWORK["model_dim"],
            inter_dim=inter_dim,
            experts=NETWORK["experts"],
            topk=NETWORK["topk"],
            w1=w1,
            w1_scale=w1_scale,
            tp_size=2,
            tp_rank=0,
            device=device,
        )
    except ValueError as exc:
        assert "tp_size" in str(exc), exc
    else:
        raise AssertionError("tp_size=2 must be rejected")

    # sort_block_m must equal the stage1 tile_m
    try:
        TPMoEStage1(
            model_dim=NETWORK["model_dim"],
            inter_dim=inter_dim,
            experts=NETWORK["experts"],
            topk=NETWORK["topk"],
            w1=w1,
            w1_scale=w1_scale,
            tp_size=8,
            tp_rank=0,
            device=device,
            sort_block_m=48,
        )
    except ValueError as exc:
        assert "sort_block_m" in str(exc), exc
    else:
        raise AssertionError("sort_block_m=48 must be rejected")

    # supplying only one of tp_size / tp_rank must be rejected with "together"
    try:
        TPMoEStage1(
            model_dim=NETWORK["model_dim"],
            inter_dim=inter_dim,
            experts=NETWORK["experts"],
            topk=NETWORK["topk"],
            w1=w1,
            w1_scale=w1_scale,
            tp_size=8,
            tp_rank=None,
            device=device,
        )
    except ValueError as exc:
        assert "together" in str(exc), exc
    else:
        raise AssertionError("tp_size=8, tp_rank=None must be rejected")

    op = TPMoEStage1(
        model_dim=NETWORK["model_dim"],
        inter_dim=inter_dim,
        experts=NETWORK["experts"],
        topk=NETWORK["topk"],
        w1=w1,
        w1_scale=w1_scale,
        tp_size=8,
        tp_rank=0,
        device=device,
        swiglu_limit=NETWORK["swiglu_limit"],
        stage1_kernel_name=STAGE1_KERNEL,
    )
    assert op.tp_size == 8
    assert op.sort_block_m == 32
    assert op.stage1_params["tile_m"] == 32
    assert op.stage1_params["tile_n"] == 64
    assert op.stage1_params["tile_k"] == 256
    assert op.stage1_params["gate_mode"] == "interleave"
    print("case_construct_validates OK")


def case_capacity():
    device = torch.device("cuda", 0)
    inter_dim = 384
    w1, w1_scale = _fake_w1(NETWORK["experts"], inter_dim, NETWORK["model_dim"], device)
    op = TPMoEStage1(
        model_dim=NETWORK["model_dim"],
        inter_dim=inter_dim,
        experts=NETWORK["experts"],
        topk=NETWORK["topk"],
        w1=w1,
        w1_scale=w1_scale,
        tp_size=8,
        tp_rank=0,
        device=device,
        stage1_kernel_name=STAGE1_KERNEL,
    )
    # M_global = tp_size * m_local; max_sorted matches moe_sorting's own formula.
    assert op.m_logical_for(1) == 8
    assert op.m_logical_for(128) == 1024
    # 8*6 + 384*32 - 6
    assert op.max_sorted_for(1) == 8 * 6 + 384 * 32 - 6
    # 1024*6 + 384*32 - 6
    assert op.max_sorted_for(128) == 1024 * 6 + 384 * 32 - 6
    print("case_capacity OK")


def _setup_dist():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group("cpu:gloo,cuda:nccl", device_id=device)
    return rank, world, device


def case_all_gather():
    rank, world, device = _setup_dist()
    assert world in (4, 8), f"run this case with 4 or 8 ranks, got {world}"
    m_local = 5
    inter_dim = 384
    w1, w1_scale = _fake_w1(NETWORK["experts"], inter_dim, NETWORK["model_dim"], device)
    op = TPMoEStage1(
        model_dim=NETWORK["model_dim"],
        inter_dim=inter_dim,
        experts=NETWORK["experts"],
        topk=NETWORK["topk"],
        w1=w1,
        w1_scale=w1_scale,
        device=device,
        stage1_kernel_name=STAGE1_KERNEL,
    )
    assert op.tp_size == world and op.tp_rank == rank

    # Rank-identifiable payloads so we can assert the concatenation order.
    x = torch.full(
        (m_local, NETWORK["model_dim"]),
        float(rank),
        dtype=torch.bfloat16,
        device=device,
    )
    ids = torch.full((m_local, NETWORK["topk"]), rank, dtype=torch.int32, device=device)
    wts = torch.full(
        (m_local, NETWORK["topk"]), float(rank), dtype=torch.float32, device=device
    )

    gx, gwts, gids = op._all_gather_inputs(x, wts, ids)

    m_global = world * m_local
    assert gx.shape == (m_global, NETWORK["model_dim"]), gx.shape
    assert gids.shape == (m_global, NETWORK["topk"]), gids.shape
    assert gwts.shape == (m_global, NETWORK["topk"]), gwts.shape
    assert gx.dtype == torch.bfloat16 and gids.dtype == torch.int32
    assert gwts.dtype == torch.float32
    assert gx.is_contiguous() and gids.is_contiguous() and gwts.is_contiguous()

    for src in range(world):
        lo, hi = src * m_local, (src + 1) * m_local
        assert torch.all(gx[lo:hi] == float(src)), f"activation block {src} misordered"
        assert torch.all(gids[lo:hi] == src), f"topk_ids block {src} misordered"
        assert torch.all(gwts[lo:hi] == float(src)), f"weights block {src} misordered"

    if rank == 0:
        print("case_all_gather OK")
    dist.barrier()
    dist.destroy_process_group()


def _random_routes(m, experts, topk, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.stack(
        [torch.randperm(experts, generator=g)[:topk] for _ in range(m)]
    ).to(device=device, dtype=torch.int32)
    w = torch.rand((m, topk), generator=g).to(device=device, dtype=torch.float32)
    return ids, w / w.sum(dim=-1, keepdim=True)


def case_forward_contract():
    rank, world, device = _setup_dist()
    inter_dim = 384
    model_dim = NETWORK["model_dim"]
    experts, topk = NETWORK["experts"], NETWORK["topk"]
    w1, w1_scale = _fake_w1(experts, inter_dim, model_dim, device)
    op = TPMoEStage1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        w1=w1,
        w1_scale=w1_scale,
        device=device,
        swiglu_limit=NETWORK["swiglu_limit"],
        stage1_kernel_name=STAGE1_KERNEL,
    )

    for m_local in (1, 2, 4, 8, 16, 32, 64, 128):
        x = torch.randn((m_local, model_dim), dtype=torch.bfloat16, device=device)
        ids, wts = _random_routes(m_local, experts, topk, device, seed=1000 + rank)
        out = op.forward(x, wts, ids)

        assert isinstance(out, TPMoEStage1Output)
        m_global = world * m_local
        assert out.m_logical == m_global
        assert out.sort_block_m == 32
        assert out.inter_dim == inter_dim

        # moe_sorting's capacity is NOT sort_block_m-aligned; the stage1 payload is.
        # out.max_sorted is the payload row count, matching what the production path
        # feeds GEMM2 (`max_sorted = inter_states.shape[0]`, aiter/fused_moe.py:2018).
        sorted_len = op.max_sorted_for(m_local)
        sbm = out.sort_block_m
        n = -(-sorted_len // sbm) * sbm
        assert out.max_sorted == n, (out.max_sorted, n)

        assert out.inter_sorted_quant.shape == (
            n,
            inter_dim,
        ), out.inter_sorted_quant.shape
        assert out.inter_sorted_quant.dtype == torch.float8_e4m3fn
        assert out.sorted_token_ids.shape == (sorted_len,)
        assert out.sorted_token_ids.dtype == torch.int32
        assert out.sorted_weights.shape == (sorted_len,)
        assert out.sorted_weights.dtype == torch.float32
        assert out.sorted_expert_ids.shape == (n // sbm,), out.sorted_expert_ids.shape
        assert out.num_valid_ids.shape == (2,)
        assert out.num_valid_ids.dtype == torch.int32
        assert out.num_valid_ids.device.type == "cuda"

        pad_rows = (n + 255) // 256 * 256
        pad_cols = ((inter_dim // 32) + 7) // 8 * 8
        assert out.inter_sorted_shuffled_scale.shape == (
            pad_rows,
            pad_cols,
        ), out.inter_sorted_shuffled_scale.shape

        # moe_sorting only writes rows [0, num_valid_ids[0]). The tail of the allocated
        # tensor is uninitialized torch.empty memory that no kernel reads: stage1
        # iterates ceil(num_valid/SBM) blocks, GEMM2 iterates cumsum[0]/BM. Scope every
        # content check to [:nvalid] — asserting over the full tensor reads garbage.
        nvalid = int(out.num_valid_ids[0].item())
        assert 0 < nvalid <= sorted_len, (nvalid, sorted_len)
        assert nvalid % sbm == 0, nvalid

        # token-id encoding: low 24 bits index the gathered batch, high 8 bits the slot
        packed = out.sorted_token_ids[:nvalid]
        tok = packed & 0x00FFFFFF
        slot = (packed >> 24) & 0xFF
        valid = tok < m_global
        assert torch.all(slot[valid] < topk), "valid rows must carry a real top-k slot"
        assert torch.all(tok[~valid] == m_global), "padding sentinel must be M_logical"
        assert torch.all(slot[~valid] == topk), "padding sentinel slot must be topk"
        assert torch.all(
            out.sorted_weights[:nvalid][~valid] == 0.0
        ), "padding weight must be 0"
        assert (
            int(valid.sum().item()) == m_global * topk
        ), f"expected {m_global * topk} routes, found {int(valid.sum().item())}"

        used = out.sorted_expert_ids[: nvalid // sbm]
        assert torch.all((used >= 0) & (used < experts)), "expert ids out of range"

    # per-call allocation: two calls must not alias
    x = torch.randn((8, model_dim), dtype=torch.bfloat16, device=device)
    ids, wts = _random_routes(8, experts, topk, device, seed=7 + rank)
    a = op.forward(x, wts, ids)
    b = op.forward(x, wts, ids)
    assert a.inter_sorted_quant.data_ptr() != b.inter_sorted_quant.data_ptr()
    assert a.sorted_token_ids.data_ptr() != b.sorted_token_ids.data_ptr()

    if rank == 0:
        print("case_forward_contract OK")
    dist.barrier()
    dist.destroy_process_group()


def case_numerics():
    rank, world, device = _setup_dist()
    inter_dim = 384
    model_dim = NETWORK["model_dim"]
    experts, topk = NETWORK["experts"], NETWORK["topk"]
    limit = NETWORK["swiglu_limit"]

    w1_ref, w1_scale_ref, w1_shuf, w1_scale_shuf = build_mxfp4_w1(
        experts, inter_dim, model_dim, device, seed=2026
    )
    op = TPMoEStage1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        w1=w1_shuf,
        w1_scale=w1_scale_shuf,
        device=device,
        swiglu_limit=limit,
        stage1_kernel_name=STAGE1_KERNEL,
    )

    worst = 0.0
    for m_local in (1, 4, 16, 64, 128):
        # Seed the activation too: an unseeded numerical regression test reports a
        # different rel_l2 every run, which makes a future regression hard to bisect.
        gx = torch.Generator(device="cpu").manual_seed(9000 + rank * 17 + m_local)
        x = torch.randn((m_local, model_dim), generator=gx).to(
            device=device, dtype=torch.bfloat16
        ) * (model_dim**-0.25)
        ids, wts = _random_routes(m_local, experts, topk, device, seed=31 + rank)
        out = op.forward(x, wts, ids)
        torch.cuda.synchronize()

        # Rebuild the gathered inputs on the host side for the reference.
        x_g, wts_g, ids_g = op._all_gather_inputs(x, wts, ids)
        x_g_f32 = x_g.float()
        x_deq = per_1x32_fp8_quant_dequant(x_g_f32)

        nvalid = int(out.num_valid_ids[0].item())
        packed = out.sorted_token_ids[:nvalid]
        tok = (packed & 0x00FFFFFF).long()
        slot = ((packed >> 24) & 0xFF).long()
        valid = (tok < out.m_logical).nonzero(as_tuple=True)[0]

        scale_cols = inter_dim // 32
        got_scale = read_shuffled_scale(
            out.inter_sorted_shuffled_scale, nvalid, scale_cols
        )
        got = out.inter_sorted_quant[:nvalid].float() * got_scale.repeat_interleave(
            32, dim=-1
        )
        # An all-NaN kernel output would otherwise slip through: max(0.0, nan)
        # returns 0.0 and nan >= threshold is False, so the aggregate check below
        # cannot see it. Fail here, where the message still names the row shape.
        assert torch.isfinite(
            got[valid]
        ).all(), f"m_local={m_local}: stage1 produced non-finite values in routed rows"

        num, den = 0.0, 0.0
        # Rows are Expert-grouped, so a single-entry cache turns O(rows) weight
        # dequantizations into O(active experts). Without it this loop re-expands a
        # [2*inter, model_dim] tensor once per row and takes minutes.
        cur_e, cur_w1_deq = -1, None
        for r in valid.tolist():
            e = int(out.sorted_expert_ids[r // out.sort_block_m].item())
            t = int(tok[r].item())
            assert int(ids_g[t, int(slot[r].item())].item()) == e, (
                f"row {r}: sorted_expert_ids says {e} but topk_ids says "
                f"{int(ids_g[t, int(slot[r].item())].item())}"
            )
            assert (
                abs(
                    float(out.sorted_weights[r].item())
                    - float(wts_g[t, int(slot[r].item())].item())
                )
                < 1e-6
            ), f"row {r}: route weight mismatch"

            if e != cur_e:
                cur_e, cur_w1_deq = e, dequant_w1_expert(
                    w1_ref, w1_scale_ref, e, inter_dim
                )
            ref = reference_inter_row(x_deq[t], cur_w1_deq, limit)
            ref_q = per_1x32_fp8_quant_dequant(ref.unsqueeze(0)).squeeze(0)
            num += float(((got[r] - ref_q) ** 2).sum())
            den += float((ref_q**2).sum())

        rel_l2 = (num / max(den, 1e-30)) ** 0.5
        if rel_l2 != rel_l2 or rel_l2 == float("inf"):
            raise AssertionError(f"m_local={m_local}: non-finite rel_l2={rel_l2}")
        worst = max(worst, rel_l2)
        if rank == 0:
            print(f"m_local={m_local:4d} rows={len(valid):6d} rel_l2={rel_l2:.5f}")

    t = torch.tensor([worst], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    worst = float(t.item())
    if rank == 0:
        print(f"case_numerics worst rel_l2={worst:.5f}")
    # measured 0.00050 on 8x gfx950, 2026-08-24; 10x headroom. A looser bound is
    # not free: with 384 experts one wholly wrong expert only moves the aggregate
    # to ~sqrt(1/384) = 0.051, i.e. right on the old 0.05 line.
    if worst >= 0.005:
        raise AssertionError(f"rel_l2={worst:.5f} exceeds 0.005")
    if rank == 0:
        print("case_numerics OK")
    dist.barrier()
    dist.destroy_process_group()


def case_prequant_equivalence():
    rank, world, device = _setup_dist()
    inter_dim = 384
    model_dim = NETWORK["model_dim"]
    experts, topk = NETWORK["experts"], NETWORK["topk"]
    _, _, w1_shuf, w1_scale_shuf = build_mxfp4_w1(
        experts, inter_dim, model_dim, device, seed=2026
    )
    op = TPMoEStage1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        w1=w1_shuf,
        w1_scale=w1_scale_shuf,
        device=device,
        swiglu_limit=NETWORK["swiglu_limit"],
        stage1_kernel_name=STAGE1_KERNEL,
    )

    worst = 0.0
    for m_local in (1, 8, 64, 128):
        gx = torch.Generator(device="cpu").manual_seed(4000 + rank * 13 + m_local)
        x = torch.randn((m_local, model_dim), generator=gx).to(
            device=device, dtype=torch.bfloat16
        ) * (model_dim**-0.25)
        ids, wts = _random_routes(m_local, experts, topk, device, seed=77 + rank)

        a = op.forward(x, wts, ids)
        x_q, x_scale = op.quantize(x)
        assert x_q.dtype == torch.float8_e4m3fn, x_q.dtype
        assert x_q.shape == (m_local, model_dim), x_q.shape
        b = op.forward_prequant(x_q, x_scale, wts, ids)
        torch.cuda.synchronize()

        # Routing metadata does not depend on quantization at all -> must match
        # exactly. Compare only the region moe_sorting actually writes: rows
        # [0, num_valid_ids[0]) of the id/weight vectors and the corresponding
        # nvalid//sort_block_m expert blocks. The tail is uninitialized
        # torch.empty memory (see case_forward_contract), so comparing the full
        # tensors asserts on garbage -- two identical forward() calls already
        # differ there. num_valid_ids itself is fully written, so it compares whole.
        assert torch.equal(a.num_valid_ids, b.num_valid_ids)
        nvalid = int(a.num_valid_ids[0].item())
        assert torch.equal(a.sorted_token_ids[:nvalid], b.sorted_token_ids[:nvalid])
        assert torch.equal(
            a.sorted_expert_ids[: nvalid // a.sort_block_m],
            b.sorted_expert_ids[: nvalid // b.sort_block_m],
        )
        assert torch.equal(a.sorted_weights[:nvalid], b.sorted_weights[:nvalid])
        assert a.m_logical == b.m_logical and a.max_sorted == b.max_sorted

        cols = inter_dim // 32
        # Compare only genuinely-routed rows. Stage1 never writes the output SCALE
        # for padding rows: poisoning the caching allocator with 0xFF makes BOTH
        # forward and forward_prequant read back 7680/7680 non-finite padding rows
        # and 0 non-finite valid rows, so that tail is uninitialized memory in both
        # paths rather than a property of either. case_numerics already scopes the
        # same way by iterating `valid` only. Padding rows carry sorted_weights == 0
        # and are masked off downstream by the token-id sentinel.
        tok = (a.sorted_token_ids[:nvalid] & 0x00FFFFFF).long()
        keep = tok < a.m_logical
        assert int(keep.sum()) == a.m_logical * topk, (
            int(keep.sum()),
            a.m_logical * topk,
        )
        va = (
            a.inter_sorted_quant[:nvalid].float()
            * read_shuffled_scale(
                a.inter_sorted_shuffled_scale, nvalid, cols
            ).repeat_interleave(32, dim=-1)
        )[keep]
        vb = (
            b.inter_sorted_quant[:nvalid].float()
            * read_shuffled_scale(
                b.inter_sorted_shuffled_scale, nvalid, cols
            ).repeat_interleave(32, dim=-1)
        )[keep]
        # A genuinely-routed row must never be inf/NaN in either path.
        assert torch.isfinite(va).all(), "forward produced non-finite routed rows"
        assert torch.isfinite(
            vb
        ).all(), "forward_prequant produced non-finite routed rows"
        rel = float(
            ((va - vb) ** 2).sum() ** 0.5 / max(float((va**2).sum() ** 0.5), 1e-30)
        )
        # max(0.0, nan) returns 0.0 in Python, so a NaN would otherwise pass silently.
        if rel != rel or rel == float("inf"):
            raise AssertionError(f"m_local={m_local}: non-finite rel_l2={rel}")
        worst = max(worst, rel)
        if rank == 0:
            print(f"m_local={m_local:4d} forward-vs-prequant rel_l2={rel:.6f}")

    t = torch.tensor([worst], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    worst = float(t.item())
    # measured exactly 0.000000 on 8x gfx950, 2026-08-24: both paths feed the
    # same kernel the same bytes, so any nonzero drift is a real divergence.
    if worst >= 1e-3:
        raise AssertionError(
            f"forward vs forward_prequant rel_l2={worst:.6f} exceeds 1e-3"
        )
    if rank == 0:
        print(f"case_prequant_equivalence OK (worst rel_l2={worst:.6f})")
    dist.barrier()
    dist.destroy_process_group()


GEMM2_BM, GEMM2_BN, GEMM2_BK = 32, 128, 128


def case_end_to_end():
    rank, world, device = _setup_dist()
    model_dim = NETWORK["model_dim"]
    experts, topk = NETWORK["experts"], NETWORK["topk"]
    limit = NETWORK["swiglu_limit"]
    inter_global = 384 * world  # TP8 -> 3072, TP4 -> 1536
    inter_dim = inter_global // world  # this rank's shard == 384

    w1_q, w1_s, w2_q, w2_s = build_global_weights(
        experts, inter_global, model_dim, device, seed=4096
    )
    w1_loc, w1_s_loc = shard_w1(w1_q, w1_s, rank, world, inter_global)
    w2_loc, w2_s_loc = shard_w2(w2_q, w2_s, rank, world, inter_global, model_dim)

    op = TPMoEStage1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        w1=shuffle_weight_a16w4(w1_loc, 16, True),
        w1_scale=shuffle_scale_a16w4(w1_s_loc, experts, True),
        device=device,
        swiglu_limit=limit,
        stage1_kernel_name=STAGE1_KERNEL,
    )
    w2_u8 = shuffle_weight_a16w4(w2_loc, 16, False).view(torch.uint8)
    w2_scale_u8 = shuffle_scale_a16w4(w2_s_loc, experts, False).view(torch.uint8)

    for m_local in (1, 8, 64, 128):
        gx = torch.Generator(device="cpu").manual_seed(5000 + rank * 11 + m_local)
        x = torch.randn((m_local, model_dim), generator=gx).to(
            device=device, dtype=torch.bfloat16
        ) * (model_dim**-0.25)
        ids, wts = _random_routes(m_local, experts, topk, device, seed=555 + rank)
        s1 = op.forward(x, wts, ids)

        # epilog="atomic" accumulates, so the buffer must start at zero.
        partial = torch.zeros(
            (s1.m_logical, model_dim), dtype=torch.bfloat16, device=device
        )
        mxfp4_moe_gemm2(
            inter_sorted_quant=s1.inter_sorted_quant,
            inter_sorted_shuffled_scale=s1.inter_sorted_shuffled_scale,
            w2_u8=w2_u8,
            w2_scale_u8=w2_scale_u8,
            sorted_expert_ids=s1.sorted_expert_ids,
            cumsum_tensor=s1.num_valid_ids,
            sorted_token_ids=s1.sorted_token_ids,
            sorted_weights=s1.sorted_weights,
            out=partial,
            M_logical=s1.m_logical,
            max_sorted=s1.max_sorted,
            NE=experts,
            D_HIDDEN=model_dim,
            D_INTER=inter_dim,
            topk=topk,
            BM=GEMM2_BM,
            BN=GEMM2_BN,
            BK=GEMM2_BK,
            a_dtype="fp8",
            b_dtype="fp4",
            epilog="atomic",
            SBM=s1.sort_block_m,
            out_dtype="bf16",
        )
        torch.cuda.synchronize()

        # Stage1 leaves the output scale of padding rows uninitialized (measured in
        # Task 5). GEMM2 gates the store on `token_id < i32_M`
        # (mxmoe_gemm_v2.py:1006-1008) so that garbage must never reach `out` -- but
        # a NaN leaking through the shared CShuffle LDS would be silent, so check.
        #
        # All three health predicates below are rank-local. Raising on one directly
        # would abort a single rank *before* a collective and leave the others
        # blocked in it until torchrun SIGTERMs them, hiding the real failure. Carry
        # them through the same all_reduce as rel_l2 so every rank raises together.
        bad_partial = float(not bool(torch.isfinite(partial).all()))

        # TP partials -> sum across ranks, keep only this rank's DP shard.
        got = torch.empty((m_local, model_dim), dtype=torch.float32, device=device)
        dist.reduce_scatter_tensor(got, partial.float().contiguous(), group=op.group)

        x_g, wts_g, ids_g = op._all_gather_inputs(x, wts, ids)
        ref_full = reference_full_moe(x_g, ids_g, wts_g, w1_q, w1_s, w2_q, w2_s, limit)
        ref = ref_full[rank * m_local : (rank + 1) * m_local]
        bad_ref = float(not bool(torch.isfinite(ref).all()))
        rel = float(
            ((got - ref) ** 2).sum() ** 0.5 / max(float((ref**2).sum() ** 0.5), 1e-30)
        )
        # ReduceOp.MAX over a NaN operand is not dependable, so flag it and send 0.
        bad_rel = float(rel != rel or rel == float("inf"))
        t = torch.tensor(
            [0.0 if bad_rel else rel, bad_partial, bad_ref, bad_rel], device=device
        )
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        rel = float(t[0].item())
        if float(t[1].item()):
            raise AssertionError(
                f"m_local={m_local}: GEMM2 output contains non-finite values"
            )
        if float(t[2].item()):
            raise AssertionError(
                f"m_local={m_local}: reference produced non-finite values"
            )
        if float(t[3].item()):
            raise AssertionError(f"m_local={m_local}: end-to-end rel_l2 is non-finite")
        if rank == 0:
            print(f"m_local={m_local:4d} end-to-end rel_l2={rel:.5f}")
        # measured 0.0037-0.0040 on 8x gfx950, 2026-08-24, over a documented
        # ~0.0031 bf16 atomic-epilog floor; 0.01 leaves ~2.5x headroom.
        if rel >= 0.01:
            raise AssertionError(
                f"m_local={m_local} end-to-end rel_l2={rel:.5f} >= 0.01"
            )

    if rank == 0:
        print("case_end_to_end OK")
    dist.barrier()
    dist.destroy_process_group()


def case_exports():
    import aiter.ops.flydsl.kernels.mega_moe as mm

    assert "TPMoEStage1" in mm.__all__
    assert "TPMoEStage1Output" in mm.__all__
    assert mm.TPMoEStage1 is TPMoEStage1
    assert mm.TPMoEStage1Output is TPMoEStage1Output
    # existing exports must survive untouched
    for name in (
        "MegaMoEConfig",
        "MegaMoEV2",
        "Stage1Config",
        "Stage2Config",
        "compile_gemm1",
        "gemm1_kernel",
        "select_mega_moe_config",
    ):
        assert name in mm.__all__, f"existing export {name} disappeared"
        assert getattr(mm, name) is not None

    # The phase-2 extension point: the knob exists, a known-but-unbuilt transport
    # raises NotImplementedError, and a typo raises ValueError instead of being
    # silently accepted.
    device = torch.device("cuda", 0)
    w1, w1_scale = _fake_w1(NETWORK["experts"], 384, NETWORK["model_dim"], device)

    def _build(transport):
        return TPMoEStage1(
            model_dim=NETWORK["model_dim"],
            inter_dim=384,
            experts=NETWORK["experts"],
            topk=NETWORK["topk"],
            w1=w1,
            w1_scale=w1_scale,
            tp_size=8,
            tp_rank=0,
            device=device,
            transport=transport,
        )

    try:
        _build("fused_p2p")
    except NotImplementedError as exc:
        assert "fused_p2p" in str(exc), exc
    else:
        raise AssertionError("unimplemented transport must raise NotImplementedError")

    try:
        _build("allgather_bf16")  # the pre-rename name is no longer valid
    except ValueError as exc:
        assert "unknown transport" in str(exc), exc
    else:
        raise AssertionError("unknown transport must raise ValueError")

    assert _build("nccl_allgather").transport == "nccl_allgather"
    print("case_exports OK")


def case_ref_fidelity():
    """TPMoEStage1NCCLRef must be a faithful copy: bit-identical to production.

    The two are the same code today, so anything other than bit-equality means
    the copy was botched. Once the fused path lands this case is what proves the
    reference still represents phase-1 behaviour.
    """
    from tp_moe_stage1_nccl_ref import TPMoEStage1NCCLRef

    rank, world, device = _setup_dist()
    model_dim = NETWORK["model_dim"]
    experts, topk = NETWORK["experts"], NETWORK["topk"]
    inter_dim, limit = 384, NETWORK["swiglu_limit"]

    _, _, w1_shuf, w1_scale_shuf = build_mxfp4_w1(
        experts, inter_dim, model_dim, device, seed=4242
    )
    common = dict(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        w1=w1_shuf,
        w1_scale=w1_scale_shuf,
        device=device,
        swiglu_limit=limit,
        stage1_kernel_name=STAGE1_KERNEL,
    )
    prod = TPMoEStage1(**common)
    ref = TPMoEStage1NCCLRef(**common)

    for m_local in (1, 8, 64):
        g = torch.Generator(device="cpu").manual_seed(9100 + rank * 17 + m_local)
        x = torch.randn((m_local, model_dim), generator=g).to(
            device=device, dtype=torch.bfloat16
        ) * (model_dim**-0.25)
        ids, wts = _random_routes(m_local, experts, topk, device, seed=71 + rank)

        a = prod.forward(x, wts, ids)
        b = ref.forward(x, wts, ids)

        assert type(b).__name__ == "TPMoEStage1Output", type(b)
        assert a.m_logical == b.m_logical, (a.m_logical, b.m_logical)
        assert a.max_sorted == b.max_sorted, (a.max_sorted, b.max_sorted)
        nvalid = int(a.num_valid_ids[0].item())
        assert nvalid == int(b.num_valid_ids[0].item())
        for name in (
            "sorted_token_ids",
            "sorted_expert_ids",
            "sorted_weights",
            "num_valid_ids",
        ):
            ta, tb = getattr(a, name), getattr(b, name)
            n = nvalid if name != "sorted_expert_ids" else nvalid // prod.sort_block_m
            n = min(n, ta.shape[0])
            assert torch.equal(ta[:n], tb[:n]), f"{name} differs at m_local={m_local}"
        # Compare only genuinely-routed rows, the same scoping case_prequant_
        # equivalence uses. Stage1 never writes the payload or the output scale of
        # padding rows, so those bytes are uninitialized torch.empty memory: two
        # identical prod.forward() calls already disagree on 1424/1472 payload rows
        # and on 64% of the shuffled-scale bytes (measured on 8x gfx950).
        tok = (a.sorted_token_ids[:nvalid] & 0x00FFFFFF).long()
        keep = tok < a.m_logical
        assert int(keep.sum()) == a.m_logical * topk, (
            int(keep.sum()),
            a.m_logical * topk,
        )
        qa = a.inter_sorted_quant.view(torch.uint8)[:nvalid][keep]
        qb = b.inter_sorted_quant.view(torch.uint8)[:nvalid][keep]
        assert torch.equal(qa, qb), f"payload differs at m_local={m_local}"
        cols = inter_dim // 32
        sa = read_shuffled_scale(a.inter_sorted_shuffled_scale, nvalid, cols)[keep]
        sb = read_shuffled_scale(b.inter_sorted_shuffled_scale, nvalid, cols)[keep]
        assert torch.isfinite(sa).all(), f"routed scale non-finite at m_local={m_local}"
        assert torch.equal(sa, sb), f"scale differs at m_local={m_local}"
        if rank == 0:
            print(
                f"  m_local={m_local} nvalid={nvalid} routed={int(keep.sum())} "
                f"bit-identical"
            )

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("case_ref_fidelity OK")


CASES = {
    "construct": case_construct_validates,
    "capacity": case_capacity,
    "all_gather": case_all_gather,
    "forward_contract": case_forward_contract,
    "numerics": case_numerics,
    "prequant": case_prequant_equivalence,
    "e2e": case_end_to_end,
    "exports": case_exports,
    "ref_fidelity": case_ref_fidelity,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="construct")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    CASES[args.case]()


if __name__ == "__main__":
    main()
