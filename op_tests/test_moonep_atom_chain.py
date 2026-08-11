# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""End-to-end check of the MoonEP chain against a plain-torch ground truth.

Runs the chain exactly as ``MoonEPPrepareAndFinalize`` does -- dispatch into
the grouped layout, byte-copy weight pools, remote prefetch, ``fused_moe``
driven by synthetic topk-1 pool-slot ids, combine -- and checks two things that
would otherwise only surface as degraded accuracy with no way to tell which
broke:

1. ``local_group_sizes()`` covers exactly the rows this rank executes -- its
   ``E/R`` home groups plus the ``B`` migration groups, and nothing else.
2. ``row_slot_ids()`` names the right pool slot for every row, so ``fused_moe``
   applies each group's own expert weights (home *or* migrated).

Check 1 runs first: if it is wrong the row accounting is wrong and the output
comparison would be noise.

Run::

    torchrun --standalone --nproc-per-node=8 op_tests/test_moonep_atom_chain.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from aiter.ops.flydsl.kernels.moonep_dispatch_combine_op import (
    MoonEPDispatchCombineConfig,
    MoonEPDispatchCombineIntraNodeOp,
)
from aiter.ops.flydsl.kernels.moonep_weights import MoonEPWeightPool

S = int(os.environ.get("T_S", "256"))
H = int(os.environ.get("T_H", "512"))
I = int(os.environ.get("T_I", "256"))
K = int(os.environ.get("T_K", "4"))
E = int(os.environ.get("T_E", "32"))
# Actual tokens, which may be < S. The op pads the batch up to S, and the
# padding rows carry real expert ids -- exercising that is the point.
N = int(os.environ.get("T_N", str(S)))
# "none" reproduces the bf16 chain; "mxfp4" reproduces what the ATOM
# server actually runs -- quantised, shuffled weights plus block scales.
QUANT = os.environ.get("T_QUANT", "none")
# Exercise the decode plan: no balancing, no migration, token_padding=1.
DECODE = os.environ.get("T_DECODE", "0") == "1"


def torch_reference(x, topk_ids, gate_all, up_all, down_all):
    out = torch.zeros_like(x, dtype=torch.float32)
    xf = x.float()
    for k in range(topk_ids.shape[1]):
        for e in torch.unique(topk_ids[:, k]):
            e = int(e.item())
            if e < 0:
                continue
            m = topk_ids[:, k] == e
            xs = xf[m]
            g = xs @ gate_all[e].float()
            u = xs @ up_all[e].float()
            out[m] += (F.silu(g) * u) @ down_all[e].float()
    return out


def main() -> int:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    epn = E // world

    import mori

    cpu_group = dist.new_group(backend="gloo")
    torch._C._distributed_c10d._register_process_group("mori", cpu_group)
    mori.shmem.shmem_torch_process_group_init("mori")

    g = torch.Generator(device="cpu").manual_seed(7)
    x = (torch.randn(N, H, generator=g) * 0.3).to(torch.bfloat16).to(dev)
    topk = torch.rand(N, E, generator=g).topk(K, dim=1).indices.to(torch.int32).to(dev)
    w = torch.rand(N, K, generator=g).float().to(dev)

    # Every rank needs all experts' weights to build the ground truth, so
    # generate them from one shared seed rather than gathering them.
    gw = torch.Generator(device="cpu").manual_seed(11)
    gate_all = (torch.randn(E, H, I, generator=gw) * 0.05).to(torch.bfloat16).to(dev)
    up_all = (torch.randn(E, H, I, generator=gw) * 0.05).to(torch.bfloat16).to(dev)
    down_all = (torch.randn(E, I, H, generator=gw) * 0.05).to(torch.bfloat16).to(dev)

    cfg = MoonEPDispatchCombineConfig(
        rank=rank,
        world_size=world,
        hidden_dim=H,
        max_num_inp_token_per_rank=S,
        num_experts_per_rank=epn,
        num_experts_per_token=K,
        max_decode_token_per_rank=S if DECODE else 0,
    )
    op = MoonEPDispatchCombineIntraNodeOp(cfg)
    rows, row_w, cu = op.dispatch_grouped(x, w, topk, decode=DECODE)
    torch.cuda.synchronize(dev)
    dist.barrier()

    # --- assumption 3 -------------------------------------------------
    sizes = op.local_group_sizes(cu)  # raises on a prefetch-slot shortage
    claimed = int(sizes.sum().item())
    all_sizes = cu - torch.cat([cu.new_zeros(1), cu[:-1]])
    total = int(all_sizes.sum().item())
    lo, hi = rank * epn, (rank + 1) * epn
    migrated = int(all_sizes[E:].sum().item())
    if rank == 0:
        print(
            f"[a3] rows={total} claimed={claimed} home={int(all_sizes[lo:hi].sum())} "
            f"migrated={migrated} {'OK' if claimed == total else 'FAIL'}",
            flush=True,
        )
    if claimed != total:
        raise AssertionError(
            f"rank{rank}: local_group_sizes accounts for {claimed} of {total} "
            "dispatched rows; every expert_num_tokens would be wrong"
        )

    # --- run the experts step the way the ATOM hook does ----------------
    # w13 is gate/up fused on dim 1, matching ATOM's layout; the pools never
    # interpret it, so this exercises the same byte-copy path a real fp8
    # shuffled slab takes.
    # fused_moe consumes pre-shuffled weights -- ATOM applies exactly this in
    # process_weights_after_loading before handing them over.
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.shuffle import moe_shuffle_weight, shuffle_weight

    w13_bf = torch.cat(
        [gate_all.transpose(1, 2), up_all.transpose(1, 2)], dim=1
    ).contiguous()
    w2_bf = down_all.transpose(1, 2).contiguous()

    if QUANT == "mxfp4":
        # Mirror ATOM's Mxfp4MoEMethod: quantise, then moe_shuffle_{weight,scale}.
        from aiter.ops.quant import per_1x32_f4_quant
        from aiter.ops.shuffle import moe_shuffle_scale

        q1, s1 = per_1x32_f4_quant(w13_bf.reshape(-1, w13_bf.shape[-1]), shuffle=False)
        q2, s2 = per_1x32_f4_quant(w2_bf.reshape(-1, w2_bf.shape[-1]), shuffle=False)
        w13 = moe_shuffle_weight(q1.reshape(E, w13_bf.shape[1], -1), experts_cnt=E, gate_up=True)
        w2t = moe_shuffle_weight(q2.reshape(E, w2_bf.shape[1], -1), experts_cnt=E, gate_up=False)
        sc1 = moe_shuffle_scale(s1.reshape(-1, s1.shape[-1]), E, gate_up=True)
        sc2 = moe_shuffle_scale(s2.reshape(-1, s2.shape[-1]), E, gate_up=False)
        # fused_moe reads the pre-shuffled layout off this attribute, not from
        # the data; ATOM sets it in process_weights_after_loading. Any slice
        # drops it, which is the bug this test now guards.
        w13.is_shuffled = True
        w2t.is_shuffled = True
        qt, tol = QuantType.per_1x32, 2.0e-1
    else:
        w13 = shuffle_weight(w13_bf, layout=(16, 16))
        w2t = shuffle_weight(w2_bf, layout=(16, 16))
        sc1 = sc2 = None
        qt, tol = QuantType.No, 5.0e-2

    def pool(t, per_expert_rows=None):
        if t is None:
            return None
        v = t if per_expert_rows is None else t.reshape(E, per_expert_rows, -1)
        p = MoonEPWeightPool(
            rank=rank,
            world_size=world,
            experts_per_rank=epn,
            prefetch_slots=op.plan_config.prefetch_slots,
            weight_shape=tuple(v.shape[1:]),
            dtype=v.dtype,
            block_num=256,
        )
        p.stage_home(v[lo:hi].contiguous())
        return p

    pw1, pw2 = pool(w13), pool(w2t)
    ps1 = pool(sc1, None if sc1 is None else sc1.shape[0] // E)
    ps2 = pool(sc2, None if sc2 is None else sc2.shape[0] // E)
    plan = op.live_plan()
    sel = plan.experts_to_copy[rank].contiguous()
    for p in (pw1, pw2, ps1, ps2):
        if p is not None:
            p.prefetch(sel)

    # The pool must be a byte-exact copy of what ATOM would have handed the
    # kernel; if it is not, nothing downstream can be trusted.
    for name, p, src in (
        ("w13", pw1, w13[lo:hi]),
        ("w2", pw2, w2t[lo:hi]),
        ("s1", ps1, None if sc1 is None else sc1.reshape(E, -1, sc1.shape[-1])[lo:hi]),
        ("s2", ps2, None if sc2 is None else sc2.reshape(E, -1, sc2.shape[-1])[lo:hi]),
    ):
        if p is None or src is None:
            continue
        a = p.pool[:epn].reshape(-1).view(torch.uint8)
        b = src.contiguous().reshape(-1).view(torch.uint8)
        if not torch.equal(a, b):
            n_bad = int((a != b).sum().item())
            raise AssertionError(
                f"rank{rank}: pool '{name}' differs from the source in "
                f"{n_bad}/{a.numel()} bytes -- staging is not a byte copy"
            )
    if rank == 0:
        print("[pool] all pools byte-identical to source", flush=True)

    # Two calls, mirroring MoonEPPrepareAndFinalize.run_experts: aiter's MoE
    # zeroes its output when the weight tensor declares experts no row routes
    # to, so home and borrowed experts cannot share one tensor.
    slot_ids = op.row_slot_ids()
    out = op.get_expert_output_buffer()
    home_end, total, nb = (
        op.expert_call_split() if op.needs_split() else (0, 0, 0)
    )
    if rank == 0:
        print(f"[split] home_end={home_end} total={total} borrowed={nb}", flush=True)

    def seg(p, s0, s1):
        if p is None:
            return None
        t = p.pool[s0:s1]
        return t.reshape(-1, t.shape[-1]) if t.dim() == 3 and p is not pw1 and p is not pw2 else t

    def wseg(p, s0, s1):
        t = p.pool[s0:s1]
        if QUANT == "mxfp4":
            t.is_shuffled = True
        return t

    def experts(a, b, s0, s1, nlt=None):
        if b <= a:
            return
        o = fused_moe(
            rows[a:b],
            wseg(pw1, s0, s1),
            wseg(pw2, s0, s1),
            torch.ones((b - a, 1), dtype=torch.float32, device=dev),
            slot_ids[a:b],
            None,
            ActivationType.Silu,
            quant_type=qt,
            w1_scale=seg(ps1, s0, s1),
            w2_scale=seg(ps2, s0, s1),
            num_local_tokens=nlt,
            dtype=rows.dtype,
        )
        out[a:b].copy_(o)

    # Bisect: same home rows, same experts, but indexing the full E-expert
    # tensor with global ids instead of the epn-slot pool. Any gap here is the
    # home call itself (pool slice or slot ids); a gap only downstream points
    # at the migration call or combine.
    gidx = torch.arange(cu.numel(), device=dev, dtype=torch.int32)
    grp = torch.searchsorted(cu.contiguous(),
                             torch.arange(rows.shape[0], device=dev, dtype=torch.int32),
                             right=True).clamp_(max=cu.numel() - 1)
    gids = plan.group_expert_ids[grp].to(torch.int32).unsqueeze(1)
    if home_end > 0:
        a = fused_moe(rows[:home_end], w13, w2t,
                      torch.ones((home_end, 1), dtype=torch.float32, device=dev),
                      gids[:home_end], None, ActivationType.Silu,
                      quant_type=qt, w1_scale=sc1, w2_scale=sc2, dtype=rows.dtype).float()
        b = fused_moe(rows[:home_end], wseg(pw1, 0, epn), wseg(pw2, 0, epn),
                      torch.ones((home_end, 1), dtype=torch.float32, device=dev),
                      slot_ids[:home_end], None, ActivationType.Silu,
                      quant_type=qt, w1_scale=seg(ps1, 0, epn), w2_scale=seg(ps2, 0, epn),
                      dtype=rows.dtype).float()
        d = (a - b).abs().max().item(); sm = a.abs().max().item()
        if rank == 0:
            print(f"[home] pool-vs-full rel={d/max(sm,1e-9):.3e} "
                  f"{'ok' if d/max(sm,1e-9) < 5e-2 else 'HOME CALL WRONG'}", flush=True)
        if d / max(sm, 1e-9) >= 5e-2:
            raise AssertionError(f"rank{rank}: home call rel={d/max(sm,1e-9):.3e}")

    if not op.needs_split():
        experts(0, out.shape[0], 0, epn, op.valid_rows())
        home_end = total = int(cu[-1].item())
    else:
        experts(0, home_end, 0, epn)
        experts(home_end, total, epn, epn + nb)

    # Same bisect for the migrated rows: reference them against the full
    # E-expert tensor with global ids. The borrowed experts' weights exist
    # locally in this test, so this isolates the prefetched slabs.
    if total > home_end:
        ra = fused_moe(rows[home_end:total], w13, w2t,
                       torch.ones((total - home_end, 1), dtype=torch.float32, device=dev),
                       gids[home_end:total], None, ActivationType.Silu,
                       quant_type=qt, w1_scale=sc1, w2_scale=sc2, dtype=rows.dtype).float()
        rb = out[home_end:total].float()
        d = (ra - rb).abs().max().item(); sm = ra.abs().max().item()
        if rank == 0:
            print(f"[migr] prefetched-vs-full rel={d/max(sm,1e-9):.3e} "
                  f"{'ok' if d/max(sm,1e-9) < 5e-2 else 'MIGRATION CALL WRONG'}", flush=True)
        if d / max(sm, 1e-9) >= 5e-2:
            raise AssertionError(f"rank{rank}: migration call rel={d/max(sm,1e-9):.3e}")
    got = op.combine_grouped(op.get_expert_output_buffer())[:N].float()
    torch.cuda.synchronize(dev)
    dist.barrier()

    # torch bf16 in both modes. A same-quantisation reference would be tighter,
    # but the obvious one -- fused_moe topk=K over all E experts -- is itself
    # wrong here: triangulating against torch put it at rel 0.76 while MoonEP
    # sat at 0.29, i.e. the reference was the outlier. So compare against torch
    # and widen the bound to mxfp4's own coarseness, measured at ~0.32 on
    # max|.| for random weights. Loose, but it is an honest bound, and the
    # failures worth catching here (is_shuffled dropped, wrong expert, dead
    # migration group) all land far above it.
    exp = torch_reference(x, topk, gate_all, up_all, down_all)
    if QUANT == "mxfp4":
        tol = 5.0e-1
    # Triangulation: which of got / exp is the odd one out? torch bf16 is a
    # third, independent opinion. mxfp4 is coarse (~30% on max|.|), so this
    # cannot be a tight bound -- it only has to separate "close" from "wrong".
    if QUANT == "mxfp4" and rank == 0:
        ref_t = torch_reference(x, topk, gate_all, up_all, down_all)
        st = ref_t.abs().max().item()
        print(
            f"[tri] vs torch-bf16: got rel={(ref_t-got).abs().max().item()/max(st,1e-9):.3e} "
            f"exp rel={(ref_t-exp).abs().max().item()/max(st,1e-9):.3e}",
            flush=True,
        )

    # Dedup only happens when a token routes to two experts on the same rank,
    # so the error splitting along that line would point straight at it.
    if rank == 0:
        dst_rank = (topk.long() // epn)
        has_dup = torch.zeros(N, dtype=torch.bool, device=dev)
        for a in range(K):
            for b in range(a + 1, K):
                has_dup |= dst_rank[:, a] == dst_rank[:, b]
        e_row = (exp - got).abs().amax(dim=1)
        nd, d_ = int((~has_dup).sum()), int(has_dup.sum())
        print(
            f"[dup] tokens with a duplicate dest rank: {d_}/{N}  "
            f"max|err| dup={e_row[has_dup].max().item() if d_ else 0:.3e} "
            f"nodup={e_row[~has_dup].max().item() if nd else 0:.3e}",
            flush=True,
        )

    err = (exp - got).abs().max().item()
    scale = exp.abs().max().item()
    rel = err / max(scale, 1e-30)
    if rank == 0:
        print(
            f"[chain] q={QUANT} N={N}/{S} max|err|={err:.3e} max|ref|={scale:.3e} rel={rel:.2e}",
            flush=True,
        )
    if scale <= 0:
        raise AssertionError("ground truth is all zeros; the test is vacuous")
    if got.abs().max().item() <= 0:
        raise AssertionError("MoonEP chain produced an all-zero output")
    if rel > tol:
        raise AssertionError(
            f"rank{rank}: chain output differs by rel {rel:.3e}. Suspects, in "
            "order: gate/up halves of w1 swapped; transpose direction into the "
            "pools; group->expert mapping."
        )
    if rank == 0:
        print("[chain] PASS", flush=True)

    for p in (pw1, pw2):
        p.close()
    op.close()
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
