# SPDX-License-Identifier: MIT
"""GPU: the Gluon V4.1 K1 launch (attention seam + front) against vLLM's FlyDSL
one, same runner, at K1's ABI. Synthetic inputs: the residual and mixes, the
previous layer's reduced output, MXFP8 wqkv / wq_b, the norms, an fp8_ds_mla
window cache, slots, window indices, a compressed block table and top-k
(ratio 4: the key table's compressed rows). FlyDSL's K1 first, the epoch moved
on (its tags would satisfy the Gluon launch's polls), every plain region, the
outputs and the cache poisoned, then the Gluon K1; compared bitwise: the seam's
outputs, LIN / PMIX / QKVP (values), X8 / QX8 and codes, q, the key table, the
written cache records. ``BENCH=1``: CUDA-graph timing of both."""

import os
import sys

import torch
import torch.distributed as dist

from deepseek_v4_gluon.tests.gpu_ffn_vs_flydsl import HC, HIDDEN, bench, poison, values
from deepseek_v4_gluon.tests.gpu_k2_vs_flydsl import DATA, HEAD_DIM, KEYS, fp8_bytes, kv_cache

Q_RANK, QKV_ROWS, TOPK = 1280, 1792, 512


def main():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import get_tp_group
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.models.deepseek_v41.amd.mono import layer as L
    from vllm.models.deepseek_v41.amd.mono.runner import DSV41MonoLayer

    from deepseek_v4_gluon.v41.ffn import ptr
    from deepseek_v4_gluon.v41.k1 import build_constexprs, k1

    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", rank)
    torch.accelerator.set_device_index(device)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(tp, rank, "env://", rank, "nccl")
        ensure_model_parallel_initialized(tp, 1)
    runner = DSV41MonoLayer(tp, rank, get_tp_group().cpu_group, device)
    health = torch.zeros(4, dtype=torch.int32, device=device)
    g = torch.Generator().manual_seed(4000 + rank)
    heads = 64 // tp
    q_rows = heads * HEAD_DIM
    mix = (2 + HC) * HC
    f32 = lambda *s, sc=1.0: (torch.randn(*s, generator=g) * sc).to(device)  # noqa: E731
    hc_fn, hc_scale, hc_base = f32(mix, HC * HIDDEN, sc=0.01), f32(3, sc=0.5).abs() + 0.5, f32(mix, sc=0.1)
    attn_w = (1 + f32(HIDDEN, sc=0.1)).bfloat16()
    wqkv = fp8_bytes((QKV_ROWS, HIDDEN), g, device)
    wqkv_s = torch.randint(118, 123, (QKV_ROWS // 32, HIDDEN // 32), generator=g, dtype=torch.uint8).to(device)
    qn_w, kvn_w = (1 + f32(Q_RANK, sc=0.1)).bfloat16(), (1 + f32(512, sc=0.1)).bfloat16()
    wqb = fp8_bytes((q_rows, Q_RANK), g, device)
    wqb_s = torch.randint(118, 123, (q_rows // 32, Q_RANK // 32), generator=g, dtype=torch.uint8).to(device)
    block, pages = 64, 24
    swa = kv_cache(pages, block, g, device)
    max_pos = 4096
    ang = torch.rand(max_pos, 32, generator=g) * 6.28
    cos_sin = torch.cat([ang.cos(), ang.sin()], 1).contiguous().to(device)
    ratio = 4
    comp_block, nblk = 64, 16
    dummy = torch.zeros(4, dtype=torch.int32, device=device)
    widths = [int(a) for a in sys.argv[1:]] or [1, 4, 16, 17, 48]
    bad = 0
    for M in widths:
        gi = torch.Generator().manual_seed(91 + M)
        res = (torch.randn(M, HC, HIDDEN, generator=gi) * 2).bfloat16().to(device)
        pend = (torch.randn(M, HIDDEN, generator=gi)).bfloat16().to(device)
        post = (torch.rand(M, HC, generator=gi) * 2).to(device)
        comb = torch.softmax(torch.randn(M, HC, HC, generator=gi), -1).to(device)
        pre = (torch.rand(M, HC, generator=gi) + 0.5).to(device)
        pos = torch.randint(0, max_pos, (M,), generator=gi, dtype=torch.int64)
        slot = torch.randperm(pages * block, generator=gi)[:M].to(torch.int64)
        slot[M // 3] = -1 if M > 2 else slot[M // 3]  # a pad row
        swa_idx = torch.randint(0, pages * block, (M, 128), generator=gi, dtype=torch.int32)
        swa_lens = torch.randint(1, 129, (M,), generator=gi, dtype=torch.int32)
        t2r = torch.randint(0, 4, (M,), generator=gi, dtype=torch.int32)
        comp_bt = torch.randint(0, pages, (4, nblk), generator=gi, dtype=torch.int32)
        topk = torch.full((M, TOPK), -1, dtype=torch.int32)
        for t in range(M):
            nt = min((int(pos[t]) + 1) // ratio, TOPK)
            topk[t, :nt] = torch.randint(0, comp_block * nblk, (nt,), generator=gi, dtype=torch.int32)
        pos, slot, swa_idx, swa_lens, t2r, comp_bt, topk = (x.to(device) for x in (
            pos, slot, swa_idx, swa_lens, t2r, comp_bt, topk))  # fmt: skip
        lay = L.scratch_layout(M, tp)
        sc = runner.scratch(M)

        def snap():
            return {k: sc[o : o + n].clone() for k, (o, n) in lay.items()}

        def outs():
            return (torch.empty_like(res), torch.empty(M, HC, device=device),
                    torch.empty(M, HC, HC, device=device), torch.empty(M, HC, device=device),
                    torch.empty(M, heads, HEAD_DIM, dtype=torch.bfloat16, device=device),
                    torch.empty(M * KEYS, dtype=torch.int32, device=device),
                    torch.empty(2 * M, dtype=torch.int32, device=device))  # fmt: skip

        def args(o):
            res_out, post_out, comb_out, pre_out, q, kt, klen = o
            return (res.data_ptr(), pend.data_ptr(), post.data_ptr(), comb.data_ptr(),
                    pre.data_ptr(), hc_fn.data_ptr(), hc_scale.data_ptr(), hc_base.data_ptr(),
                    attn_w.data_ptr(), res_out.data_ptr(), post_out.data_ptr(),
                    comb_out.data_ptr(), pre_out.data_ptr(), wqkv.data_ptr(), wqkv_s.data_ptr(),
                    qn_w.data_ptr(), kvn_w.data_ptr(), wqb.data_ptr(), wqb_s.data_ptr(),
                    cos_sin.data_ptr(), pos.data_ptr(), slot.data_ptr(), swa.data_ptr(),
                    swa.stride(0), block, q.data_ptr(), swa_idx.data_ptr(), swa_lens.data_ptr(),
                    comp_block, comp_bt.data_ptr(), nblk, topk.data_ptr(), t2r.data_ptr(),
                    kt.data_ptr(), klen.data_ptr(), dummy.data_ptr(), dummy.data_ptr(),
                    sc.data_ptr(), runner.epoch.data_ptr(), 0)  # fmt: skip

        INTS = {23, 24, 28, 30, 39}  # swa_stride, swa_block, comp_block, bt_stride, tl
        k1_fly = runner.kernels(M, ratio)[0]
        fly = outs()
        torch.accelerator.synchronize()
        dist.barrier()
        k1_fly(*args(fly), stream=torch.cuda.current_stream())
        torch.accelerator.synchronize()
        s_fly = snap()
        swa_fly = swa.clone()
        runner.epoch[0] += 1  # K2 would have: new tags for the Gluon launch
        poison(sc, lay, ("x8", "x8s", "qx8", "qx8s"))
        swa.fill_(0xFF)
        mine = outs()
        for x in mine:
            x.view(torch.uint8).fill_(0xFF)
        c = build_constexprs(M, tp, ratio)

        def run_gl(o):
            aa = [a if i in INTS else ptr(a) for i, a in enumerate(args(o))]
            k1[(256,)](*aa, ptr(health.data_ptr()), **c, num_warps=8, enable_fp_fusion=False)

        dist.barrier()
        run_gl(mine)
        torch.accelerator.synchronize()
        s_gl = snap()
        rep = [f"rank {rank} S={M}: health {health.tolist()}"]
        names = ("res_out", "post", "comb", "pre", "q", "kt", "klen")
        for name, a, b in zip(names, fly, mine):
            ok = torch.equal(a, b)
            bad += not ok
            nd = int((a.view(torch.uint8) != b.view(torch.uint8)).sum()) if not ok else 0
            rep.append(f"  {name:8s} bitwise {'OK' if ok else f'DIFF ({nd} bytes)'}")
        for k in ("lin", "pmix", "x8", "x8s", "qkvp", "qx8", "qx8s"):
            a, b = s_fly[k], s_gl[k]
            if k in ("lin", "pmix", "qkvp"):
                a, b = values(a), values(b)
            ok = torch.equal(a, b)
            bad += not ok
            rep.append(f"  {k:8s} bitwise {'OK' if ok else f'DIFF ({int((a != b).sum())} of {a.numel()})'}")
        # the records written: data rows and code words of the live slots
        live = slot[slot >= 0]
        pg, off = live // block, live % block
        data = lambda x: torch.stack([x[p, o * DATA : (o + 1) * DATA] for p, o in zip(pg, off)])  # noqa: E731
        codes = lambda x: torch.stack([x[p, block * DATA + o * 8 : block * DATA + o * 8 + 7]  # noqa: E731
                                       for p, o in zip(pg, off)])  # fmt: skip
        for name, f in (("kv rows", data), ("kv codes", codes)):
            ok = torch.equal(f(swa_fly), f(swa))
            bad += not ok
            rep.append(f"  {name:8s} bitwise {'OK' if ok else 'DIFF'}")
        print("\n".join(rep), flush=True)
        if os.environ.get("BENCH"):
            bench(M, rank, lambda: k1_fly(*args(fly), stream=torch.cuda.current_stream()),
                  lambda: run_gl(mine))  # fmt: skip
    dist.barrier()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
