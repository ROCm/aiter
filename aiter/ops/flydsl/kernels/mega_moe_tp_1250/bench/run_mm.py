"""torchrun --nproc_per_node=4 run_mm.py --tokens 8 64 ... [--check] [--bench]
MegaMoE TP gfx1250 (ag_rs) vs the torch reference, and its timing (CUDA graph)."""
import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, "/home/binding2/moe_1026/aiter/op_tests/flydsl_tests")
import test_flydsl_grouped_gemm as tg  # noqa: E402

from aiter import ActivationType  # noqa: E402
from aiter.ops.flydsl.kernels.mega_moe_tp_1250.engine import MegaMoeTP1250Engine  # noqa
from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--tokens", type=int, nargs="+", default=[8, 64, 256, 1024])
p.add_argument("--inter-total", type=int, default=2560)
p.add_argument("--pad-inter", type=int, default=0, help="kernel inter per rank (zero-padded)")
p.add_argument("--experts", type=int, default=256)
p.add_argument("--topk", type=int, default=6)
p.add_argument("--shared-expert", action="store_true", help="last expert always routed (fused shared)")
p.add_argument("--model-dim", type=int, default=4096)
p.add_argument("--act", default="swiglu", choices=("swiglu", "situv2"), help="stage1 activation")
p.add_argument("--situ-beta", type=float, default=4.0, help="SiTUv2 gate beta (Kimi-K3)")
p.add_argument("--situ-linear-beta", type=float, default=25.0, help="SiTUv2 up beta (Kimi-K3)")
p.add_argument("--check", action="store_true")
p.add_argument("--bench", action="store_true")
p.add_argument("--iters", type=int, default=20)
p.add_argument("--stamps", action="store_true")
p.add_argument("--debug-sync", type=int, default=0)
p.add_argument("--ar", action="store_true", help="AR mode: replicated tokens in / out")
p.add_argument("--sh8", action="store_true",
               help="MiniMax-M3.1: the last expert is the shared expert and runs A8W8 "
                    "(MXFP8 weights x MXFP8 activations); the others stay A4W4")
p.add_argument("--debug-burst", type=int, default=1)
p.add_argument("--data-init", default="uniform", help="weights: uniform (random) | constant")
p.add_argument("--scale-init", default="auto")
p.add_argument("--warm", type=int, default=0,
               help="eager launches (ranks synced before each) first, e.g. for ATT on a warm one")
args = p.parse_args()


def route(score, K, shared):
    """topk routing; shared: the last expert is a fused shared expert every token takes
    (weight 1), the other K-1 routes topk over the rest (MiniMax-M3: 128 + 1, top-5)"""
    if not shared:
        return torch.softmax(score, -1).topk(K, -1)
    tw, ids = torch.softmax(score[:, :-1], -1).topk(K - 1, -1)
    n = score.shape[0]
    tw = torch.cat([tw, torch.ones((n, 1), dtype=tw.dtype, device=tw.device)], 1)
    ids = torch.cat([ids, torch.full((n, 1), score.shape[1] - 1, dtype=ids.dtype, device=ids.device)], 1)
    return tw, ids

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
dev = torch.device("cuda", torch.cuda.current_device())
dist.init_process_group("nccl", device_id=dev)
E, K, H = args.experts, args.topk, args.model_dim
I = args.inter_total // world
gen = tg.bench_init.make_generator(1000 + rank)
ER = E - 1 if args.sh8 else E  # routed (A4W4) experts
w1 = tg._pattern_packed(ER, 2 * I, H // 2, data_init=args.data_init, generator=gen)
w2 = tg._pattern_packed(ER, H, I // 2, data_init=args.data_init, generator=gen)
w1s = tg.init_weight_scales(ER, 2 * I, H // 32, scale_init=args.scale_init, generator=gen)
w2s = tg.init_weight_scales(ER, H, I // 32, scale_init=args.scale_init, generator=gen)
shkw = {}
if args.sh8:  # the shared expert: logical MXFP8 codes (GGUU w1) + raw e8m0 scales (as baseline_m31_sh8.py)
    assert args.shared_expert and not args.pad_inter
    sw1 = tg.bench_init.fill_fp8((2 * I, H), "uniform", gen)
    sw2 = tg.bench_init.fill_fp8((H, I), "uniform", gen)
    sw1s = tg.init_weight_scales(1, 2 * I, H // 32, scale_init="auto", generator=gen)
    sw2s = tg.init_weight_scales(1, H, I // 32, scale_init="auto", generator=gen)
    shkw = dict(
        shared_w1=moe_shuffle_weight(sw1.view(1, 2 * I, H), experts_cnt=1, is_guinterleave=True,
                                     gate_up=True).view(torch.uint8),
        shared_w1_scale=moe_shuffle_scale(sw1s.contiguous(), experts_cnt=1, is_guinterleave=True,
                                          gate_up=True).view(torch.uint8),
        shared_w2=moe_shuffle_weight(sw2.view(1, H, I), experts_cnt=1).view(torch.uint8),
        shared_w2_scale=moe_shuffle_scale(sw2s.contiguous(), experts_cnt=1).view(torch.uint8))
P = args.pad_inter or I  # the kernel's inter (zero-padded shard; the reference keeps I)
from pad_inter import pad_inter  # noqa: E402
w1k, w1sk, w2k, w2sk = pad_inter(w1, w1s, w2, w2s, I, P)
from pad_inter import pad_w2_scale  # noqa: E402
w2sk = pad_w2_scale(w2sk, P)  # P % 128: the native split (the weights stay at P)
w1g = moe_shuffle_weight(w1k, experts_cnt=ER, is_guinterleave=True, gate_up=True)
w2g = moe_shuffle_weight(w2k, experts_cnt=ER)
w1sg = moe_shuffle_scale(w1sk.contiguous(), experts_cnt=ER, is_guinterleave=True, gate_up=True)
w2sg = moe_shuffle_scale(w2sk.contiguous(), experts_cnt=ER)
mmax = max(args.tokens) // world
eng = MegaMoeTP1250Engine(
    rank=rank, world_size=world, model_dim=H, inter_dim=P, experts=E, topk=K,
    max_local_tokens=mmax, w1=w1g.view(torch.uint8), w1_scale=w1sg.view(torch.uint8),
    w2=w2g.view(torch.uint8), w2_scale=w2sg.view(torch.uint8), activation=args.act,
    swiglu_limit=7.0 if args.act == "swiglu" else None, situ_beta=args.situ_beta,
    situ_linear_beta=args.situ_linear_beta, device=dev, **shkw)


def timed(fn, iters):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(iters):
            fn()
    g.replay()
    torch.cuda.synchronize()
    dist.barrier()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = 1e30
    for _ in range(5):
        dist.barrier()
        st.record()
        g.replay()
        en.record()
        torch.cuda.synchronize()
        best = min(best, st.elapsed_time(en) * 1e3 / iters)
    t = torch.tensor([best], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t)


for T in args.tokens:
    m = T // world
    g2 = torch.Generator(device=dev).manual_seed(T * 31 + rank)
    x = (torch.randn((m, H), device=dev, generator=g2) * 0.5).to(torch.bfloat16)
    score = torch.randn((m, E), device=dev, generator=g2)
    tw, ids = route(score, K, args.shared_expert)
    tw, ids = tw.float(), ids.to(torch.int32)
    if args.ar:  # every rank: all T tokens (the ranks' rows gathered)
        xf = torch.cat([torch.empty_like(x) for _ in range(world)])
        dist.all_gather_into_tensor(xf, x)
        twf = torch.cat([torch.empty_like(tw) for _ in range(world)])
        dist.all_gather_into_tensor(twf, tw)
        idsf = torch.cat([torch.empty_like(ids) for _ in range(world)])
        dist.all_gather_into_tensor(idsf, ids)

        def run():
            return eng.forward_ar(xf, twf, idsf)
    else:
        def run():
            return eng(x, tw, ids)
    for _ in range(args.warm):
        torch.cuda.synchronize()
        dist.barrier()
        run()
    torch.cuda.synchronize()
    for it in range(args.debug_sync):  # ctrl state after each of a few launches
        from aiter.ops.flydsl.kernels.mega_moe_tp_1250 import config as C
        for _ in range(args.debug_burst):
            eng(x, tw, ids)
        torch.cuda.synchronize()
        c = eng.ctrl.cpu()
        cnt = c[C.CT_CNT:C.CT_CNT + E]
        fill = c[C.CT_FILL:C.CT_FILL + E]
        print(f"rank{rank} it{it} cnt_sum={int(cnt.sum())} fill_sum={int(fill.sum())} "
              f"psum_last={int(eng.psum[E - 1])} rpos_max={int(eng.rpos[:T * K].max())}", flush=True)
        sy = [c[C.CT_SYNC + k * C.SYNC_INTS:][: C.SYNC_INTS] for k in range(7)]
        print(f"rank{rank} it{it} ep={int(c[C.CT_EPOCH])} err={int(c[C.CT_ERR])} " + " | ".join(
            f"s{k}: g0={int(s[0])} g31={int(s[31 * C.LINE])} top={int(s[C.SYNC_TOP])} "
            f"flag={int(s[C.SYNC_FLAG])}" for k, s in enumerate(sy)), flush=True)
    y = run().clone()
    torch.cuda.synchronize()
    err = eng.poll_errors()
    msg = f"T={T:6d} err={err}"
    if args.check:
        xs = [torch.empty_like(x) for _ in range(world)]
        ws = [torch.empty_like(tw) for _ in range(world)]
        is_ = [torch.empty_like(ids) for _ in range(world)]
        dist.all_gather(xs, x)
        dist.all_gather(ws, tw)
        dist.all_gather(is_, ids)
        KR = K - 1 if args.sh8 else K  # routed columns
        ref = tg._torch_moe_ref(
            torch.cat(xs), w1, w1s, torch.zeros(ER, 2 * I, device=dev), w2, w2s,
            torch.zeros(ER, H, device=dev), torch.cat(ws)[:, :KR].contiguous().to(torch.bfloat16),
            torch.cat(is_)[:, :KR].contiguous(), data_format="a4w4", activation=ActivationType.Swiglu if args.act == "swiglu" else ActivationType.Situv2,
            swiglu_limit=7.0 if args.act == "swiglu" else float("inf"), situ_beta=args.situ_beta,
            situ_linear_beta=args.situ_linear_beta).float()
        if args.sh8:  # + the shared expert's fp32 reference: MXFP8 activations x MXFP8 weights
            xa = torch.cat(xs)
            ref = ref + tg._torch_moe_ref(
                xa, sw1.view(1, 2 * I, H), sw1s, torch.zeros(1, 2 * I, device=dev),
                sw2.view(1, H, I), sw2s, torch.zeros(1, H, device=dev),
                torch.cat(ws)[:, KR:].contiguous().to(torch.bfloat16),
                torch.zeros((xa.shape[0], 1), dtype=torch.int32, device=dev), data_format="a8w4",
                activation=ActivationType.Swiglu, swiglu_limit=7.0, situ_beta=args.situ_beta,
                situ_linear_beta=args.situ_linear_beta).float()
        dist.all_reduce(ref)
        if not args.ar:
            ref = ref[rank * m:(rank + 1) * m]
        d = torch.stack([((y.float() - ref) ** 2).sum(), (ref ** 2).sum(), (y.float() ** 2).sum()])
        dist.all_reduce(d)
        msg += f" rel_l2={float((d[0] / d[1]).sqrt()):.4f} logits_diff={float(d[0] / (d[1] + d[2])):.3e}"
    if args.bench:
        msg += f" us={timed(run, args.iters):.1f}"
    if args.stamps:  # needs MEGA_TS=1: phase stamps of one more (warm) launch
        run()
        torch.cuda.synchronize()
        ts = eng.phase_stamps()
        order = [(0, "start"), (15, "agrows"), (16, "agwb"), (17, "agarr"), (1, "ag"), (2, "agwait"), (12, "hist"), (3, "b1"), (13, "scan"),
                 (14, "place"), (4, "b2"), (5, "G1"), (6, "b3"), (7, "G2"), (8, "b4"),
                 (18, "rsrows"), (19, "rswb"), (20, "rsarr"), (9, "rs"), (10, "rswait"), (21, "y1"), (22, "y1wait"), (11, "y")]
        med, mx = ts.median(0).values, ts.max(0).values
        msg += "\n   " + " ".join(f"{n}={med[i]:.1f}/{mx[i]:.1f}" for i, n in order)
        q = torch.quantile(ts[:, 0].float().cpu(), torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], device="cpu"))
        msg += "\n   start pct 0/10/25/50/75/90/100: " + " ".join(f"{v:.1f}" for v in q.tolist())
        bysm = ts[:, 0].view(-1).tolist()
        msg += "\n   start by cta 0..7: " + " ".join(f"{v:.1f}" for v in bysm[:8]) + " | 256..263: " + " ".join(f"{v:.1f}" for v in bysm[256:264])
        e2 = torch.tensor([eng.poll_errors()], device=dev)
        el = [torch.zeros_like(e2) for _ in range(world)]
        dist.all_gather(el, e2)
        msg += f"\n   err per rank after: {[int(v) for v in el]}"
        mr = med.to(dev).float()
        allm = [torch.zeros_like(mr) for _ in range(world)]
        dist.all_gather(allm, mr)
        for r_, v in enumerate(allm):
            msg += f"\n   rank{r_} med: " + " ".join(f"{n}={v[i]:.1f}" for i, n in order)
        mxr = mx.to(dev).float()
        allx = [torch.zeros_like(mxr) for _ in range(world)]
        dist.all_gather(allx, mxr)
        for r_, v in enumerate(allx):
            msg += f"\n   rank{r_} max: " + " ".join(f"{n}={v[i]:.1f}" for i, n in order)
        slow = int(ts[:, 11].argmax())
        msg += f"\n   slowest cta {slow}: " + " ".join(f"{n}={ts[slow, i]:.1f}" for i, n in order)
    if rank == 0:
        print(msg, flush=True)

dist.barrier()
eng.close()
dist.destroy_process_group()
