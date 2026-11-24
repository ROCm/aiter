"""MiniMax-M3.1 baseline with the A8W8 shared expert (routed A4W4 E128 top-4 + shared MXFP8
weights x MXFP8 activations). sep: upstream main's way (routed fused_moe + the shared expert as its
own chain of public ops); fuse: branch binding_moe_m3.1_opt_gfx1250's fused_moe(shared_w*) path
(<= 819 tokens). Derived from baseline_moe_ar.py:

Split-MoE TP baseline: every rank runs fused_moe (grouped gfx1250 path, a4w4) on the
replicated tokens with its inter shard, then all-reduces the output. Timed as one
CUDA graph (MoE + AR) with cuda events, plus each part alone. AR: aiter custom
all-reduce when it takes the shape, and RCCL; the faster one counts.

    torchrun --nproc_per_node=4 baseline_moe_ar.py --tokens 8 32 ... [--inter-total 2560]
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, "/home/binding2/moe_1026/aiter/op_tests/flydsl_tests")
import test_flydsl_grouped_gemm as tg  # noqa: E402

from aiter import ActivationType, QuantType  # noqa: E402
from aiter.fused_moe import fused_moe  # noqa: E402
from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight  # noqa: E402
from aiter.utility import dtypes  # noqa: E402
from aiter.ops.flydsl.moe_common import GateMode  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--tokens", type=int, nargs="+", default=[8, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
p.add_argument("--inter-total", type=int, default=2560)
p.add_argument("--pad-inter", type=int, default=0, help="inter per rank as run (zero-padded)")
p.add_argument("--check-pad", type=int, default=0, help="also run weights zero-padded to this inter and compare outputs")
p.add_argument("--experts", type=int, default=256)
p.add_argument("--topk", type=int, default=6)
p.add_argument("--shared-expert", action="store_true", help="last expert always routed (fused shared)")
p.add_argument("--model-dim", type=int, default=4096)
p.add_argument("--iters", type=int, default=20)
p.add_argument("--act", default="swiglu", choices=("swiglu", "situv2"), help="stage1 activation")
p.add_argument("--situ-beta", type=float, default=4.0, help="SiTUv2 gate beta (Kimi-K3)")
p.add_argument("--situ-linear-beta", type=float, default=25.0, help="SiTUv2 up beta (Kimi-K3)")
p.add_argument("--mode", default="ar", choices=("ar", "ag_rs"),
               help="ar: replicated tokens, MoE + all-reduce; ag_rs: each rank's T/tp own tokens, "
                    "all-gather (x, routing) + MoE + reduce-scatter")
p.add_argument("--variant", default="sep", choices=("sep", "fuse"),
               help="sep: routed fused_moe + the shared expert's own A8W8 chain; fuse: fused_moe(shared_w*)")
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
tg.set_data_format("a4w4")
E, K, H = args.experts, args.topk, args.model_dim
I = args.inter_total // world

from aiter.ops.quant import per_group_quant_hip  # noqa: E402
from aiter.ops.shuffle import shuffle_mxfp8_bpreshuffle_scale, shuffle_weight  # noqa: E402
import aiter  # noqa: E402

assert args.shared_expert and not args.pad_inter and not args.check_pad
ER = E - 1  # routed experts; expert id ER is the shared one
gen = tg.bench_init.make_generator(1000 + rank)
w1 = tg._pattern_packed(ER, 2 * I, H // 2, data_init="uniform", generator=gen)
w2 = tg._pattern_packed(ER, H, I // 2, data_init="uniform", generator=gen)
w1s = tg.init_weight_scales(ER, 2 * I, H // 32, scale_init="auto", generator=gen)
w2s = tg.init_weight_scales(ER, H, I // 32, scale_init="auto", generator=gen)
# the shared expert: logical MXFP8 codes (GGUU w1) + raw e8m0 scales (as moe_1121 m31_bench.shared_weights)
sw1 = tg.bench_init.fill_fp8((2 * I, H), "uniform", gen)
sw2 = tg.bench_init.fill_fp8((H, I), "uniform", gen)
sw1s = tg.init_weight_scales(1, 2 * I, H // 32, scale_init="auto", generator=gen)
sw2s = tg.init_weight_scales(1, H, I // 32, scale_init="auto", generator=gen)
w1g = moe_shuffle_weight(w1, experts_cnt=ER, is_guinterleave=True, gate_up=True).view(dtypes.fp4x2)
w2g = moe_shuffle_weight(w2, experts_cnt=ER).view(dtypes.fp4x2)
w1sg = moe_shuffle_scale(w1s.contiguous(), experts_cnt=ER, is_guinterleave=True, gate_up=True)
w2sg = moe_shuffle_scale(w2s.contiguous(), experts_cnt=ER)
# fused path layouts (GUGU w1, n32k4 scales)
dsh = dict(
    shared_w1=moe_shuffle_weight(sw1.view(1, 2 * I, H), experts_cnt=1, is_guinterleave=True, gate_up=True),
    shared_w2=moe_shuffle_weight(sw2.view(1, H, I), experts_cnt=1),
    shared_w1_scale=moe_shuffle_scale(sw1s.contiguous(), experts_cnt=1, is_guinterleave=True, gate_up=True),
    shared_w2_scale=moe_shuffle_scale(sw2s.contiguous(), experts_cnt=1), shared_expert_id=ER)
# separate chain layouts: dense MXFP8 GEMM operands (GGUU w1 -> gate | up halves)
dw1 = shuffle_weight(sw1.view(2 * I, H), layout=(16, 16))
dw2 = shuffle_weight(sw2.view(H, I), layout=(16, 16))
dw1s = shuffle_mxfp8_bpreshuffle_scale(sw1s.view(2 * I, H // 32).view(dtypes.fp8_e8m0))
dw2s = shuffle_mxfp8_bpreshuffle_scale(sw2s.view(H, I // 32).view(dtypes.fp8_e8m0))
del w1, w2

ca = None
cpu_group = dist.new_group(backend="gloo")
try:
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    ca = CustomAllreduce(cpu_group, dev)
    if ca.disabled:
        ca = None
except Exception as e:  # noqa: BLE001
    if rank == 0:
        print("custom AR unavailable:", e)


# upstream's opt-in plain Gluon all-reduce (gfx1250 TP4, bf16, <= 4 MiB; CudaCommunicator with
# AITER_GLUON_FUSED_AR=1): one more AR candidate. BASE_GLUON_AR=0 skips it.
glu = None
if os.environ.get("BASE_GLUON_AR", "1") == "1" and world == 4:
    try:
        from aiter.ops.triton.comms.all_reduce import all_reduce as gluon_all_reduce, all_reduce_init

        glu = all_reduce_init(cpu_group, 4 << 20)
    except Exception as e:  # noqa: BLE001
        glu = None
        if rank == 0:
            print("gluon AR unavailable:", str(e)[:200])


def timed(fn, iters, cap=None):
    """us per call of fn, as a graph of iters calls replayed (cuda events). cap: capture
    under the custom-AR context (its kernels' graph buffers); default: AR mode only."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    # (ag_rs uses RCCL only: no custom-AR capture context, whose capture breaks large
    # all-gathers in the graph: ncclUnhandledCudaError from 8192 tokens)
    if cap is None:
        cap = args.mode == "ar"
    ctx = ca.capture() if ca is not None and cap else torch.no_grad()
    with ctx:
        with torch.cuda.graph(g, stream=s):
            for _ in range(iters):
                fn()
    torch.cuda.synchronize()
    for _ in range(2):
        g.replay()
    dist.barrier()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = 1e30
    for _ in range(5):
        st.record()
        g.replay()
        en.record()
        torch.cuda.synchronize()
        best = min(best, st.elapsed_time(en) * 1e3 / iters)
    t = torch.tensor([best], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t)


# stage1 activation (Kimi-K3: SiTUv2 with its betas; others: SwiGLU, limit 7)
ACT_KW = (dict(activation=ActivationType.Swiglu, swiglu_limit=7.0) if args.act == "swiglu" else
          dict(activation=ActivationType.Situv2, beta=args.situ_beta, linear_beta=args.situ_linear_beta))
def timed_eager(fn, iters):
    """us per call of fn, iters back-to-back eager calls between cuda events (rank max): the
    ag_rs baseline at >= 8192 tokens, where RCCL all-gather cannot be graph-captured here."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = 1e30
    for _ in range(5):
        dist.barrier()
        torch.cuda.synchronize()
        st.record()
        for _ in range(iters):
            fn()
        en.record()
        torch.cuda.synchronize()
        best = min(best, st.elapsed_time(en) * 1e3 / iters)
    t = torch.tensor([best], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t)


rows = []
for T in args.tokens:
    g2 = torch.Generator(device=dev).manual_seed(T)
    x = (torch.randn((T, H), device=dev, generator=g2) * 0.5).to(torch.bfloat16)
    dist.broadcast(x, 0)
    score = torch.randn((T, E), device=dev, generator=g2)
    dist.broadcast(score, 0)
    tw, ids = route(score, K, args.shared_expert)
    tw, ids = tw.to(torch.bfloat16), ids.to(torch.int32)
    if args.variant == "sep":  # the routed MoE takes the 4 routed columns (the shared weight is 1)
        tw, ids = tw[:, :K - 1].contiguous(), ids[:, :K - 1].contiguous()
    out = {}
    y1b = torch.empty((T, 2 * I), dtype=torch.bfloat16, device=dev)
    actb = torch.empty((T, I), dtype=torch.bfloat16, device=dev)
    y2b = torch.empty((T, H), dtype=torch.bfloat16, device=dev)

    def moe_impl(xx, tww, idss):
        if args.variant == "fuse":
            return fused_moe(xx, w1g, w2g, tww, idss, quant_type=QuantType.per_1x32, w1_scale=w1sg,
                             w2_scale=w2sg, gate_mode=GateMode.INTERLEAVE.value, dtype=dtypes.bf16,
                             **ACT_KW, **dsh)
        o = fused_moe(xx, w1g, w2g, tww, idss, quant_type=QuantType.per_1x32, w1_scale=w1sg,
                      w2_scale=w2sg, gate_mode=GateMode.INTERLEAVE.value, dtype=dtypes.bf16, **ACT_KW)
        xq, xs = per_group_quant_hip(xx, quant_dtype=dtypes.fp8, group_size=32, scale_type=dtypes.fp8_e8m0)
        aiter.gemm_a8w8_mxfp8_bpreshuffle(xq, dw1, xs, dw1s, dtypes.bf16, out=y1b)
        aiter.swiglu_and_mul(actb, y1b)
        aq, a_s = per_group_quant_hip(actb, quant_dtype=dtypes.fp8, group_size=32, scale_type=dtypes.fp8_e8m0)
        aiter.gemm_a8w8_mxfp8_bpreshuffle(aq, dw2, a_s, dw2s, dtypes.bf16, out=y2b)
        return o.add_(y2b)

    def moe():
        out["y"] = moe_impl(x, tw, ids)
        return out["y"]

    moe()
    y0 = out["y"].clone()
    if args.check_pad:
        yr = fused_moe(x, rw1g, rw2g, tw, ids, quant_type=QuantType.per_1x32, w1_scale=rw1sg,
                       w2_scale=rw2sg, gate_mode=GateMode.INTERLEAVE.value, dtype=dtypes.bf16,
                       **ACT_KW)
        d = (y0.float() - yr.float()).norm() / yr.float().norm().clamp_min(1e-20)
        nan = int(torch.isnan(y0).any())
        if rank == 0:
            print(f"T={T:6d} check_pad rel_l2_vs_pad{args.check_pad}={float(d):.3e} nan={nan}", flush=True)

    if args.mode == "ag_rs":
        # each rank owns T / tp tokens (and their routing): all-gather them, the MoE on every
        # token (this rank's inter shard), reduce-scatter the output back to the owners
        m_own = T // world
        x_own = x[rank * m_own:(rank + 1) * m_own].contiguous()
        tw_own = tw[rank * m_own:(rank + 1) * m_own].contiguous()
        ids_own = ids[rank * m_own:(rank + 1) * m_own].contiguous()
        xg, twg, idsg = torch.empty_like(x), torch.empty_like(tw), torch.empty_like(ids)
        y_own = torch.empty((m_own, H), dtype=torch.bfloat16, device=dev)

        gat = {"x": xg, "tw": twg, "ids": idsg}

        def ag():
            dist.all_gather_into_tensor(xg, x_own)
            dist.all_gather_into_tensor(twg, tw_own)
            dist.all_gather_into_tensor(idsg, ids_own)
            gat["x"], gat["tw"], gat["ids"] = xg, twg, idsg

        # upstream's dispatch (CudaCommunicator.all_gather / reduce_scatter): the custom kernel
        # (gfx1250: Lamport all-gather) where it takes the tensor, RCCL otherwise
        ca_ag = [ca is not None and bool(ca.should_custom_ag(t)) for t in (x_own, tw_own, ids_own)]

        def ag_up():
            for k, own, dst, c in (("x", x_own, xg, ca_ag[0]), ("tw", tw_own, twg, ca_ag[1]),
                                   ("ids", ids_own, idsg, ca_ag[2])):
                if c:
                    gat[k] = ca.custom_all_gather(own, 0)
                else:
                    dist.all_gather_into_tensor(dst, own)
                    gat[k] = dst

        def moe_g():
            out["y"] = moe_impl(gat["x"], gat["tw"], gat["ids"])

        def rs():
            dist.reduce_scatter_tensor(y_own, out["y"])

        def rs_up():
            ca.custom_reduce_scatter(out["y"], y_own, 0)

        ag()
        moe_g()
        ca_rs = ca is not None and bool(ca.should_custom_rs(out["y"], 0))
        big = T >= 8192
        tf = timed_eager if big else timed
        tfc = timed_eager if big else (lambda f, n: timed(f, n, cap=True))
        t_ag_r = tf(ag, args.iters)
        t_ag_c = tfc(ag_up, args.iters) if any(ca_ag) else float("inf")
        ag()
        t_moe = tf(moe_g, args.iters)
        t_rs_r = tf(rs, args.iters)
        t_rs_c = tfc(rs_up, args.iters) if ca_rs else float("inf")
        ag_fn = ag_up if t_ag_c < t_ag_r else ag
        rs_fn = rs_up if t_rs_c < t_rs_r else rs
        up = ag_fn is ag_up or rs_fn is rs_up
        t_tot_r = tf(lambda: (ag(), moe_g(), rs()), args.iters)  # RCCL only (the 11-14 baseline)
        t_tot = (tfc if up else tf)(lambda: (ag_fn(), moe_g(), rs_fn()), args.iters) if up else t_tot_r
        t_tot = min(t_tot, t_tot_r)
        t_ag, t_rs = min(t_ag_r, t_ag_c), min(t_rs_r, t_rs_c)
        rows.append((T, t_moe, t_ag, t_rs, t_tot, "ag_rs"))
        if rank == 0:
            print(f"T={T:6d} moe={t_moe:8.1f} ag={t_ag:8.1f} rs={t_rs:8.1f} "
                  f"ag_rccl={t_ag_r:.1f} ag_ca={t_ag_c:.1f}{ca_ag} rs_rccl={t_rs_r:.1f} rs_ca={t_rs_c:.1f} "
                  f"rccl_tot={t_tot_r:.1f} ag+moe+rs={t_tot:8.1f} (ag_rs{', eager' if big else ''})", flush=True)
        continue

    def ar_rccl():
        dist.all_reduce(out["y"])

    def ar_ca():
        out["r"] = ca.custom_all_reduce(out["y"])

    def ar_glu():
        out["r"] = gluon_all_reduce(out["y"], cpu_group, *glu)

    use_ca = ca is not None and ca.should_custom_ar(y0)
    use_glu = (glu is not None and y0.is_contiguous() and y0.numel() % 4 == 0
               and y0.data_ptr() % 16 == 0 and y0.numel() * 2 <= (4 << 20))
    lam = ca.select_lamport_ar(y0) if use_ca and hasattr(ca, "select_lamport_ar") else None
    t_moe = timed(moe, args.iters)
    t_rccl = timed(ar_rccl, args.iters)
    t_ca = timed(ar_ca, args.iters) if use_ca else float("inf")
    t_glu = float("inf")
    if use_glu:
        try:
            ar_glu()
            t_glu = timed(ar_glu, args.iters)
        except Exception as e:  # noqa: BLE001
            if rank == 0:
                print("gluon AR failed:", str(e)[:200])
    cands = {"rccl": (t_rccl, ar_rccl), "ca": (t_ca, ar_ca), "gluon": (t_glu, ar_glu)}
    best = min(cands, key=lambda k: cands[k][0])
    ar_fn = cands[best][1]
    t_tot = timed(lambda: (moe(), ar_fn()), args.iters)
    rows.append((T, t_moe, t_rccl, t_ca, t_tot, best))
    if rank == 0:
        print(f"T={T:6d} moe={t_moe:8.1f} ar_rccl={t_rccl:8.1f} ar_ca={t_ca:8.1f} "
              f"ar_gluon={t_glu:.1f} lamport={ {None: 'no', False: '1shot', True: '2shot'}[lam] } "
              f"moe+ar={t_tot:8.1f} ({best})", flush=True)

dist.barrier()
if glu is not None:
    try:
        from aiter.ops.triton.comms.all_reduce import all_reduce_free

        all_reduce_free(cpu_group, glu[0])
    except Exception:  # noqa: BLE001
        pass
if ca is not None:
    ca.close()
dist.destroy_process_group()
