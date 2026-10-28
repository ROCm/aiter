"""One synthetic sparse layer on both backends, intermediate hand-offs compared (TP4).

    torchrun --nproc_per_node 4 bench/debug_layer.py [rows] [ctx] [topk 0|1]"""
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m3_inputs import StandaloneRunner, make_layers, make_step, make_spec_step  # noqa: E402


def nrmse(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def pairs(buf, off, n):
    """n tagged pairs at byte offset off of a uint8 scratch: (values i32, tags i32)."""
    w = buf[off: off + 8 * n].view(torch.int32).view(n, 2)
    return w[:, 0], w[:, 1]


def bf16_pairs(buf, off, n):
    v, t = pairs(buf, off, n)
    return v.contiguous().view(torch.bfloat16).float(), t


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo")
    group = dist.new_group(backend="gloo")
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 1536
    topk = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    L = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    ql = int(sys.argv[5]) if len(sys.argv) > 5 else 1  # > 1: a speculative-verify step, rows = requests * ql
    reqs = rows
    rows = reqs * ql
    nb = int(os.environ.get("M3_NB", 16 * ((ctx + 127) // 128) + 8))  # M3_NB: the bench's page count
    layers = make_layers(L, nb, rank, dev, max_pos=ctx + ql + 16)
    layers[0].index_topk = bool(topk)
    from vllm.models.minimax_m3.amd.mono.gluon import layout as LY
    from vllm.models.minimax_m3.amd.mono.gluon.runner import GluonStandalone
    from vllm.models.minimax_m3.amd.mono.kernels.pre_attn import SCRATCH_QKV
    from vllm.models.minimax_m3.amd.mono.layout import SCRATCH

    fly = StandaloneRunner(layers, rank, group, dev)
    glu = GluonStandalone(layers, rank, group, dev)
    if ql > 1:
        glu.ql = ql
        md, positions = make_spec_step(reqs, ql, ctx, nb, dev)
    else:
        md, positions = make_step(rows, ctx, nb, dev)
    fly.md = glu.md = md
    assert fly.prepare(rows) and glu.prepare(rows)
    g = torch.Generator(device=dev).manual_seed(5)
    hidden = torch.randn(rows, 6144, generator=g, device=dev).to(torch.bfloat16)
    resid = (torch.randn(rows, 6144, generator=g, device=dev) * 4).to(torch.bfloat16)
    kv0 = [lw.attn.kv_cache.clone() for lw in layers]
    ic0 = [lw.index_cache.kv_cache.clone() for lw in layers]
    hf, rf = fly.forward(positions, hidden, resid)
    torch.cuda.synchronize()
    hf, rf = hf.clone(), rf.clone()
    kv_f = layers[0].attn.kv_cache.clone()
    ic_f = layers[0].index_cache.kv_cache.clone()
    qkv_f = pairs(fly.scratch1, SCRATCH_QKV, rows * 2560)[0].view(torch.float32).view(rows, 2560)
    h_f = fly.h[:rows].float()
    q_f = fly.q[:rows].float()
    a_f = bf16_pairs(fly.scratch4, SCRATCH["a"], rows * 3072)[0].view(rows, 6144)
    sel_f = pairs(fly.scratch4, SCRATCH["sel"], rows * 10)[0].view(rows, 10)
    mid_f = pairs(fly.scratch4, SCRATCH["mid"], rows * 5 * 768)[0].view(torch.float32).view(rows, 5, 768).clone()
    xn8_f = pairs(fly.scratch4, SCRATCH["xn8"], rows * 1536)[0].view(rows, 1536).clone()
    for lw, a, b in zip(layers, kv0, ic0):
        lw.attn.kv_cache.copy_(a)
        lw.index_cache.kv_cache.copy_(b)
    if os.environ.get("PTRS"):
        lw = layers[0]
        mp = dict(w2=lw.w2, s2=lw.s2, w13=lw.w13, s13=lw.s13, scratch=glu.scratch, errors=glu.errors,
                  trace=glu.trace, ar_out=glu.ar_out, hm_out=glu.hm_out, epoch=glu.epoch, ptab=glu._ptab_tensor())
        for k_, v_ in mp.items():
            print(f"PTR r{rank} {k_} {v_.data_ptr():#x} .. {v_.data_ptr() + v_.numel() * v_.element_size():#x}",
                  flush=True)
        print(f"PTR r{rank} peers.local {glu.peers.local:#x} half {LY.peer_half():#x}", flush=True)
    hg, rg = glu.forward(positions, hidden, resid)
    torch.cuda.synchronize()
    glu.raise_on_errors()
    if int(os.environ.get("VLLM_M3_GLUON_STAGES", "0")) & (131072 | 4194304):
        tv = glu.trace.view(-1, 256, 32)[0, :192].view(192, 8, 4)
        if rank == 0:
            for i in range(4):
                u, c = tv[:, :, i].unique(return_counts=True)
                print(f"  E[{i}] values across CTA x wave: {dict(zip(u.tolist(), c.tolist()))}", flush=True)
    kv_g = layers[0].attn.kv_cache.clone()
    ic_g = layers[0].index_cache.kv_cache.clone()
    par = (L - 1) % 2  # the last layer's regions (FlyDSL's scratch holds the last layer's too)
    half_ = LY.StepLayout().half()
    sc = glu.scratch[par * half_:(par + 1) * half_]
    R = LY.StepLayout().regions()
    qkv_g = bf16_pairs(sc, R["qkv"][0], rows * 1280)[0].view(rows, 2560)
    h_g = bf16_pairs(sc, R["h"][0], rows * 3072)[0].view(rows, 6144)
    a_g = bf16_pairs(sc, R["a"][0], rows * 3072)[0].view(rows, 6144)
    hm_g = bf16_pairs(sc, R["hm"][0], rows * 3072)[0].view(rows, 6144)
    mid_g = bf16_pairs(sc, R["mid"][0], rows * 5 * 384)[0].view(rows, 5, 768)
    xn8_g = pairs(sc, R["xn8"][0], rows * 1536)[0].view(rows, 1536)
    scv, _ = pairs(sc, R["sc"][0], 8 * 4 * 128 * 2)
    sig_g = scv.view(8, 4, 128, 2)[0, :rows, :, 1].contiguous().view(torch.float32)
    if rank == 0:
        def rep(name, a, b):
            print(f"  {name:10s} nRMSE {nrmse(a, b):.3e}   |gluon| {a.float().norm():.3e} |fly| {b.float().norm():.3e}",
                  flush=True)

        print(f"rows={rows} (reqs {reqs} x ql {ql}) ctx={ctx} topk={topk} L={L}")
        rep("qkv rows", qkv_g, qkv_f)
        rep("q rows", qkv_g[:, :2048], qkv_f[:, :2048])
        rep("k rows", qkv_g[:, 2048:2176], qkv_f[:, 2048:2176])
        rep("v rows", qkv_g[:, 2176:2304], qkv_f[:, 2176:2304])
        rep("ik rows", qkv_g[:, 2432:], qkv_f[:, 2432:])
        rep("h", h_g, h_f)
        rep("kv cache", kv_g.float(), kv_f.float())
        rep("idx cache", ic_g.float(), ic_f.float())
        rep("a (sum)", a_g, a_f)
        print("   a per row:", ["%.2e" % nrmse(a_g[r_], a_f[r_]) for r_ in range(rows)])
        print("   h_mid out per row:", ["%.2e" % nrmse(rg[r_], rf[r_]) for r_ in range(rows)])
        rep("h_mid", hm_g, rf)
        rep("h_mid out", rg, rf)
        rep("ar out", hg, hf)
        print("  xn8 words equal", float((xn8_g == xn8_f).float().mean()))
        for sl in range(5):
            rep(f"mid s{sl}", mid_g[:, sl], mid_f[:, sl])
            bad = ((mid_g[0, sl] - mid_f[0, sl]).abs() > 1e-6).view(48, 16).any(1).nonzero().flatten().tolist()
            if bad:
                print(f"    slot {sl} wrong 16-row tasks: {bad[:6]}.. ({len(bad)})")
                d = (mid_g[0, sl] - mid_f[0, sl]).abs() > 1e-6
                print(f"    wrong rows {int(d.sum())}/768; by row%16: {d.view(48, 16).sum(0).tolist()}")
                print(f"    task0 glu {mid_g[0, sl, :8].tolist()}")
                print(f"    task0 fly {mid_f[0, sl, :8].tolist()}")
        from bench_agentx import sparse_tables
        if topk or L > 4:
            tb, tf = sparse_tables("gluon", glu, rows), sparse_tables("flydsl", fly, rows)
            print("  tables", "EQUAL" if torch.equal(tb, tf) else "DIFFER")
            for r_ in range(rows):
                if not torch.equal(tb[r_], tf[r_]):
                    print(f"   row {r_} glu {tb[r_].tolist()}\n   row {r_} fly {tf[r_].tolist()}")
            nblk = (ctx + ql + 127) // 128
            isc_g = pairs(sc, R["isc"][0], rows * 8192)[0].view(torch.float32).view(rows, 8192)[:, :nblk]
            isc_f = pairs(fly.scratch4, SCRATCH["iscore"], rows * 8192)[0].view(torch.float32).view(rows, 8192)[:, :nblk]
            rep("idx scores", isc_g, isc_f)
            d = (isc_g - isc_f).abs()
            print("   max |dscore|", d.max().item(), "n diff", int((d > 0).sum()), "/", d.numel())
            print("   glu scores row0", isc_g[0, :6].tolist())
            print("   fly scores row0", isc_f[0, :6].tolist())
        print("  fly route", sel_f[:, :5].tolist(), sel_f[:, 5:].view(torch.float32).tolist())
        top = sig_g.topk(6, dim=-1)
        print("  glu sig top", top.indices.tolist(), top.values.tolist())
    dist.barrier(group=group)
    fly.close()
    glu.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
