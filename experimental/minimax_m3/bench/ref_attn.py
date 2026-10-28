"""One synthetic sparse layer on both backends, intermediate hand-offs compared (TP4).

    torchrun --nproc_per_node 4 bench/debug_layer.py [rows] [ctx] [topk 0|1]"""
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m3_inputs import StandaloneRunner, make_layers, make_step, make_spec_step  # noqa: E402
from m3_inputs import HEAD_DIM as HEAD_DIM_  # noqa: E402


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

        # ---- float reference of the split attention (rank 0's 16 heads), per row, vs both backends' merged outputs
        import math
        lw_ = layers[L - 1]
        eps_ = float(getattr(glu, "eps", 1e-6))
        kvc = lw_.attn.kv_cache.float()  # [nb, 2, 128, 128]
        qraw = qkv_g[:, :2048].view(rows, 16, 128).float()
        gq = lw_.g_q.float()
        cs = lw_.cos_sin.float()  # [pos, 64]: cos 32 | sin 32
        bt = md.decode.block_table
        tbl = sparse_tables("gluon", glu, rows)  # [rows, 17]: page ids by slot, sparse context
        def norm_rope(x, pos):
            rr = torch.rsqrt((x * x).mean(-1, keepdim=True) + eps_)
            n = x * rr * (1.0 + gq)
            c, s = cs[pos, :32], cs[pos, 32:]
            x1, x2 = n[:, :32], n[:, 32:64]
            r1 = x1 * c - x2 * s
            r2 = x2 * c + x1 * s
            return torch.cat([r1, r2, n[:, 64:]], -1)
        def fp8r(x):
            return x.clamp(-448, 448).to(torch.float8_e4m3fn).float()
        def keys_of(r_):
            pos = int(positions[r_]); seq = pos + 1
            if seq > 2048:
                row = tbl[r_].tolist(); sctx = row[16]
                ki = torch.arange(sctx, device=dev)
                pages = torch.tensor(row[:16], device=dev)[ki // 128]
            else:
                ki = torch.arange(seq, device=dev)
                pages = bt[r_ // ql][ki // 128]
            return pages.long(), ki % 128, pos
        def decode_oq(buf, off_w, off_s, r_):
            w = buf[off_w + r_ * 2048: off_w + (r_ + 1) * 2048].view(torch.float8_e4m3fn).float().view(16, 128)
            s = buf[off_s + 8 * r_: off_s + 8 * r_ + 4].view(torch.float32)[0]
            return w * s
        sm = HEAD_DIM_ ** -0.5
        for r_ in range(rows):
            pages, t, pos = keys_of(r_)
            # page-16 layout: K page16 j = [8 chunks][16 keys][16 dims], V page16 j = [128 dims][16 keys]
            kk = kvc[:, 0].reshape(-1, 8, 8, 16, 16)  # [page, page16, chunk, key, dim]
            vv_ = kvc[:, 1].reshape(-1, 8, 128, 16)  # [page, page16, dim, key]
            K = kk[pages, t // 16, :, t % 16, :].reshape(-1, 128)
            V = vv_[pages, t // 16, :, t % 16]
            q = norm_rope(qraw[r_], pos)  # [16, 128]
            s_ = (q @ K.T) * sm
            p = torch.softmax(s_, -1)
            o_f = p @ V
            # the kernels' fp8 path: q bf16 -> fp8, P as fp8(448 p) / 448 (exact products)
            q8 = fp8r(q.to(torch.bfloat16).float())
            s8 = (q8 @ K.T) * sm
            m8 = s8.max(-1, keepdim=True).values
            e8 = torch.exp(s8 - m8)
            l8 = e8.sum(-1, keepdim=True)
            o_8 = (fp8r(448.0 * e8) @ V) / 448.0 / l8
            # the kernels' pipeline: 8 partitions of 256 keys (slot order), per-partition max, fp8 P,
            # bf16 partial o, merge by l exp(m - max m), bf16 average
            n_ = s8.shape[1]
            parts = []
            for p_ in range(0, n_, 256):
                sp = s8[:, p_:p_ + 256]
                mp = sp.max(-1, keepdim=True).values
                ep = torch.exp2((sp - mp) * 1.4426950408889634)
                lp_ = ep.sum(-1, keepdim=True)
                op = ((fp8r(448.0 * ep) @ V[p_:p_ + 256]) / 448.0 / lp_).to(torch.bfloat16).float()
                parts.append((mp, lp_, op))
            gm = torch.cat([m for m, _, _ in parts], 1).max(-1, keepdim=True).values
            ws_ = [l_ * torch.exp2((m - gm) * 1.4426950408889634) for m, l_, _ in parts]
            gs_ = sum(ws_)
            o_k = sum(w / gs_ * o for w, (_, _, o) in zip(ws_, parts)).to(torch.bfloat16).float()
            def oquant(o):  # the kernels' per-token FP8 output (amax / 448 over this rank's 16 heads)
                sc_ = o.abs().max() / 448.0
                return fp8r(o / sc_) * sc_
            o_f = oquant(o_f)
            o_8 = oquant(o_8)
            o_k = oquant(o_k)
            og = decode_oq(sc, R["oq"][0], R["oqs"][0], r_)
            of = decode_oq(fly.scratch4, SCRATCH["attnq"], SCRATCH["attnq_s"], r_)
            print(f"   ref row {r_} (keys {len(t)}): gluon vs ref {nrmse(og, o_f):.2e} emu {nrmse(og, o_8):.2e} kern {nrmse(og, o_k):.2e} | "
                  f"fly vs ref {nrmse(of, o_f):.2e} emu {nrmse(of, o_8):.2e} kern {nrmse(of, o_k):.2e} | gluon vs fly {nrmse(og, of):.2e}")
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
