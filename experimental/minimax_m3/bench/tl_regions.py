import torch
# boundary: (name, fly slot, fly stat, gluon slot, gluon stat)
B = [("input arrived", 27, "med", 20, "med"), ("x normed+quantized", 28, "med", 21, "med"),
     ("qkv gemv done", 29, "med", 1, "med"), ("split q ready", 12, "med", 13, "med"),
     ("split end (slowest)", 1, "max", 2, "max"), ("o has parts", 3, "med", 11, "med"),
     ("o end (slowest)", 4, "max", 3, "max"), ("router has input", 22, "med", 17, "med"),
     ("scores published", 23, "max", 19, "max"), ("routed", 2, "med", 6, "med"),
     ("up/gate done", 8, "med", 8, "med"), ("mids staged", 10, "med", 9, "med"),
     ("down+allreduce done", 11, "med", 5, "med")]
SEG = ["layer boundary: launch gap + input poll + residual", "RMSnorm + FP8 quant + stage x", "qkv GEMV",
       "head hand-off, q/k norm+RoPE, KV loads", "split attention (slowest split)", "split -> o merge hop",
       "o GEMV + all-reduce", "o -> router hop", "router logits/scores/publish", "scores -> routed (top-k)",
       "xn stage + routed up/gate", "up/gate -> mids hop", "down GEMV + all-reduce"]
PER = {1: (39.1, 46.1)}
def pt(tl, k, stat):
    t = tl[1].double(); t0 = t[:, 0][t[:, 0] > 0].min(); v = t[:, k]; v = (v[v > 0] - t0) / 100
    return (v.median() if stat == "med" else v.max()).item()
for rows in (1,):
    f = torch.load(f"bench/logs/tl_flydsl_r0_rows{rows}_ctx1536.pt"); g = torch.load(f"bench/logs/tl_gluon_r0_rows{rows}_ctx1536.pt")
    pf, pg = PER[rows]
    fb = [pt(f, b[1], b[2]) for b in B]; gb = [pt(g, b[3], b[4]) for b in B]
    fb = [fb[-1] - pf] + fb; gb = [gb[-1] - pg] + gb
    print(f"rows {rows}")
    for i, s in enumerate(SEG):
        df, dg = fb[i+1] - fb[i], gb[i+1] - gb[i]
        print(f"  {i+1:2d} {s:48s} {df:6.2f} {dg:6.2f} {dg-df:+6.2f}")
    print(f"     {'total':48s} {pf:6.2f} {pg:6.2f} {pg-pf:+6.2f}")
