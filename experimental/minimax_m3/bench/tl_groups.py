"""Group-to-group layer breakdown, FlyDSL vs Gluon, from saved per-layer timelines (bench_agentx
--timeline with M3_TL_SAVE). Each group ends at a boundary = the latest CTA's stamp of that
event (critical path); a group's time = its boundary minus the previous one, so the groups sum to
the layer period (layer start to the next layer's start).

    python3 tl_groups.py PREFIX CTX   (reads PREFIX_flydsl_CTX.pt and PREFIX_gluon_CTX.pt)"""
import sys
import torch

# boundary -> stamp(s) (max over CTAs; a list: the latest of them); "sel" only on selecting layers
B = {
    "flydsl": {"norm": [27], "x": [28], "qkv": {"reuse": [31], "select": [30]}, "sel": [19], "split": [15],
               "oin": [3], "o": [4], "ra": [22], "router": [23], "routed": [6], "ug": [8], "down": [11]},
    "gluon": {"norm": [51], "x": [21], "qkv": [1], "sel": [45], "split": [16],
              "oin": [33], "o": [3], "ra": [17], "router": [19], "routed": [6], "ug": [8, 12], "down": [39, 11]},
}
# S = 8 (wide MoE: slots 49-56 reused, down ends at 11, the norm tasks' last stamp 55)
# (ROUTER_AAG, S >= 8: the o tasks end at their partial puts; E2 = merge + o GEMM + partials, F1 = AG_RS + hop to the router)
B8 = {"flydsl": B["flydsl"],
      "gluon": {"norm": [55], "x": [21], "qkv": [1], "sel": [37], "split": [16],
                "oin": [33], "o": [3], "ra": [17], "router": [19], "routed": [6], "ug": [8], "down": [11]}}
import os
if os.environ.get("TLG_S8"):
    B = B8
MODE = "max"
GROUPS = [
    ("A1", "  down + FFN all-reduce end -> next launch start", None),
    ("A2", "  layer start -> residual + RMSNorm done", "norm"),
    ("A3", "  normed x landed at the QKV CTAs", "x"),
    ("B", "QKV GEMV + q/k norm, RoPE, KV write", "qkv"),
    ("C", "Indexer scores + top-16 selection", "sel"),
    ("D", "Split (flash-decode) attention", "split"),
    ("E1", "  split end -> partitions landed at the o CTAs", "oin"),
    ("E2", "  merge + o-proj + o all-reduce", "o"),
    ("F1", "  o end -> attention output landed at the router", "ra"),
    ("F2", "  router: residual, RMSNorm, gate logits", "router"),
    ("G", "Routing: top-4 + hand-off to experts", "routed"),
    ("H", "Expert up/gate GEMV (routed + shared)", "ug"),
    ("I", "Expert down GEMV + FFN all-reduce", "down"),
]


def load(path):
    tl = torch.load(path).double()
    n = int(sum(bool((tl[l, :, 0] > 0).any()) for l in range(tl.shape[0])))
    return tl[:n]


def bounds(tl, be):
    """Per layer: t0 (us, absolute) and each boundary (us after that layer's t0)."""
    out = []
    for l in range(tl.shape[0]):
        kind = "select" if l % 4 == 0 else "reuse"
        t0 = tl[l, :, 0][tl[l, :, 0] > 0].min()
        d = {"t0": t0 / 100, "kind": kind}
        for g, st in B[be].items():
            if isinstance(st, dict):
                st = st[kind]
            if g == "sel" and kind == "reuse":
                continue
            vs = [tl[l, :, k][tl[l, :, k] > 0] for k in st]
            vs = [(v.max() if MODE == "max" else v.median()) for v in vs if len(v)]
            d[g] = (max(vs) - t0).item() / 100 if vs else None
        out.append(d)
    return out


def groups(bl):
    """Per layer (1 .. n-2) group times; returns {kind: {group: mean us}}."""
    acc = {"select": {}, "reuse": {}}
    cnt = {"select": 0, "reuse": 0}
    for l in range(1, len(bl) - 1):
        d, p = bl[l], bl[l - 1]
        prev_period = (d["t0"] - p["t0"]).item()
        period = (bl[l + 1]["t0"] - d["t0"]).item()
        t = {}
        # A spans the boundary: previous layer's down end -> this layer's x staged
        t["A1"] = prev_period - p["down"]
        last = 0.0
        for key, _, b in GROUPS[1:]:
            if b not in d:
                t[key] = 0.0
                continue
            if d[b] is None:  # (e.g. PRE_NORM: the norm ran in the previous launch's tail)
                t[key] = 0.0
                continue
            t[key] = max(d[b] - last, 0.0)
            last = max(last, d[b])
        t["rest"] = period - last  # down end -> next layer start (counted in the next layer's A)
        t["period"] = period
        k = d["kind"]
        cnt[k] += 1
        for g, v in t.items():
            acc[k][g] = acc[k].get(g, 0.0) + v
    return {k: {g: v / cnt[k] for g, v in acc[k].items()} for k in acc}, cnt


def main():
    global MODE
    pre, ctx = sys.argv[1], sys.argv[2]
    MODE = sys.argv[3] if len(sys.argv) > 3 else "max"
    res, cnt = {}, None
    for be in ("flydsl", "gluon"):
        res[be], cnt = groups(bounds(load(f"{pre}_{be}_{ctx}.pt"), be))
    nsel, nre = 15, 42  # of 57 layers: 0, 4, .. 56 select
    print(f"ctx {ctx}: us per layer ({MODE} over CTAs at each boundary), selecting / reusing layers; per step = 15 sel + 42 reuse")
    print(f"  {'group':52s} {'fly sel':>8s} {'glu sel':>8s} {'fly re':>8s} {'glu re':>8s} {'fly/step':>9s} {'glu/step':>9s} {'delta':>7s}")
    tot = [0.0, 0.0]
    for key, name, _ in GROUPS:
        fs, gs = res["flydsl"]["select"][key], res["gluon"]["select"][key]
        fr, gr = res["flydsl"]["reuse"][key], res["gluon"]["reuse"][key]
        fstep, gstep = fs * nsel + fr * nre, gs * nsel + gr * nre
        tot[0] += fstep; tot[1] += gstep
        print(f"  {key:2s} {name:50s} {fs:8.2f} {gs:8.2f} {fr:8.2f} {gr:8.2f} {fstep:9.1f} {gstep:9.1f} {gstep - fstep:+7.1f}")
    fs, gs = res["flydsl"]["select"]["period"], res["gluon"]["select"]["period"]
    fr, gr = res["flydsl"]["reuse"]["period"], res["gluon"]["reuse"]["period"]
    print(f"  {'  layer period':52s} {fs:8.2f} {gs:8.2f} {fr:8.2f} {gr:8.2f} {fs*nsel+fr*nre:9.1f} {gs*nsel+gr*nre:9.1f} {(gs*nsel+gr*nre)-(fs*nsel+fr*nre):+7.1f}")


if __name__ == "__main__":
    main()
