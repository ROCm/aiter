"""SILOTIGER-1277 iter028: end-to-end TP8 gate for the FUSED producer+acc_agg
high-T router branch (``attn_res_allreduce_router(..., fuse_producer_agg=True)``).

The non-overlapped high-T branch normally runs the online producer then acc_agg as
TWO kernels. With ``fuse_producer_agg=True`` and no overlapped ``bank_acc`` supplied,
the router folds both into ONE post-AR kernel (``fused_producer_accagg``), saving the
acc[T,H] HBM round-trip and one dispatch (~1.27-1.32x on the producer+agg marginal
cost, iter028). This test proves that fused path is correct against the SAME torch
fp32 reference used by the acc_agg gate, on the REAL production (gloo/one-shot) AR
output, and that it dispatches only on the high-T branch (fused one-shot still wins
T<=t_fused regardless of the flag).

Run: TP=8 python op_tests/multigpu_tests/test_attn_res_fused_producer_agg.py [out.json]
"""
import os
import sys
import json
import logging
from multiprocessing import Pool, freeze_support, set_start_method

import torch
import torch.distributed as dist

sys.path.insert(0, "/home/mvedenev/aiter")
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.ops.flydsl.one_shot_allreduce import OneShotAllReduce, OneShotAllReduceAttnRes
from aiter.ops.flydsl.attn_res_acc_agg import (
    fused_producer_accagg,
    attn_res_allreduce_router,
    DEFAULT_T_FUSED,
)

logger = logging.getLogger("attn_res_fused_producer_agg")

H = 7168
NB = 9
SCORE_EPS = 1e-6
OUT_EPS = 1e-6
MAX_BYTES = 8 * 1024 * 1024
TOKENS = [1, 8, 9, 12, 16, 20, 24, 32, 64]
NVBS = [8, 4]
SUPPORTED_WORLDS = (2, 4, 8)


def _ulp(a, b):
    return ((a.float() - b.float()).abs()
            / (b.float().abs().clamp_min(1e-3) * 2.0 ** -7)).max().item()


def _agg_reference(prefix, addend, bank, nvb, cw, ow, out_norm):
    pv = (prefix.float() + addend.float()).to(torch.bfloat16).float()
    v = bank[:, :nvb, :].float()
    cwf = cw.float().view(1, 1, H)
    sr = (v * cwf).sum(-1) * torch.rsqrt((v * v).sum(-1) * (1.0 / H) + SCORE_EPS)
    psc = (pv * cw.float()).sum(-1) * torch.rsqrt((pv * pv).sum(-1) * (1.0 / H) + SCORE_EPS)
    mb = sr.max(-1).values
    m = torch.maximum(mb, psc)
    resc = torch.exp(mb - m)
    pw = torch.exp(psc - m)
    w = torch.exp(sr - mb.unsqueeze(-1))
    sb = w.sum(-1)
    acc = torch.einsum("tr,trh->th", w, v)
    inv = 1.0 / (sb * resc + pw)
    mix = (pw * inv).unsqueeze(-1) * pv + (resc * inv).unsqueeze(-1) * acc
    if out_norm:
        o = mix * torch.rsqrt((mix * mix).sum(-1) * (1.0 / H) + OUT_EPS).unsqueeze(-1) * ow.float()
    else:
        o = mix
    return o


def _run_rank(rank, tp, init_method, t_fused):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("gloo", init_method=init_method, world_size=tp, rank=rank)
    group = dist.group.WORLD
    eng = OneShotAllReduceAttnRes(group=group, device=device, rank=rank,
                                  world_size=tp, max_bytes=MAX_BYTES)
    plain = OneShotAllReduce(group=group, device=device, rank=rank,
                             world_size=tp, max_bytes=MAX_BYTES)

    def all_reduce_fn(x):
        red = torch.empty_like(x)
        plain.allreduce(x, red)
        return red

    torch.manual_seed(0)
    cw = (torch.randn(H, device=device) * 0.05).float()
    ow = (1 + 0.1 * torch.randn(H, device=device)).bfloat16()
    rows = []
    try:
        for t in TOKENS:
            for nvb in NVBS:
                for out_norm in (True, False):
                    torch.manual_seed(1000 + t * 10 + nvb)
                    prefix = (4 * torch.randn(t, H, device=device)).bfloat16()
                    bank = (4 * torch.randn(t, NB, H, device=device)).bfloat16()
                    torch.manual_seed(5000 + rank * 97 + t * 10 + nvb)
                    inp = torch.randn(t, H, device=device).bfloat16()
                    owv = ow if out_norm else None

                    # Router with the fused flag + NO overlapped producer (non-overlapped
                    # path). For T>t_fused it must take the fused producer+agg kernel.
                    out_r, pout_r = attn_res_allreduce_router(
                        eng, inp.clone(), prefix, bank, nvb, cw, owv,
                        SCORE_EPS, OUT_EPS, t_fused=t_fused,
                        all_reduce_fn=all_reduce_fn, fuse_producer_agg=True,
                    )

                    addend = all_reduce_fn(inp)
                    ref = _agg_reference(prefix, addend, bank, nvb, cw,
                                         ow if out_norm else None, out_norm)
                    pref = (prefix.float() + addend.float()).to(torch.bfloat16)
                    u = _ulp(out_r, ref)
                    pe = torch.equal(pout_r, pref)
                    fin = bool(torch.isfinite(out_r.float()).all())

                    if t > t_fused:
                        # route_match: router output == a direct fused_producer_accagg call
                        out_d, pout_d = fused_producer_accagg(
                            prefix, addend, bank, cw, owv, nvb, SCORE_EPS, OUT_EPS,
                            write_prefix=True)
                        route_match = torch.equal(out_r, out_d) and torch.equal(pout_r, pout_d)
                        branch = "fused_pa"
                    else:
                        # flag inert at low T: router with the flag OFF must be BIT-IDENTICAL
                        # (both take the one-shot fused AR with the SAME online producer).
                        out_d, pout_d = attn_res_allreduce_router(
                            eng, inp.clone(), prefix, bank, nvb, cw, owv,
                            SCORE_EPS, OUT_EPS, t_fused=t_fused,
                            all_reduce_fn=all_reduce_fn, fuse_producer_agg=False)
                        route_match = torch.equal(out_r, out_d) and torch.equal(pout_r, pout_d)
                        branch = "fused_1shot"

                    ok = bool(u <= 1.0 and pe and fin and route_match)
                    rows.append(dict(t=t, nvb=nvb, out_norm=out_norm, branch=branch,
                                     ulp=round(u, 3), prefix_exact=pe, finite=fin,
                                     route_match=bool(route_match), ok=ok))
    finally:
        eng.close()
        plain.close()
        dist.destroy_process_group()
    return {"rank": rank, "rows": rows}


def main():
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch not in ("gfx942", "gfx950"):
        logger.warning("fused producer+agg gate unsupported on %s; skipping", arch)
        return
    tp = int(os.environ.get("TP", "8"))
    if tp not in SUPPORTED_WORLDS:
        raise ValueError(f"unsupported world_size={tp}")
    if torch.cuda.device_count() < tp:
        logger.warning("need %d GPUs, have %d; skipping", tp, torch.cuda.device_count())
        return
    t_fused = int(os.environ.get("T_FUSED", str(DEFAULT_T_FUSED)))
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    set_start_method("spawn", force=True)
    with Pool(processes=tp) as pool:
        res = pool.starmap(_run_rank, [(r, tp, init_method, t_fused) for r in range(tp)])

    agg = {}
    for per in res:
        for row in per["rows"]:
            k = (row["t"], row["nvb"], row["out_norm"])
            cur = agg.get(k)
            if cur is None:
                agg[k] = dict(row)
            else:
                cur["ulp"] = max(cur["ulp"], row["ulp"])
                cur["prefix_exact"] &= row["prefix_exact"]
                cur["finite"] &= row["finite"]
                cur["route_match"] &= row["route_match"]
                cur["ok"] &= row["ok"]
    out = []
    worst = 0.0
    fails = 0
    for k in sorted(agg):
        row = agg[k]
        out.append(row)
        worst = max(worst, row["ulp"])
        fails += 0 if row["ok"] else 1
        print(f"T={row['t']:>2} nvb={row['nvb']} out_norm={int(row['out_norm'])} "
              f"[{row['branch']:>11}] ulp={row['ulp']:.3f} prefix_exact={row['prefix_exact']} "
              f"route_match={row['route_match']} finite={row['finite']} "
              f"{'OK' if row['ok'] else 'FAIL'}")
    print(f"SUMMARY tp={tp} t_fused={t_fused} cases={len(out)} "
          f"max_ulp_vs_ref={worst:.3f} FAIL={fails}")
    if len(sys.argv) > 1:
        json.dump(out, open(sys.argv[1], "w"), indent=1)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    freeze_support()
    main()
