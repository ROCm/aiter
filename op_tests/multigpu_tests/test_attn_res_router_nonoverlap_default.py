"""SILOTIGER-1277 iter031: gate the NON-OVERLAPPED router DEFAULT.

The 3-way router's high-T branch, when called with NO overlapped producer
(``bank_acc=None``), used to default to the online-producer + separate acc_agg
TWO-kernel path. iter030 measured that path at 0.75x vs production raw agg (a 33%
pessimization), while the single fused producer+agg kernel is ~0.98x (near-parity).
iter031 flips the router default to ``fuse_producer_agg=True`` so a naive caller who
does NOT precompute/overlap the producer gets the dominating fused kernel instead of
the pessimized two-kernel path.

This test exercises the router EXACTLY as a naive non-overlapped caller would -- no
``bank_acc``, no ``fuse_producer_agg`` kwarg (pure default) -- and proves:
  * T <= t_fused  -> fused one-shot AR branch, out <= 1 ulp vs torch fp32 ref;
  * T >  t_fused  -> the DEFAULT now takes the fused producer+agg kernel
    (route_match: router out == a direct ``fused_producer_accagg`` call),
    out <= 1 ulp vs the SAME torch fp32 ref, prefix exact, finite.

Run: TP=8 python op_tests/multigpu_tests/test_attn_res_router_nonoverlap_default.py [out.json]
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

logger = logging.getLogger("attn_res_router_nonoverlap_default")

H = 7168
NB = 9
SCORE_EPS = 1e-6
OUT_EPS = 1e-6
MAX_BYTES = 8 * 1024 * 1024
TOKENS = [1, 4, 8, 9, 12, 16, 24, 32, 64]
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

                    # Naive non-overlapped caller: NO bank_acc, NO fuse_producer_agg
                    # kwarg -> pure router default (iter031: default is now True).
                    out_r, pout_r = attn_res_allreduce_router(
                        eng, inp.clone(), prefix, bank, nvb, cw, owv,
                        SCORE_EPS, OUT_EPS, t_fused=t_fused,
                        all_reduce_fn=all_reduce_fn,
                    )

                    addend = all_reduce_fn(inp)
                    ref = _agg_reference(prefix, addend, bank, nvb, cw,
                                         ow if out_norm else None, out_norm)
                    pref = (prefix.float() + addend.float()).to(torch.bfloat16)
                    u = _ulp(out_r, ref)
                    pe = torch.equal(pout_r, pref)
                    fin = bool(torch.isfinite(out_r.float()).all())

                    if t > t_fused:
                        # The DEFAULT must now take the fused producer+agg kernel.
                        out_d, pout_d = fused_producer_accagg(
                            prefix, addend, bank, cw, owv, nvb, SCORE_EPS, OUT_EPS,
                            write_prefix=True)
                        route_match = torch.equal(out_r, out_d) and torch.equal(pout_r, pout_d)
                        branch = "fused_pa"
                    else:
                        # Low T: default must still take the one-shot fused AR, which
                        # is BIT-IDENTICAL to the explicit-flag-OFF call (flag inert).
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
        logger.warning("router non-overlap default gate unsupported on %s; skipping", arch)
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
