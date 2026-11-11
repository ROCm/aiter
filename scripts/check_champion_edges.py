#!/usr/bin/env python3
"""Edge checks for champions: --check 1 (batch-size boundary), --check 2 (large-logit / EXP_APPROX)."""

import argparse
import importlib
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
bc = importlib.import_module("bench_champions")
import torch

bc.DT.update(bf16=torch.bfloat16, f16=torch.float16)

_ut = importlib.import_module("op_tests.test_unified_attention")
make_case = _ut.make_case
ref_paged_attn = _ut.ref_paged_attn
shuffle_kv_cache = _ut.shuffle_kv_cache

AUTH = "/home/jograner/.claude/jobs/178b5e41/tmp/author-results/silotiger-877-flydsl-prefill-attention/"
for _n, _f in (
    ("author_approx32", "approx/attention_approx.py"),
    ("author_exact", "exact/attention_exact.py"),
):
    _sp = importlib.util.spec_from_file_location(_n, AUTH + _f)
    _m = importlib.util.module_from_spec(_sp)
    sys.modules[_n] = _m
    _sp.loader.exec_module(_m)
    bc.CHAMP_MODULE[_n] = _n
KERNELS = (
    ["author_approx32", "author_exact", "noslack", "flydsl"]
    if os.environ.get("CHK_AUTH")
    else ["exact", "noslack", "flydsl"]
)
if os.environ.get("CHK_ONLY"):
    KERNELS = os.environ["CHK_ONLY"].split(",")


def run_all(case, qls, kvs, layout, want, extra=None):
    k, v = case["k"], case["v"]
    if layout == "shuffled":
        k, v = shuffle_kv_cache(k, v)
    total = sum(qls)
    res = {}
    for kern in KERNELS:
        kw = dict(case, k=k, v=v, shuffled_kv_cache=layout == "shuffled")
        kw["out"] = torch.full(
            (total, bc.H, bc.D), float("nan"), device="cuda", dtype=torch.bfloat16
        )
        try:
            bc.make_call(kern, kw, for_corr=True)()
            torch.cuda.synchronize()
            res[kern] = kw["out"].float()
        except Exception as ex:  # noqa
            res[kern] = f"error {type(ex).__name__}: {str(ex)[:150]}"
            torch.cuda.synchronize()
    return res


def check1():
    print(
        "| B | layout | kernel | pass | NaN | max_err all | max_err last seq | max_err other | q_first/q_last |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for B in (63, 64, 65, 128, 129):
        qls, kvs = bc.ragged_B(
            B
        )  # same generator as the c_B* cases in bench_champions.py
        case = make_case(qls, kvs, torch.bfloat16, causal=True)
        want = torch.nan_to_num(
            ref_paged_attn(
                case["q"],
                case["k"],
                case["v"],
                qls,
                kvs,
                case["block_table"],
                case["softmax_scale"],
                q_descale=case["q_descale"],
                k_descale=case["k_descale"],
                v_descale=case["v_descale"],
                causal=1,
            ).float()
        )
        atol = bc.FP8_REL_ERR * want.abs().max().item()
        last0 = sum(qls[:-1])
        for layout in ("linear", "shuffled"):
            res = run_all(case, qls, kvs, layout, want)
            for kern, got in res.items():
                if isinstance(got, str):
                    print(f"| {B} | {layout} | {kern} | {got} |||||")
                    continue
                d = torch.nan_to_num((got - want).abs(), nan=float("inf"))
                nan = bool(torch.isnan(got).any())
                print(
                    f"| {B} | {layout} | {kern} | {bool(d.max() <= atol and not nan)} | {nan} | {d.max().item():.4g} | "
                    f"{d[last0:].max().item():.4g} | {d[:last0].max().item():.4g} | {qls[0]}/{qls[-1]} |",
                    flush=True,
                )


def check2():
    print(
        "| scenario | B | layout | kernel | NaN count | max_err | ref max abs | atol | pass |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for B in (1, 2):
        qls = [256] * B
        kvs = [4096 + 64 * i for i in range(B)]
        for scen, qd, spike in [
            ("base", 1, 0),
            ("desc8", 8, 0),
            ("desc32", 32, 0),
            ("desc128", 128, 0),
            ("spike", 1, 1),
            ("desc8+spike", 8, 1),
            ("desc32+spike", 32, 1),
        ]:
            case = make_case(qls, kvs, torch.bfloat16, causal=True)
            case["q_descale"] = case["q_descale"] * qd if qd != 1 else case["q_descale"]
            if spike:
                bt = case["bt"] if "bt" in case else case["block_table"]
                for s in range(B):
                    p = 2000 + 37 * s  # mid sequence, not in first/last two tiles
                    page, slot = int(bt[s, p // 64]), p % 64
                    qrow = case["q"][sum(qls[:s]) + qls[s] - 1].float()  # (H,D)
                    for h in range(bc.HKV):
                        sign = qrow[h * (bc.H // bc.HKV)] >= 0
                        kf = torch.where(sign, 448.0, -448.0).to(case["k"].dtype)
                        case["k"].view(torch.uint8)[page, slot, h] = kf.view(
                            torch.uint8
                        )
            want = ref_paged_attn(
                case["q"],
                case["k"],
                case["v"],
                qls,
                kvs,
                case["block_table"],
                case["softmax_scale"],
                q_descale=case["q_descale"],
                k_descale=case["k_descale"],
                v_descale=case["v_descale"],
                causal=1,
            ).float()
            want = torch.nan_to_num(want)
            atol = bc.FP8_REL_ERR * want.abs().max().item()
            for layout in ("linear", "shuffled"):
                res = run_all(case, qls, kvs, layout, want)
                for kern, got in res.items():
                    if isinstance(got, str):
                        print(f"| {scen} | {B} | {layout} | {kern} | {got} ||||")
                        continue
                    nn = int(torch.isnan(got).sum())
                    ni = int(torch.isinf(got).sum())
                    d = torch.nan_to_num((got - want).abs(), nan=float("inf"))
                    print(
                        f"| {scen} | {B} | {layout} | {kern} | {nn} (inf {ni}) | {d.max().item():.4g} | "
                        f"{want.abs().max().item():.4g} | {atol:.3g} | {bool(d.max() <= atol and nn == 0)} |",
                        flush=True,
                    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int, required=True)
    a = ap.parse_args()
    (check1 if a.check == 1 else check2)()
