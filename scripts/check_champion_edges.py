#!/usr/bin/env python3
"""Edge checks for champions: --check 1 (batch-size boundary), --check 2 (large-logit / EXP_APPROX),
--check 3 (incumbent packed BN64 compact grid: large batches that reach flydsl, ragged and empty q-sequences
under both interleaved settings), --check 4 (packed BN64 scalar page ids: garbage-padded, exact-width and
too-narrow block tables, empty KV segments)."""

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


def _ref(case, qls, kvs, causal):
    return torch.nan_to_num(
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
            causal=int(causal),
        ).float()
    )


def _run_bn64(case, layout, interleaved, group):
    """Packed BN64 launch: via the router (interleaved=None) or with interleaved forced on the router's arguments."""
    import aiter.ops.flydsl.unified_attention_kernels as uak
    import aiter.ops.unified_attention as ua

    k, v = case["k"], case["v"]
    if layout == "shuffled":
        k, v = shuffle_kv_cache(k, v)
    out = torch.full_like(case["out"], float("nan"))
    if interleaved is None:
        kw = dict(case, k=k, v=v, out=out, shuffled_kv_cache=layout == "shuffled")
        ua.unified_attention(**kw, backend="flydsl")
    else:
        q = case["q"]
        nseq = case["seqused_k"].numel()
        kernel = uak._get_kernel(
            q.shape[1],
            bc.HKV,
            bool(case["causal"]),
            "bf16" if out.dtype == torch.bfloat16 else "f16",
            layout == "shuffled",
            uak._fp8_rescale_threshold(case["max_seqlen_k"]),
            128,
            packed_bn64=True,
            batch_chunks=-(-(nseq + 1) // 64),
            interleaved=interleaved,
        )
        bt = case["block_table"]
        kernel(
            uak._as_i8(q).reshape(-1),
            uak._as_i8(k).reshape(k.shape[0], -1),
            uak._as_i8(v).reshape(v.shape[0], -1),
            out.reshape(-1),
            nseq,
            int(case["max_seqlen_q"]),
            bc.HKV * bc.D,
            q.stride(0),
            softmax_scale=float(case["softmax_scale"]),
            seq_len_kv=int(case["max_seqlen_k"]),
            cu_seqlens_q=case["cu_seqlens_q"],
            cu_seqlens_kv=case["seqused_k"],
            q_descale=case["q_descale"],
            k_descale=case["k_descale"],
            v_descale=case["v_descale"],
            block_table=bt.reshape(-1),
            block_table_stride=int(bt.stride(0)),
            total_q=int(q.shape[0]),
            stream=torch.cuda.current_stream(),
        )
    torch.cuda.synchronize()
    return out.float()


def _check3_cases():
    import random

    cases = []
    for B in (63, 64, 65, 127, 128, 129):
        qls, kvs = bc.ragged_B(B)
        groups = (16, 4, 8) if B in (64, 128) else (16,)
        for g in groups:
            # Non-causal ragged: the router never declines non-causal.
            cases.append((f"B{B}_noncausal_ragged", qls, kvs, False, g, None))
        # Causal uniform: M = B*S > (B-1)*S + 1, so the router routes it, with interleaved=True.
        S = 17
        ukv = [S + random.Random(B).randint(0, 300) for _ in range(B)]
        cases.append((f"B{B}_causal_uniform", [S] * B, ukv, True, 16, None))
        # Causal ragged the router admits (M > (B-1)*max + 1): one short sequence, the rest at max.
        rq = [S] * B
        rq[B // 2] = 2
        rq[-1] = S
        cases.append((f"B{B}_causal_ragged_router", rq, ukv, True, 16, None))
    for B in (64, 128):
        qls, kvs = bc.ragged_B(B)
        # The kernel must not trust the interleaved hint: ragged (a graph replay) under both settings.
        for il in (False, True):
            cases.append((f"B{B}_causal_ragged_il{int(il)}", qls, kvs, True, 16, il))
    # Empty q-sequences (cu_seqlens_q repeats), first, middle and last, under both interleaved settings.
    eq = [0, 40, 0, 17, 64, 0, 3, 0]
    ekv = [10, 300, 0, 17, 64, 5, 100, 70]
    for il in (False, True):
        cases.append((f"empty_q_causal_il{int(il)}", eq, ekv, True, 16, il))
    cases.append(("empty_q_noncausal_router", eq, ekv, False, 16, None))
    # Packed slot starts (prefix sum of ceil(len / BLOCK_Q)): many sequences shorter than BLOCK_Q, and
    # short/long mixes, across one, two and three scan chunks.
    for B in (40, 64, 129):
        rnd = random.Random(1000 + B)
        sq = [rnd.randint(1, 7) for _ in range(B)]
        skv = [q + rnd.randint(0, 200) for q in sq]
        for g in (16, 4):
            cases.append((f"B{B}_short_q_noncausal_G{g}", sq, skv, False, g, None))
            cases.append((f"B{B}_short_q_causal_G{g}", sq, skv, True, g, False))
        mq = [rnd.choice((1, 3, 92, 93, 300)) for _ in range(B)]
        mkv = [q + rnd.randint(0, 500) for q in mq]
        cases.append((f"B{B}_mixed_q_causal", mq, mkv, True, 16, False))
        cases.append((f"B{B}_mixed_q_noncausal_G4", mq, mkv, False, 4, None))
    # Uniform q with ragged KV under forced rank-major: the uniform test passes and rank-major runs.
    cases.append(
        (
            "B64_uniform_il1_forced",
            [33] * 64,
            [33 + 7 * i for i in range(64)],
            True,
            16,
            True,
        )
    )
    return cases


def check3():
    print(
        "| case | B | G | causal | interleaved | layout | pass | NaN | max_err all | max_err last seq | atol |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, qls, kvs, causal, g, il in _check3_cases():
        case = make_case(qls, kvs, torch.bfloat16, causal=causal, group=g)
        want = _ref(case, qls, kvs, causal)
        atol = bc.FP8_REL_ERR * want.abs().max().item()
        last0 = sum(qls[:-1])
        for layout in ("linear", "shuffled"):
            try:
                got = _run_bn64(case, layout, il, g)
            except Exception as ex:  # noqa: BLE001
                print(
                    f"| {name} | {len(qls)} | {g} | {int(causal)} | {il} | {layout} | error {type(ex).__name__}: {str(ex)[:120]} |||||"
                )
                continue
            d = torch.nan_to_num((got - want).abs(), nan=float("inf"))
            nan = bool(torch.isnan(got).any())
            last = d[last0:].max().item() if qls[-1] else float("nan")
            print(
                f"| {name} | {len(qls)} | {g} | {int(causal)} | {'router' if il is None else il} | {layout} | "
                f"{bool(d.max() <= atol and not nan)} | {nan} | {d.max().item():.4g} | {last:.4g} | {atol:.3g} |",
                flush=True,
            )


def _with_table(case, kvs, extra_cols, fill):
    """Copy of case whose table has extra_cols more columns and every entry past a row's pages set to fill."""
    bt = case["block_table"]
    need = torch.tensor([-(-n // 64) for n in kvs], device="cuda").unsqueeze(1)
    width = max(1, bt.shape[1] + extra_cols)
    out = torch.full((bt.shape[0], width), fill, device="cuda", dtype=torch.int32)
    w = min(width, bt.shape[1])
    cols = torch.arange(w, device="cuda").unsqueeze(0)
    out[:, :w] = torch.where(cols < need, bt[:, :w], out[:, :w])
    return dict(case, block_table=out.contiguous())


def check4():
    print("| case | G | causal | layout | pass | NaN | max_err all | atol |")
    print("|---|---|---|---|---|---|---|---|")
    shapes = [
        ("ragged", [93, 80, 71, 60], [157, 100, 90, 700]),
        ("page_multiple", [64, 128], [1024, 640]),
        ("empty_kv_mixed", [64, 32, 48], [0, 200, 0]),
        ("empty_kv_all", [64, 32], [0, 0]),
    ]
    tables = [
        ("exact", 0, None),
        ("pad_huge", 5, 2**31 - 1),
        ("pad_neg1", 5, -1),
    ]
    for g in (16, 8, 4):
        for causal in (True, False):
            for sname, qls, kvs in shapes:
                case = make_case(qls, kvs, torch.bfloat16, causal=causal, group=g)
                want = _ref(case, qls, kvs, causal)
                atol = max(bc.FP8_REL_ERR * want.abs().max().item(), 1e-6)
                for tname, extra, fill in tables:
                    c = case if fill is None else _with_table(case, kvs, extra, fill)
                    # make_case sizes the table to exactly ceil(max_seqlen_k / 64) columns.
                    assert fill is not None or c["block_table"].shape[1] == max(
                        1, -(-max(kvs) // 64)
                    )
                    for layout in ("linear", "shuffled"):
                        name = f"{sname}_{tname}"
                        try:
                            got = _run_bn64(c, layout, None, g)
                        except Exception as ex:  # noqa: BLE001
                            print(
                                f"| {name} | {g} | {int(causal)} | {layout} | error {type(ex).__name__}: {str(ex)[:100]} ||||"
                            )
                            continue
                        d = torch.nan_to_num((got - want).abs(), nan=float("inf"))
                        nan = bool(torch.isnan(got).any())
                        print(
                            f"| {name} | {g} | {int(causal)} | {layout} | {bool(d.max() <= atol and not nan)} | {nan} | "
                            f"{d.max().item():.4g} | {atol:.3g} |",
                            flush=True,
                        )
    # Too narrow: one column short of ceil(max_seqlen_k / 64). The router must decline (RuntimeError under
    # backend="flydsl"), not launch.
    for g in (16, 8, 4):
        qls, kvs = [64, 128], [1024, 700]
        case = make_case(qls, kvs, torch.bfloat16, causal=True, group=g)
        c = dict(case, block_table=case["block_table"][:, :-1].contiguous())
        for layout in ("linear", "shuffled"):
            try:
                _run_bn64(c, layout, None, g)
                ok, msg = False, "launched"
            except RuntimeError as ex:
                ok, msg = "does not support" in str(ex), str(ex)[:60]
            except Exception as ex:  # noqa: BLE001
                ok, msg = False, f"{type(ex).__name__}: {str(ex)[:60]}"
            print(
                f"| narrow_declined | {g} | 1 | {layout} | {ok} | - | {msg} | - |",
                flush=True,
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int, required=True)
    a = ap.parse_args()
    {1: check1, 2: check2, 3: check3, 4: check4}[a.check]()
