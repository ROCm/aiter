# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sparse MLA training backward (OPUS), Q/K 576 and V/O 512, through
``aiter.ops.opus.sparse_mla_bwd``: correctness.

    python3 op_tests/test_sparse_mla_bwd_opus.py                         # every check
    python3 op_tests/test_sparse_mla_bwd_opus.py --checks grads -m glm53-tp2 k31-tp1

Cases (``-m``) use the head counts of the models: GLM-5.3 at TP2 and TP1, Kimi-K3.1 at TP1, TP4
and TP8, a rope-free case and a case with an attention sink.

Checks (``--checks``):
  reference   ``sparse_mla_ref`` equals ``sparse_mla_dsv4_ref`` when V is as wide as K
  errors      the interface rejects the inputs that it cannot run
  grads       dQ, dKV and d_sink against autograd through the fp32 reference. The 64-column
              rope tail of dQ and dKV is also checked alone, so the latent cannot hide it.
  accumulate  ``dkv_out`` gets dKV added, and it is the returned dKV

The kernels are not in this revision. ``grads`` and ``accumulate`` log a skip until they land.
gfx950 only.
"""

import argparse
import re
from dataclasses import dataclass

import torch

from aiter import logger
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.opus.sparse_mla_bwd import sparse_mla_bwd
from aiter.test_mha_common import sparse_mla_dsv4_ref, sparse_mla_ref

D_V = 512
COS_TOL = 0.999
# The absorbed MLA scale comes from the unabsorbed head: qk_nope 192 + qk_rope 64.
SCALE = 1.0 / (256**0.5)


@dataclass(frozen=True)
class Case:
    T: int
    H: int
    topk: int
    npool: int = 0  # KV rows past T, as with context parallelism
    d_rope: int = 64
    has_sink: bool = False


CASES = {
    "glm53-tp2": Case(T=128, H=32, topk=128),
    "glm53-tp1": Case(T=128, H=64, topk=256, npool=128),
    "k31-tp1": Case(T=64, H=96, topk=128),
    "k31-tp4": Case(T=64, H=24, topk=128),
    "k31-tp8": Case(T=64, H=12, topk=128),
    "rope-free": Case(T=64, H=32, topk=128, d_rope=0),
    "sink": Case(T=64, H=64, topk=128, has_sink=True),
}


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().reshape(-1), b.float().reshape(-1), dim=0
    ).item()


def make_inputs(c: Case, seed=0, invalid_frac=0.1):
    torch.manual_seed(seed)
    d_qk = D_V + c.d_rope
    num_kv = c.T + c.npool
    q = torch.randn((c.T, c.H, d_qk), dtype=torch.bfloat16, device="cuda")
    kv = torch.randn((num_kv, d_qk), dtype=torch.bfloat16, device="cuda")
    do = torch.randn((c.T, c.H, D_V), dtype=torch.bfloat16, device="cuda")
    sink = (
        torch.randn((c.H,), dtype=torch.float32, device="cuda") * 0.1
        if c.has_sink
        else None
    )
    idx = torch.randint(0, num_kv, (c.T, c.topk), dtype=torch.int32, device="cuda")
    invalid = (
        torch.rand((c.T, c.topk), dtype=torch.float32, device="cuda") < invalid_frac
    )
    idx = torch.where(invalid, torch.full_like(idx, -1), idx).contiguous()
    return q, kv, do, sink, idx


def run_reference(q, kv, do, sink, idx):
    """fp32 forward and autograd backward. Returns bf16 ``o``, ``lse`` and the grads."""
    qg = q.float().requires_grad_(True)
    kvg = kv.float().requires_grad_(True)
    sg = sink.clone().requires_grad_(True) if sink is not None else None
    o_ref, lse = sparse_mla_ref(qg, kvg, idx, SCALE, kv_lora_rank=D_V, attn_sink=sg)
    o_ref.backward(do.float())
    o = o_ref.detach().to(torch.bfloat16).contiguous()
    return o, lse, qg.grad, kvg.grad, (sg.grad if sg is not None else None)


def check_reference():
    """With V as wide as K, ``sparse_mla_ref`` must equal ``sparse_mla_dsv4_ref``."""
    torch.manual_seed(0)
    T, H, topk, D = 32, 16, 64, 512
    q = torch.randn((T, H, D), dtype=torch.float32, device="cuda")
    kv = torch.randn((T, D), dtype=torch.float32, device="cuda")
    do = torch.randn((T, H, D), dtype=torch.float32, device="cuda")
    sink = torch.randn((H,), dtype=torch.float32, device="cuda") * 0.1
    idx = torch.randint(0, T, (T, topk), dtype=torch.int32, device="cuda")
    idx[:, ::7] = -1

    results = []
    for fn in (
        lambda q_, kv_, s_: sparse_mla_ref(q_, kv_, idx, 0.05, D, s_),
        lambda q_, kv_, s_: sparse_mla_dsv4_ref(q_, kv_, idx, s_, 0.05),
    ):
        qg, kvg, sg = (t.clone().requires_grad_(True) for t in (q, kv, sink))
        o, lse = fn(qg, kvg, sg)
        o.backward(do)
        results.append((o.detach(), lse, qg.grad, kvg.grad, sg.grad))
    for got, want in zip(*results):
        torch.testing.assert_close(got, want)
    return []


def check_errors():
    """Each bad input must raise ValueError with the expected message."""
    q, kv, do, _, idx = make_inputs(Case(T=64, H=32, topk=64))
    o = torch.randn((64, 32, D_V), dtype=torch.bfloat16, device="cuda")
    lse = torch.randn((64, 32), dtype=torch.float32, device="cuda")
    good = dict(q=q, kv=kv, o=o, do=do, lse=lse, topk_indices=idx)
    bad_dkv_out = torch.zeros((64, D_V + 64), dtype=torch.bfloat16, device="cuda")
    cases = [
        ("q", q[..., :D_V].contiguous(), "q width"),
        ("o", torch.cat([o, o[..., :64]], -1), "V is the latent"),
        ("do", do[:, :16].contiguous(), "V is the latent"),
        ("lse", lse[:, :16].contiguous(), "lse must be"),
        ("topk_indices", idx[:32], "topk_indices must be"),
        ("topk_indices", idx.long(), "int32"),
        ("kv", kv.float(), "kv must be bf16"),
        ("dkv_out", bad_dkv_out, "dkv_out must be"),
    ]
    failures = []
    for field, value, match in cases:
        args = dict(good, **{field: value})
        try:
            sparse_mla_bwd(**args, softmax_scale=SCALE)
        except ValueError as e:
            if not re.search(match, str(e)):
                failures.append(f"errors/{field}: message {e!r} lacks {match!r}")
            continue
        except NotImplementedError:
            pass
        failures.append(f"errors/{field}: accepted a bad input")
    return failures


def check_grads(name, c: Case):
    q, kv, do, sink, idx = make_inputs(c)
    o, lse, dq_ref, dkv_ref, dsink_ref = run_reference(q, kv, do, sink, idx)
    try:
        dq, dkv, d_sink = sparse_mla_bwd(
            q,
            kv,
            o,
            do,
            lse,
            idx,
            SCALE,
            kv_lora_rank=D_V,
            qk_rope_head_dim=c.d_rope,
            attn_sink=sink,
        )
    except NotImplementedError:
        logger.warning("grads/%s: skipped, the kernels are not in this revision", name)
        return []

    failures = []
    if dq.dtype != torch.bfloat16 or dkv.dtype != torch.float32:
        failures.append(f"grads/{name}: dtypes dq {dq.dtype}, dkv {dkv.dtype}")
    checks = [("dq", dq, dq_ref), ("dkv", dkv, dkv_ref)]
    if c.d_rope:
        checks += [
            ("dq rope tail", dq[..., D_V:], dq_ref[..., D_V:]),
            ("dkv rope tail", dkv[:, D_V:], dkv_ref[:, D_V:]),
        ]
    if c.has_sink:
        checks.append(("d_sink", d_sink, dsink_ref))
    elif d_sink is not None:
        failures.append(f"grads/{name}: d_sink is not None without a sink")
    for what, got, want in checks:
        cos = _cos(got, want)
        logger.info("grads/%s: %s cos %.6f", name, what, cos)
        if not cos > COS_TOL:
            failures.append(f"grads/{name}: {what} cos {cos:.6f} <= {COS_TOL}")
    return failures


def check_accumulate():
    c = Case(T=64, H=32, topk=128)
    q, kv, do, sink, idx = make_inputs(c)
    o, lse, _, dkv_ref, _ = run_reference(q, kv, do, sink, idx)
    base = torch.randn((c.T, D_V + c.d_rope), dtype=torch.float32, device="cuda")
    acc = base.clone()
    try:
        _, dkv, _ = sparse_mla_bwd(q, kv, o, do, lse, idx, SCALE, dkv_out=acc)
    except NotImplementedError:
        logger.warning("accumulate: skipped, the kernels are not in this revision")
        return []
    failures = []
    if dkv.data_ptr() != acc.data_ptr():
        failures.append("accumulate: the returned dkv is not dkv_out")
    cos = _cos(acc - base, dkv_ref)
    if not cos > COS_TOL:
        failures.append(f"accumulate: dkv cos {cos:.6f} <= {COS_TOL}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--checks",
        nargs="+",
        choices=["reference", "errors", "grads", "accumulate"],
        default=["reference", "errors", "grads", "accumulate"],
    )
    parser.add_argument(
        "-m", "--cases", nargs="+", choices=list(CASES), default=list(CASES)
    )
    args = parser.parse_args()

    if get_gfx() != "gfx950":
        logger.info(
            "test_sparse_mla_bwd_opus: skipped, gfx950 only (got %s)", get_gfx()
        )
        return

    failures = []
    if "reference" in args.checks:
        failures += check_reference()
    if "errors" in args.checks:
        failures += check_errors()
    if "grads" in args.checks:
        for name in args.cases:
            failures += check_grads(name, CASES[name])
    if "accumulate" in args.checks:
        failures += check_accumulate()

    for f in failures:
        logger.error("%s", f)
    assert not failures, f"{len(failures)} check(s) failed"
    logger.info("test_sparse_mla_bwd_opus: %s passed", ", ".join(args.checks))


if __name__ == "__main__":
    main()
