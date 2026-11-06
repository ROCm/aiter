# SPDX-License-Identifier: MIT
"""Validate an explicitly loaded TDM table through AITER's public GEMM APIs.

AITER_CONFIG_GEMM_BF16=/tmp/tdm.csv python -m csrc.gemm_a16w16.validate_tdm_csv
"""

import csv
import json
import os
from pathlib import Path
from unittest import mock

import torch

import aiter.tuned_gemm as tuned
from aiter.ops.flydsl import gemm_kernels as catalog


def main():
    path = Path(os.environ["AITER_CONFIG_GEMM_BF16"])
    rows = list(csv.DictReader(path.open()))
    torch.set_num_threads(4)
    torch.manual_seed(1120)
    results = []
    for row in rows:
        m, n, k = (int(row[key]) for key in ("M", "N", "K"))
        bias_on = row["bias"].lower() == "true"
        cfg = catalog.get_flydsl_hgemm_kernel_params(row["kernelName"])
        assert cfg is not None and cfg["policy"] == "tdm"
        a = (torch.randint(-4, 5, (m, k), device="cuda").float() / 16).bfloat16()
        w = (torch.randint(-4, 5, (n, k), device="cuda").float() / 16).bfloat16()
        bias = (torch.rand(n, device="cuda") + 0.5).bfloat16() if bias_on else None
        sk = cfg["split_k"]
        part = k // sk
        ref = sum(
            (
                a[:, i * part : (i + 1) * part].float()
                @ w[:, i * part : (i + 1) * part].float().t()
            )
            .bfloat16()
            .float()
            for i in range(sk)
        ).bfloat16()
        if bias is not None:
            ref = (ref.float() + bias.float()).bfloat16()
        picked = tuned.get_GEMM_A16W16_config(
            m,
            n,
            k,
            bias_on,
            str(a.dtype),
            str(a.dtype),
            gfx=row["gfx"],
            cu_num=int(row["cu_num"]),
        )
        assert picked["kernelName"] == row["kernelName"], picked
        with mock.patch.object(
            catalog, "flydsl_hgemm", wraps=catalog.flydsl_hgemm
        ) as call:
            out = tuned.gemm_a16w16(a, w, bias)
            torch.cuda.synchronize()
            assert call.call_count == 1
            assert call.call_args.kwargs["policy"] == "tdm"
        torch.testing.assert_close(out, ref, atol=0, rtol=0)
        fixed = torch.empty_like(out)
        tuned.gemm_a16w16_out(a, w, fixed, bias)
        torch.cuda.synchronize()
        torch.testing.assert_close(fixed, ref, atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            tuned.gemm_a16w16_out(a, w, fixed, bias)
        a.neg_()
        if bias is not None:
            bias.neg_()
        fixed.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(fixed, -ref, atol=0, rtol=0)
        graph.reset()
        results.append(
            {
                "M": m,
                "N": n,
                "K": k,
                "bias": bias_on,
                "passed": True,
                "kernelName": row["kernelName"],
            }
        )
        print("PASS", m, n, k, bias_on, flush=True)
        del a, w, bias, out, ref, fixed
        torch.cuda.empty_cache()
    result = path.with_suffix(".validation.json")
    result.write_text(json.dumps(results, indent=2))
    print("VALIDATED", len(results), "rows", result, flush=True)


if __name__ == "__main__":
    main()
