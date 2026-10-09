# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU CSV-selection coverage for public padded-token integration reports."""

from __future__ import annotations

import ast
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HARNESS = Path(__file__).resolve().parents[1] / "test_mxfp4_flydsl_public_csv.py"
RUNTIME = Path(__file__).resolve().parents[2] / "aiter/fused_moe.py"


def load_csv_boundary() -> Any:
    # Execute only the CSV boundary; importing the GPU harness is unnecessary.
    runtime_tree = ast.parse(RUNTIME.read_text(), filename=str(RUNTIME))
    runtime_tree.body = [
        node
        for node in runtime_tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_PADDED_M_TIERS"
                for target in node.targets
            )
        )
        or (
            isinstance(node, ast.FunctionDef)
            and node.name in ("nextPow2", "get_padded_M")
        )
    ]
    runtime = {}
    exec(  # noqa: S102 - trusted runtime token-padding boundary
        compile(runtime_tree, str(RUNTIME), "exec"), runtime
    )
    tree = ast.parse(HARNESS.read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("csv_rows", "select_csv_lookup")
    ]
    namespace = {
        "Path": Path,
        "csv": csv,
        "fm": SimpleNamespace(**runtime),
    }
    exec(  # noqa: S102 - trusted checkout CSV boundary
        compile(tree, str(HARNESS), "exec"), namespace
    )
    return SimpleNamespace(**namespace)


def row(token: int, kernel: str) -> dict[str, Any]:
    return {
        "gfx": "gfx950",
        "cu_num": "256",
        "token": str(token),
        "model_dim": "3072",
        "inter_dim": "512",
        "expert": "128",
        "topk": "4",
        "act_type": "ActivationType.Swiglu",
        "dtype": "torch.bfloat16",
        "q_dtype_a": "torch.float8_e4m3fn",
        "q_dtype_w": "torch.float4_e2m1fn_x2",
        "q_type": "QuantType.per_1x32",
        "use_g1u1": "1",
        "doweight_stage1": "0",
        "kernelName1": kernel,
        "kernelName2": kernel + "_g2",
        "_tag": "",
    }


class TestPublicCsvLookup(unittest.TestCase):
    def test_request_token3_reports_actual_token4_primary_in_same_csv(self) -> None:
        harness = load_csv_boundary()
        request, lookup = row(3, "requested_pair"), row(4, "lookup_pair")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actual.csv"
            with path.open("w", newline="") as source:
                writer = csv.DictWriter(source, fieldnames=request)
                writer.writeheader()
                writer.writerows(
                    [request, dict(lookup, _tag="flydsl_fallback"), lookup]
                )
            rows = harness.csv_rows(path)
            lookup_token = harness.fm.get_padded_M(int(rows[0]["token"]))
            self.assertEqual(lookup_token, 4)
            selection = harness.select_csv_lookup(rows, rows[0], lookup_token)
            self.assertEqual(selection["csv_row"], 2)
            self.assertEqual(selection["token"], 4)
            self.assertEqual(selection["kernelName1"], "lookup_pair")
            self.assertEqual(rows[0]["kernelName1"], "requested_pair")

    def test_large_token_lookup_and_smaller_tier_do_not_claim_requested_pair(
        self,
    ) -> None:
        harness = load_csv_boundary()
        request = row(65536, "requested_pair")
        lookup_token = harness.fm.get_padded_M(int(request["token"]))
        self.assertEqual(lookup_token, 32768)
        self.assertEqual(
            harness.select_csv_lookup(
                [request, row(32768, "tier_pair")], request, lookup_token
            )["token"],
            32768,
        )
        request = row(196608, "requested_pair")
        lookup_token = harness.fm.get_padded_M(int(request["token"]))
        self.assertEqual(lookup_token, 131072)
        exact = harness.select_csv_lookup(
            [row(32768, "tier_pair"), row(131072, "large_pair")],
            request,
            lookup_token,
        )
        self.assertEqual(exact["kernelName1"], "large_pair")
        self.assertEqual(exact["kernelName2"], "large_pair_g2")
        selection = harness.select_csv_lookup(
            [request, row(32768, "tier_pair")], request, lookup_token
        )
        self.assertEqual(selection["token"], 32768)
        self.assertEqual(selection["kernelName1"], "tier_pair")
        self.assertEqual(selection["kernelName2"], "tier_pair_g2")
        self.assertIsNone(
            harness.select_csv_lookup(
                [row(4, "wrong_shape")],
                dict(row(3, "x"), topk="8"),
                harness.fm.get_padded_M(3),
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
