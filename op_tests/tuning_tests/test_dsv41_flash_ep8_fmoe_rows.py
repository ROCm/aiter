# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4.1-Flash EP8 MXFP4 (a4w4) fused-MoE rows are what fused_moe looks up.

Under 8-way expert parallelism the target's MoE has 48 local experts (top-6) and the
DSpark drafter's 16 (top-3), hidden 5120, intermediate 2304. Their rows, the target's
at M = 512..32768 and the drafter's at M = 256..32768, live in
model_configs/dsv41_fp4_tuned_fmoe.csv. After AITER merges every tuned_fmoe table,
each of these keys must resolve to exactly one row, that row must be the one below,
and no other table may carry the key. get_padded_M maps every buffer of 257-32768
rows onto the target's keys and of 129-32768 rows onto the drafter's.

Requires torch (importing aiter pulls it in). No GPU: ``GPU_ARCHS`` and
``CU_NUM`` stand in for the device.

Run:
    GPU_ARCHS=gfx950 CU_NUM=256 python3 -m unittest \
        op_tests.tuning_tests.test_dsv41_flash_ep8_fmoe_rows -v
"""

import csv
import glob
import os
import unittest

os.environ.setdefault("GPU_ARCHS", "gfx950")
os.environ.setdefault("CU_NUM", "256")

try:
    from aiter.fused_moe import get_padded_M
    from aiter.jit.core import AITER_CONFIGS

    _IMPORT_ERR = None
except Exception as e:  # noqa: BLE001
    get_padded_M = AITER_CONFIGS = None
    _IMPORT_ERR = e

# (expert, topk, token): (block_m, kernelName1, kernelName2)
EXPECTED = {
    (48, 6, 512): (
        32,
        "flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w2_fp4",
        "flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32",
    ),
    (48, 6, 1024): (
        32,
        "flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w2_fp4",
        "flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32",
    ),
    (48, 6, 2048): (
        64,
        "flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w3_fp4",
        "flydsl_moe2_afp4_wfp4_bf16_t64x128x256_atomic_bnt2",
    ),
    (48, 6, 4096): (
        128,
        "flydsl_moe1_afp4_wfp4_bf16_t128x128x256_fp4",
        "flydsl_moe2_afp4_wfp4_bf16_t64x128x256_atomic_bnt2_sbm128",
    ),
    (48, 6, 8192): (
        128,
        "flydsl_moe1_afp4_wfp4_bf16_t128x128x256_fp4",
        "flydsl_moe2_afp4_wfp4_bf16_t64x128x256_atomic_bnt2_sbm128",
    ),
    (48, 6, 16384): (
        128,
        "flydsl_moe1_afp4_wfp4_bf16_t128x128x256_fp4",
        "flydsl_moe2_afp4_wfp4_bf16_t64x128x256_atomic_bnt2_sbm128",
    ),
    (48, 6, 32768): (
        128,
        "flydsl_moe1_afp4_wfp4_bf16_t128x128x256_bnt0_fp4",
        "flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x256_atomic_persist_sbm128",
    ),
    (16, 3, 256): (
        32,
        "flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w3_bnt0_kw2_fp4",
        "flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32",
    ),
    (16, 3, 512): (
        32,
        "flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w4_bnt0_kw4_fp4",
        "flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32",
    ),
    (16, 3, 1024): (
        32,
        "flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w3_bnt0_kw4_fp4",
        "flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_persist_sbm32",
    ),
    (16, 3, 2048): (
        64,
        "flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w2_fp4",
        "flydsl_moe2_afp4_wfp4_bf16_t64x128x256_atomic",
    ),
    (16, 3, 4096): (
        128,
        "flydsl_moe1_afp4_wfp4_bf16_t128x128x256_fp4",
        "flydsl_moe2_afp4_wfp4_bf16_t64x128x256_atomic_bnt2_sbm128",
    ),
    (16, 3, 8192): (
        128,
        "flydsl_moe1_afp4_wfp4_bf16_t128x128x256_fp4",
        "flydsl_moe2_afp4_wfp4_bf16_t64x128x256_atomic_bnt2_sbm128",
    ),
    (16, 3, 16384): (
        128,
        "flydsl_moe1_afp4_wfp4_bf16_t128x128x256_fp4",
        "flydsl_moe2_afp4_wfp4_bf16_t64x128x256_atomic_bnt2_sbm128",
    ),
    (16, 3, 32768): (
        128,
        "flydsl_moe1_afp4_wfp4_bf16_t128x128x256_bnt0_fp4",
        "flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x256_atomic_persist_sbm128",
    ),
}

# The rest of the key, shared by all fifteen rows.
SHAPE = {
    "gfx": "gfx950",
    "cu_num": "256",
    "model_dim": "5120",
    "inter_dim": "2304",
    "act_type": "ActivationType.Silu",
    "dtype": "torch.bfloat16",
    "q_dtype_a": "torch.float4_e2m1fn_x2",
    "q_dtype_w": "torch.float4_e2m1fn_x2",
    "q_type": "QuantType.per_1x32",
    "use_g1u1": "1",
    "doweight_stage1": "0",
}


def _matches(row, expert, topk, token):
    if any(row.get(k, "").strip() != v for k, v in SHAPE.items()):
        return False
    return (row["expert"].strip(), row["topk"].strip(), row["token"].strip()) == (
        str(expert),
        str(topk),
        str(token),
    )


class Dsv41FlashEp8FmoeRowsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _IMPORT_ERR is not None:
            raise unittest.SkipTest(f"aiter import failed: {_IMPORT_ERR}")

    def test_keys_resolve_to_these_rows(self):
        with open(AITER_CONFIGS.AITER_CONFIG_FMOE_FILE) as f:
            merged = list(csv.DictReader(f))
        for (expert, topk, token), (block_m, kn1, kn2) in EXPECTED.items():
            with self.subTest(expert=expert, topk=topk, token=token):
                hits = [r for r in merged if _matches(r, expert, topk, token)]
                self.assertEqual(
                    len(hits), 1, "the key must resolve to exactly one row"
                )
                row = hits[0]
                self.assertEqual(
                    (
                        int(row["block_m"]),
                        row["kernelName1"].strip(),
                        row["kernelName2"].strip(),
                    ),
                    (block_m, kn1, kn2),
                )

    def test_no_other_table_carries_these_keys(self):
        import aiter

        root = os.path.join(os.path.dirname(os.path.abspath(aiter.__file__)), "configs")
        tables = [
            p
            for p in glob.glob(
                os.path.join(root, "**", "*tuned_fmoe*.csv"), recursive=True
            )
            if "untuned" not in os.path.basename(p)
            and "grouped" not in os.path.basename(p)
        ]
        self.assertTrue(tables, f"no tuned_fmoe tables under {root}")
        for path in tables:
            with open(path) as f:
                rows = list(csv.DictReader(f))
            for expert, topk, token in EXPECTED:
                with self.subTest(
                    table=os.path.basename(path), expert=expert, token=token
                ):
                    n = sum(_matches(r, expert, topk, token) for r in rows)
                    if os.path.basename(path) == "dsv41_fp4_tuned_fmoe.csv":
                        self.assertEqual(n, 1)
                    else:
                        self.assertEqual(n, 0)

    def test_buffers_map_onto_these_keys(self):
        for expert, first, last in ((48, 257, 32768), (16, 129, 32768)):
            keys = {token for e, _, token in EXPECTED if e == expert}
            for rows in (first, 384, 512, 1000, 2304, 4096, 6000, 12000, 20000, last):
                with self.subTest(expert=expert, rows=rows):
                    self.assertIn(get_padded_M(rows), keys)
            self.assertNotIn(get_padded_M(first - 1), keys)


if __name__ == "__main__":
    unittest.main()
