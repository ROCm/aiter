# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 Triton 3D lookup for plain head-dim <= 64 decode.

Requires torch. It does not need a GPU: the table is selected with arch="gfx1250".

Run:
    python3 -m unittest op_tests.triton_tests.attention.test_unified_attention_config -v
"""

import unittest
from types import SimpleNamespace

import torch

from aiter.ops.triton.utils.types import e4m3_dtype
from aiter.ops.triton.utils.unified_attention_utils import (
    explain,
    get_unified_attention_config,
)

BF16 = torch.bfloat16
FP16 = torch.float16
FP8 = e4m3_dtype

# num_sms == num_2d_prgms, so NUM_SEGMENTS lands on SEGMENTS_PER_CU.
_SMS = 8


def _params(q_dtype, kv_dtype, *, head_size=64, max_seqlen_q=1, shuffled=False):
    return SimpleNamespace(
        head_size=head_size,
        head_size_v=head_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=8192,
        sliding_window=0,
        shuffled_kv_cache=shuffled,
        q_dtype=q_dtype,
        kv_cache_dtype=kv_dtype,
        block_size=64,
        num_2d_prgms=_SMS,
        num_sms=_SMS,
        num_queries_per_kv=1,
        num_seqs=1,
    )


def _matched(op, params) -> str:
    for line in explain(op, params, arch="gfx1250").splitlines():
        if "matched:" in line:
            return line.split("matched:", 1)[1].strip()
    raise AssertionError("explain() did not report a matched key")


class TestGfx1250Ua3dPlainDecode(unittest.TestCase):
    def _attn(self, params):
        return get_unified_attention_config("attn_3d", params, arch="gfx1250")

    def _split(self, params):
        return get_unified_attention_config("kv_split", params, arch="gfx1250")

    def test_plain_decode_uses_the_tuned_entry(self):
        pairs = (
            (BF16, BF16, "bf16_bf16"),
            (BF16, FP8, "bf16_fp8"),
            (FP8, BF16, "fp8_bf16"),
            (FP8, FP8, "fp8_fp8"),
        )
        for q_dtype, kv_dtype, tag in pairs:
            for head_size in (32, 64):
                with self.subTest(tag=tag, head_size=head_size):
                    params = _params(q_dtype, kv_dtype, head_size=head_size)
                    key = f"D_LEQ_64.DT_{tag}.Q_LEQ_1"
                    self.assertEqual(_matched("attn_3d", params), key)
                    self.assertEqual(_matched("kv_split", params), key)
                    attn = self._attn(params)
                    self.assertEqual((attn["num_stages"], attn["waves_per_eu"]), (3, 2))
                    self.assertEqual(self._split(params)["NUM_SEGMENTS"], 16)

    def test_fp16_decode_shares_the_bf16_entry(self):
        params = _params(FP16, FP16)
        key = "D_LEQ_64.DT_bf16_bf16.Q_LEQ_1"
        self.assertEqual(_matched("attn_3d", params), key)
        self.assertEqual(self._attn(params)["num_stages"], 3)
        self.assertEqual(self._split(params)["NUM_SEGMENTS"], 16)

    def test_query_longer_than_one_keeps_the_generic_entry(self):
        # 16 is the longest query gfx1250 still sends to the 3D kernel.
        params = _params(BF16, BF16, max_seqlen_q=16)
        self.assertEqual(_matched("attn_3d", params), "any")
        self.assertEqual(_matched("kv_split", params), "any")
        attn = self._attn(params)
        self.assertEqual((attn["num_stages"], attn["waves_per_eu"]), (2, 1))
        self.assertEqual(self._split(params)["NUM_SEGMENTS"], 4)

    def test_head_dim_above_64_keeps_the_generic_entry(self):
        params = _params(BF16, BF16, head_size=128)
        self.assertNotIn("Q_LEQ_1", _matched("attn_3d", params))
        self.assertNotIn("Q_LEQ_1", _matched("kv_split", params))

    def test_shuffled_decode_keeps_its_own_entry(self):
        bf16 = _params(BF16, BF16, shuffled=True)
        self.assertEqual(_matched("attn_3d", bf16), "D_LEQ_64.SHUF")
        self.assertEqual(_matched("kv_split", bf16), "D_LEQ_64.SHUF")
        self.assertEqual(self._attn(bf16)["num_stages"], 2)
        self.assertEqual(self._split(bf16)["NUM_SEGMENTS"], 8)

        fp8 = _params(BF16, FP8, shuffled=True)
        self.assertEqual(_matched("attn_3d", fp8), "D_LEQ_64.SHUF.DT_any_fp8")
        self.assertEqual(_matched("kv_split", fp8), "D_LEQ_64.SHUF.DT_any_fp8")
        self.assertEqual(self._split(fp8)["NUM_SEGMENTS"], 8)


if __name__ == "__main__":
    unittest.main()
