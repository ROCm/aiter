# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 Gluon 2D lookup on the num_seqs (NS) axis.

Requires torch. It does not need a GPU: the table is selected with arch="gfx1250".

Run:
    python3 -m unittest op_tests.triton_tests.attention.test_unified_attention_gluon_config -v
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
NVFP4 = torch.uint8
NS_KEY = "DT_bf16_bf16.NS_LEQ_8"


def _params(
    num_seqs,
    *,
    q_dtype=BF16,
    kv_dtype=BF16,
    head_size=128,
    num_queries_per_kv=4,
    max_seqlen_q=4096,
    max_seqlen_k=65536,
    sliding_window=0,
):
    return SimpleNamespace(
        head_size=head_size,
        head_size_v=head_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        sliding_window=sliding_window,
        shuffled_kv_cache=True,
        q_dtype=q_dtype,
        kv_cache_dtype=kv_dtype,
        block_size=64,
        num_2d_prgms=1024,
        num_sms=96,
        num_queries_per_kv=num_queries_per_kv,
        num_seqs=num_seqs,
    )


def _matched(params) -> str:
    report = explain("attn_2d", params, backend="gluon", arch="gfx1250")
    for line in report.splitlines():
        if "matched:" in line:
            return line.split("matched:", 1)[1].strip()
    raise AssertionError("explain() did not report a matched key")


def _block_m(params) -> int:
    config = get_unified_attention_config(
        "attn_2d", params, backend="gluon", arch="gfx1250"
    )
    return config["BLOCK_M"]


class TestGfx1250GluonNumSeqs(unittest.TestCase):
    def test_few_bf16_sequences_take_block_m_256(self):
        for num_seqs in (1, 8):
            with self.subTest(num_seqs=num_seqs):
                params = _params(num_seqs)
                self.assertEqual(_matched(params), NS_KEY)
                self.assertEqual(_block_m(params), 256)

    def test_more_than_8_sequences_keep_block_m_128(self):
        params = _params(9)
        self.assertEqual(_matched(params), "any")
        self.assertEqual(_block_m(params), 128)

    def test_fp16_shares_the_bf16_entry(self):
        params = _params(1, q_dtype=FP16, kv_dtype=FP16)
        self.assertEqual(_matched(params), NS_KEY)

    def test_other_dtypes_are_unchanged(self):
        for q_dtype, kv_dtype in ((FP8, FP8), (FP8, NVFP4), (NVFP4, NVFP4)):
            with self.subTest(q=q_dtype, kv=kv_dtype):
                one = _matched(_params(1, q_dtype=q_dtype, kv_dtype=kv_dtype))
                many = _matched(_params(64, q_dtype=q_dtype, kv_dtype=kv_dtype))
                self.assertNotIn("NS_", one)
                self.assertEqual(one, many)

    def test_specific_keys_still_win(self):
        cases = {
            "decode": {"max_seqlen_q": 1},
            "sliding window": {"sliding_window": 128},
            "short context": {"max_seqlen_k": 1024},
            "head 256": {"head_size": 256},
        }
        for name, overrides in cases.items():
            with self.subTest(case=name):
                one = _matched(_params(1, **overrides))
                many = _matched(_params(64, **overrides))
                self.assertNotIn("NS_", one)
                self.assertEqual(one, many)


if __name__ == "__main__":
    unittest.main()
