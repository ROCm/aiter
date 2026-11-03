# SPDX-License-Identifier: MIT
"""Real-source WMMA exact/padded dispatch traces (CPU only, raw K).

The production wrapper allocates/pads/copies; its _run seam is replaced by the
existing trace executor, which instruments the actual selected kernel body.
Every descriptor/access is checked and every store must be exactly once. TDM
payload safety is covered independently by test_pad_bucket_cpu and stock gates.
Requires QSA_GOLDEN_DIR pointing to the frozen golden directory (read-only).
"""
from __future__ import annotations

import os
import re
import unittest
from unittest.mock import patch

os.environ["QSA_BF16_K_LOAD"] = "raw"
os.environ["QSA_BF16_PERSIST"] = "0"
# ROUND-MERGE1: the trace executor instruments the WMMA kernel bodies only; the
# default draft kernel (rows <= requests) is covered by the lane-level emulation
# in test_draft_cpu.py and by GPU correctness. Pin the champion routing here.
os.environ["QSA_DRAFT_PATH"] = "0"
# ROUND-DECG2: likewise the decode-group kernel (now default QSA_DECODE_GROUP=all,
# rows > requests in the r16 regime) is covered by the lane-level emulation and
# exactly-once store plan in test_decg_cpu.py and by GPU correctness (22/22,
# root run 2026-11-03). Pin the previous WMMA routing for this trace gate.
os.environ["QSA_DECODE_GROUP"] = "0"

import torch
import trace_check_cpu as T

torch.set_num_threads(1)
K = T.load_module()


class PadBucketTraceCPU(unittest.TestCase):
    def test_real_wrapper_exact_and_padded_bitwise(self):
        records = []
        def run(launch, *args):
            q, cache, pt, t2r, qp, cl, logits, vis, rst, gx, gy, rows, cols, pages, reqs, width, stream = args
            geometry = int(re.search(r"_r(16|32|64)_", launch.kernel_name).group(1))
            with patch.dict(os.environ, {"QSA_TRACE_ROWS_PER_BLOCK": str(geometry)}):
                out = T.run_kernel(K, q.view(torch.bfloat16).view(rows, 4, 128),
                    cache.view(torch.bfloat16).view(pages, 4, 1, 128), pt, t2r, qp, cl, cols)
            logits.copy_(out[0]); vis.copy_(out[1]); rst.copy_(out[2])
            records.append((geometry, rows, cols, reqs, width, out))
        cases = [(1, 1, 1, 1), (3, 3, 17, 33), (37, 3, 17, 67),
                 (37, 1, 17, 67), (31, 1, 1, 3), (33, 1, 17, 65), (16, 4, 16, 64),
                 (32, 1, 16, 64), (64, 2, 16, 64)]
        with patch.object(K, "_run", run), patch.object(torch.cuda, "current_stream", lambda dev: None):
            for rows, reqs, width, cols in cases:
                with self.subTest(rows=rows, requests=reqs, columns=cols):
                    torch.manual_seed(rows + cols)
                    q = torch.randn(rows, 4, 128, dtype=torch.bfloat16)
                    cache = torch.randn(4, 4, 1, 128, dtype=torch.bfloat16)
                    pt = torch.arange(reqs * width, dtype=torch.int32).view(reqs, width) % 4
                    pt[:, 0] = -1
                    if width > 1:
                        pt[:, 1] = 4  # invalid physical page => clamped page zero
                    t2r = torch.arange(rows, dtype=torch.int32) % reqs
                    if rows > 1:
                        t2r[-1] = reqs  # invalid TRUE request must not resurrect
                    qp = torch.arange(rows, dtype=torch.int32) * 7 - 4
                    ctx = torch.full((reqs,), 300, dtype=torch.int32)
                    ctx[0] = 9  # short visible/context, negative positions
                    def call(env):
                        starts = torch.full((rows,), -999, dtype=torch.int32)
                        with patch.dict(os.environ, {"QSA_PAD_BUCKET": env}):
                            logits, vis = K.qsa_paged_mqa_logits_page4(q, cache, pt, t2r, qp, ctx,
                                max_columns=cols, row_starts=starts)
                        return logits, vis, starts
                    exact, padded = call("0"), call("1")
                    self.assertTrue(torch.equal(exact[0].view(torch.int32), padded[0].view(torch.int32)))
                    self.assertTrue(torch.equal(exact[1], padded[1]))
                    self.assertTrue(torch.equal(exact[2], padded[2]))
                    p = K.qsa_pad_plan((rows, 4, cols, reqs, width, 4), "bf16")
                    geometry, pr, pc, pn, pw, out = records[-1]
                    self.assertEqual(geometry, p["rows_per_block"])
                    native = not (rows % geometry or cols % 64 or width % 16)
                    expected = ((rows, cols, reqs, width) if native else tuple(p["padded_shape"][k]
                        for k in ("rows", "columns", "requests", "page_table_width")))
                    self.assertEqual((pr, pc, pn, pw), expected)
                    if native:
                        self.assertEqual(int(out[1][-1]), 0)  # True-invalid request stays invalid.
                    self.assertTrue(bool((out[1][rows:] == 0).all()))
                    self.assertTrue(bool(torch.isneginf(out[0][rows:]).all()))


if __name__ == "__main__":
    unittest.main()
