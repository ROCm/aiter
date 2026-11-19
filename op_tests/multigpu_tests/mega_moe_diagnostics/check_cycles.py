"""Reuse changing-input EP4 correctness checks with the fine probes active."""

from pathlib import Path
import os
import sys
from unittest import mock
import install_overlay

install_overlay.install()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_mega_moe_gfx1250 as suite

suite.HIDDEN, suite.INTER, suite.TOPK = 5120, 2304, 6
suite.TOKENS = int(os.environ.get("CHECK_TPR", "257"))
suite.EXPERTS_PER_RANK = 96


def check(ctx):
    from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped

    original = grouped._find_grouped_config

    def configured(**kwargs):
        row = dict(original(**kwargs) or {})
        row["tdm_as_in_prologue"] = "0"
        return row

    with mock.patch.dict(
        os.environ,
        {
            "AITER_TDM_TILE_M": "256",
            "AITER_TDM_TILE_N": "256",
            "AITER_TDM_TILE_K": "256",
            "AITER_TDM_M_WARP": "2",
            "AITER_TDM_N_WARP": "2",
            "AITER_TDM_NUM_BUFFERS": "4" if suite.TOKENS >= 4096 else "2",
        },
    ), mock.patch.object(grouped, "_find_grouped_config", configured):
        suite._check_pull(ctx, fused=True, expected_affinity=(8, 3))


suite.CHECKS = (check,)
if __name__ == "__main__":
    raise SystemExit(suite.main())
