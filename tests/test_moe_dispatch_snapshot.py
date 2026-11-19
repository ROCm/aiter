# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Legacy MoE dispatch must not change unless the golden snapshot is updated with it.

CPU only. Regenerate after an intentional change, and list the changed keys in the PR:

    python tests/moe_dispatch/snapshot.py --update
"""

import dataclasses
import os
import pathlib
import subprocess
import sys

_REPO = pathlib.Path(__file__).resolve().parents[1]
_SNAPSHOT = _REPO / "tests" / "moe_dispatch" / "snapshot.py"


def test_moe_metadata_defaults_are_frozen_snapshot_inputs():
    from aiter.moev2.execute.metadata import AsmMoeMode, MOEMetadata

    defaults = {
        field.name: field.default
        for field in dataclasses.fields(MOEMetadata)
        if field.default is not dataclasses.MISSING
    }
    assert defaults == {
        "run_1stage": False,
        "has_bias": False,
        "use_non_temporal_load": True,
        "fuse_quant": "",
        "stage2_has_bias": False,
        "flat": AsmMoeMode.SORTED,
        "output_aux": False,
        "prequant": True,
        "skip_inter_quant": False,
        "route_bucket": "",
        "expected_sorted_blocks": None,
        "min_sorted_blocks": None,
        "max_sorted_blocks": None,
        "full_impl": None,
    }


def test_legacy_dispatch_matches_golden_snapshot():
    result = subprocess.run(
        [sys.executable, str(_SNAPSHOT), "--check"],
        cwd=_REPO,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    report = "\n".join(
        line for line in result.stdout.splitlines() if not line.startswith("[aiter]")
    )
    assert result.returncode == 0, (
        f"{report}\n{result.stderr[-3000:]}\n"
        "Legacy MoE dispatch changed. If intended, run "
        "`python tests/moe_dispatch/snapshot.py --update` and list the changes in the PR."
    )
