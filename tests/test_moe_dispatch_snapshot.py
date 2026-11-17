# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Legacy MoE dispatch must not change unless the golden snapshot is updated with it.

CPU only. Regenerate after an intentional change, and list the changed keys in the PR:

    python tests/moe_dispatch/snapshot.py --update
"""

import os
import pathlib
import subprocess
import sys

_REPO = pathlib.Path(__file__).resolve().parents[1]
_SNAPSHOT = _REPO / "tests" / "moe_dispatch" / "snapshot.py"


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
