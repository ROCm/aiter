# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Legacy MoE dispatch must not change unless the golden snapshot is updated with it.

CPU only. Regenerate after an intentional change, and paste the changed keys it prints
into the PR:

    python tests/moe_dispatch/snapshot.py --update

NOTE: part of the aiter.moev2 migration (under active development, not graduated).
Please coordinate changes with the moev2 owners (@qli88). See aiter/moev2/README.md.
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


def test_snapshot_files_depend_on_their_content_only(tmp_path):
    from moe_dispatch import snapshot

    files = sorted(snapshot.GOLDEN_DIR.rglob("*.gz"))
    assert files
    for path in files:
        header = path.read_bytes()[:10]
        # Byte 3 holds the flags (0x08: a file name follows); bytes 4-7 the mtime.
        assert header[3] & 0x08 == 0 and header[4:8] == bytes(4), (
            f"{path.name}: gzip header carries a file name or mtime; rewrite it with "
            "tests/moe_dispatch/snapshot.py"
        )
    data = b'{"format": 1}\n'
    snapshot._write_gz(tmp_path / "a.jsonl.gz", data)
    snapshot._write_gz(tmp_path / "b.jsonl.gz.tmp", data)
    assert (tmp_path / "a.jsonl.gz").read_bytes() == (
        tmp_path / "b.jsonl.gz.tmp"
    ).read_bytes()


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
    lines = [ln for ln in result.stdout.splitlines() if not ln.startswith("[aiter]")]
    # A wide dispatch change lists ~30k keys per arch; the CLI prints them all.
    cap = 200
    if len(lines) > cap:
        more = f"... {len(lines) - cap} more lines"
        lines = lines[:cap] + [f"{more}: run tests/moe_dispatch/snapshot.py --check"]
    report = "\n".join(lines)
    assert result.returncode == 0, (
        f"{report}\n{result.stderr[-3000:]}\n"
        "Legacy MoE dispatch changed. If intended, run "
        "`python tests/moe_dispatch/snapshot.py --update` and paste the changes it "
        "prints into the PR."
    )
