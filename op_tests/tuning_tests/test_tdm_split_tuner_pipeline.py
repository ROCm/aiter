# SPDX-License-Identifier: MIT
"""Real AITER tuner -> CSV -> exact-contract runtime graph test on gfx1250."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1250":
        pytest.skip("gfx1250 required")


def test_tune_save_load_replay_and_retune(tmp_path):
    require_gpu()
    from aiter.ops.flydsl import TdmTunedPolicy

    root = Path(__file__).resolve().parents[2]
    script = root / "op_tests/tuners/tune_gemm_a16w16_tdm.py"
    table = tmp_path / "tuned.csv"
    env = dict(os.environ, PYTHONPATH=str(root), AITER_LOG_LEVEL="WARNING")
    cmd = [
        sys.executable,
        str(script),
        "--shapes",
        "64,64,1024",
        "--dtype",
        "fp16",
        "--splits",
        "1,2",
        "--rounds",
        "1",
        "--shortlist",
        "2",
        "--iters",
        "4",
        "--workspace-mib",
        "8",
        "-o",
        str(table),
    ]
    # Keep the entire subprocess output on failure; no tune result is fabricated.
    for _ in range(2):
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=300, check=False
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        rows = list(csv.DictReader(table.open()))
        assert len(rows) == 1
        assert float(rows[0]["tflops"]) == round(
            2 * 64 * 64 * 1024 / float(rows[0]["us"]) / 1e6, 2
        )
    audit = json.loads(Path(str(table) + ".audit.json").read_text())
    assert audit and any("confirmation_us" in r for r in audit)
    policy = TdmTunedPolicy(table)
    a = torch.randn(64, 1024, device="cuda", dtype=torch.float16) * 0.1
    b = (torch.randn(64, 1024, device="cuda", dtype=torch.float16) * 0.1).t()
    config = policy.select(a, b, max_workspace_bytes=8 << 20)
    assert config.split_k_policy == "workspace"
    out = policy(a, b, max_workspace_bytes=8 << 20)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        policy(a, b, out=out, max_workspace_bytes=8 << 20)
    for _ in range(3):
        a.neg_()
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            out.float(), a.float() @ b.float(), atol=0.002, rtol=0.002
        )
    graph.reset()
    with pytest.raises(KeyError):
        policy.select(a, b, allow_atomic=True, max_workspace_bytes=8 << 20)
    with pytest.raises(KeyError):
        policy.select(a, b, max_workspace_bytes=8 << 20, relative_l2_limit=0.0001)
    with pytest.raises(KeyError):
        policy.select(a[:32], b, max_workspace_bytes=8 << 20)
    # Extend a table with a different contract, preserving the first measured row.
    proc = subprocess.run(
        [*cmd[:3], "32,64,1024", *cmd[4:]],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert len(list(csv.DictReader(table.open()))) == 2
    assert len(TdmTunedPolicy(table).rows) == 2


@pytest.mark.parametrize("policy_name", ["workspace", "atomic"])
def test_loader_matches_explicit_policy_and_budget(tmp_path, policy_name):
    require_gpu()
    from aiter.ops.flydsl import TdmTunedPolicy
    from aiter.ops.flydsl import flydsl_gemm_a16w16_tdm as gemm
    from aiter.ops.flydsl.gemm_a16w16_tdm_policy import TdmConfig, contract

    a = torch.randn(128, 1024, device="cuda", dtype=torch.bfloat16) * 0.1
    b = (torch.randn(128, 1024, device="cuda", dtype=torch.bfloat16) * 0.1).t()
    cfg = TdmConfig(split_k=2, split_k_policy=policy_name)
    allow = policy_name == "atomic"
    # This test constructs metadata; no claim that its arbitrary timing is measured.
    record = contract(a, b, None, None, "nt", allow, 8 << 20)
    record.update(kernelName=cfg.encode(), us=1.0, errRatio=0.0)
    table = tmp_path / "fixture.csv"
    with table.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)
    selected = TdmTunedPolicy(table)
    y = selected(a, b, allow_atomic=allow, max_workspace_bytes=8 << 20)
    ref = gemm(a, b, **cfg.kwargs())
    torch.cuda.synchronize()
    torch.testing.assert_close(y, ref, atol=0, rtol=0)  # two splits: same sum order
    with pytest.raises(KeyError):
        selected.select(a, b, allow_atomic=allow, max_workspace_bytes=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", __file__]))
