# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GPU test: AITER_DUMP_MHA_FWD_INFO must be safe under HIP/CUDA graph capture.

The dumper reads env vars once per process (static locals), so every case runs
in a fresh subprocess. Each child runs the same sequence:

    eager -> capture -> capture -> eager -> replay x2 (each graph)

Expectations (per mode = group | batch):
  * no exception (group mode used to fail with a D2H copy inside
    torch.cuda.graph(): "operation not permitted when stream is capturing");
  * the "skipping dump during stream capture" warning is printed exactly once;
  * only eager calls produce records, captures/replays produce none;
  * captured calls do not consume the sampling counter:
      stride=1 -> 2 records, stride=2 -> 1 record (call #0 only).
"""

import os
import subprocess
import sys
import tempfile
import textwrap

import pytest
import torch

CAPTURE_WARNING = "skipping dump during stream capture"

_CHILD = textwrap.dedent("""
    import sys
    import torch
    import aiter

    mode = sys.argv[1]
    dev = "cuda"
    dt = torch.float16  # fp16 keeps both paths on CK (asm v3 is bf16-only)
    nh, nhk, d, s = 4, 2, 128, 128
    scale = d ** -0.5

    if mode == "group":
        cu = torch.tensor([0, s, 2 * s], dtype=torch.int32, device=dev)
        q = torch.randn(2 * s, nh, d, dtype=dt, device=dev)
        k = torch.randn(2 * s, nhk, d, dtype=dt, device=dev)
        v = torch.randn(2 * s, nhk, d, dtype=dt, device=dev)

        def run():
            return aiter.mha_varlen_fwd(
                q, k, v, cu, cu, s, s, 0, 0.0, scale, 0.0,
                False, True, -1, -1, 0, False, False,
            )[0]

    else:
        q = torch.randn(2, s, nh, d, dtype=dt, device=dev)
        k = torch.randn(2, s, nhk, d, dtype=dt, device=dev)
        v = torch.randn(2, s, nhk, d, dtype=dt, device=dev)

        def run():
            return aiter.mha_fwd(
                q, k, v, 0.0, scale, True, -1, -1, 0, False, False
            )[0]

    ref = run()  # eager #1 (also JIT / module-load warmup)
    torch.cuda.synchronize()

    graphs = []
    for _ in range(2):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = run()
        graphs.append((g, out))

    run()  # eager #2
    for g, out in graphs:
        g.replay()
        g.replay()
    torch.cuda.synchronize()
    for _, out in graphs:
        torch.testing.assert_close(out, ref)
    print("CAPTURE_OK")
    """)


def _run_child(mode: str, stride: int, log_path: str):
    env = dict(os.environ)
    env["AITER_DUMP_MHA_FWD_INFO"] = str(stride)
    env["AITER_DUMP_MHA_FWD_INFO_FILE"] = log_path
    return subprocess.run(
        [sys.executable, "-c", _CHILD, mode],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("stride,expected_records", [(1, 2), (2, 1)])
@pytest.mark.parametrize("mode", ["group", "batch"])
def test_dump_skipped_during_graph_capture(mode, stride, expected_records):
    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "mha_dump.log")
        proc = _run_child(mode, stride, log_path)
        diag = f"stdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-4000:]}"

        assert proc.returncode == 0, diag
        assert "CAPTURE_OK" in proc.stdout, diag
        assert proc.stderr.count(CAPTURE_WARNING) == 1, diag

        with open(log_path) as f:
            records = [ln for ln in f if ln.startswith(f"[MHA_FWD] mode={mode} ")]
        assert len(records) == expected_records, "".join(records) + diag


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
