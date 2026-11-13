# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""GPU-free checks of the roofline tool: TransferBench config generation, byte
accounting and the output parser.

    python3 op_tests/op_benchmarks/collective_bw/test_collective_bw.py
    pytest op_tests/op_benchmarks/collective_bw/test_collective_bw.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _path in (str(_HERE), str(_HERE.parents[2])):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import patterns
import tb_text

from aiter.ops.flydsl.comm_traffic import (
    Pattern,
    PeerMem,
    Phase,
    RcclCall,
    Traffic,
)

BF16 = 2


def _phase(pattern, chunk, *, self_=False, repeat=1, wg=128, waves=8, mem=PeerMem.FINE):
    return Phase(pattern, chunk, self_, repeat, mem, wg, waves)


def _a2a_trio(n, role_bytes, ratio=1.0):
    chunk = int(role_bytes * ratio) // n
    phase = _phase(Pattern.PUSH_ALL, chunk, self_=True)
    return Traffic(
        n,
        "all_to_all",
        3 * role_bytes,
        (phase,) * 3,
        (RcclCall("all_to_all", role_bytes),) * 3,
    )


def _two_shot(n, nbytes, ring, ratio=1.0):
    chunk = int(nbytes * ratio) // n
    pattern = Pattern.PUSH_NEXT if ring else Pattern.PUSH_ALL
    lap = _phase(pattern, chunk, repeat=n - 1 if ring else 1, wg=64, waves=4)
    return Traffic(
        n, "all_reduce", nbytes, (lap, lap), (RcclCall("all_reduce", nbytes),)
    )


def test_fabric_bytes_match_busbw_convention():
    """Bytes sent to other GPUs equal the busbw factor times the payload, scaled
    by the wire ratio, for every pattern an op can declare."""
    n, nbytes = 4, 4096 * 7168 * BF16
    cases = (
        _a2a_trio(n, nbytes),
        _a2a_trio(n, nbytes, ratio=0.28125),
        _two_shot(n, nbytes, ring=False),
        _two_shot(n, nbytes, ring=True),
        _two_shot(n, nbytes, ring=True, ratio=0.40625),
    )
    ratios = (1.0, 0.28125, 1.0, 1.0, 0.40625)
    for traffic, ratio in zip(cases, ratios):
        want = (
            patterns.busbw_factor(traffic.collective, n) * traffic.logical_bytes * ratio
        )
        got = patterns.fabric_bytes_per_rank(traffic)
        assert math.isclose(got, want, rel_tol=1e-6), (traffic.phases[0], got, want)


def test_a2a_push_line_and_waves():
    """Attention A2A at TP4: every GPU pushes into all four buffers, own
    included, with the op's waves split evenly over them."""
    traffic = _a2a_trio(4, 12_124_160)
    test = patterns.tb_tests(traffic)[0]
    assert test.line.startswith("-16 (G0 G0 F0 256 3031040) (G0 G0 F1 256 3031040)")
    assert "(G3 G3 F2 256 3031040)" in test.line
    env = dict(test.env)
    assert env["GFX_SE_TYPE"] == "1" and env["GFX_BLOCK_SIZE"] == "512"
    assert test.repeat == 1
    # 128 workgroups x 8 waves over 4 destinations.
    assert patterns.waves_per_transfer(traffic.phases[0], 4) == 256


def test_ring_and_pull_lines():
    ring = _phase(
        Pattern.PUSH_NEXT, 1000, repeat=3, wg=8, waves=4, mem=PeerMem.UNCACHED
    )
    test = patterns.tb_test(ring, 4)
    # Rounded up to a multiple of 4 bytes, all 32 waves on the one transfer.
    assert test.line == (
        "-4 (G0 G0 U1 32 1000) (G1 G1 U2 32 1000) (G2 G2 U3 32 1000) (G3 G3 U0 32 1000)"
    )
    assert test.repeat == 3
    pull = _phase(Pattern.PULL_ALL, 1002, wg=2, waves=4, mem=PeerMem.COARSE)
    line = patterns.tb_test(pull, 2).line
    assert line == "-2 (G1 G0 G0 8 1004) (G0 G1 G1 8 1004)"


def test_subiterations_amortize_small_chunks():
    small = _phase(Pattern.PUSH_ALL, 4096)
    big = _phase(Pattern.PUSH_ALL, 256 << 20)
    assert patterns.subiterations(small, 4) == patterns.MAX_SUBITERATIONS
    assert patterns.subiterations(big, 4) == 1


def test_parser_on_captured_output():
    text = (_HERE / "testdata" / "transferbench_v1.71.02.txt").read_text()
    assert tb_text.version(text) == "1.71.02"
    tests = tb_text.parse(text)
    assert [t.num for t in tests] == [1, 2]
    assert set(tests[0].exec_ms) == {"GPU 0", "GPU 1"}
    assert set(tests[1].exec_ms) == {"GPU 0", "GPU 1", "GPU 2", "GPU 3"}
    # Duration comes from bytes / bandwidth, not the 3-decimal ms column.
    assert math.isclose(tests[0].exec_ms["GPU 1"], 4194304 / 1e6 / 214.911)
    assert math.isclose(tests[1].slowest_ms, 262144 / 1e6 / 100.336)


def test_parser_without_borders():
    text = "Test 1:\n  Executor: GPU 00 | 10.000 GB/s | 0.100 ms | 1000000 bytes | x\n"
    (test,) = tb_text.parse(text)
    assert test.exec_ms == {"GPU 0": 0.1}


def test_grade_columns():
    """eff and rccl-vs-roof both use the op's roof; busbw uses the
    collective's factor; wire bandwidth uses the op's wire bytes."""
    import report
    from measure import Record
    from roofline import Roof

    n, nbytes = 4, 1 << 20
    op_t = _two_shot(n, nbytes, ring=False, ratio=0.28125)
    records = [
        Record("c", "fly", 100.0, "graph", op_t, "v", 2.0),
        Record("c", "rccl", 200.0, "graph", op_t),
    ]
    (row,) = report.grade(records, {op_t: Roof(50.0, (25.0, 25.0))})
    assert row["op eff"] == 0.5 and row["rccl vs roof"] == 0.25
    assert math.isclose(row["op busbw GB/s"], nbytes / 100.0 / 1e3 * 1.5)
    assert math.isclose(row["rccl busbw GB/s"], nbytes / 200.0 / 1e3 * 1.5)
    assert math.isclose(row["op wire GB/s"], 1.5 * nbytes * 0.28125 / 100.0 / 1e3)
    # Without a roof the roof-derived columns are blank, the rest stay.
    (row,) = report.grade(records, {})
    assert math.isnan(row["op eff"]) and row["op us"] == 100.0
    assert "|  |" in report.markdown([row])


def test_find_binary_respects_env(tmp_path=None):
    old = os.environ.get(tb_text.ENV_BIN)
    os.environ[tb_text.ENV_BIN] = "/nonexistent/TransferBench"
    try:
        assert tb_text.find_binary() is None
    finally:
        if old is None:
            del os.environ[tb_text.ENV_BIN]
        else:
            os.environ[tb_text.ENV_BIN] = old


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)
