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
        n,
        "all_reduce",
        nbytes,
        (lap, lap),
        (RcclCall("all_reduce", nbytes),),
        pipelined=True,
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
    test = patterns.tb_test(traffic.phases[0], 4)
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


def test_one_pass_per_launch():
    """Every roof test is one pass per launch: sub-iterations would let
    cacheable peer memory absorb the rewrites in L2."""
    for traffic in (_two_shot(4, 4096, ring=False), _a2a_trio(8, 256 << 20)):
        tests = [t for c in patterns.candidates(traffic) for t in c.tests]
        rccl = patterns.rccl_candidates(patterns.rccl_traffic(traffic))
        tests += [t for c in rccl for t in c.tests]
        assert {dict(t.env)["NUM_SUBITERATIONS"] for t in tests} == {"1"}


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
    # In-kernel times per transfer; the roof takes the slowest.
    assert sorted(tests[0].xfer_ms) == list(range(8))
    assert sorted(tests[1].xfer_ms) == list(range(4))
    assert math.isclose(tests[0].kernel_ms, 1048576 / 1e6 / 53.493)
    assert math.isclose(tests[1].kernel_ms, 262144 / 1e6 / 104.289)


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
    roofs = {op_t: Roof(50.0, (25.0, 25.0), "fused w64")}
    rccl_roofs = {patterns.rccl_traffic(op_t): Roof(160.0, (160.0,), "fused w32")}
    (row,) = report.grade(records, roofs, rccl_roofs)
    assert row["op eff"] == 0.5
    # RCCL is graded against its own (bf16) roof, not the op's quantized one.
    assert row["rccl roof us"] == 160.0 and row["rccl eff"] == 0.8
    assert row["roof cfg"] == "fused w64"
    assert row["rccl roof cfg"] == "fused w32"
    assert math.isclose(row["op busbw GB/s"], nbytes / 100.0 / 1e3 * 1.5)
    assert math.isclose(row["rccl busbw GB/s"], nbytes / 200.0 / 1e3 * 1.5)
    assert math.isclose(row["op wire GB/s"], 1.5 * nbytes * 0.28125 / 100.0 / 1e3)
    # Without a roof the roof-derived columns are blank, the rest stay.
    (row,) = report.grade(records, {})
    assert math.isnan(row["op eff"]) and row["op us"] == 100.0
    assert math.isnan(row["rccl eff"]) and row["rccl us"] == 200.0
    assert "|  |" in report.markdown([row])


def test_fused_flows_carry_every_lap():
    """A pipelined two-shot fuses into one test whose transfers carry both
    laps: mesh RS + AG per peer, ring RS + AG x (N - 1) on the next hop."""
    n, nbytes = 4, 4096 * 7168 * BF16
    mesh = _two_shot(n, nbytes, ring=False, ratio=0.28125)
    chunk = mesh.phases[0].chunk_bytes
    fused = [c for c in patterns.candidates(mesh) if c.label == "fused w64"]
    ((test,),) = [c.tests for c in fused]
    assert test.line.startswith(f"-12 (G0 G0 F1 64 {2 * chunk}) ")
    assert test.repeat == 1
    ring = _two_shot(n, nbytes, ring=True)
    (test,) = next(c for c in patterns.candidates(ring) if c.label == "fused own").tests
    chunk = ring.phases[0].chunk_bytes
    # 64 workgroups x 4 waves, all on the one fused transfer per GPU.
    assert test.line.startswith(f"-4 (G0 G0 F1 256 {2 * (n - 1) * chunk}) ")
    # Either way the fused test moves the same fabric bytes as the laps.
    for t in (mesh, ring):
        flows = patterns._fused_flows(t)
        assert sum(f[3] for f in flows) // n == patterns.fabric_bytes_per_rank(t)


def test_candidates_sweep_widths_and_schedules():
    """The op roof tries its own grid and every sweep width, serial and,
    for a pipelined op only, fused."""
    ar = _two_shot(4, 1 << 24, ring=False)
    labels = [c.label for c in patterns.candidates(ar)]
    widths = ["own", *(f"w{w}" for w in patterns.WAVE_SWEEP)]
    assert labels == [f"{s} {w}" for w in widths for s in ("serial", "fused")]
    a2a = _a2a_trio(4, 12_124_160)
    labels = [c.label for c in patterns.candidates(a2a)]
    assert labels == [f"serial {w}" for w in widths]
    # The own serial candidate is the op's replay, phase by phase, in the
    # op's own memory type.
    own = patterns.candidates(a2a)[0].tests
    assert own == tuple(patterns.tb_test(p, 4) for p in a2a.phases)
    assert "(G0 G0 F1 256 " in own[0].line


def test_rccl_traffic_is_unquantized_and_op_independent():
    """RCCL's roof moves bf16 bytes in the busbw convention, whatever the op's
    wire format or schedule, and sweeps the memory type."""
    n, nbytes = 8, 1024 * 7168 * BF16
    int4_mesh = _two_shot(n, nbytes, ring=False, ratio=0.28125)
    ring = _two_shot(n, nbytes, ring=True, ratio=0.40625)
    rt = patterns.rccl_traffic(int4_mesh)
    assert rt == patterns.rccl_traffic(ring)
    assert rt.pipelined and rt.logical_bytes == nbytes
    want = patterns.busbw_factor("all_reduce", n) * nbytes
    assert math.isclose(patterns.fabric_bytes_per_rank(rt), want, rel_tol=1e-6)
    a2a = patterns.rccl_traffic(_a2a_trio(n, 12_124_160, ratio=0.28125))
    assert not a2a.pipelined and len(a2a.phases) == 3
    assert all(p.include_self for p in a2a.phases)
    want = patterns.busbw_factor("all_to_all", n) * 3 * 12_124_160
    assert math.isclose(patterns.fabric_bytes_per_rank(a2a), want, rel_tol=1e-6)
    labels = {c.label for c in patterns.rccl_candidates(rt)}
    assert labels == {
        f"{s} w{w} {m}"
        for s in ("serial", "fused")
        for w in patterns.WAVE_SWEEP
        for m in "UFG"
    }
    lap = next(c for c in patterns.rccl_candidates(rt) if c.label == "serial w32 G")
    assert lap.tests[0].line.startswith(f"-56 (G0 G0 G1 32 {nbytes // n}) ")


def test_roofline_takes_fastest_candidate():
    """Each distinct test runs once; the roof is the candidate with the
    smallest sum, and its label is kept."""
    import roofline

    ar = _two_shot(4, 1 << 24, ring=False)
    lap = ar.phases[0].chunk_bytes
    calls = []

    def fake_run(binary, lines, env, **kw):
        """1 ms per lap of bytes at 64 waves, 10% slower per octave away
        from it, and a fused test 10% faster than its two laps."""
        calls.append(list(lines))
        out = []
        for line in lines:
            _, _, _, _, waves, nbytes = line.split(")")[0].split()
            laps = int(nbytes) / lap
            ms = laps * (1 + abs(math.log2(int(waves) / 64)) / 10)
            ms *= 0.9 if laps > 1 else 1.0
            res = tb_text.TestResult(len(out) + 1, {"GPU 0": ms + 1}, {0: ms})
            out.append(res)
        return out, "1.71.02"

    real = roofline.tb_text.run
    roofline.tb_text.run = fake_run
    try:
        # Any executable will do: the run itself is faked.
        rl = roofline.Roofline(sys.executable, iters=1, warmup=0)
        roof = rl.measure([ar])[ar]
        n_lines = sum(len(c) for c in calls)
        assert roof.label == "fused w64"
        assert math.isclose(roof.us, 0.9 * 2 * 1e3)
        # Serial laps are identical lines: run once, counted twice.
        distinct = {t.line for c in patterns.candidates(ar) for t in c.tests}
        assert n_lines == len(distinct)
        # Cached: a second call runs nothing.
        calls.clear()
        assert rl.measure([ar])[ar] == roof and not calls
    finally:
        roofline.tb_text.run = real


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
