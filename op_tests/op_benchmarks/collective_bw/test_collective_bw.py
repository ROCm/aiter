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
from roof_settings import DEFAULT, RoofSettings

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
    widths = ["own", *(f"w{w}" for w in DEFAULT.wave_sweep)]
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
        for w in DEFAULT.wave_sweep
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
        rl = roofline.Roofline(sys.executable, RoofSettings(iterations=1, warmups=0))
        roof = rl.measure([ar])[ar]
        n_lines = sum(len(c) for c in calls)
        assert roof.label == "fused w64"
        assert math.isclose(roof.us, 0.9 * 2 * 1e3)
        # Every candidate's time is kept, in candidate order.
        labels = [c.label for c in patterns.candidates(ar)]
        assert [label for label, _us in roof.candidates] == labels
        assert min(us for _l, us in roof.candidates) == roof.us
        # Serial laps are identical lines: run once, counted twice.
        distinct = {t.line for c in patterns.candidates(ar) for t in c.tests}
        assert n_lines == len(distinct)
        # Cached: a second call runs nothing.
        calls.clear()
        assert rl.measure([ar])[ar] == roof and not calls
    finally:
        roofline.tb_text.run = real


def test_settings_reach_every_test():
    """Iterations, extra env and the sweep come from ``RoofSettings``; the
    method's own settings are fixed."""
    s = RoofSettings(iterations=7, warmups=1, wave_sweep=(48,), env=(("X", "1"),))
    ar = _two_shot(4, 1 << 24, ring=False)
    cands = patterns.candidates(ar, s)
    labels = [c.label for c in cands]
    assert labels == ["serial own", "fused own", "serial w48", "fused w48"]
    rccl = patterns.rccl_candidates(patterns.rccl_traffic(ar), s)
    for test in (t for c in cands + rccl for t in c.tests):
        env = dict(test.env)
        assert env["NUM_ITERATIONS"] == "7" and env["NUM_WARMUPS"] == "1"
        assert env["X"] == "1" and "ALWAYS_VALIDATE" not in env
        assert env["GFX_SE_TYPE"] == "1" and env["NUM_SUBITERATIONS"] == "1"
    # The RCCL block comes from the settings too.
    s4 = RoofSettings(rccl_waves_per_workgroup=2, rccl_mems=(PeerMem.FINE,))
    rccl = patterns.rccl_candidates(patterns.rccl_traffic(ar), s4)
    assert {dict(t.env)["GFX_BLOCK_SIZE"] for c in rccl for t in c.tests} == {"128"}
    assert {c.label.split()[-1] for c in rccl} == {"F"}


def test_cli_overrides_settings():
    import argparse

    import report

    parser = argparse.ArgumentParser()
    report.add_args(parser)
    assert report.settings_from(parser.parse_args([])) == DEFAULT
    args = parser.parse_args(["--tb-iters", "5", "--link-gbps", "64"])
    assert report.settings_from(args) == RoofSettings(iterations=5, link_gbps=64.0)


def test_dump_writes_runnable_configs():
    """Every candidate of every roof gets a .cfg that runs as is, and the
    index lists them with their times and marks the roof."""
    import tempfile

    import tb_dump
    from measure import Record
    from roofline import Roof

    ar = _two_shot(4, 1 << 24, ring=False, ratio=0.28125)
    records = [
        Record("1024x7168", "fly_mesh", 50.0, "graph", ar),
        Record("1024x7168", "rccl", 100.0, "graph", ar),
    ]
    cands = patterns.candidates(ar)
    times = tuple((c.label, 10.0 + i) for i, c in enumerate(cands))
    roofs = {ar: Roof(12.0, (12.0,), cands[2].label, times)}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "out.tb"
        (root / "1024x7168").mkdir(parents=True)
        (root / tb_dump.MARKER).touch()
        (root / "stale.cfg").write_text("old")
        index = tb_dump.write(root, records, DEFAULT, roofs, {})
        assert not (root / "stale.cfg").exists()
        op_dir = root / "1024x7168" / "fly_mesh"
        assert sorted(f.stem for f in op_dir.glob("*.cfg")) == sorted(
            tb_dump.slug(c.label) for c in cands
        )
        text = (op_dir / "serial_w16.cfg").read_text().splitlines()
        title = "1024x7168 / op fly_mesh / serial w16"
        assert text[0] == f"## collective_bw roof candidate: {title}"
        assert text[1] == "## measured: 12.000 us (the roof)"
        run = next(line for line in text if "TransferBench serial_w16.cfg" in line)
        assert "NUM_SUBITERATIONS=1" in run and "NUM_ITERATIONS=20" in run
        # Serial laps: two tests in one file, each counted once.
        tests = [line for line in text if line.startswith("-")]
        assert tests == [t.line for t in cands[2].tests] and len(tests) == 2
        rccl = patterns.rccl_candidates(patterns.rccl_traffic(ar))
        assert len(list((root / "1024x7168" / "rccl").glob("*.cfg"))) == len(rccl)
        rows = [r for r in index.read_text().splitlines() if r.startswith("| 1024")]
        assert len(rows) == len(cands) + len(rccl)
        assert "| serial w16 | 12.000 | **roof** |" in rows[2]
        rccl_cfg = root / "1024x7168" / "rccl" / "fused_w64_U.cfg"
        assert "not measured" in rccl_cfg.read_text()
        # The fabric-peak probes are written too.
        for kind in patterns.PEAK_KINDS:
            assert len(list((root / "peak" / kind).glob("*.cfg"))) == 15


def test_dump_default_dir():
    import tb_dump

    assert tb_dump.default_dir("a/b/tp4.md", None) == Path("a/b/tp4.tb")
    assert tb_dump.default_dir(None, "x.csv") == Path("x.tb")
    assert tb_dump.default_dir(None, None) == Path("collective_bw_tb")


def test_peak_probes():
    """All-peers pushes the probe size to every other rank, one-peer to the
    next rank only; both sweep widths and memory types."""
    s = RoofSettings(peak_bytes_per_peer=1 << 20)
    allp = patterns.peak_traffic(4, "all_peers", s)
    one = patterns.peak_traffic(4, "one_peer", s)
    assert patterns.fabric_bytes_per_rank(allp) == 3 << 20
    assert patterns.fabric_bytes_per_rank(one) == 1 << 20
    cands = patterns.sweep_candidates(allp, s)
    assert len(cands) == len(s.wave_sweep) * len(s.rccl_mems)
    line = next(c for c in cands if c.label == "serial w64 G").tests[0].line
    assert line.startswith(f"-12 (G0 G0 G1 64 {1 << 20}) (G0 G0 G2 64 ")
    line = patterns.sweep_candidates(one, s)[0].tests[0].line
    assert line.startswith(f"-4 (G0 G0 U1 16 {1 << 20}) (G1 G1 U2 16 ")


def test_measure_peaks():
    """Peak GB/s is the probe's per-rank egress over its fastest candidate."""
    import roofline

    def fake_run(binary, lines, env, **kw):
        # 1 ms per test, 2 ms for 256-wave ones.
        out = []
        for line in lines:
            ms = 2.0 if " 256 " in line else 1.0
            out.append(tb_text.TestResult(len(out) + 1, {"GPU 0": ms}, {0: ms}))
        return out, "1.71.02"

    s = RoofSettings(peak_bytes_per_peer=1_000_000)
    real = roofline.tb_text.run
    roofline.tb_text.run = fake_run
    try:
        peaks = roofline.Roofline(sys.executable, s).measure_peaks(8)
    finally:
        roofline.tb_text.run = real
    assert set(peaks) == set(patterns.PEAK_KINDS)
    allp, one = peaks["all_peers"], peaks["one_peer"]
    assert allp.peers == 7 and one.peers == 1
    assert math.isclose(allp.gbps, 7.0) and math.isclose(allp.per_link_gbps, 1.0)
    assert math.isclose(one.gbps, 1.0)
    assert "w256" not in allp.roof.label


def test_summary_against_peaks():
    """Per implementation at its largest message: wire rate over the fabric
    peak, the best known rate and the nominal rate; busbw over the peak."""
    import report
    from measure import Record
    from roofline import Peak, Roof

    n = 4
    small, big = 1 << 20, 1 << 24
    t_small = _two_shot(n, small, ring=False, ratio=0.28125)
    t_big = _two_shot(n, big, ring=False, ratio=0.28125)
    records = [
        Record("s", "fly", 10.0, "graph", t_small),
        Record("s", "rccl", 20.0, "graph", t_small),
        Record("b", "fly", 100.0, "graph", t_big),
        Record("b", "rccl", 200.0, "graph", t_big),
    ]
    peak = Peak(n, "all_peers", 150.0, 3, Roof(1.0, (1.0,), "serial w64 U"))
    peaks = {"all_peers": peak}
    s = RoofSettings(link_gbps=64.0)
    rows = report.summary(records, peaks, "xgmi", s)
    fly, rccl = rows
    assert fly["case"] == "b" and rccl["case"] == "b"
    fly_wire = patterns.fabric_bytes_per_rank(t_big) / 100.0 / 1e3
    rccl_wire = 1.5 * big / 200.0 / 1e3  # bf16, 2(N-1)/N
    assert math.isclose(fly["wire GB/s per rank"], fly_wire)
    assert math.isclose(rccl["wire GB/s per rank"], rccl_wire)
    assert math.isclose(fly["wire/peak"], fly_wire / 150.0)
    assert math.isclose(fly["wire/nominal"], fly_wire / (3 * 64.0))
    best, where = report.best_known(records, peaks)
    assert best == max(150.0, rccl_wire, fly_wire)
    assert math.isclose(fly["wire/best known"], fly_wire / best)
    assert math.isclose(fly["busbw/peak"], 1.5 * big / 100.0 / 1e3 / 150.0)
    # The same per-rank peak is in every graded row.
    (_, row) = report.grade(records, {}, {}, 150.0)
    assert math.isclose(row["op wire/peak"], fly_wire / 150.0)
    assert math.isclose(row["rccl wire/peak"], rccl_wire / 150.0)
    # Nominal: N - 1 links on xGMI, one on PCIe, none when unknown.
    assert report.nominal_gbps("xgmi", 8, s) == 7 * 64.0
    assert report.nominal_gbps("pcie", 8, s) == 64.0
    assert math.isnan(report.nominal_gbps("unknown", 8, s))
    assert math.isnan(report.nominal_gbps("xgmi", 8, DEFAULT))
    lines = report.peak_lines(records, peaks, "xgmi", s)
    assert lines[0] == "- link: xgmi, 4 ranks"
    assert "150.0 GB/s (50.0 per link" in lines[1]
    assert "not measured" in lines[2]
    assert "nominal per rank: 192.0 GB/s" in lines[-1]


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
