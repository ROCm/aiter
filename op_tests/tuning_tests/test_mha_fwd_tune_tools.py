# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Level 1: Unit tests for the MHA forward tuning tools in
`csrc/cpp_itfs/mha/tools/` (mha_count_shape.py, mha_tune.py,
mha_gen_runtime_json.py).

All inputs are synthetic (see `mha_fwd_tune_synth.py`); no GPU, torch or
aiter import is required.  Run:
    python3 -m unittest op_tests.tuning_tests.test_mha_fwd_tune_tools -v
"""

import argparse
import contextlib
import csv
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
_TOOLS_DIR = _REPO_ROOT / "csrc" / "cpp_itfs" / "mha" / "tools"

for _p in (_TOOLS_DIR, _THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mha_count_shape
import mha_fwd_tune_synth as synth
import mha_gen_runtime_json
import mha_tune  # must stay stdlib-only (no torch / aiter import)


def _rows(*seqlens):
    return [{"max_seqlen": m} for m in seqlens]


def _owner(intervals, x):
    """Indices of intervals [low, high) containing x (None = unbounded)."""
    return [
        i
        for i, (lo, hi) in enumerate(intervals)
        if (lo is None or x >= lo) and (hi is None or x < hi)
    ]


class TestRowIntervals(unittest.TestCase):
    """R5: `_row_intervals` midpoint must keep each sample in its own bucket."""

    def test_empty_and_single(self):
        self.assertEqual(mha_gen_runtime_json._row_intervals([]), [])
        self.assertEqual(
            mha_gen_runtime_json._row_intervals(_rows(1024)), [(None, None)]
        )

    def test_adjacent_512_513(self):
        iv = mha_gen_runtime_json._row_intervals(_rows(512, 513))
        self.assertEqual(iv, [(None, 513), (513, None)])
        self.assertEqual(_owner(iv, 512), [0])
        self.assertEqual(_owner(iv, 513), [1])

    def test_odd_and_even_gap(self):
        # even gap: 512/516 -> mid 514; odd gap: 512/515 -> mid 514 (round up)
        self.assertEqual(
            mha_gen_runtime_json._row_intervals(_rows(512, 516)),
            [(None, 514), (514, None)],
        )
        self.assertEqual(
            mha_gen_runtime_json._row_intervals(_rows(512, 515)),
            [(None, 514), (514, None)],
        )

    def test_interval_properties(self):
        cases = [
            (512, 513),
            (1, 2, 3, 4),
            (64, 128, 129, 1000, 1001, 1002, 4096),
            (256, 320, 384, 448, 512, 1024, 4176),
        ]
        for samples in cases:
            with self.subTest(samples=samples):
                iv = mha_gen_runtime_json._row_intervals(_rows(*samples))
                self.assertEqual(len(iv), len(samples))
                # every sample lies in exactly its own non-empty interval
                for i, m in enumerate(samples):
                    self.assertEqual(_owner(iv, m), [i])
                    lo, hi = iv[i]
                    if lo is not None and hi is not None:
                        self.assertLess(lo, hi)
                # intervals tile the line: disjoint + contiguous + unbounded ends
                self.assertIsNone(iv[0][0])
                self.assertIsNone(iv[-1][1])
                for i in range(1, len(iv)):
                    self.assertEqual(iv[i - 1][1], iv[i][0])
                # exhaustive coverage over a window around the samples
                for x in range(max(0, samples[0] - 8), samples[-1] + 8):
                    self.assertEqual(len(_owner(iv, x)), 1, msg=f"x={x}")

    def test_merged_payload_adjacent_constraint(self):
        with tempfile.TemporaryDirectory() as td:
            path = synth.write_tuned_csv(
                Path(td),
                {},
                [
                    {"max_seqlen": 512, "tile": synth.DEFAULT_TILE},
                    {"max_seqlen": 513, "tile": synth.ALT_TILE},
                ],
            )
            payload = mha_gen_runtime_json.build_merged_payload(
                [path],
                target="gfx942",
                schema_version=2,
                constraint_var="a.max_seqlen_q",
            )
        tiles = payload["tiles"]["bf16"]["80,96"]
        self.assertEqual(
            [t["cpp_constraint"] for t in tiles],
            ["a.max_seqlen_q < 513", "a.max_seqlen_q >= 513"],
        )
        self.assertEqual([t["_max_seqlen_samples"] for t in tiles], [[512], [513]])


def _write_tuned(
    d,
    meta=None,
    max_seqlens=(1024,),
    tile=synth.DEFAULT_TILE,
    gid=0,
    **kw,
):
    """Tuned csv under the canonical `mha_tuned_<gid>_<signature>.csv` name
    (so the loader does not warn about the filename shape)."""
    meta = dict(meta or {})
    rows = [{"max_seqlen": m, "tile": tile} for m in max_seqlens]
    path = Path(d) / synth.tuned_csv_name({**synth.SIGNATURE_DEFAULTS, **meta}, gid)
    return synth.write_tuned_csv(path, meta, rows, **kw)


def _merge(*paths):
    return mha_gen_runtime_json.build_merged_payload(
        list(paths),
        target="gfx942",
        schema_version=2,
        constraint_var="a.max_seqlen_q",
    )


class TestMergeConflicts(unittest.TestCase):
    """C2: one bucket may only be fed by csvs sharing the kernel signature."""

    def _merge_error(self, *paths):
        with self.assertRaises(ValueError) as cm:
            _merge(*paths)
        return str(cm.exception)

    def test_no_mask_vs_causal(self):
        with tempfile.TemporaryDirectory() as td:
            a = _write_tuned(td, {"mask_type": 0}, (1024,))
            b = _write_tuned(td, {"mask_type": 2}, (4096,))
            msg = self._merge_error(a, b)
        self.assertIn("dtype=bf16 hdim=(80,96)", msg)
        self.assertIn("2 incompatible tuned csvs", msg)
        self.assertIn("differing in: mask_type", msg)
        self.assertIn("mask_type=0", msg)
        self.assertIn("mask_type=2", msg)
        self.assertIn(a.name, msg)
        self.assertIn(b.name, msg)

    def test_report_is_order_independent(self):
        with tempfile.TemporaryDirectory() as td:
            a = _write_tuned(td, {"mask_type": 0}, (1024,))
            b = _write_tuned(td, {"mask_type": 2}, (4096,))
            self.assertEqual(self._merge_error(a, b), self._merge_error(b, a))

    def test_different_raw_hdim_same_compiled_bucket(self):
        with tempfile.TemporaryDirectory() as td:
            a = _write_tuned(td, {"hdim_q": 72, "hdim_v": 72}, (1024,))
            b = _write_tuned(td, {"hdim_q": 80, "hdim_v": 96}, (4096,))
            msg = self._merge_error(a, b)
        self.assertIn("differing in: hdim_q, hdim_v", msg)

    def test_gqa_vs_mha(self):
        with tempfile.TemporaryDirectory() as td:
            a = _write_tuned(td, {"nhead_k": 16}, (1024,))
            b = _write_tuned(td, {"nhead_k": 2}, (4096,))
            msg = self._merge_error(a, b)
        self.assertIn("differing in: nhead_k", msg)
        self.assertIn("nhead_k=2", msg)
        self.assertIn("nhead_k=16", msg)

    def test_every_conflict_column_is_detected(self):
        alt = {
            "mode": "batch",
            "hdim_q": 80,
            "hdim_v": 96,
            "mask_type": 2,
            "nhead_q": 8,
            "nhead_k": 2,
            "has_logits_soft_cap": 1,
            "bias_type": 1,
            "has_lse": 1,
            "has_dropout": 1,
            "skip_min_seqlen_q": 1,
            "qscale_type": 1,
            "has_sink": 1,
            "bench_variant": "skip",
        }
        self.assertEqual(sorted(alt), sorted(mha_gen_runtime_json.CONFLICT_COLS))
        for col, value in alt.items():
            with self.subTest(col=col), tempfile.TemporaryDirectory() as td:
                a = _write_tuned(td, None, (1024,), gid=0)
                b = _write_tuned(td, {col: value}, (4096,), gid=1)
                msg = self._merge_error(a, b)
                self.assertIn(f"differing in: {col})", msg)
                self.assertIn(f"{col}={value} : {b.name}", msg)

    def test_distinct_buckets_are_kept(self):
        with tempfile.TemporaryDirectory() as td:
            a = _write_tuned(td, {"mask_type": 0}, (1024,))
            b = _write_tuned(
                td,
                {
                    "mask_type": 2,
                    "hdim_q": 256,
                    "hdim_v": 256,
                    "best_hdim_q": 256,
                    "best_hdim_v": 256,
                },
                (4096,),
            )
            payload = _merge(a, b)
        self.assertEqual(payload["dtypes"], ["bf16"])
        self.assertEqual(sorted(payload["tiles"]["bf16"]), ["256,256", "80,96"])

    def test_dtypes_are_sorted(self):
        with tempfile.TemporaryDirectory() as td:
            a = _write_tuned(td, {"dtype": "fp16"}, (1024,))
            b = _write_tuned(td, {"dtype": "bf16"}, (4096,))
            payload = _merge(a, b)
            flipped = _merge(b, a)
        self.assertEqual(payload["dtypes"], ["bf16", "fp16"])
        self.assertEqual(payload["dtypes"], flipped["dtypes"])

    def test_same_signature_two_ranges_merge(self):
        with tempfile.TemporaryDirectory() as td:
            a = _write_tuned(td, None, (256, 512), gid=0)
            b = _write_tuned(td, None, (2048, 4096), tile=synth.ALT_TILE, gid=1)
            payload = _merge(a, b)
            flipped = _merge(b, a)
        tiles = payload["tiles"]["bf16"]["80,96"]
        self.assertEqual(
            [t["_max_seqlen_samples"] for t in tiles], [[256, 512], [2048, 4096]]
        )
        self.assertEqual(
            [t["cpp_constraint"] for t in tiles],
            ["a.max_seqlen_q < 1280", "a.max_seqlen_q >= 1280"],
        )
        for p in (payload, flipped):
            p["meta"].pop("generated_at")
        self.assertEqual(payload, flipped)

    def test_duplicate_max_seqlen_across_csvs(self):
        with tempfile.TemporaryDirectory() as td:
            a = _write_tuned(td, None, (256, 1024), gid=0)
            b = _write_tuned(td, None, (1024,), tile=synth.ALT_TILE, gid=1)
            names = ", ".join(sorted([a.name, b.name]))
            msg = self._merge_error(a, b)
        self.assertIn("same max_seqlen from several tuned csvs", msg)
        self.assertIn(f"max_seqlen=1024 : {names}", msg)
        self.assertIn("must be tuned together", msg)

    def test_duplicate_max_seqlen_inside_one_csv(self):
        with tempfile.TemporaryDirectory() as td:
            a = _write_tuned(td, None, (256, 1024, 1024))
            msg = self._merge_error(a)
        self.assertIn("duplicate max_seqlen [1024]", msg)

    def test_legacy_csv_without_signature_rejected(self):
        for dropped in (
            ["nhead_q", "nhead_k", "has_sink"],
            ["bench_variant"],
        ):
            with self.subTest(dropped=dropped), tempfile.TemporaryDirectory() as td:
                a = _write_tuned(td, None, (1024,), drop_columns=dropped)
                msg = self._merge_error(a)
                self.assertIn(f"missing required columns: {sorted(dropped)}", msg)
                self.assertIn("mha_tune.py bench", msg)

    def test_main_returns_2_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            a = _write_tuned(td, {"mask_type": 0}, (1024,))
            b = _write_tuned(td, {"mask_type": 2}, (4096,))
            out = Path(td) / "deploy.json"
            argv = [
                "mha_gen_runtime_json.py",
                "--in",
                str(a),
                "--in",
                str(b),
                "--out",
                str(out),
            ]
            err = io.StringIO()
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(err):
                rc = mha_gen_runtime_json.main()
            self.assertEqual(rc, 2)
            self.assertFalse(out.exists())
        self.assertIn("incompatible tuned csvs", err.getvalue())


class TestSynthRoundTrip(unittest.TestCase):
    """The synthetic generators must stay parseable by the real tools."""

    def test_dump_line_parses(self):
        line = synth.make_dump_line(seqlens_q=[100, 200], mask_type=2, nhead_k=2)
        with tempfile.TemporaryDirectory() as td:
            log = synth.write_dump_log(Path(td) / "dump.log", [line, line])
            recs = mha_count_shape.parse_log(log)
        self.assertEqual(len(recs), 2)
        r = recs[0]
        self.assertEqual(r["mode"], "group")
        self.assertEqual(r["dtype"], "bf16")
        self.assertEqual(r["mask_type"], 2)
        self.assertEqual(r["nhead_k"], 2)
        self.assertEqual(r["batch"], 2)
        self.assertEqual(r["max_seqlen_q"], 200)
        self.assertEqual(r["seqlens_q"], [100, 200])
        self.assertEqual(r["seqlens_k"], [100, 200])
        self.assertEqual(r["total_q"], 300)

    def test_dump_line_drop_field(self):
        line = synth.make_dump_line(has_dropout=None)
        self.assertNotIn("has_dropout", line)

    def test_tuned_csv_loads(self):
        with tempfile.TemporaryDirectory() as td:
            path = synth.write_tuned_csv(
                Path(td), {"mask_type": 2}, _rows(2048, 1024, 4096)
            )
            self.assertTrue(
                mha_gen_runtime_json._TUNED_NAME_RE.match(path.name), path.name
            )
            lc = mha_gen_runtime_json.TunedCsv.load(path)
        self.assertEqual(lc.mask_type, 2)
        self.assertEqual((lc.compiled_hdim_q, lc.compiled_hdim_v), (80, 96))
        self.assertEqual([r["max_seqlen"] for r in lc.rows], [1024, 2048, 4096])


def _parse_lines(*lines, noise=True):
    with tempfile.TemporaryDirectory() as td:
        log = synth.write_dump_log(Path(td) / "dump.log", lines, noise=noise)
        return mha_count_shape.parse_log(log)


class TestParseLog(unittest.TestCase):
    """C4: strict parsing of the extended dump format (group + batch)."""

    def test_group_new_fields(self):
        line = synth.make_dump_line(
            seqlens_q=[300, 700],
            seqlens_k=[310, 720],
            mask_type=2,
            window_left=-1,
            window_right=0,
            sink_size=4,
            has_sink=1,
            has_logits_soft_cap=1,
            qscale_type=1,
            min_seqlen_q=2,
        )
        (r,) = _parse_lines(line)
        self.assertEqual(r["mode"], "group")
        self.assertEqual((r["window_left"], r["window_right"]), (-1, 0))
        self.assertEqual(r["sink_size"], 4)
        self.assertEqual(r["has_sink"], 1)
        self.assertEqual(r["has_logits_soft_cap"], 1)
        self.assertEqual(r["qscale_type"], 1)
        self.assertEqual(r["min_seqlen_q"], 2)
        self.assertEqual(r["seqlens_q"], [300, 700])
        self.assertEqual(r["seqlens_k"], [310, 720])
        self.assertEqual((r["total_q"], r["total_k"]), (1000, 1030))

    def test_group_defaults_no_mask(self):
        (r,) = _parse_lines(synth.make_dump_line())
        self.assertEqual((r["window_left"], r["window_right"]), (-1, -1))
        self.assertEqual(
            (r["sink_size"], r["has_sink"], r["has_logits_soft_cap"]), (0, 0, 0)
        )
        self.assertEqual((r["qscale_type"], r["min_seqlen_q"]), (0, 0))

    def test_batch_new_format(self):
        line = synth.make_dump_line(
            seqlens_q=[512] * 4, mode="batch", seqlens_k=[1024] * 4
        )
        (r,) = _parse_lines(line)
        self.assertEqual(r["mode"], "batch")
        self.assertEqual(r["batch"], 4)
        self.assertEqual((r["seqlen_q"], r["seqlen_k"]), (512, 1024))
        for f in ("min_seqlen_q", "total_q", "total_k", "seqlens_q", "seqlens_k"):
            self.assertNotIn(f, r)
        self.assertEqual(r["window_left"], -1)

    def test_mixed_group_and_batch(self):
        recs = _parse_lines(
            synth.make_dump_line(seqlens_q=[128]),
            synth.make_dump_line(seqlens_q=[256, 256], mode="batch"),
        )
        self.assertEqual([r["mode"] for r in recs], ["group", "batch"])

    def test_banner_line_skipped(self):
        banner = "[MHA_FWD] AITER_DUMP_MHA_FWD_INFO enabled, writing to /tmp/x.log"
        recs = _parse_lines(banner, synth.make_dump_line())
        self.assertEqual(len(recs), 1)

    def test_capture_warning_line_skipped(self):
        # Printed once on stderr by mha_dump_should_emit_on() when a call is
        # graph-captured; must be ignored if stderr is merged into the log.
        warning = (
            "[MHA_FWD] AITER_DUMP_MHA_FWD_INFO: skipping dump during stream "
            "capture (calls captured into or replayed from a HIP/CUDA graph "
            "are not recorded). This warning is printed once."
        )
        recs = _parse_lines(warning, synth.make_dump_line())
        self.assertEqual(len(recs), 1)

    def _assert_missing(self, line, field):
        with self.assertRaises(ValueError) as cm:
            _parse_lines(line, noise=False)
        msg = str(cm.exception)
        self.assertIn(field, msg)
        self.assertIn("re-dump", msg)
        self.assertIn(":1:", msg)  # line number of the offending record

    def test_missing_common_field(self):
        for field in mha_count_shape.COMMON_REQUIRED:
            if field == "mode":
                continue
            with self.subTest(field=field):
                self._assert_missing(synth.make_dump_line(drop=[field]), field)

    def test_old_format_rejected(self):
        old = synth.make_dump_line(
            drop=[
                "window_left",
                "window_right",
                "sink_size",
                "has_sink",
                "has_logits_soft_cap",
                "qscale_type",
                "min_seqlen_q",
            ]
        )
        self._assert_missing(old, "window_left")

    def test_group_missing_specific(self):
        for field in mha_count_shape.MODE_REQUIRED["group"]:
            with self.subTest(field=field):
                self._assert_missing(synth.make_dump_line(drop=[field]), field)

    def test_batch_missing_specific(self):
        for field in mha_count_shape.MODE_REQUIRED["batch"]:
            with self.subTest(field=field):
                self._assert_missing(
                    synth.make_dump_line(mode="batch", drop=[field]), field
                )

    def test_unknown_or_missing_mode(self):
        for mode in ("splitkv", None):
            with self.subTest(mode=mode):
                line = synth.make_dump_line(mode=mode)
                with self.assertRaises(ValueError) as cm:
                    _parse_lines(line)
                self.assertIn("unknown mode", str(cm.exception))

    def test_cmd_group_exits_on_bad_log(self):
        with tempfile.TemporaryDirectory() as td:
            log = synth.write_dump_log(
                Path(td) / "dump.log", [synth.make_dump_line(drop=["qscale_type"])]
            )
            args = argparse.Namespace(
                input_log=str(log), out_dir=str(Path(td) / "out"), topk=5
            )
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
                mha_count_shape.cmd_group(args)
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("[ERROR]", buf.getvalue())
        self.assertIn("qscale_type", buf.getvalue())


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _run_cmd_group(td, lines, strict=False):
    """Run `cmd_group` on a synthetic log; return (out_dir, stdout)."""
    log = synth.write_dump_log(Path(td) / "dump.log", lines)
    out_dir = Path(td) / "out"
    args = argparse.Namespace(
        input_log=str(log), out_dir=str(out_dir), topk=5, strict=strict
    )
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            mha_count_shape.cmd_group(args)
    except SystemExit as e:
        e.stdout = buf.getvalue()
        raise
    return out_dir, buf.getvalue()


# Causal record as emitted for a GQA hdim-256 vLLM layer (window -1/0).
_CAUSAL = {"mask_type": 2, "window_left": -1, "window_right": 0}


class TestGroupFilter(unittest.TestCase):
    """C5: `group` keeps only group-mode fwd in the supported trait scope."""

    def _reason(self, **kw):
        (r,) = _parse_lines(synth.make_dump_line(**kw))
        return mha_count_shape.drop_reason(r)

    def test_supported_records_kept(self):
        self.assertIsNone(self._reason())  # no-mask, window -1/-1
        self.assertIsNone(self._reason(**_CAUSAL))  # causal top-left
        self.assertIsNone(self._reason(mask_type=1, window_left=-1, window_right=0))
        # D3: min_seqlen_q != 0 selects the skip variant, not a drop reason
        self.assertIsNone(self._reason(min_seqlen_q=1, **_CAUSAL))

    def test_drop_reasons(self):
        cases = {
            "batch": {"mode": "batch", "seqlens_q": [256, 256]},
            "sliding_window": {"mask_type": 2, "window_left": 128, "window_right": 0},
            "sink": dict(sink_size=4, has_sink=1, **_CAUSAL),
            "logits_soft_cap": {"has_logits_soft_cap": 1},
            "qscale": {"qscale_type": 1},
            "dropout": {"has_dropout": 1},
        }
        for reason, kw in cases.items():
            with self.subTest(reason=reason):
                self.assertEqual(self._reason(**kw), reason)

    def test_sliding_window_variants(self):
        for kw in (
            {"mask_type": 2, "window_left": -1, "window_right": 64},  # right window
            {"mask_type": 3, "window_left": 32, "window_right": 32},  # generic local
            {"mask_type": 0, "window_left": 128, "window_right": -1},  # inconsistent
        ):
            with self.subTest(**kw):
                self.assertEqual(self._reason(**kw), "sliding_window")

    def test_sink_flag_or_size(self):
        self.assertEqual(self._reason(has_sink=1), "sink")
        self.assertEqual(self._reason(sink_size=8), "sink")

    def test_reason_priority(self):
        # batch wins over every trait; sliding window wins over sink etc.
        self.assertEqual(self._reason(mode="batch", qscale_type=1, has_sink=1), "batch")
        self.assertEqual(
            self._reason(mask_type=2, window_left=8, window_right=0, has_sink=1),
            "sliding_window",
        )
        self.assertEqual(self._reason(has_sink=1, qscale_type=1), "sink")
        self.assertEqual(self._reason(qscale_type=1, has_dropout=1), "qscale")

    def test_cmd_group_mixed_log(self):
        lines = [
            synth.make_dump_line(seqlens_q=[1000, 2000]),
            synth.make_dump_line(seqlens_q=[1000, 2000]),
            synth.make_dump_line(
                seqlens_q=[512],
                hdim_q=256,
                hdim_v=256,
                nhead_k=2,
                min_seqlen_q=1,
                **_CAUSAL,
            ),
            synth.make_dump_line(seqlens_q=[300, 300], mode="batch"),
            synth.make_dump_line(
                seqlens_q=[100], mask_type=2, window_left=64, window_right=0
            ),
            synth.make_dump_line(seqlens_q=[100], has_sink=1, sink_size=4),
            synth.make_dump_line(seqlens_q=[100], has_logits_soft_cap=1),
            synth.make_dump_line(seqlens_q=[100], qscale_type=1),
            synth.make_dump_line(seqlens_q=[200], qscale_type=1),
            synth.make_dump_line(seqlens_q=[400], has_dropout=1),
        ]
        with tempfile.TemporaryDirectory() as td:
            out_dir, out = _run_cmd_group(td, lines)
            summary = _read_csv(out_dir / mha_count_shape.SUMMARY_NAME)
            dropped = _read_csv(out_dir / mha_count_shape.DROPPED_SUMMARY_NAME)
            group_csvs = sorted(p.name for p in out_dir.glob("mha_group_*.csv"))

        self.assertIn(
            "kept 3 / dropped 7 (batch=1, sliding_window=1, sink=1, "
            "logits_soft_cap=1, qscale=2, dropout=1)",
            out,
        )
        # only group-mode groups; the min_seqlen_q=1 causal record is kept
        self.assertEqual({r["mode"] for r in summary}, {"group"})
        self.assertEqual(
            sorted((r["hdim_q"], r["mask_type"], r["num_calls"]) for r in summary),
            [("256", "2", "1"), ("72", "0", "2")],
        )
        self.assertEqual(len(group_csvs), 2)
        self.assertFalse(any("batch" in n for n in group_csvs))

        self.assertEqual(
            list(dropped[0].keys()),
            [*mha_count_shape.DROPPED_COLS, "num_calls", "total_q_tokens"],
        )
        got = [
            (r["reason"], r["mode"], r["num_calls"], r["total_q_tokens"])
            for r in dropped
        ]
        self.assertEqual(
            got,
            [
                ("batch", "batch", "1", "600"),  # batch * seqlen_q
                ("sliding_window", "group", "1", "100"),
                ("sink", "group", "1", "100"),
                ("logits_soft_cap", "group", "1", "100"),
                ("qscale", "group", "2", "300"),  # same bucket aggregated
                ("dropout", "group", "1", "400"),
            ],
        )

    def test_cmd_group_nothing_dropped(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir, out = _run_cmd_group(td, [synth.make_dump_line()])
            dropped_path = out_dir / mha_count_shape.DROPPED_SUMMARY_NAME
            self.assertTrue(dropped_path.is_file())
            self.assertEqual(_read_csv(dropped_path), [])  # header only
        self.assertIn("kept 1 / dropped 0", out)

    def test_cmd_group_only_unsupported_exits(self):
        lines = [
            synth.make_dump_line(seqlens_q=[256] * 2, mode="batch"),
            synth.make_dump_line(qscale_type=1),
        ]
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit) as cm:
                _run_cmd_group(td, lines)
            out_dir = Path(td) / "out"
            self.assertTrue((out_dir / mha_count_shape.DROPPED_SUMMARY_NAME).is_file())
            self.assertFalse((out_dir / mha_count_shape.SUMMARY_NAME).exists())
        self.assertEqual(cm.exception.code, 1)


class TestSeqlenMismatch(unittest.TestCase):
    """C7: seqlen_q != seqlen_k is warned about and tuned by seqlen_q."""

    def _rec(self, **kw):
        (r,) = _parse_lines(synth.make_dump_line(**kw))
        return r

    def test_has_seqlen_mismatch(self):
        self.assertFalse(mha_count_shape.has_seqlen_mismatch(self._rec()))
        self.assertTrue(
            mha_count_shape.has_seqlen_mismatch(
                self._rec(seqlens_q=[100, 200], seqlens_k=[100, 300])
            )
        )
        self.assertTrue(
            mha_count_shape.has_seqlen_mismatch(
                self._rec(mode="batch", seqlens_q=[64], seqlens_k=[128])
            )
        )
        self.assertFalse(
            mha_count_shape.has_seqlen_mismatch(
                self._rec(mode="batch", seqlens_q=[64, 64])
            )
        )

    def test_stats(self):
        recs = [
            self._rec(seqlens_q=[100, 300]),
            self._rec(seqlens_q=[100], seqlens_k=[900]),
            self._rec(seqlens_q=[50, 50], seqlens_k=[50, 60]),
            self._rec(seqlens_q=[400]),
        ]
        st = mha_count_shape.seqlen_mismatch_stats(recs)
        self.assertEqual(
            st,
            {
                "num_calls": 4,
                "mismatch_calls": 2,
                "total_q_tokens": 1000,
                "mismatch_q_tokens": 200,
            },
        )

    def _lines(self):
        return [
            synth.make_dump_line(seqlens_q=[1000, 2000]),
            synth.make_dump_line(seqlens_q=[1000, 2000]),
            synth.make_dump_line(seqlens_q=[500], seqlens_k=[1500]),
            synth.make_dump_line(
                seqlens_q=[512], hdim_q=256, hdim_v=256, nhead_k=2, **_CAUSAL
            ),
        ]

    def test_cmd_group_warns_and_keeps(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir, out = _run_cmd_group(td, self._lines())
            summary = _read_csv(out_dir / mha_count_shape.SUMMARY_NAME)
            hq72_csv = next(out_dir.glob("mha_group_*_hq72_*.csv"))
            rows = _read_csv(hq72_csv)
        self.assertIn(
            "[WARN] 1 / 4 kept calls (25.0%, 7.1% of total_q tokens) "
            "have seqlen_q != seqlen_k",
            out,
        )
        self.assertIn("tuned by seqlen_q", out)
        self.assertIn("full seqlen_q != seqlen_k support is pending", out)
        by_hq = {r["hdim_q"]: r for r in summary}
        self.assertEqual(by_hq["72"]["seqlen_mismatch_calls"], "1")
        self.assertEqual(by_hq["72"]["num_calls"], "3")  # not dropped
        self.assertEqual(by_hq["256"]["seqlen_mismatch_calls"], "0")
        self.assertIn(("500", "1500"), {(r["seqlens_q"], r["seqlens_k"]) for r in rows})

    def test_cmd_group_no_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            _, out = _run_cmd_group(td, self._lines()[:2])
        self.assertIn("seqlen_q == seqlen_k for every kept call", out)
        self.assertNotIn("[WARN]", out)

    def test_cmd_group_strict(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit) as cm:
                _run_cmd_group(td, self._lines(), strict=True)
            self.assertFalse((Path(td) / "out" / mha_count_shape.SUMMARY_NAME).exists())
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("[ERROR] 1 / 4 kept calls", cm.exception.stdout)
        self.assertIn("Aborting (--strict)", cm.exception.stdout)
        # strict passes when everything is equal-length
        with tempfile.TemporaryDirectory() as td:
            _run_cmd_group(td, self._lines()[:2], strict=True)

    def test_dropped_mismatch_not_counted(self):
        lines = [
            synth.make_dump_line(seqlens_q=[100]),
            synth.make_dump_line(mode="batch", seqlens_q=[64], seqlens_k=[128]),
        ]
        with tempfile.TemporaryDirectory() as td:
            _run_cmd_group(td, lines, strict=True)  # batch dropped first

    def test_strict_cli_flag(self):
        with tempfile.TemporaryDirectory() as td:
            log = synth.write_dump_log(Path(td) / "d.log", self._lines())
            argv = ["mha_count_shape.py", "group", "-i", str(log), "-d", td, "--strict"]
            buf = io.StringIO()
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(
                buf
            ), self.assertRaises(SystemExit) as cm:
                mha_count_shape.main()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("[ERROR]", buf.getvalue())


_SIG_HQ72 = (
    "group_bf16_hq72_hv72_mask0_nh16_nhk16_"
    "nlogits_nbias_nlse_ndropout_nskip_nqscale_nsink"
)
_SIG_HQ256 = (
    "group_bf16_hq256_hv256_mask2_nh16_nhk2_"
    "nlogits_nbias_nlse_ndropout_skip_nqscale_nsink"
)


def _key(**over):
    vals = {
        "mode": "group",
        "dtype": "bf16",
        "hdim_q": 72,
        "hdim_v": 72,
        "mask_type": 0,
        "nhead_q": 16,
        "nhead_k": 16,
        "has_logits_soft_cap": 0,
        "bias_type": 0,
        "has_lse": 0,
        "has_dropout": 0,
        "skip_min_seqlen_q": 0,
        "qscale_type": 0,
        "has_sink": 0,
    }
    vals.update(over)
    return tuple(vals[c] for c in mha_count_shape.GROUP_COLS)


class TestGroupSignature(unittest.TestCase):
    """C5: group key / filename carry every kernel-selecting field."""

    def test_group_cols(self):
        self.assertEqual(
            mha_count_shape.GROUP_COLS,
            (
                "mode",
                "dtype",
                "hdim_q",
                "hdim_v",
                "mask_type",
                "nhead_q",
                "nhead_k",
                "has_logits_soft_cap",
                "bias_type",
                "has_lse",
                "has_dropout",
                "skip_min_seqlen_q",
                "qscale_type",
                "has_sink",
            ),
        )

    def test_default_signature(self):
        self.assertEqual(mha_count_shape.group_signature(_key()), _SIG_HQ72)

    def test_each_token(self):
        cases = [
            ({"nhead_q": 32, "nhead_k": 4}, "_nh32_nhk4_"),
            ({"has_logits_soft_cap": 1}, "_logits_"),
            ({"bias_type": 1}, "_bias_"),
            ({"bias_type": 2}, "_alibi_"),
            ({"has_lse": 1}, "_lse_"),
            ({"has_dropout": 1}, "_dropout_"),
            ({"skip_min_seqlen_q": 1}, "_skip_"),
            ({"qscale_type": 1}, "_pertensor_"),
            ({"qscale_type": 4}, "_mx_"),
            ({"mask_type": 1}, "_mask1_"),
        ]
        for over, token in cases:
            with self.subTest(**over):
                sig = mha_count_shape.group_signature(_key(**over))
                self.assertIn(token, sig + "_")
                self.assertEqual(len(sig.split("_")), len(_SIG_HQ72.split("_")))
        self.assertTrue(
            mha_count_shape.group_signature(_key(has_sink=1)).endswith("_sink")
        )

    def test_string_values_from_csv(self):
        as_str = tuple(str(v) for v in _key(nhead_k=2, skip_min_seqlen_q=1))
        self.assertEqual(
            mha_count_shape.group_signature(as_str),
            mha_count_shape.group_signature(_key(nhead_k=2, skip_min_seqlen_q=1)),
        )

    def test_legacy_five_field_key(self):
        self.assertEqual(
            mha_count_shape.group_signature(("group", "bf16", 72, 72, 0)),
            "group_bf16_hq72_hv72_mask0",
        )

    def test_invalid_key(self):
        with self.assertRaises(ValueError):
            mha_count_shape.group_signature(("group", "bf16", 72))
        with self.assertRaises(ValueError):
            mha_count_shape.group_signature(_key(bias_type=7))

    def test_group_key_skip_from_min_seqlen_q(self):
        (r0,) = _parse_lines(synth.make_dump_line(min_seqlen_q=0))
        (r1,) = _parse_lines(synth.make_dump_line(min_seqlen_q=5))
        cols = mha_count_shape.GROUP_COLS
        i = cols.index("skip_min_seqlen_q")
        self.assertEqual(mha_count_shape.group_key(r0)[i], 0)
        self.assertEqual(mha_count_shape.group_key(r1)[i], 1)

    def test_single_field_difference_splits_groups(self):
        # The two real-workload groups plus variants differing in exactly one
        # kernel-selecting field must all end up in separate files.
        base256 = dict(hdim_q=256, hdim_v=256, nhead_k=2, min_seqlen_q=1, **_CAUSAL)
        lines = [
            synth.make_dump_line(seqlens_q=[1000]),
            synth.make_dump_line(seqlens_q=[1000], nhead_k=2),
            synth.make_dump_line(seqlens_q=[1000], has_lse=1),
            synth.make_dump_line(seqlens_q=[1000], bias_type=2),
            synth.make_dump_line(seqlens_q=[1000], nhead_q=32),
            synth.make_dump_line(seqlens_q=[1000], min_seqlen_q=1),
            synth.make_dump_line(seqlens_q=[512], **base256),
            synth.make_dump_line(seqlens_q=[512], **{**base256, "min_seqlen_q": 0}),
        ]
        with tempfile.TemporaryDirectory() as td:
            out_dir, out = _run_cmd_group(td, lines)
            names = sorted(p.name for p in out_dir.glob("mha_group_*.csv"))
            summary = _read_csv(out_dir / mha_count_shape.SUMMARY_NAME)
            hq256 = [r for r in summary if r["signature"] == _SIG_HQ256]
            self.assertEqual(len(hq256), 1)
            rows = _read_csv(out_dir / hq256[0]["group_csv"])
        self.assertEqual(len(names), len(lines))
        self.assertEqual(len({r["signature"] for r in summary}), len(lines))
        self.assertTrue(any(n.endswith(f"_{_SIG_HQ72}.csv") for n in names))
        self.assertEqual(
            hq256[0]["group_csv"], f"mha_group_{hq256[0]['group_id']}_{_SIG_HQ256}.csv"
        )
        # group CSV carries every GROUP_COLS column with the group values
        for c, v in zip(
            mha_count_shape.GROUP_COLS,
            _key(hdim_q=256, hdim_v=256, mask_type=2, nhead_k=2, skip_min_seqlen_q=1),
        ):
            self.assertEqual(rows[0][c], str(v), msg=c)
        self.assertIn(_SIG_HQ256, out)  # overview prints the signature

    def _gen_range(self, group_csv):
        args = argparse.Namespace(
            input_csv=str(group_csv),
            output="",
            range=["512:640:64"],
            singletons="4096",
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mha_count_shape.cmd_generate_tune_range(args)
        return buf.getvalue()

    def test_generate_tune_range_keeps_signature(self):
        line = synth.make_dump_line(
            seqlens_q=[512],
            hdim_q=256,
            hdim_v=256,
            nhead_k=2,
            min_seqlen_q=1,
            **_CAUSAL,
        )
        with tempfile.TemporaryDirectory() as td:
            out_dir, _ = _run_cmd_group(td, [line])
            (gcsv,) = out_dir.glob("mha_group_*.csv")
            self._gen_range(gcsv)
            untune = out_dir / f"mha_untune_0_{_SIG_HQ256}.csv"
            self.assertTrue(untune.is_file(), sorted(os.listdir(out_dir)))
            rows = _read_csv(untune)
            self.assertTrue(mha_tune._UNTUNE_NAME_RE.match(untune.name))
            meta, ms = mha_tune.parse_untune_csv(untune)
            tuned = mha_tune.tuned_csv_path(meta, out_dir)
        self.assertEqual(
            list(rows[0].keys()), ["max_seqlen", *mha_count_shape.GROUP_COLS]
        )
        self.assertEqual([r["max_seqlen"] for r in rows], ["512", "576", "640", "4096"])
        self.assertEqual({r["skip_min_seqlen_q"] for r in rows}, {"1"})
        self.assertEqual({r["nhead_k"] for r in rows}, {"2"})
        self.assertEqual(meta.gid, 0)
        self.assertEqual((meta.hdim_q, meta.mask_type), (256, 2))
        self.assertEqual(ms, [512, 576, 640, 4096])
        self.assertEqual(tuned.name, f"mha_tuned_0_{_SIG_HQ256}.csv")
        self.assertTrue(mha_gen_runtime_json._TUNED_NAME_RE.match(tuned.name))

    def test_generate_tune_range_legacy_group_csv_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            gcsv = Path(td) / "mha_group_3_group_bf16_hq72_hv72_mask0.csv"
            with gcsv.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([*mha_count_shape.BASE_GROUP_COLS, "count"])
                w.writerow(["group", "bf16", 72, 72, 0, 1])
            args = argparse.Namespace(
                input_csv=str(gcsv), output="", range=["512:512:1"], singletons=""
            )
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit):
                mha_count_shape.cmd_generate_tune_range(args)
            self.assertFalse(
                (Path(td) / "mha_untune_3_group_bf16_hq72_hv72_mask0.csv").exists()
            )
        self.assertIn("missing columns", buf.getvalue())
        self.assertIn("nhead_q", buf.getvalue())

    def test_generate_tune_range_partial_traits_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            gcsv = Path(td) / "mha_group_0_x.csv"
            with gcsv.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([*mha_count_shape.BASE_GROUP_COLS, "nhead_q"])
                w.writerow(["group", "bf16", 72, 72, 0, 16])
            args = argparse.Namespace(
                input_csv=str(gcsv), output="", range=["512:512:1"], singletons=""
            )
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit):
                mha_count_shape.cmd_generate_tune_range(args)
        self.assertIn("missing columns", buf.getvalue())
        self.assertIn("nhead_k", buf.getvalue())


class TestTuneModeGuard(unittest.TestCase):
    """C5: mha_tune refuses anything but group mode."""

    def test_parse_untune_csv_group_ok(self):
        with tempfile.TemporaryDirectory() as td:
            p = synth.write_untune_csv(Path(td), [512, 1024])
            meta, ms = mha_tune.parse_untune_csv(p)
        self.assertEqual(meta.mode, "group")
        self.assertEqual(ms, [512, 1024])

    def test_parse_untune_csv_rejects_batch(self):
        with tempfile.TemporaryDirectory() as td:
            p = synth.write_untune_csv(Path(td), [512], {"mode": "batch"})
            with self.assertRaises(ValueError) as cm:
                mha_tune.parse_untune_csv(p)
        self.assertIn("unsupported mode='batch'", str(cm.exception))

    def _bench_args(self, mode):
        return mha_tune._build_bench_args(
            dtype="bf16",
            hdim_q_bench=80,
            hdim_v_bench=96,
            nhead_q=16,
            nhead_k=16,
            max_seqlen=1024,
            mask_letter="0",
            lse=0,
            p_drop=0.0,
            bias="n",
            warmup=1,
            repeat=1,
            mode=mode,
        )

    def test_build_bench_args(self):
        self.assertIn("-mode=1", self._bench_args("group"))
        for mode in ("batch", "splitkv"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self._bench_args(mode)

    def test_filters_from_meta_rejects_batch(self):
        meta = _meta(mode="batch")
        with self.assertRaises(ValueError):
            mha_tune._filters_from_meta(meta)
        meta.mode = "group"
        self.assertEqual(mha_tune._filters_from_meta(meta)["mode"], ["group"])


def _meta(**over):
    """UntuneMeta built from synth.signature(**over)."""
    return mha_tune.UntuneMeta(
        gid=0,
        **synth.signature(**over),
        input_stem="mha_untune_0_x",
        input_path=Path("mha_untune_0_x.csv"),
    )


# hq256 GQA causal group from the real vLLM log (min_seqlen_q=1 -> skip).
_HQ256 = {
    "hdim_q": 256,
    "hdim_v": 256,
    "mask_type": 2,
    "nhead_k": 2,
    "skip_min_seqlen_q": 1,
}


class TestSignatureConsts(unittest.TestCase):
    """C6: the three tools agree on the signature columns."""

    def test_same_signature_cols(self):
        self.assertEqual(mha_tune.SIGNATURE_COLS, mha_count_shape.GROUP_COLS)
        self.assertEqual(
            mha_gen_runtime_json.SIGNATURE_COLS, mha_count_shape.GROUP_COLS
        )
        self.assertEqual(tuple(synth.SIGNATURE_DEFAULTS), mha_count_shape.GROUP_COLS)
        self.assertEqual(
            tuple(
                mha_tune.UntuneMeta(
                    gid=0, **synth.signature(), input_stem="", input_path=Path(".")
                ).signature()
            ),
            mha_count_shape.GROUP_COLS,
        )

    def test_tuned_fieldnames(self):
        self.assertEqual(mha_tune.TUNED_CSV_COLS, synth.TUNED_FIELDNAMES)

    def test_synth_name_matches_count_shape(self):
        for over in ({}, _HQ256, {"bias_type": 2, "has_lse": 1}):
            sig = synth.signature(**over)
            with self.subTest(**over):
                self.assertEqual(
                    synth.signature_name(sig),
                    mha_count_shape.group_signature(
                        tuple(sig[c] for c in mha_count_shape.GROUP_COLS)
                    ),
                )


class TestParseSignature(unittest.TestCase):
    """C6: filename signature <-> GROUP_COLS round trip and strictness."""

    def test_round_trip(self):
        cases = [
            {},
            _HQ256,
            {"nhead_q": 32, "nhead_k": 4, "bias_type": 1, "has_lse": 1},
            {
                "has_logits_soft_cap": 1,
                "has_dropout": 1,
                "qscale_type": 2,
                "has_sink": 1,
            },
            {"dtype": "fp16", "bias_type": 2, "mask_type": 1},
        ]
        for over in cases:
            sig = synth.signature(**over)
            with self.subTest(**over):
                name = mha_count_shape.group_signature(
                    tuple(sig[c] for c in mha_count_shape.GROUP_COLS)
                )
                self.assertEqual(mha_tune.parse_signature(name), sig)

    def test_kv_blockscale_token(self):
        name = synth.signature_name(synth.signature()).replace(
            "_nqscale_", "_kv_blockscale_"
        )
        self.assertEqual(mha_tune.parse_signature(name)["qscale_type"], 3)

    def test_legacy_and_missing(self):
        with self.assertRaises(ValueError) as cm:
            mha_tune.parse_signature("group_bf16_hq72_hv72_mask0")
        self.assertIn("nhead_q", str(cm.exception))
        self.assertIn("has_sink", str(cm.exception))  # lists remaining fields
        full = synth.signature_name(synth.signature())
        with self.assertRaises(ValueError) as cm:
            mha_tune.parse_signature(full.replace("_nlse", ""))
        self.assertIn("has_lse", str(cm.exception))

    def test_unknown_reordered_trailing(self):
        full = synth.signature_name(synth.signature())
        bad = [
            full.replace("_nbias_", "_foo_"),
            full.replace("nlse_ndropout", "ndropout_nlse"),
            full + "_extra",
            full.replace("_nh16_", "_nhx_"),
        ]
        for s in bad:
            with self.subTest(sig=s), self.assertRaises(ValueError):
                mha_tune.parse_signature(s)


class TestParseUntuneCsvSignature(unittest.TestCase):
    """C6: parse_untune_csv reads + cross-checks the full signature."""

    def _parse(self, td, ms=(512, 1024), meta=None, name=None, **kw):
        path = Path(td) / name if name else Path(td)
        p = synth.write_untune_csv(path, list(ms), meta, **kw)
        return mha_tune.parse_untune_csv(p)

    def _parse_err(self, **kw):
        """Run _parse in a temp dir and return the ValueError message."""
        with tempfile.TemporaryDirectory() as td, self.assertRaises(ValueError) as cm:
            self._parse(td, **kw)
        return str(cm.exception)

    def test_full_signature_loaded(self):
        with tempfile.TemporaryDirectory() as td:
            meta, ms = self._parse(td, meta=_HQ256)
        self.assertEqual(meta.signature(), synth.signature(**_HQ256))
        self.assertEqual((meta.nhead_q, meta.nhead_k), (16, 2))
        self.assertEqual(meta.skip_min_seqlen_q, 1)
        self.assertEqual(meta.gid, 0)
        self.assertEqual(ms, [512, 1024])

    def test_legacy_filename_rejected(self):
        msg = self._parse_err(name="mha_untune_0_group_bf16_hq72_hv72_mask0.csv")
        self.assertIn("legacy", msg)
        self.assertIn("nhead_q", msg)

    def test_bad_filename_rejected(self):
        self._parse_err(name="untune_hq72.csv")

    def test_missing_columns_rejected(self):
        cols = [
            c for c in mha_count_shape.GROUP_COLS if c not in ("nhead_k", "has_lse")
        ]
        msg = self._parse_err(columns=cols)
        self.assertIn("missing required columns", msg)
        self.assertIn("nhead_k", msg)
        self.assertIn("has_lse", msg)

    def test_legacy_five_columns_rejected(self):
        msg = self._parse_err(columns=mha_count_shape.BASE_GROUP_COLS)
        self.assertIn("nhead_q", msg)

    def test_filename_vs_columns_mismatch(self):
        # filename says nhk2, columns say 16
        name = synth.untune_csv_name(synth.signature(nhead_k=2))
        msg = self._parse_err(name=name)
        self.assertIn("nhead_k", msg)
        self.assertIn("filename=2", msg)
        self.assertIn("csv=16", msg)

    def test_mixed_rows_rejected(self):
        msg = self._parse_err(ms=(512, 1024, 2048), row_over={2: {"has_lse": 1}})
        self.assertIn("line 4", msg)
        self.assertIn("has_lse", msg)

    def test_unsupported_traits_rejected(self):
        for over in (
            {"has_dropout": 1},
            {"has_logits_soft_cap": 1},
            {"qscale_type": 1},
            {"has_sink": 1},
        ):
            with self.subTest(**over):
                self.assertIn("unsupported", self._parse_err(meta=over))


class TestTuneFromSignature(unittest.TestCase):
    """C6: filters / bench args / tuned csv all come from the signature."""

    def test_filters_follow_meta(self):
        f = mha_tune._filters_from_meta(_meta(bias_type=2, has_lse=1))
        self.assertEqual(
            f,
            {
                "mode": ["group"],
                "vlayout": ["row"],
                "mask": ["s_no"],
                "bias": ["alibi"],
                "lse": ["t"],
                "dropout": ["f"],
                "logits": ["f"],
                "qscale": ["no"],
                "skip": ["f"],
                "sink": ["f"],
            },
        )

    def test_filters_skip_group_uses_nskip(self):
        f = mha_tune._filters_from_meta(_meta(**_HQ256))
        self.assertEqual(f["mask"], ["s_mask"])
        self.assertEqual(f["skip"], ["f"])
        self.assertEqual(f["sink"], ["f"])
        self.assertEqual(mha_tune.bench_variant(_meta(**_HQ256)), "nskip")

    def test_gen_filter_glob(self):
        self.assertEqual(
            mha_tune._gen_filter_from_meta(_meta()),
            "*bf16*_nbias*_nlse*_ndropout*",
        )
        self.assertEqual(
            mha_tune._gen_filter_from_meta(_meta(bias_type=2, has_lse=1)),
            "*bf16*_alibi*_lse*_ndropout*",
        )
        self.assertNotIn(
            "FMHA_FWD_GEN_FILTER", " ".join(mha_tune.DEFAULT_CMAKE_OPTIONS)
        )

    def test_gqa_bench_args(self):
        a = mha_tune.bench_args_for_meta(
            _meta(**_HQ256),
            hdim_q_bench=256,
            hdim_v_bench=256,
            max_seqlen=2048,
            warmup=1,
            repeat=2,
        )
        for tok in (
            "-h=16",
            "-h_k=2",
            "-mask=2",
            "-d=256",
            "-s=2048",
            "-s_k=2048",
            "-bias=n",
            "-lse=0",
            "-p_drop=0.0",
            "-mode=1",
        ):
            self.assertIn(tok, a)
        b = mha_tune.bench_args_for_meta(
            _meta(bias_type=1, has_lse=1),
            hdim_q_bench=80,
            hdim_v_bench=96,
            max_seqlen=512,
            warmup=1,
            repeat=1,
        )
        self.assertIn("-bias=e", b)
        self.assertIn("-lse=1", b)
        self.assertIn("-h_k=16", b)

    def test_bench_args_reject_dropout_and_window(self):
        for over in ({"has_dropout": 1}, {"mask_type": 3}):
            with self.subTest(**over), self.assertRaises(ValueError):
                mha_tune.bench_args_for_meta(
                    _meta(**over),
                    hdim_q_bench=80,
                    hdim_v_bench=96,
                    max_seqlen=512,
                    warmup=1,
                    repeat=1,
                )

    def test_old_cli_options_rejected(self):
        parser = mha_tune.build_parser()
        base = [
            "bench",
            "-i",
            "x.csv",
            "--work-dir",
            "w",
            "--ck-root",
            "c",
            "--tune-hdim-q",
            "80",
            "--tune-hdim-v",
            "96",
        ]
        ns = parser.parse_args(base)
        for attr in ("nhead_q", "nhead_k", "lse", "p_drop", "bias"):
            self.assertFalse(hasattr(ns, attr), attr)
        for extra in (
            ["--nhead-k", "2"],
            ["--nhead-q", "16"],
            ["--lse", "1"],
            ["--p-drop", "0.1"],
            ["--bias", "n"],
        ):
            err = io.StringIO()
            sub = self.subTest(extra=extra)
            with sub, contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
                parser.parse_args(base + extra)

    def _run_bench_stage(self, td, sig_over, max_seqlens=(512, 1024)):
        td = Path(td)
        untune = synth.write_untune_csv(td, list(max_seqlens), sig_over)
        meta, ms = mha_tune.parse_untune_csv(untune)
        tile = mha_tune.TileSize(*synth.DEFAULT_TILE)
        pair = mha_tune.PairPlan(
            hdim_q=256,
            hdim_v=256,
            sub_dir=td / "hq256_hv256",
            tiles_json=td / "hq256_hv256" / "tile_candidates.json",
            build_root=td / "hq256_hv256",
        )
        bdir = pair.build_root / f"build_{tile.name}"
        binary = Path(mha_tune._binary_path(str(bdir), "tile_example_fmha_fwd"))
        binary.parent.mkdir(parents=True)
        binary.write_text("")
        plan = mha_tune.TilePlan(
            pair=pair,
            tile=tile,
            build_dir=bdir,
            cfg_json_path=td / "cfg.json",
            cfg_json_text="{}",
        )
        args = argparse.Namespace(
            ck_root=str(td),
            work_dir=str(td / "work"),
            tune_hdim_q=256,
            tune_hdim_v=256,
            warmup=1,
            repeat=1,
            build_target="tile_example_fmha_fwd",
            dry_run=False,
        )
        calls = []

        def fake_bench(binary, bench_args, ck_root, dry_run):
            calls.append(list(bench_args))
            return "ok", {"time_ms": 0.5, "tflops": 10.0, "gbps": 1.0, "kname": "k"}

        err = io.StringIO()
        with mock.patch.object(
            mha_tune, "_do_bench", side_effect=fake_bench
        ), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            rc = mha_tune._bench_stage_and_dump([plan], meta, ms, args)
        tuned = mha_tune.tuned_csv_path(meta, Path(args.work_dir))
        return rc, calls, err.getvalue(), tuned, _read_csv(tuned)

    def test_bench_stage_skip_group(self):
        with tempfile.TemporaryDirectory() as td:
            rc, calls, err, tuned, rows = self._run_bench_stage(td, _HQ256)
            lc = mha_gen_runtime_json.TunedCsv.load(tuned)
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 2)
        for c in calls:
            self.assertIn("-h_k=2", c)
            self.assertIn("-mask=2", c)
        self.assertIn("[WARN] bench", err)
        self.assertIn("bench_variant=nskip", err)
        self.assertEqual(list(rows[0].keys()), list(mha_tune.TUNED_CSV_COLS))
        self.assertEqual({r["bench_variant"] for r in rows}, {"nskip"})
        self.assertEqual({r["skip_min_seqlen_q"] for r in rows}, {"1"})
        self.assertEqual({r["nhead_k"] for r in rows}, {"2"})
        self.assertEqual({r["status"] for r in rows}, {"ok"})
        self.assertEqual(
            tuned.name,
            synth.untune_csv_name(synth.signature(**_HQ256)).replace(
                "mha_untune_", "mha_tuned_"
            ),
        )
        # mha_gen_runtime_json reads the new columns
        self.assertEqual(lc.signature, synth.signature(**_HQ256))
        self.assertEqual(lc.bench_variant, "nskip")

    def test_bench_stage_nskip_group_no_warn(self):
        with tempfile.TemporaryDirectory() as td:
            rc, _, err, _, rows = self._run_bench_stage(td, {})
        self.assertEqual(rc, 0)
        self.assertNotIn("[WARN]", err)
        self.assertEqual({r["bench_variant"] for r in rows}, {"nskip"})
        self.assertEqual({r["skip_min_seqlen_q"] for r in rows}, {"0"})

    def test_build_stage_uses_meta_gen_filter_and_warns(self):
        args = argparse.Namespace(
            cmake_opt=[],
            ck_root="/nonexistent",
            workers=1,
            build_target="tile_example_fmha_fwd",
            jobs=1,
            no_fresh=False,
            dry_run=True,
            stop_on_error=False,
        )
        tile = mha_tune.TileSize(*synth.DEFAULT_TILE)
        pair = mha_tune.PairPlan(256, 256, Path("s"), Path("s/t.json"), Path("s"))
        plan = mha_tune.TilePlan(pair, tile, Path("s/b"), Path("s/c.json"), "{}")
        seen = []

        def fake_one(p, hipcc, ck_root, extra, *rest):
            seen.append(list(extra))
            return {
                "tile_name": p.tile.name,
                "did_configure": False,
                "did_make": False,
                "configure_ok": True,
                "build_ok": True,
                "log": None,
            }

        err = io.StringIO()
        with mock.patch.object(
            mha_tune, "_configure_and_build_one", side_effect=fake_one
        ), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            rc, ok = mha_tune._build_stage(
                [plan],
                args,
                do_configure=False,
                do_make=False,
                meta=_meta(**_HQ256, has_lse=1),
            )
        self.assertEqual((rc, len(ok)), (0, 1))
        self.assertIn("-DFMHA_FWD_GEN_FILTER=*bf16*_nbias*_lse*_ndropout*", seen[0])
        self.assertIn("-DFMHA_FWD_GEN_OPTDIM=256", seen[0])
        self.assertIn("[WARN] build", err.getvalue())


class TestBuildStamp(unittest.TestCase):
    """C3 / R4: `bench` reuses a build dir only when its stamp matches.

    The stamp covers MHA dimensions + configuration only (no CK source,
    compiler or arch); a mismatch is a hard error, never an auto-rebuild.
    """

    TARGET = "tile_example_fmha_fwd"

    # -- fixtures ---------------------------------------------------------
    def _args(self, td, **over):
        ns = argparse.Namespace(
            input_csv="",
            work_dir=str(Path(td) / "work"),
            ck_root=str(td),
            tune_hdim_q=80,
            tune_hdim_v=96,
            build_target=self.TARGET,
            cmake_opt=[],
            jobs=1,
            workers=1,
            no_fresh=False,
            dry_run=False,
            stop_on_error=False,
            warmup=1,
            repeat=1,
            limit=0,
            occupancy=None,
            allow_mfma_16=True,
        )
        for k, v in over.items():
            setattr(ns, k, v)
        return ns

    def _untune(self, base, sig_over=None, sub=""):
        """Write an untune CSV under `base`/`sub` and parse it."""
        d = Path(base) / sub if sub else Path(base)
        d.mkdir(parents=True, exist_ok=True)
        return mha_tune.parse_untune_csv(synth.write_untune_csv(d, [512], sig_over))

    def _plans(self, td, meta, tiles=(synth.DEFAULT_TILE, synth.ALT_TILE)):
        """TilePlans with their tune-config JSON written for `meta`."""
        root = Path(td) / "work" / "hq80_hv96"
        pair = mha_tune.PairPlan(
            hdim_q=80,
            hdim_v=96,
            sub_dir=root,
            tiles_json=root / "tile_candidates.json",
            build_root=root,
        )
        return mha_tune._tile_plans_for(
            pair,
            [mha_tune.TileSize(*t) for t in tiles],
            meta.dtype,
            filters=mha_tune._filters_from_meta(meta),
        )

    def _touch_binary(self, plan):
        b = Path(mha_tune._binary_path(str(plan.build_dir), self.TARGET))
        b.parent.mkdir(parents=True, exist_ok=True)
        b.write_text("")
        return b

    def _stamp_for(self, plan, meta, args):
        return mha_tune._build_stamp(
            plan,
            mha_tune._extra_cmake_opts_for_pair(plan.pair, meta, args.cmake_opt),
            args.build_target,
        )

    def _pretend_built(self, plans, meta, args):
        """Binary + matching stamp, as a successful build would leave it."""
        for p in plans:
            self._touch_binary(p)
            mha_tune._write_stamp(p.build_dir, self._stamp_for(p, meta, args))

    def _classify(self, td, sig_over=None, built_sig_over=None, **args_over):
        """Build with `built_sig_over`, then classify against `sig_over`."""
        td = Path(td)
        args = self._args(td, **args_over)
        built_meta, _ = self._untune(td, built_sig_over, sub="built")
        self._pretend_built(self._plans(td, built_meta), built_meta, args)
        meta, _ = self._untune(td, sig_over, sub="now")
        plans = self._plans(td, meta)
        return mha_tune._classify_existing_builds(plans, meta, args), plans

    # -- classification ---------------------------------------------------
    def test_missing_binary_is_built(self):
        with tempfile.TemporaryDirectory() as td:
            meta, _ = self._untune(td)
            plans = self._plans(td, meta)
            reusable, missing, stale = mha_tune._classify_existing_builds(
                plans, meta, self._args(td)
            )
        self.assertEqual((reusable, missing, stale), ([], plans, []))

    def test_matching_stamp_is_reused(self):
        with tempfile.TemporaryDirectory() as td:
            (reusable, missing, stale), plans = self._classify(td)
        self.assertEqual(len(reusable), len(plans))
        self.assertEqual((missing, stale), ([], []))

    def test_stamp_missing_is_stale(self):
        """A build dir from before C3 carries a binary but no stamp."""
        with tempfile.TemporaryDirectory() as td:
            meta, _ = self._untune(td)
            plans = self._plans(td, meta)
            for p in plans:
                self._touch_binary(p)
            reusable, missing, stale = mha_tune._classify_existing_builds(
                plans, meta, self._args(td)
            )
        self.assertEqual((reusable, missing), ([], []))
        self.assertEqual(
            [d for _, d in stale], [["<no build stamp found>"]] * len(plans)
        )

    def test_corrupt_or_foreign_stamp_is_stale(self):
        for payload in ("not json at all", "[1, 2, 3]"):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as td:
                args = self._args(td)
                meta, _ = self._untune(td)
                plans = self._plans(td, meta)
                self._pretend_built(plans, meta, args)
                mha_tune._stamp_path(plans[0].build_dir).write_text(payload)
                reusable, missing, stale = mha_tune._classify_existing_builds(
                    plans, meta, args
                )
                self.assertEqual(missing, [])
                self.assertEqual(len(reusable), len(plans) - 1)
                self.assertEqual([d for _, d in stale], [["<no build stamp found>"]])

    def test_schema_bump_is_stale(self):
        with tempfile.TemporaryDirectory() as td:
            args = self._args(td)
            meta, _ = self._untune(td)
            plans = self._plans(td, meta)
            self._pretend_built(plans, meta, args)
            old = self._stamp_for(plans[0], meta, args)
            old["schema"] = mha_tune.BUILD_STAMP_SCHEMA + 1
            mha_tune._write_stamp(plans[0].build_dir, old)
            _, missing, stale = mha_tune._classify_existing_builds(plans, meta, args)
        self.assertEqual(missing, [])
        self.assertEqual([d for _, d in stale], [["schema"]])

    def test_lse_switch_is_stale(self):
        """Same work-dir, same hdim, but the group signature flipped lse."""
        with tempfile.TemporaryDirectory() as td:
            (reusable, missing, stale), plans = self._classify(
                td, sig_over={"has_lse": 1}, built_sig_over={"has_lse": 0}
            )
        self.assertEqual((reusable, missing), ([], []))
        self.assertEqual(len(stale), len(plans))
        for _, diff in stale:
            self.assertIn("tune_config.filters.lse", diff)
            self.assertIn("cmake_args", diff)

    def test_mask_switch_is_stale(self):
        with tempfile.TemporaryDirectory() as td:
            (_, _, stale), plans = self._classify(
                td, sig_over={"mask_type": 2}, built_sig_over={"mask_type": 0}
            )
        self.assertEqual(len(stale), len(plans))
        for _, diff in stale:
            self.assertIn("tune_config.filters.mask", diff)

    def test_nhead_change_is_reused(self):
        """nhead only affects the bench CLI, not the compiled kernel."""
        with tempfile.TemporaryDirectory() as td:
            (reusable, missing, stale), plans = self._classify(
                td, sig_over={"nhead_k": 2}, built_sig_over={"nhead_k": 16}
            )
        self.assertEqual(len(reusable), len(plans))
        self.assertEqual((missing, stale), ([], []))

    def test_cmake_opt_change_is_stale(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            meta, _ = self._untune(td)
            plans = self._plans(td, meta)
            self._pretend_built(plans, meta, self._args(td))
            args = self._args(td, cmake_opt=["-DCMAKE_HIP_ARCHITECTURES=gfx950"])
            _, missing, stale = mha_tune._classify_existing_builds(plans, meta, args)
        self.assertEqual(missing, [])
        self.assertEqual([d for _, d in stale], [["cmake_args"]] * len(plans))

    def test_stamp_diff_reports_nested_keys(self):
        self.assertEqual(mha_tune._stamp_diff({"a": 1}, {"a": 1}), [])
        self.assertEqual(
            mha_tune._stamp_diff({"a": {"b": 1, "c": 2}}, {"a": {"b": 9, "c": 2}}),
            ["a.b"],
        )
        self.assertEqual(mha_tune._stamp_diff({"a": 1}, {"b": 1}), ["a", "b"])

    # -- stamp writing ----------------------------------------------------
    def _build_one(self, plan, extra, make_rc=0, dry_run=False):
        with mock.patch.object(mha_tune, "_do_make", return_value=make_rc):
            return mha_tune._configure_and_build_one(
                plan,
                None,
                "/ck",
                extra,
                False,  # do_configure
                True,  # do_make
                self.TARGET,
                1,
                True,
                dry_run,
                True,  # buffered
            )

    def test_successful_make_writes_stamp(self):
        with tempfile.TemporaryDirectory() as td:
            args = self._args(td)
            meta, _ = self._untune(td)
            plan = self._plans(td, meta)[0]
            extra = mha_tune._extra_cmake_opts_for_pair(plan.pair, meta, args.cmake_opt)
            res = self._build_one(plan, extra)
            stamp = mha_tune._read_stamp(plan.build_dir)
        self.assertTrue(res["build_ok"])
        self.assertEqual(stamp, self._stamp_for(plan, meta, args))
        self.assertEqual(stamp["schema"], mha_tune.BUILD_STAMP_SCHEMA)
        self.assertEqual((stamp["hdim_q"], stamp["hdim_v"]), (80, 96))
        self.assertEqual(stamp["tile"], plan.tile.name)
        self.assertEqual(stamp["build_target"], self.TARGET)
        self.assertEqual(stamp["tune_config"]["filters"]["lse"], ["f"])
        for opt in (*mha_tune.DEFAULT_CMAKE_OPTIONS, *extra):
            self.assertIn(opt, stamp["cmake_args"])

    def test_failed_make_drops_old_stamp(self):
        with tempfile.TemporaryDirectory() as td:
            args = self._args(td)
            meta, _ = self._untune(td)
            plan = self._plans(td, meta)[0]
            self._pretend_built([plan], meta, args)
            res = self._build_one(plan, [], make_rc=1)
            self.assertFalse(res["build_ok"])
            self.assertIsNone(mha_tune._read_stamp(plan.build_dir))
            self.assertFalse(mha_tune._stamp_path(plan.build_dir).exists())

    def test_dry_run_does_not_touch_stamp(self):
        with tempfile.TemporaryDirectory() as td:
            args = self._args(td)
            meta, _ = self._untune(td)
            plan = self._plans(td, meta)[0]
            self._pretend_built([plan], meta, args)
            before = mha_tune._read_stamp(plan.build_dir)
            self._build_one(plan, [], dry_run=True)
            self.assertEqual(mha_tune._read_stamp(plan.build_dir), before)

    # -- cmd_bench integration -------------------------------------------
    def _run_cmd_bench(self, td, plans, meta, args):
        in_dir = Path(td) / "in"
        in_dir.mkdir(parents=True, exist_ok=True)
        args.input_csv = str(synth.write_untune_csv(in_dir, [512], meta.signature()))
        built = []
        benched = []

        def fake_build_stage(ps, a, do_configure, do_make, *, meta):
            built.extend(ps)
            for p in ps:
                self._touch_binary(p)
            return 0, list(ps)

        def fake_bench(ps, m, ms, a):
            benched.extend(ps)
            return 0

        err = io.StringIO()
        with mock.patch.object(
            mha_tune, "_load_pair_plans_from_disk", return_value=plans
        ), mock.patch.object(
            mha_tune, "_build_stage", side_effect=fake_build_stage
        ), mock.patch.object(
            mha_tune, "_bench_stage_and_dump", side_effect=fake_bench
        ), contextlib.redirect_stdout(
            io.StringIO()
        ), contextlib.redirect_stderr(
            err
        ):
            rc = mha_tune.cmd_bench(args)
        return rc, built, benched, err.getvalue()

    def test_cmd_bench_reuses_matching_builds(self):
        with tempfile.TemporaryDirectory() as td:
            args = self._args(td)
            meta, _ = self._untune(td)
            plans = self._plans(td, meta)
            self._pretend_built(plans, meta, args)
            rc, built, benched, err = self._run_cmd_bench(td, plans, meta, args)
        self.assertEqual(rc, 0)
        self.assertEqual(built, [])
        self.assertEqual(benched, plans)
        self.assertIn("reusing 2 tile binaries", err)

    def test_cmd_bench_builds_only_missing_tiles(self):
        with tempfile.TemporaryDirectory() as td:
            args = self._args(td)
            meta, _ = self._untune(td)
            plans = self._plans(td, meta)
            self._pretend_built(plans[:1], meta, args)
            rc, built, benched, err = self._run_cmd_bench(td, plans, meta, args)
        self.assertEqual(rc, 0)
        self.assertEqual(built, plans[1:])
        self.assertEqual(benched, plans)
        self.assertIn("1/2 tile binaries missing", err)

    def test_cmd_bench_rejects_stale_without_building(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            args = self._args(td)
            built_meta, _ = self._untune(td, {"has_lse": 0}, sub="built")
            self._pretend_built(self._plans(td, built_meta), built_meta, args)
            meta, _ = self._untune(td, {"has_lse": 1}, sub="now")
            plans = self._plans(td, meta)
            rc, built, benched, err = self._run_cmd_bench(td, plans, meta, args)
        self.assertEqual(rc, 2)
        self.assertEqual((built, benched), ([], []))
        self.assertIn("stale build dir", err)
        self.assertIn("tune_config.filters.lse", err)
        self.assertIn("2/2 tile build dir(s)", err)
        self.assertIn("--work-dir", err)

    def test_cmd_bench_rejects_when_only_one_tile_is_stale(self):
        with tempfile.TemporaryDirectory() as td:
            args = self._args(td)
            meta, _ = self._untune(td)
            plans = self._plans(td, meta)
            self._pretend_built(plans, meta, args)
            mha_tune._remove_stamp(plans[1].build_dir)
            rc, built, benched, err = self._run_cmd_bench(td, plans, meta, args)
        self.assertEqual(rc, 2)
        self.assertEqual((built, benched), ([], []))
        self.assertIn("1/2 tile build dir(s)", err)
        self.assertIn(str(plans[1].build_dir), err)
        self.assertNotIn(str(plans[0].build_dir), err)


def _gen_range(group_csv, range_spec, singletons=""):
    args = argparse.Namespace(
        input_csv=str(group_csv),
        output="",
        range=list(range_spec),
        singletons=singletons,
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mha_count_shape.cmd_generate_tune_range(args)
    return buf.getvalue()


def _fake_tuned(untune_path, work_dir, best_hdim=(80, 96), tiles=None, **row_over):
    """Write the tuned CSV `bench` would produce for one untune CSV.

    The signature columns are copied verbatim from the untune CSV (that is
    what C6 makes `bench` do), so the merge stage sees the real signature.
    """
    meta, max_seqlens = mha_tune.parse_untune_csv(untune_path)
    out = mha_tune.tuned_csv_path(meta, Path(work_dir))
    tiles = list(tiles or [synth.DEFAULT_TILE])
    synth.write_tuned_csv(
        out,
        {
            **{c: getattr(meta, c) for c in mha_tune.SIGNATURE_COLS},
            "bench_variant": mha_tune.bench_variant(meta),
            "best_hdim_q": best_hdim[0],
            "best_hdim_v": best_hdim[1],
        },
        [
            {
                "max_seqlen": m,
                "tile": tiles[i % len(tiles)],
                "best_tflops": 100.0 - i,
            }
            for i, m in enumerate(max_seqlens)
        ],
        **row_over,
    )
    return out, meta


class TestEndToEnd(unittest.TestCase):
    """C9: synthetic dump -> group -> tune range -> tuned CSV -> merged JSON.

    Covers the three shapes the reviewers asked about (causal, GQA and
    seqlen_q != seqlen_k) in one pass, without a GPU or a CK checkout.
    """

    # no-mask MHA at hdim 72 plus one prefix-cached call (sq != sk)
    _LINES = (
        [synth.make_dump_line(seqlens_q=[1024], hdim_q=72, hdim_v=72)] * 2
        + [synth.make_dump_line(seqlens_q=[512])]
        + [
            synth.make_dump_line(
                seqlens_q=[2048], seqlens_k=[4096]
            )  # seqlen_q != seqlen_k
        ]
        # GQA causal at hdim 256 with min_seqlen_q != 0 (skip variant)
        + [
            synth.make_dump_line(
                seqlens_q=[512],
                hdim_q=256,
                hdim_v=256,
                nhead_k=2,
                min_seqlen_q=1,
                **_CAUSAL,
            )
        ]
    )

    def _run_pipeline(self, td):
        out_dir, out = _run_cmd_group(td, self._LINES)
        self.assertIn("kept 5 / dropped 0", out)
        self.assertIn("seqlen_q != seqlen_k", out)

        untunes = []
        for gcsv in sorted(out_dir.glob("mha_group_*.csv")):
            _gen_range(gcsv, ["256:1024:256"])
            untunes.append(out_dir / gcsv.name.replace("mha_group_", "mha_untune_", 1))
        self.assertEqual(len(untunes), 2)
        return out_dir, untunes

    def test_full_chain(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir, untunes = self._run_pipeline(td)
            summary = {
                r["hdim_q"]: r
                for r in _read_csv(out_dir / mha_count_shape.SUMMARY_NAME)
            }

            # 1) grouping: causal+GQA+skip and no-mask end up in separate groups
            self.assertEqual(sorted(summary), ["256", "72"])
            self.assertEqual(summary["256"]["mask_type"], "2")
            self.assertEqual(summary["256"]["nhead_k"], "2")
            self.assertEqual(summary["256"]["skip_min_seqlen_q"], "1")
            self.assertEqual(
                summary["72"]["signature"],
                "group_bf16_hq72_hv72_mask0_nh16_nhk16_"
                "nlogits_nbias_nlse_ndropout_nskip_nqscale_nsink",
            )
            # the prefix-cached call is kept and reported, not dropped
            self.assertEqual(summary["72"]["seqlen_mismatch_calls"], "1")
            self.assertEqual(summary["72"]["num_calls"], "4")

            # 2) every untune CSV round-trips through the strict parser
            metas = {}
            for u in untunes:
                meta, ms = mha_tune.parse_untune_csv(u)
                self.assertEqual(ms, [256, 512, 768, 1024])
                self.assertEqual(meta.mode, "group")
                metas[meta.hdim_q] = meta
            self.assertEqual(mha_tune.bench_variant(metas[256]), "nskip")
            self.assertEqual(mha_tune._filters_from_meta(metas[256])["skip"], ["f"])

            # 3) "bench" output -> merged JSON, one bucket per compiled hdim
            tuned = {}
            for u in untunes:
                best = (80, 96) if "_hq72_" in u.name else (256, 256)
                path, _ = _fake_tuned(
                    u,
                    out_dir / "work",
                    best_hdim=best,
                    tiles=[synth.DEFAULT_TILE, synth.ALT_TILE],
                )
                tuned[best] = path

            payload = _merge(*tuned.values())

        self.assertEqual(payload["dtypes"], ["bf16"])
        self.assertEqual(sorted(payload["tiles"]["bf16"]), ["256,256", "80,96"])
        # hq72 is compiled for (80,96): the raw 72 never becomes a bucket key
        tiles = payload["tiles"]["bf16"]["80,96"]
        self.assertEqual(
            sorted(s for t in tiles for s in t["_max_seqlen_samples"]),
            [256, 512, 768, 1024],
        )
        self.assertEqual(payload["meta"]["sources"][0]["orig_hdim_q"], 72)
        self.assertEqual(payload["meta"]["sources"][0]["compiled_hdim_q"], 80)

    def test_full_chain_is_order_independent(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir, untunes = self._run_pipeline(td)
            paths, metas = [], {}
            for u in untunes:
                best = (80, 96) if "_hq72_" in u.name else (256, 256)
                path, meta = _fake_tuned(u, out_dir / "work", best_hdim=best)
                paths.append(path)
                metas[best] = meta
            forward = _merge(*paths)
            backward = _merge(*reversed(paths))
        for p in (forward, backward):
            p["meta"].pop("generated_at")
        self.assertEqual(forward, backward)

    def test_strict_rejects_seqlen_mismatch(self):
        with tempfile.TemporaryDirectory() as td, self.assertRaises(SystemExit) as cm:
            _run_cmd_group(td, self._LINES, strict=True)
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("seqlen_q != seqlen_k", cm.exception.stdout)

    def test_conflicting_group_is_rejected_end_to_end(self):
        """Two vLLM layers, same raw hdim, different mask -> different buckets."""
        lines = [
            synth.make_dump_line(seqlens_q=[1024], hdim_q=72, hdim_v=72),
            synth.make_dump_line(seqlens_q=[1024], hdim_q=72, hdim_v=72, **_CAUSAL),
        ]
        with tempfile.TemporaryDirectory() as td:
            out_dir, _ = _run_cmd_group(td, lines)
            paths = []
            for gcsv in sorted(out_dir.glob("mha_group_*.csv")):
                _gen_range(gcsv, ["512:1024:512"])
                u = out_dir / gcsv.name.replace("mha_group_", "mha_untune_", 1)
                paths.append(_fake_tuned(u, out_dir / "work")[0])
            # both compile to (80,96) but differ in mask_type: refuse to merge
            self.assertEqual(len({p.name for p in paths}), 2)
            with self.assertRaises(ValueError) as cm:
                _merge(*paths)
        self.assertIn("differing in: mask_type", str(cm.exception))
        self.assertIn("incompatible tuned csvs", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
