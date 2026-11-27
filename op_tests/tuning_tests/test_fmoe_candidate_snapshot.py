# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Snapshot of the candidates FmoeTuner.tune() submits, per shape and arch.

tune() runs for real, but mp_tuner is replaced, so no candidate is built or
timed. The tasks it would have timed are compared, in order, with a snapshot
taken on the base commit. Each task is recorded whole except its shape: the
tag, the data, kernel and reference functions by qualified name, their
arguments, the tolerances and the compare function. Tensors are recorded by
shape, dtype and device, not by contents.

Functions that move to another module change their qualified name. MOVED maps
each old name in the snapshot to its new one; the snapshot is rewritten with
it before the comparison, so only the location of a function may differ.

The snapshot is not committed: it is more than 1.3 MB per arch and goes stale
whenever main changes a candidate. AITER_FMOE_CANDIDATE_SNAPSHOT names its
path, with no default. Take it on the base commit, the commit on the
refactor branch that adds this test (main plus this safety net), then run
the test on the branch, on the same node:
    export AITER_FMOE_CANDIDATE_SNAPSHOT=<dir>/fmoe_snapshot_<base>.json
    git checkout <base>
    python3 op_tests/tuning_tests/test_fmoe_candidate_snapshot.py --regenerate
    git checkout <branch>
    python3 op_tests/tuning_tests/test_fmoe_candidate_snapshot.py -v

The snapshot records the commit it was taken at. --regenerate refuses to run
with uncommitted changes under aiter/ or csrc/, and to replace an arch entry
or a snapshot of another commit unless given --force. The test fails when the
snapshot's commit is not an ancestor of HEAD, and when gemm_moe_tune.py at
that commit is not the base blob. Regenerating the snapshot after that file
changes does not make the test pass.

Only the arch of the GPU in the machine is collected. GPU_ARCHS cannot stand
in for another arch: dtypes.fp8 and many other checks read the real GPU
through get_gfx_runtime(), so a gfx942 run on gfx950 mixes both. The second
arch is added to the same file on its own node, at the same commit.
"""

import contextlib
import enum
import functools
import importlib.util
import inspect
import io
import json
import numbers
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

SNAPSHOT_ENV = "AITER_FMOE_CANDIDATE_SNAPSHOT"
_SOURCE_DIRS = ("aiter", "csrc")
# Blob of gemm_moe_tune.py at 04500613c6, the last main commit that changed it.
# Update it whenever that file changes, and take the snapshot again.
_TUNER_PATH = "csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py"
_BASE_TUNER_BLOB = "c57fc28b98a10274157c3a2803a037c4769e6359"
# Version 1 kept only the tags; a snapshot in it has to be taken again.
_FORMAT = 2


def _moved(old_module, new_module, *names):
    """{old_module.name: new_module.<name without its class>} for each name."""
    return {f"{old_module}.{n}": f"{new_module}.{n.rsplit('.', 1)[-1]}" for n in names}


_TUNER = "csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune"
_MOE = "aiter.tuning.moe"

# Task functions that moved since the snapshot was taken: old qualified name
# -> new one.
MOVED = {
    **_moved(
        _TUNER,
        f"{_MOE}.backends.asm_1stage",
        "FmoeTuner.run_1stage_fmoe_fp8_blockscale_g1u1",
        "FmoeTuner.run_1stage_fmoe_g1u1",
        "FmoeTuner.run_1stage_fmoe_g1u1_tkw1",
    ),
    **_moved(_TUNER, f"{_MOE}.backends.asm_stage1", "FmoeTuner.run_asm_stage1"),
    **_moved(
        _TUNER,
        f"{_MOE}.backends.ck",
        "FmoeTuner.ck_moe_stage1_fwd_out",
        "FmoeTuner.ck_moe_stage2_fwd_out",
    ),
    **_moved(
        _TUNER,
        f"{_MOE}.backends.cktile",
        "FmoeTuner.cktile_moe_stage1_out",
        "FmoeTuner.cktile_moe_stage2_out",
    ),
    **_moved(
        _TUNER,
        f"{_MOE}.backends.flydsl",
        "FmoeTuner.run_a16w_stage1_sorted_ref",
        "FmoeTuner.run_flydsl_stage1_out",
        "FmoeTuner.run_flydsl_stage2_out",
    ),
    **_moved(
        _TUNER,
        f"{_MOE}.backends.flydsl_v2",
        "FmoeTuner.generate_v2_stage1_data",
        "FmoeTuner.generate_v2_stage2_data",
        "FmoeTuner.run_flydsl_v2_stage1_out",
        "FmoeTuner.run_flydsl_v2_stage2_out",
        "FmoeTuner.run_v2_stage1_sorted_ref",
        "FmoeTuner.run_v2_stage2_ref",
    ),
    **_moved(
        _TUNER,
        f"{_MOE}.backends.opus",
        "FmoeTuner.run_opus_stage1_out",
        "FmoeTuner.run_opus_stage2_out",
    ),
    **_moved(
        _TUNER,
        f"{_MOE}.compare",
        "FmoeTuner.run_torch_moe_stage1",
        "FmoeTuner.run_torch_moe_stage1_ref",
        "FmoeTuner.run_torch_moe_stage2",
        "_a16w_sorted_cos",
        "sorted_stage1_cosine_diff_compare",
    ),
    **_moved(
        _TUNER,
        f"{_MOE}.inputs",
        "FmoeTuner.generate_asm_stage1",
        "FmoeTuner.generate_data_1stage",
        "FmoeTuner.generate_data_2stages",
        "FmoeTuner.generate_opus_sorted_stage2_data",
    ),
    **_moved(
        "csrc.ck_gemm_moe_2stages_codegen.mxfp4_v2_tune_utils",
        f"{_MOE}.mxfp4_v2_tune_utils",
        "v2_stage1_dequant_cosine_err",
    ),
    **_moved(
        _TUNER,
        "aiter.fused_moe",
        "FmoeTuner.torch_moe_2stages",
        "FmoeTuner.torch_moe_blockscale",
    ),
    **_moved(_TUNER, "aiter.test_common", "cosine_diff_compare"),
}


def snapshot_path():
    value = os.environ.get(SNAPSHOT_ENV)
    return Path(value) if value else None


def _git(*args):
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )


def head_commit():
    return _git("rev-parse", "HEAD").stdout.strip()


def source_changes():
    """Tracked changes under aiter/ and csrc/; JIT build outputs are untracked."""
    status = _git("status", "--porcelain", "--untracked-files=no", "--", *_SOURCE_DIRS)
    return status.stdout.strip()


# Same order as the key list in aiter.tuning.moe.fused_moe.main().
KEYS = [
    "gfx",
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
]

CU_NUM = {"gfx950": 256, "gfx942": 304}
# Removed from the child so the arch and CU count come from the real GPU.
_ARCH_ENV = ("GPU_ARCHS", "CU_NUM")

# Unset runs FlyDSL v2 alone when it has candidates; empty runs every backend.
TUNE_ONLY_MODES = {"default": None, "all": ""}
_TUNER_ENV = (
    "TUNE_ONLY",
    "OPUS_ONLY",
    "OPUS_SKIP_CKTILE",
    "TUNE_MOE_KERNEL_REGEX",
    "AITER_FMOE_TUNE_EXCLUDE_KERNELS",
)

GENERATORS = {
    "asm_1stage": "gen_1stage_asm_task",
    "asm_stage1": "gen_2stages_asm1_task",
    "ck": "gen_2stages_task",
    "flydsl": "gen_flydsl_2stages_task",
    "flydslv2": "gen_flydsl_v2_2stages_task",
    "flydsli4": "gen_flydsl_i4_2stages_task",
    "opus": "gen_opus_2stages_task",
}
BACKENDS = (*GENERATORS, "cktile")


def _shape(gfx, token, dims, act, q_type, q_dtype_a, q_dtype_w, doweight=0):
    model_dim, inter_dim, expert, topk = dims
    return (
        gfx,
        CU_NUM[gfx],
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        f"ActivationType.{act}",
        "torch.bfloat16",
        f"torch.{q_dtype_a}",
        f"torch.{q_dtype_w}",
        f"QuantType.{q_type}",
        1,
        doweight,
    )


BF16 = "bfloat16"
FP8 = "float8_e4m3fn"
FNUZ = "float8_e4m3fnuz"
FP4 = "float4_e2m1fn_x2"

# One shipped row per dtype / quant / activation group of the tuned fmoe CSVs,
# Coverage is checked after collection; these rows are not exhaustive.
# The opus a8w4 stage2 list is filtered by token, hence the token 1 and 2048 pair.
SHAPES = [
    # gfx950
    _shape("gfx950", 1, (6144, 512, 192, 8), "Silu", "No", BF16, BF16),
    _shape("gfx950", 1, (5120, 2560, 32, 8), "Silu", "per_Token", FP8, FP8),
    _shape("gfx950", 1, (5120, 1024, 128, 1), "Silu", "per_Token", FP8, FP8, 1),
    _shape("gfx950", 4096, (4096, 384, 128, 8), "Silu", "per_1x128", FP8, FP8),
    _shape("gfx950", 1, (2048, 512, 512, 10), "Gelu", "per_1x128", FP8, FP8),
    _shape("gfx950", 1, (7168, 256, 257, 9), "Silu", "per_1x32", FP4, FP4),
    _shape("gfx950", 256, (3072, 3072, 128, 4), "Swiglu", "per_1x32", FP4, FP4),
    _shape("gfx950", 1, (5120, 1152, 385, 7), "Silu", "per_1x32", FP8, FP4),
    _shape("gfx950", 2048, (5120, 1152, 385, 7), "Silu", "per_1x32", FP8, FP4),
    _shape("gfx950", 512, (3072, 3072, 128, 4), "Swiglu", "per_1x32", FP8, FP4),
    _shape("gfx950", 1, (3584, 512, 896, 16), "Situv2", "per_1x32", FP8, FP4),
    _shape("gfx950", 1, (6144, 512, 257, 9), "Silu", "per_1x32", FP8, FP8),
    _shape("gfx950", 1, (3072, 3072, 128, 4), "Swiglu", "per_1x32", BF16, FP4),
    _shape("gfx950", 1, (7168, 256, 384, 8), "Silu", "per_1x32", BF16, "int4"),
    # gfx942
    _shape("gfx942", 1, (4096, 384, 128, 8), "Silu", "No", BF16, BF16),
    _shape("gfx942", 2048, (2048, 192, 128, 8), "Gelu", "No", BF16, BF16, 1),
    _shape("gfx942", 1, (4096, 192, 128, 8), "Silu", "per_Token", FNUZ, FNUZ),
    _shape("gfx942", 1, (6144, 384, 128, 4), "Swiglu", "per_Token", FNUZ, FNUZ),
    _shape("gfx942", 1, (7168, 2048, 33, 9), "Silu", "per_1x128", FNUZ, FNUZ),
    _shape("gfx942", 2048, (4096, 384, 128, 8), "Silu", "per_Tensor", FNUZ, FNUZ),
    _shape("gfx942", 512, (6144, 4096, 8, 2), "Silu", "per_Tensor", "int8", "int8"),
    # tune() returns [] for per_1x32 outside gfx950.
    _shape("gfx942", 1, (3584, 384, 896, 16), "Situv2", "per_1x32", BF16, FP4),
]


def _shape_id(shape):
    return ",".join(str(v) for v in shape)


def _function_name(fn):
    """module.qualname of a task function.

    A guarded aiter op gives the op it wraps: torch_compile_guard names every
    wrapper wrapper_custom.
    """
    fn = inspect.unwrap(fn)
    qualname = getattr(fn, "__qualname__", None)
    if qualname is None:
        raise TypeError(f"no qualified name for {type(fn).__qualname__}: {fn!r:.80}")
    return f"{fn.__module__}.{qualname}"


def _record_value(value):
    """JSON form of a task element."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, enum.Enum):
        return f"{type(value).__qualname__}.{value.name}"
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        return {
            "tensor": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, (list, tuple)):
        return [_record_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _record_value(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, functools.partial):
        return {
            "partial": _function_name(value.func),
            "args": _record_value(value.args),
            "keywords": _record_value(value.keywords),
        }
    if callable(value):
        return {"function": _function_name(value)}
    text = str(value)
    # A default repr names the memory address, which differs between runs.
    if " at 0x" in text or (text.startswith("<") and text.endswith(">")):
        raise TypeError(f"no stable record for {type(value).__qualname__}: {text:.80}")
    return text


def _record(task):
    """Everything mp_tuner reads from a task, except the shape in task[0][0]."""
    _info, *tag = task[0]
    return [_record_value(tag), *(_record_value(v) for v in task[1:])]


def _is_function(value):
    return isinstance(value, dict) and set(value) in (
        {"function"},
        {"partial", "args", "keywords"},
    )


def _renamed(value, moved):
    """value with every recorded function name looked up in moved."""
    if isinstance(value, list):
        return [_renamed(v, moved) for v in value]
    if not isinstance(value, dict):
        return value
    out = {k: _renamed(v, moved) for k, v in value.items()}
    if _is_function(value):
        key = "function" if "function" in value else "partial"
        out[key] = moved.get(value[key], value[key])
    return out


def _function_names(value):
    if isinstance(value, list):
        for v in value:
            yield from _function_names(v)
    elif isinstance(value, dict):
        if _is_function(value):
            yield value["function" if "function" in value else "partial"]
        for v in value.values():
            yield from _function_names(v)


def collect():
    """Return the GPU arch, and ordered task tags and per-backend counts for
    every shape of that arch."""
    import pandas as pd

    from aiter.jit.utils.chip_info import get_gfx, get_gfx_runtime
    from aiter.tuning.moe import fused_moe
    from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

    gfx = get_gfx_runtime()
    if gfx not in CU_NUM:
        raise RuntimeError(f"no snapshot shapes for {gfx}; known: {list(CU_NUM)}")
    if get_gfx() != gfx:
        raise RuntimeError(f"get_gfx() is {get_gfx()} but the GPU is {gfx}")

    tuner = gemm_moe_tune.FmoeTuner("snapshotFmoe", KEYS, [], "candidate snapshot")
    with mock.patch.object(sys, "argv", ["gemm_moe_tune.py"]):
        args = tuner.parse_args()

    captured = []

    def fake_mp_tuner(tasks, *_args, **_kwargs):
        captured.extend(_record(task) for task in tasks)
        return []

    base_env = {k: v for k, v in os.environ.items() if k not in _TUNER_ENV}
    snapshot = {}
    coverage = {}
    counts = {}

    def observe(backend, generate):
        def wrapped(*args, **kwargs):
            tasks = generate(*args, **kwargs)
            if backend == "ck":
                # gen_2stages_task delegates a8w4 to CK-tile. Attribute tasks
                # by their runner, rather than counting CK-tile as CK coverage.
                for task in tasks:
                    family = (
                        "cktile" if task[3].__name__.startswith("cktile_") else "ck"
                    )
                    counts[family] += 1
            else:
                counts[backend] += len(tasks)
            return tasks

        return wrapped

    for shape in SHAPES:
        if shape[0] != gfx:
            continue
        untunedf = pd.DataFrame([shape], columns=KEYS)
        for mode, tune_only in TUNE_ONLY_MODES.items():
            env = dict(base_env)
            if tune_only is not None:
                env["TUNE_ONLY"] = tune_only
            captured.clear()
            counts = dict.fromkeys(BACKENDS, 0)
            with (
                mock.patch.dict(os.environ, env, clear=True),
                mock.patch.object(fused_moe, "mp_tuner", fake_mp_tuner),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.ExitStack() as patches,
            ):
                if mode == "all":
                    for backend, method in GENERATORS.items():
                        patches.enter_context(
                            mock.patch.object(
                                tuner, method, observe(backend, getattr(tuner, method))
                            )
                        )
                tuner.tune(untunedf, pd.DataFrame(columns=KEYS), args)
            snapshot[f"{_shape_id(shape)} TUNE_ONLY={mode}"] = list(captured)
            if mode == "all":
                coverage[_shape_id(shape)] = dict(counts)
    return {"gfx": gfx, "candidates": snapshot, "coverage": coverage}


def collect_in_subprocess():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "collected.json")
        # Exclusions are cached at module import, so clean the child environment
        # before importing the tuner, not only around tune().
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in _TUNER_ENV and k not in _ARCH_ENV
        }
        result = subprocess.run(
            [sys.executable, __file__, "--collect", out],
            env=env,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"collecting candidates exited {result.returncode}:\n"
                f"{result.stderr[-4000:]}"
            )
        with open(out) as f:
            data = json.load(f)
    return data.pop("gfx"), data


def coverage_gaps(snapshot):
    """Backends without an emitting or an empty case across the snapshot's archs.

    An empty case may be rejected by tune() before reaching a generator.
    This is a minimum coverage check, not proof of every internal branch.
    """
    rows = [
        counts for arch in snapshot.values() for counts in arch["coverage"].values()
    ]
    gaps = []
    for backend in BACKENDS:
        if not any(row[backend] > 0 for row in rows):
            gaps.append(f"{backend}: no case emits candidates")
        if not any(row[backend] == 0 for row in rows):
            gaps.append(f"{backend}: no empty case")
    return gaps


def _diff(expected, actual):
    exp = Counter(json.dumps(t, sort_keys=True) for t in expected)
    act = Counter(json.dumps(t, sort_keys=True) for t in actual)
    missing = sorted((exp - act).elements())
    extra = sorted((act - exp).elements())
    if not missing and not extra:
        return "same candidates in a different order"
    return (
        f"{len(missing)} missing, {len(extra)} new\n"
        f"missing: {missing[:5]}\nnew: {extra[:5]}"
    )


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None, "gemm_moe_tune.py needs torch"
)
class TestFmoeCandidateSnapshot(unittest.TestCase):
    def _baseline(self):
        path = snapshot_path()
        if path is None:
            self.skipTest(f"{SNAPSHOT_ENV} not set")
        if not path.exists():
            self.skipTest(f"{path} missing; run --regenerate on the base commit")
        snapshot = json.loads(path.read_text())
        self.assertIn("base", snapshot, f"{path} records no base commit")
        self.assertEqual(
            snapshot.get("format", 1),
            _FORMAT,
            f"{path} has another record format; retake it on the base commit",
        )
        base, head = snapshot["base"], head_commit()
        sys.stderr.write(f"\n{path}: taken at {base}, HEAD is {head}\n")
        self.assertEqual(
            _git("merge-base", "--is-ancestor", base, head).returncode,
            0,
            f"snapshot commit {base} is not an ancestor of HEAD {head}",
        )
        blob = _git("rev-parse", f"{base}:{_TUNER_PATH}")
        self.assertEqual(
            blob.returncode,
            0,
            f"snapshot commit {base} has no {_TUNER_PATH}; retake the snapshot "
            "on the base commit",
        )
        self.assertEqual(
            blob.stdout.strip(),
            _BASE_TUNER_BLOB,
            "snapshot was taken after gemm_moe_tune.py changed; retake it on "
            f"the base commit (blob {_BASE_TUNER_BLOB})",
        )
        return snapshot["archs"]

    def test_snapshot_covers_every_backend(self):
        baseline = self._baseline()
        # Every backend has gfx950 kernels; gfx942 alone lacks the per_1x32 ones.
        if "gfx950" not in baseline:
            self.skipTest("backend coverage needs the gfx950 entry")
        gaps = coverage_gaps(baseline)
        self.assertFalse(gaps, "Incomplete backend coverage:\n" + "\n".join(gaps))

    def test_candidates_match_snapshot_on_this_gpu(self):
        baseline = self._baseline()
        gfx, collected = collect_in_subprocess()
        if gfx not in baseline:
            self.skipTest(f"no {gfx} entry; run --regenerate on a {gfx} node")
        self.assertEqual(collected["coverage"], baseline[gfx]["coverage"])
        expected = _renamed(baseline[gfx]["candidates"], MOVED)
        actual = collected["candidates"]
        self.assertEqual(sorted(actual), sorted(expected), "snapshot keys changed")
        before = set(_function_names(list(expected.values())))
        after = set(_function_names(list(actual.values())))
        self.assertEqual(
            before,
            after,
            "task functions changed; declare a moved function in MOVED\n"
            f"only in the snapshot: {sorted(before - after)}\n"
            f"only on HEAD: {sorted(after - before)}",
        )
        for key, records in expected.items():
            with self.subTest(shape=key):
                self.assertEqual(actual[key], records, _diff(records, actual[key]))


def _regenerate(force):
    path = snapshot_path()
    if path is None:
        sys.exit(f"set {SNAPSHOT_ENV} to the snapshot path")
    changes = source_changes()
    if changes:
        sys.exit(f"uncommitted changes under aiter/ or csrc/:\n{changes}")
    base = head_commit()
    snapshot = {"format": _FORMAT, "base": base, "archs": {}}
    if path.exists():
        old = json.loads(path.read_text())
        if (old.get("base"), old.get("format", 1)) == (base, _FORMAT):
            snapshot = old
        elif not force:
            sys.exit(
                f"{path} was taken at {old.get('base')} in format "
                f"{old.get('format', 1)}, HEAD is {base} with format {_FORMAT}; "
                "use a new path, or --force to replace it"
            )
    gfx, data = collect_in_subprocess()
    if gfx in snapshot["archs"] and not force:
        sys.exit(f"{path} already has {gfx} at {base}; --force replaces it")
    snapshot["archs"][gfx] = data
    for gap in coverage_gaps(snapshot["archs"]):
        print(f"coverage gap: {gap}")
    snapshot["archs"] = {
        a: snapshot["archs"][a] for a in CU_NUM if a in snapshot["archs"]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=1)
        f.write("\n")
    print(f"wrote {gfx} at {base} to {path}")
    print(gfx, {k: len(v) for k, v in data["candidates"].items()})
    for backend in BACKENDS:
        emitting = sum(row[backend] > 0 for row in data["coverage"].values())
        empty = len(data["coverage"]) - emitting
        print(f"  {backend}: {emitting} emitting, {empty} empty")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--collect"]:
        with open(sys.argv[2], "w") as f:
            json.dump(collect(), f)
    elif sys.argv[1:2] == ["--regenerate"] and sys.argv[2:] in ([], ["--force"]):
        _regenerate(force=sys.argv[2:] == ["--force"])
    else:
        unittest.main()
