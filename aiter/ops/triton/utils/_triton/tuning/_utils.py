"""Helpers shared by sweep_configs.py, write_best_configs.py and verify_configs.py.

These run on the host side only; everything that touches the GPU is in
``_worker.py``, which the scripts start under ``rocprofv3``.
"""

import inspect
import itertools
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from gemm_cases import CASES
from parse_kernel_trace import segment_times

TUNING_DIR = Path(__file__).resolve().parent
# aiter/ops/triton/configs, next to utils/ in the checkout these scripts sit in.
CONFIGS_ROOT = TUNING_DIR.parents[2] / "configs"
SWEEPS_DIR = Path("sweeps")
TUNED_DIR = Path("tuned_configs")
N_RUN = 250

BLOCK_KEY = re.compile(r"BLOCK_(SIZE_)?[MNK]")


def parse_dims(tokens):
    """``["N=7168", "K=2048"]`` -> ``{"N": 7168, "K": 2048}``."""
    dims = {}
    for token in tokens:
        name, _, value = token.partition("=")
        if not name or not value.isdigit():
            sys.exit(f"shape dims are NAME=INT, got {token!r}")
        dims[name] = int(value)
    return dims


def case_dims(op):
    """The shape dims a case takes besides M, in signature order."""
    params = list(inspect.signature(CASES[op].make_fn).parameters)
    return [p for p in params[1:] if p != "backend"]


def get_case(op, dims, backend):
    if op not in CASES:
        sys.exit(f"no tuning case {op!r}; cases: {', '.join(sorted(CASES))}")
    names = case_dims(op)
    if set(dims) != set(names):
        sys.exit(f"{op} takes M plus {' '.join(f'{n}=<int>' for n in names)}")
    if backend and "backend" not in inspect.signature(CASES[op].make_fn).parameters:
        sys.exit(f"{op} has a single backend; drop --backend")
    return CASES[op]


def shape_tag(op, dims):
    return "-".join(f"{name}={dims[name]}" for name in case_dims(op))


def sweep_dir(op, arch, backend, dims):
    return SWEEPS_DIR / op / f"{arch}-{backend}" / shape_tag(op, dims)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, entry):
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def tuned_filename(lookup):
    """The specialized file ``get_gemm_config()`` tries first for this lookup."""
    name, N, K, B = (lookup[k] for k in ("config_name", "N", "K", "B"))
    if lookup["specialized_filename"] is not None:
        return f"{name}-{lookup['specialized_filename']}.json"
    if N is None or K is None:
        return None
    if B is not None:
        return f"{name}-B={B}-N={N}-K={K}.json"
    return f"{name}-N={N}-K={K}.json"


# ---------------------------------------------------------------------------
# Search space


def builtin_values(key, M, N, K):
    """What the sweep tries for the standard Triton GEMM knobs, or None for a key
    it has no range for. Tile sizes stop at the problem size."""
    if key == "BLOCK_SIZE_M":
        return [4, 8] + [v for v in (16, 32, 64, 128, 256, 512) if v <= M]
    if key == "BLOCK_SIZE_N":
        return [16] + [v for v in (32, 64, 128, 256) if N is None or v <= N]
    if key == "BLOCK_SIZE_K":
        return [128] + [v for v in (256, 512, 1024) if K is None or v <= K]
    if key == "NUM_KSPLIT":
        return [1] + [v for v in (3, 4, 7, 8, 14, 16, 28) if K and K % v == 0]
    return {
        "GROUP_SIZE_M": [1, 4, 8],
        "num_warps": [1, 4, 8],
        "num_stages": [1, 2],
        "waves_per_eu": [1, 2, 4, 6, 8],
        "matrix_instr_nonkdim": [16],
        "cache_modifier": [".cg", None],
        "kpack": [1, 2],
    }.get(key)


def build_space(family, M, N, K, *overrides):
    """One value list per key of the family's DEFAULT.json. Keys without a
    built-in range take the values the family's files already use. Each
    override (the case's, then the command line's) replaces a key's list.
    Returns the space and the override keys the family does not have."""
    space = {}
    for key in family["keys"]:
        # configs/CLAUDE.md: kpack is deprecated off gfx942 and dropped on retune.
        if key == "kpack" and family["arch"] != "gfx942":
            continue
        values = builtin_values(key, M, N, K)
        if values is None:
            values = family["observed"].get(key, [])
            if all(type(v) in (int, float) for v in values):
                values = sorted(values)
        space[key] = values
    unknown = []
    for override in overrides:
        for key, values in override.items():
            if key in space:
                space[key] = list(values)
            else:
                unknown.append(key)
    return space, unknown


def prune_reason(config, K):
    """Why a config cannot win, for the split-K rules of the Triton GEMMs; None
    when no rule applies. Keys a config lacks simply skip their rule."""
    ksplit = config.get("NUM_KSPLIT", 1)
    if ksplit > 1 and config.get("GROUP_SIZE_M", 1) > 1:
        return "NUM_KSPLIT > 1 and GROUP_SIZE_M > 1"
    bk = config.get("BLOCK_SIZE_K")
    if bk is None or not K:
        return None
    k_split = K // ksplit
    if bk >= 2 * k_split:
        return "BLOCK_SIZE_K >= 2 * (K // NUM_KSPLIT)"
    stages = config.get("num_stages")
    if stages is not None and bk == k_split and stages > 1:
        return "BLOCK_SIZE_K == K // NUM_KSPLIT and num_stages > 1"
    if stages is not None and bk < k_split and stages == 1:
        return "BLOCK_SIZE_K < K // NUM_KSPLIT and num_stages == 1"
    return None


def block_keys(keys):
    return [key for key in keys if BLOCK_KEY.fullmatch(key)]


def candidates(space, K, verbose=False):
    """Every config in the space that no rule prunes, grouped by tile sizes."""
    blocks = block_keys(space)
    keys = blocks + [key for key in space if key not in blocks]
    configs, pruned = [], 0
    for values in itertools.product(*(space[key] for key in keys)):
        config = dict(zip(keys, values))
        reason = prune_reason(config, K)
        if reason is None:
            configs.append(config)
            continue
        pruned += 1
        if verbose:
            print(f"Remove case {config} because {reason}")
    return configs, pruned


def parse_space(tokens):
    """``["BLOCK_SIZE_K=128,256", "cache_modifier=.cg,null"]`` -> value lists.
    Values are read as JSON (numbers, null, true, quoted strings), anything else
    as a plain string."""
    space = {}
    for token in tokens:
        key, sep, values = token.partition("=")
        if not sep or not key or not values:
            sys.exit(f"--space takes KEY=V1,V2,..., got {token!r}")
        space[key] = [_json_or_str(v) for v in values.split(",")]
    return space


def _json_or_str(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


# ---------------------------------------------------------------------------
# Running the worker


@dataclass
class WorkerRun:
    returncode: int | None  # None: killed at the timeout
    stderr: str
    record: dict | None  # None: the worker died before running the op
    statuses: list[dict]
    segments: list[tuple[list[str], float | None]] | None  # None: no trace


def run_worker(spec, gpu, timeout, keywords):
    """Run one worker process under ``rocprofv3`` on GPU ``gpu``."""
    with tempfile.TemporaryDirectory(prefix="tune-") as tmp:
        spec = dict(
            spec,
            n_run=N_RUN,
            record=os.path.join(tmp, "record.json"),
            status=os.path.join(tmp, "status.jsonl"),
        )
        spec_path = os.path.join(tmp, "spec.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)
        trace = f"trace-{spec['op']}-{os.getpid()}"
        trace_csv = Path(f"{trace}_kernel_trace.csv")
        trace_csv.unlink(missing_ok=True)
        cmd = ["rocprofv3", "--kernel-trace", "-f", "csv", "-o", trace, "--"]
        cmd += [sys.executable, str(TUNING_DIR / "_worker.py"), spec_path]
        env = dict(os.environ, HIP_VISIBLE_DEVICES=str(gpu))
        try:
            # Own session, so a timeout kills the worker too, not just rocprofv3.
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except FileNotFoundError:
            sys.exit("rocprofv3 is not on PATH")
        try:
            _, stderr = proc.communicate(timeout=timeout)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            returncode, stderr = None, f"timed out after {timeout}s"

        record = None
        if os.path.isfile(spec["record"]):
            with open(spec["record"]) as f:
                record = json.load(f)
        statuses = read_jsonl(spec["status"]) if os.path.isfile(spec["status"]) else []
        segments = None
        if trace_csv.is_file():
            segments = segment_times(trace_csv, keywords)
            trace_csv.unlink()
    return WorkerRun(returncode, stderr, record, statuses, segments)


def resolve_family(run, op):
    """The single config lookup the op made, and its family, from a run of the
    op with the config the library resolves. Exits with the reason when the op
    cannot be tuned this way. The op failing with the resolved config is left to
    the caller (``run.record["error"]``): a sweep can still find one that works."""
    record = run.record
    if record is None:
        tail = "\n".join(run.stderr.strip().splitlines()[-20:])
        sys.exit(f"{op}: the worker failed before running the op:\n{tail}")
    lookups = record["lookups"]
    if not lookups:
        sys.exit(f"{op}: {record['error']}")
    if len(lookups) > 1:
        names = ", ".join(sorted({lookup["config_name"] for lookup in lookups}))
        sys.exit(
            f"{op} makes {len(lookups)} different config lookups ({names}); "
            "tune the ops that own those configs instead"
        )
    family = record["family"]
    if not family["default_exists"]:
        sys.exit(
            f"{op}: no DEFAULT.json in {family['config_dir']}. Add one first "
            "(configs/CLAUDE.md, section 6): its keys are the knobs the sweep tunes."
        )
    return lookups[0], family


def describe_lookup(lookup):
    parts = [f"{k}={lookup[k]}" for k in ("B", "N", "K") if lookup[k] is not None]
    if lookup["specialized_filename"] is not None:
        parts.append(lookup["specialized_filename"])
    return f"{lookup['config_name']} ({lookup['backend']}; {', '.join(parts)})"
