"""Tune one GEMM at one shape and keep the fastest config.

    HIP_VISIBLE_DEVICES=0 python3 tune_gemm.py <op> M=<int> N=<int> K=<int> [B=<int>]

<op> is a case in gemm_cases.py, named after the GEMM wrapper it runs. What
happens:

1. The op runs with the config the library resolves today: the tuned file for
   this shape if there is one, else DEFAULT.json. Its kernel time is the time
   to beat.
2. Every candidate config runs, with the op's own get_gemm_config() lookup
   handing back the candidate instead of reading JSON. A candidate that raises,
   or whose output differs from step 1's, goes to errors-<op>-<shape>.txt and
   is skipped. The keys tried are the keys of the family's DEFAULT.json for
   this GPU's arch and the backend the op runs, so the same command tunes
   every arch and both Triton and Gluon kernels.
3. The best candidate and the current config are timed again back to back. If
   the candidate wins, it is written into the file get_gemm_config() reads for
   this shape, in the M bucket for this M. A shape with no tuned file gets one,
   made from DEFAULT.json plus the winner.

The op runs in a separate process under rocprofv3 (profile_configs.py), in
batches, and only the GEMM kernels' time counts. A batch that crashes or hangs
costs just the config it was on; the rest of the batch runs again in a new
process.
"""

import argparse
import collections
import csv
import glob
import inspect
import itertools
import json
import os
import re
import signal
import subprocess
import sys
import tempfile

from gemm_cases import CASES

from aiter.ops.triton.utils import config_utils, gemm_config_utils
from aiter.ops.triton.utils._triton import arch_info

# A candidate replaces the current config only when it is at least this much
# faster; smaller differences are run-to-run noise.
MIN_GAIN = 0.03
# Configs per rocprofv3 process.
BATCH = 100
# Only kernels whose name contains this are timed (the op's kernels, not the
# cache flush or the output zeroing around them).
KERNEL_NAME = "gemm"
RESOURCE_ERRORS = ("OutOfResources", "exceeds triton maximum tensor numel")
TILE_KEY = re.compile(r"BLOCK_(SIZE_)?[MNK]")
HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS_ROOT = config_utils.AITER_TRITON_CONFIGS_PATH


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("op", help="case in gemm_cases.py, e.g. gemm_a8w8")
    parser.add_argument("dims", nargs="+", metavar="DIM=INT", help="M=16 N=1024 K=1024")
    parser.add_argument(
        "--backend",
        choices=("triton", "gluon"),
        help="for ops that have both; default: whatever the wrapper picks",
    )
    parser.add_argument(
        "--space",
        nargs="+",
        default=[],
        metavar="KEY=V1,V2",
        help="values to try for a key instead of the defaults, e.g. BLOCK_SIZE_K=128,256",
    )
    parser.add_argument(
        "--timeout", type=int, default=900, help="seconds per batch (default 900)"
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="do not compare each candidate's output with the current config's",
    )
    return parser.parse_args()


def parse_dims(tokens):
    """["M=16", "N=1024"] -> {"M": 16, "N": 1024}"""
    dims = {}
    for token in tokens:
        name, _, value = token.partition("=")
        if not name or not value.isdigit():
            sys.exit(f"dims are NAME=INT, got {token!r}")
        dims[name] = int(value)
    if "M" not in dims:
        sys.exit("M is required")
    return dims


def parse_space(tokens):
    """["BLOCK_SIZE_K=128,256", "cache_modifier=.cg,null"] -> {key: [values]}
    Values are read as JSON (numbers, null, true); anything else is a string."""
    space = {}
    for token in tokens:
        key, _, values = token.partition("=")
        if not key or not values:
            sys.exit(f"--space takes KEY=V1,V2,..., got {token!r}")
        space[key] = [json_or_str(v) for v in values.split(",")]
    return space


def json_or_str(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def get_case(op, dims, backend):
    if op not in CASES:
        sys.exit(f"no case {op!r} in gemm_cases.py; cases: {', '.join(CASES)}")
    case = CASES[op]
    params = list(inspect.signature(case).parameters)
    dim_names = [p for p in params if p != "backend"]
    if sorted(dims) != sorted(dim_names):
        sys.exit(f"{op} takes {' '.join(f'{p}=<int>' for p in dim_names)}")
    if backend and "backend" not in params:
        sys.exit(f"{op} has one backend; drop --backend")
    return case


# ---------------------------------------------------------------------------
# Running the op under rocprofv3

Child = collections.namedtuple("Child", "statuses times record ended stderr")


def run_child(spec, candidates, run_current, timeout):
    """Run profile_configs.py under rocprofv3 with these candidate configs.

    statuses: "ready", then one line per config (the current config first).
      A config without a line is the one the process died on.
    times: median GEMM kernel time in us per config, in the same order.
    record: the op's config lookup and the config it resolved.
    ended: how the process ended when it did not exit normally, else None.
    """
    with tempfile.TemporaryDirectory() as tmp:
        spec = dict(
            spec,
            candidates=candidates,
            run_current=run_current,
            status=os.path.join(tmp, "status.txt"),
            record=os.path.join(tmp, "record.json"),
        )
        spec_path = os.path.join(tmp, "spec.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)
        trace = f"tune-{spec['op']}-{os.getpid()}"
        cmd = ["rocprofv3", "--kernel-trace", "-f", "csv", "-o", trace, "--"]
        cmd += [sys.executable, os.path.join(HERE, "profile_configs.py"), spec_path]
        try:
            # Its own process group, so a timeout kills the op too, not just rocprofv3.
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except FileNotFoundError:
            sys.exit("rocprofv3 is not on PATH")
        try:
            _, stderr = proc.communicate(timeout=timeout)
            ended = (
                None if proc.returncode == 0 else f"exited with code {proc.returncode}"
            )
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            stderr, ended = "", f"timed out after {timeout} s"

        statuses, record, times = [], None, []
        if os.path.isfile(spec["status"]):
            with open(spec["status"]) as f:
                statuses = f.read().splitlines()
        if os.path.isfile(spec["record"]):
            with open(spec["record"]) as f:
                record = json.load(f)
        trace_csv = f"{trace}_kernel_trace.csv"
        if os.path.isfile(trace_csv):
            times = kernel_times(trace_csv)
            os.remove(trace_csv)
    return Child(statuses, times, record, ended, stderr)


def kernel_times(trace_csv):
    """Median GEMM kernel time (us) per segment of a rocprofv3 kernel trace.

    A segment is what ran between two split_dummy launches. The time of one
    run is the sum over the GEMM kernels it launched (e.g. main + reduce)."""
    with open(trace_csv) as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["Start_Timestamp"]))
    segments, kernels = [], {}
    for row in rows:
        name = row["Kernel_Name"]
        if name == "split_dummy":
            segments.append(kernels)
            kernels = {}
        elif KERNEL_NAME in name.lower():
            duration = int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
            kernels.setdefault(name, []).append(duration)
    times = []
    for kernels in segments:
        if not kernels:
            times.append(None)
            continue
        per_run = [sum(durations) for durations in zip(*kernels.values())]
        times.append(sorted(per_run)[len(per_run) // 2] / 1000)
    return times


def stderr_tail(text, lines=5):
    return "\n".join(text.strip().splitlines()[-lines:])


# ---------------------------------------------------------------------------
# The config family and the search space


def find_family(record, op):
    """Where the op's config family lives, from the one lookup the op made."""
    if record is None:
        sys.exit(f"{op} never called get_gemm_config(), so there is nothing to tune")
    lookups = record["lookups"]
    if len(lookups) > 1:
        names = ", ".join(lookup["config_name"] for lookup in lookups)
        sys.exit(f"{op} looks up more than one config ({names}); tune those ops")
    lookup = lookups[0]
    name, backend = lookup["config_name"], lookup["backend"]
    config_dir = config_utils.resolve_config_dir("gemm", name, backend=backend)
    default_path = os.path.join(config_dir, "DEFAULT.json")
    default = config_utils.load_config_json(default_path, required=False)
    if default is None:
        sys.exit(
            f"{default_path} does not exist. Add it first (configs/CLAUDE.md, "
            "section 6); its keys are the config keys this script tunes."
        )
    keys = []
    for bucket in default.values():
        if isinstance(bucket, dict):
            keys += [key for key in bucket if key not in keys]

    # The specialized files get_gemm_config() tries for this shape, in order.
    if lookup["specialized_filename"] is not None:
        tuned_files = [f"{name}-{lookup['specialized_filename']}.json"]
    elif lookup["N"] is None or lookup["K"] is None:
        sys.exit(f"{name} is looked up without N and K, so there is no file to tune")
    else:
        N, K, B = lookup["N"], lookup["K"], lookup["B"]
        tuned_files = [f"{name}-N={N}-K={K}.json"]
        if B is not None:
            tuned_files.insert(0, f"{name}-B={B}-N={N}-K={K}.json")
    tuned_paths = [os.path.join(config_dir, f) for f in tuned_files]
    existing = [p for p in tuned_paths if os.path.isfile(p)]
    return {
        "lookup": lookup,
        "config_dir": config_dir,
        "keys": keys,
        "default_path": default_path,
        # The file the config comes from today, and the file the winner goes to.
        "source_path": existing[0] if existing else default_path,
        "target_path": tuned_paths[0],
    }


def resolve(lookup, M):
    """The config the library resolves for this lookup right now."""
    config_utils.load_config_json.cache_clear()
    gemm_config_utils._get_gemm_config_cached.cache_clear()
    config, _ = gemm_config_utils._get_gemm_config_cached(
        lookup["config_name"],
        M,
        lookup["N"],
        lookup["K"],
        tuple(lookup["bounds"]) if lookup["bounds"] else None,
        lookup["specialized_filename"],
        lookup["backend"],
        lookup["B"],
    )
    return config


def search_space(family, M, N, K, overrides):
    """Values to try for each key of the family's DEFAULT.json."""
    arch = arch_info.get_arch()
    seen = values_in_family_files(family["config_dir"], family["lookup"]["backend"])
    space = {}
    for key in family["keys"]:
        # configs/CLAUDE.md: kpack is deprecated off gfx942 and dropped on retune.
        if key == "kpack" and arch != "gfx942":
            continue
        space[key] = candidate_values(key, M, N, K, seen)
    for key, values in overrides.items():
        if key in space:
            space[key] = values
        else:
            print(f"Note: {key} is not a key of this family's DEFAULT.json; ignored")
    return space


def candidate_values(key, M, N, K, seen):
    """Values to try for one config key."""
    if key == "BLOCK_SIZE_M":
        return [4, 8] + [v for v in (16, 32, 64, 128, 256, 512) if v <= M]
    if key == "BLOCK_SIZE_N":
        return [16] + [v for v in (32, 64, 128, 256) if N is None or v <= N]
    if key == "BLOCK_SIZE_K":
        return [128] + [v for v in (256, 512, 1024) if K is None or v <= K]
    if key == "NUM_KSPLIT":
        return [1] + [v for v in (3, 4, 7, 8, 14, 16, 28) if K is None or K % v == 0]
    fixed = {
        "GROUP_SIZE_M": [1, 4, 8],
        "num_warps": [1, 4, 8],
        "num_stages": [1, 2],
        "waves_per_eu": [1, 2, 4, 6, 8],
        "matrix_instr_nonkdim": [16],
        "cache_modifier": [".cg", None],
        "kpack": [1, 2],
    }
    # Any other key (Gluon tiles, buffer counts, kernel variants, ...) tries
    # the values the family's config files already use.
    return fixed.get(key, seen.get(key, []))


def values_in_family_files(config_dir, backend):
    """Every value each key takes in this family's config files, on any arch."""
    family = os.path.basename(config_dir)
    seen = {}
    for path in sorted(glob.glob(f"{CONFIGS_ROOT}/*/{backend}/gemm/{family}/*.json")):
        for bucket in config_utils.load_config_json(path).values():
            if isinstance(bucket, dict):
                for key, value in bucket.items():
                    if value not in seen.setdefault(key, []):
                        seen[key].append(value)
    return seen


def skip_reason(config, K):
    """Split-K combinations that cannot win; None when the config is worth trying."""
    ksplit = config.get("NUM_KSPLIT", 1)
    if ksplit > 1 and config.get("GROUP_SIZE_M", 1) > 1:
        return "NUM_KSPLIT > 1 with GROUP_SIZE_M > 1"
    block_k, stages = config.get("BLOCK_SIZE_K"), config.get("num_stages")
    if block_k is None or K is None:
        return None
    k_per_split = K // ksplit
    if block_k >= 2 * k_per_split:
        return "BLOCK_SIZE_K is more than twice the K of one split"
    if stages is not None and block_k == k_per_split and stages > 1:
        return "one K iteration, so num_stages > 1 pipelines nothing"
    if stages is not None and block_k < k_per_split and stages == 1:
        return "several K iterations with no pipelining"
    return None


# ---------------------------------------------------------------------------
# Trying the candidates


class ErrorLog:
    """errors-<op>-<shape>.txt: each config that failed, and the error."""

    def __init__(self, path):
        self.path = path
        self.count = 0
        if os.path.exists(path):
            os.remove(path)

    def add(self, config, error):
        with open(self.path, "a") as f:
            f.write(f"{json.dumps(config)}\n    {error}\n")
        self.count += 1

    def close(self):
        if self.count:
            print(f"{self.count} configs failed; see {self.path}")


def try_candidates(spec, configs, current_works, timeout, errors):
    """Time every config, BATCH per rocprofv3 process; returns the fastest and
    its time.

    The process writes one status line per config. If it dies, the config it
    was on gets the blame and the rest of the batch runs again in a new
    process. A config that ran out of resources rules out every config with
    the same tile sizes."""
    tile_keys = spec["tile_keys"]
    pending = list(configs)
    tried = collections.Counter()
    bad_tiles = set()
    best, best_us = None, float("inf")
    done = 0
    while pending:
        batch, skipped = [], 0
        while pending and len(batch) < BATCH:
            config = pending.pop(0)
            if tuple(config.get(key) for key in tile_keys) in bad_tiles:
                skipped += 1
            else:
                batch.append(config)
        done += skipped
        if skipped:
            print(
                f"[{done}/{len(configs)}] skipped {skipped} configs whose tile sizes ran out of resources"
            )
        if not batch:
            continue

        child = run_child(spec, batch, run_current=current_works, timeout=timeout)
        if child.statuses[:1] != ["ready"]:
            sys.exit(
                f"profile_configs.py could not set up the op:\n{stderr_tail(child.stderr, 20)}"
            )
        results = child.statuses[2:]  # after "ready" and the current config
        retry = []
        for i, config in enumerate(batch):
            tiles = tuple(config.get(key) for key in tile_keys)
            if i >= len(results):
                # The process died before finishing this config. The config it
                # was on is to blame, unless it stopped on purpose after a GPU
                # fault (exit code 3), which the config before already reported.
                if i == len(results) and child.ended != "exited with code 3":
                    done += 1
                    errors.add(
                        config,
                        f"process {child.ended}\n    {stderr_tail(child.stderr)}",
                    )
                    print(f"[{done}/{len(configs)}] process {child.ended}")
                else:
                    retry.append(config)
                continue
            status = results[i]
            if status == "ok":
                us = child.times[i + 1] if i + 1 < len(child.times) else None
                if us is None:
                    # The trace was lost with the process; run this one again.
                    tried[json.dumps(config)] += 1
                    if tried[json.dumps(config)] < 2:
                        retry.append(config)
                        continue
                    status = "error: no kernel time recorded, twice"
                else:
                    done += 1
                    print(f"[{done}/{len(configs)}] {us:9.3f} us  {json.dumps(config)}")
                    if us < best_us:
                        best, best_us = config, us
                    continue
            done += 1
            if status.startswith("skipped"):
                bad_tiles.add(tiles)  # the process saw these tiles fail
                continue
            error = status[len("error: ") :]
            errors.add(config, error)
            print(f"[{done}/{len(configs)}] error: {error[:120]}")
            if any(s in error for s in RESOURCE_ERRORS):
                bad_tiles.add(tiles)
        pending = retry + pending
    return best, best_us


# ---------------------------------------------------------------------------
# Writing the winner


def write_config(family, M, config):
    """Put `config` in the M bucket for this M of the file get_gemm_config()
    reads for this shape, creating that file from DEFAULT.json if needed."""
    table = dict(config_utils.load_config_json(family["source_path"]))
    bounds = (
        family["lookup"]["bounds"]
        or table.get("M_BOUNDS")
        or gemm_config_utils.STANDARD_M_BOUNDS
    )
    bucket = next((f"M_LEQ_{b}" for b in bounds if b >= M), "any")
    table[bucket] = {k: v for k, v in config.items() if k in family["keys"]}
    with open(family["target_path"], "w") as f:
        json.dump(dict(sorted(table.items(), key=bucket_order)), f, indent=4)
        f.write("\n")
    print(f"Wrote {bucket} in {family['target_path']}")
    return table[bucket]


def bucket_order(item):
    """M_LEQ_* ascending, then M_GEQ_* descending, then the rest."""
    key = item[0]
    if key.startswith("M_LEQ_"):
        return (0, int(key[6:]))
    if key.startswith("M_GEQ_"):
        return (1, -int(key[6:]))
    return (2, 0)


def main():
    args = parse_args()
    dims = parse_dims(args.dims)
    case = get_case(args.op, dims, args.backend)
    M = dims["M"]
    shape = " ".join(f"{k}={v}" for k, v in dims.items())
    print(f"{args.op} {shape} on {arch_info.get_arch()}")
    errors = ErrorLog(f"errors-{args.op}-{shape.replace(' ', '-')}.txt")
    spec = {
        "op": args.op,
        "dims": dims,
        "backend": args.backend,
        "check": not args.no_check,
        "tile_keys": [],
    }

    # 1. The current config: what the library resolves for this shape today.
    child = run_child(spec, [], run_current=True, timeout=args.timeout)
    if child.statuses[:1] != ["ready"]:
        sys.exit(
            f"profile_configs.py could not set up the op:\n{stderr_tail(child.stderr, 20)}"
        )
    family = find_family(child.record, args.op)
    lookup = family["lookup"]
    current = child.record["config"]
    source = os.path.relpath(family["source_path"], CONFIGS_ROOT)
    print(f"Config family {lookup['config_name']} ({lookup['backend']})")
    print(f"Current config, from {source}:")
    print(f"    {json.dumps(current)}")
    if len(child.statuses) < 2:
        current_error = f"process {child.ended}\n    {stderr_tail(child.stderr)}"
    elif child.statuses[1] != "ok":
        current_error = child.statuses[1][len("error: ") :]
    elif not child.times or child.times[0] is None:
        sys.exit(f"rocprofv3 recorded no kernel whose name contains '{KERNEL_NAME}'")
    else:
        current_error = None
    current_works = current_error is None
    if current_works:
        current_us = child.times[0]
        print(f"    {current_us:.3f} us")
    else:
        current_us = float("inf")
        print(f"    fails: {current_error.splitlines()[0]}")
        errors.add(current, current_error)

    # 2. Every candidate config.
    N = lookup["N"] if lookup["N"] is not None else dims.get("N")
    K = lookup["K"] if lookup["K"] is not None else dims.get("K")
    space = search_space(family, M, N, K, {**case.space, **parse_space(args.space)})
    print("Values tried:")
    for key, values in space.items():
        print(f"    {key} = {values}")
    configs = [
        dict(zip(space, values))
        for values in itertools.product(*space.values())
        if skip_reason(dict(zip(space, values)), K) is None
    ]
    print(f"{len(configs)} configs to try, {BATCH} per rocprofv3 process")
    spec["tile_keys"] = [key for key in space if TILE_KEY.fullmatch(key)]
    best, best_us = try_candidates(spec, configs, current_works, args.timeout, errors)
    errors.close()

    # 3. Keep the winner.
    if best is not None and current_works:
        # Back to back in one process, so a GPU that got warmer during the
        # sweep does not decide.
        child = run_child(spec, [best], run_current=True, timeout=args.timeout)
        if len(child.times) >= 2 and None not in child.times[:2]:
            current_us, best_us = child.times[:2]
        print(f"Best candidate {best_us:.3f} us, current config {current_us:.3f} us")
    if best is not None and best_us < current_us * (1 - MIN_GAIN):
        winner = best
    elif family["source_path"] != family["default_path"]:
        print("The current config is still the fastest; nothing written.")
        return
    elif current_works:
        print("No candidate beats DEFAULT.json; recording it as this shape's config.")
        winner = current
    else:
        sys.exit(f"No config works for this shape; see {errors.path}")
    written = write_config(family, M, winner)
    if resolve(lookup, M) == written:
        print("Checked: get_gemm_config() now returns it for this shape.")
    else:
        print(f"Warning: another bucket in the file still wins for M={M}.")


if __name__ == "__main__":
    main()
