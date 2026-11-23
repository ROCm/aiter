# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Opt-in live hipBLASLt index/name check; never launches a GEMM.

Run in a ROCm environment with aiter installed:
    python3 -m op_tests.tuning_tests.validate_hipblaslt_config --report report.json

Exit codes: 0 = every selected row matches, 1 = rows need attention,
2 = setup failed or no hipBLASLt rows exist for this GPU's gfx/CU count.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from unittest.mock import patch

import pandas as pd

_PREFIX = "AITER_HIPBLASLT_CHECK "
_ENV = "AITER_CONFIG_GEMM_BF16"


def _load_config():
    """Use the runtime family merge on copies, including env-selected files."""
    from aiter.jit import core

    original_root = Path(core.AITER_ROOT_DIR)
    override = os.environ.get(_ENV, "").strip()
    with tempfile.TemporaryDirectory(prefix="aiter_hipblaslt_check_") as tmp:
        config_dir = Path(tmp) / "aiter" / "configs"
        shutil.copytree(original_root / "aiter" / "configs", config_dir)
        copied_paths = []
        if override:
            for i, name in enumerate(override.split(os.pathsep)):
                source = Path(name)
                destination = Path(tmp) / "overrides" / str(i) / source.name
                destination.parent.mkdir(parents=True)
                # Do not silently skip a mistyped override and validate less.
                shutil.copy2(source, destination)
                copied_paths.append(str(destination))
        with (
            patch.object(core, "AITER_ROOT_DIR", tmp),
            patch.dict(os.environ, {_ENV: os.pathsep.join(copied_paths)}),
            patch("tempfile.gettempdir", return_value=tmp),
        ):
            merged = core.AITER_CONFIG().get_config_file(
                _ENV, str(config_dir / "bf16_tuned_gemm.csv"), "bf16_tuned_gemm"
            )
            content = Path(merged).read_bytes()
            df = pd.read_csv(merged, dtype=str, keep_default_na=False)
    return df, {
        "aiter_root": str(original_root),
        "config_override": override or None,
        "merged_csv_sha256": hashlib.sha256(content).hexdigest(),
    }


def _emit(event):
    print(_PREFIX + json.dumps(event), flush=True)


def _query_child(indices):
    import torch

    import aiter
    from aiter.jit.core import get_module

    aiter.hipb_create_extension()
    module = get_module("module_hipbsolgemm")
    # Header version is build provenance, not the loaded library's version.
    libraries = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and Path(fields[5]).name.startswith("libhipblaslt.so"):
            path = Path(fields[5])
            if str(path) not in libraries:
                libraries[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    _emit(
        {
            "event": "ready",
            "environment": {
                "torch_version": str(torch.__version__),
                "torch_hip_version": torch.version.hip,
                "hipblaslt_build_version": getattr(
                    module, "hipblaslt_build_version", None
                ),
                "loaded_hipblaslt_sha256": libraries,
            },
        }
    )
    for index in indices:
        # An unknown native index can exit() before returning a name. Mark
        # the attempted index so it is distinguishable from setup failure.
        _emit({"event": "query", "index": index})
        name = aiter.getHipblasltKernelName(index)
        _emit({"event": "result", "index": index, "name": name})
    aiter.hipb_destroy_extension()


def _query_indices(indices, timeout):
    """Keep completed queries when a native exit kills a batch; retry the rest.

    Only a process that initialized successfully and identified the failing
    query may be retried. Setup/protocol failures and timeouts fail all remaining
    rows rather than being misreported as unknown indexes.
    """
    pending = sorted(set(indices))
    results, environment = {}, None
    while pending:
        command = [sys.executable, str(Path(__file__).resolve()), "--query"]
        command.extend(map(str, pending))
        timed_out = False
        try:
            process = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout, check=False
            )
            stdout, stderr, code = process.stdout, process.stderr, process.returncode
        except subprocess.TimeoutExpired as exc:
            stdout, stderr, code = exc.stdout or "", exc.stderr or "", None
            timed_out = True
        # TimeoutExpired's captured streams can be bytes even with text=True.
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        active, ready, protocol_error = None, False, False
        for line in stdout.splitlines():
            if not line.startswith(_PREFIX):
                continue
            try:
                event = json.loads(line[len(_PREFIX) :])
                if event["event"] == "ready" and not ready:
                    if not isinstance(event["environment"], dict):
                        raise ValueError("missing environment metadata")
                    if environment is not None and environment != event["environment"]:
                        raise ValueError("library environment changed between batches")
                    ready = True
                    environment = event["environment"]
                elif event["event"] == "query" and ready and active is None:
                    active = event["index"]
                    if (
                        type(active) is not int
                        or active not in pending
                        or active in results
                    ):
                        raise ValueError("unexpected query index")
                elif (
                    event["event"] == "result"
                    and active is not None
                    and active == event["index"]
                ):
                    name = event["name"]
                    if not isinstance(name, str) or not name.strip():
                        raise ValueError("empty/non-string kernel name")
                    results[active] = {"name": name.strip(), "error": None}
                    active = None
                else:
                    raise ValueError("unexpected child event")
            except (KeyError, TypeError, ValueError):
                protocol_error = True
                break
        remaining = [index for index in pending if index not in results]
        if not remaining and code == 0 and not protocol_error and not timed_out:
            break
        error = (
            f"child timed out after {timeout}s"
            if timed_out
            else f"child exit={code}; protocol_error={protocol_error}"
        )
        if stderr.strip():
            error += ": " + stderr.strip()[-2000:]
        if (
            ready
            and active in remaining
            and code != 0
            and not (timed_out or protocol_error)
        ):
            # Do not claim that every exit is an invalid index: it might be a
            # library failure. Preserve the diagnostics for this query.
            results[active] = {"name": None, "error": error}
            pending = [index for index in remaining if index != active]
        else:
            # Even a failure after the last result (e.g. native cleanup) must
            # not produce a passing report.
            for index in pending if not remaining else remaining:
                results[index] = {"name": None, "error": error}
            break
    return results, environment


def _check_rows(df, gfx, cu_num, timeout):
    required = {"gfx", "cu_num", "libtype", "solidx"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing columns: {sorted(required - set(df.columns))}")
    rows = df[
        df.libtype.eq("hipblaslt")
        & df.gfx.eq(gfx)
        & pd.to_numeric(df.cu_num, errors="raise").eq(cu_num)
    ]
    if rows.empty:
        raise ValueError(f"no hipBLASLt rows for {gfx}/{cu_num} CUs")
    checked = []
    for row_index, row in rows.iterrows():
        saved = str(row.get("kernelName", "")).strip()
        # The runtime merge uses 0 for a column absent from a legacy file.
        saved = "" if saved in ("0", "0.0") else saved
        item = {
            "merged_row": int(row_index) + 2,
            "config_row": {key: str(value) for key, value in row.items()},
            "solidx": str(row.solidx),
            "kernelName": saved,
        }
        try:
            number = Decimal(str(row.solidx))
            if (
                not number.is_finite()
                or number != number.to_integral_value()
                or not 0 <= number <= 2**31 - 1
            ):
                raise ValueError("solidx must be a nonnegative int32")
            item["index"] = int(number)
        except (InvalidOperation, ValueError, TypeError) as exc:
            item.update(status="query_failed", live_kernelName=None, error=str(exc))
        checked.append(item)
    queries, environment = _query_indices(
        [row["index"] for row in checked if "index" in row], timeout
    )
    for item in checked:
        if "index" not in item:
            continue
        result = queries[item.pop("index")]
        item.update(live_kernelName=result["name"], error=result["error"])
        if result["error"]:
            item["status"] = "query_failed"
        elif not item["kernelName"]:
            item["status"] = "missing_name"
        elif item["kernelName"] == result["name"]:
            item["status"] = "match"
        else:
            item["status"] = "mismatch"
    return checked, environment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="also save the JSON report here")
    parser.add_argument(
        "--timeout",
        type=float,
        default=300,
        help="seconds per child batch (default 300)",
    )
    parser.add_argument("--query", nargs="+", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.query is not None:
        _query_child(args.query)
        return 0
    if not 0 < args.timeout < float("inf"):
        parser.error("--timeout must be finite and positive")
    report = {}
    try:
        from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

        gfx, cu_num = get_gfx_runtime(), get_cu_num()
        df, config = _load_config()
        rows, environment = _check_rows(df, gfx, cu_num, args.timeout)
        report.update(gfx=gfx, cu_num=cu_num, config=config, environment=environment)
        report["rows"] = rows
        report["counts"] = {
            status: sum(row["status"] == status for row in rows)
            for status in ("match", "mismatch", "missing_name", "query_failed")
        }
        code = int(any(row["status"] != "match" for row in rows))
    except Exception as exc:  # noqa: BLE001 -- report dependency/setup errors as JSON.
        report["setup_error"] = f"{type(exc).__name__}: {exc}"
        code = 2
    content = json.dumps(report, indent=2) + "\n"
    print(content, end="")
    if args.report is not None:
        args.report.write_text(content)
    return code


if __name__ == "__main__":
    sys.exit(main())
