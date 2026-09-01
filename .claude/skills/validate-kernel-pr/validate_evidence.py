"""Structured evidence helpers for validate-kernel-pr."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import pathlib
import platform
import sys
import xml.etree.ElementTree as ET


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pytest_stats(path: pathlib.Path) -> dict:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    totals = {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    totals["executed"] = totals["tests"] - totals["skipped"]
    totals["junit_xml"] = str(path)
    return totals


def validate_receipt(
    path: pathlib.Path,
    expected_route: str,
    grid: str,
) -> dict:
    if not expected_route:
        return {
            "status": "skip",
            "note": "--expected-route was not supplied",
        }
    if not path.is_file():
        return {
            "status": "skip",
            "note": "selected test did not write an execution receipt",
        }
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return {"status": "skip", "note": f"execution receipt is invalid: {error}"}
    if receipt.get("schema_version") != 1:
        return {"status": "skip", "note": "execution receipt schema_version must be 1"}
    if receipt.get("producer") != "validate-kernel-pr.validation_probe":
        return {
            "status": "skip",
            "note": "execution receipt was not produced by validator instrumentation",
        }
    if receipt.get("route") != expected_route:
        return {
            "status": "skip",
            "note": (
                f"expected route {expected_route!r}, "
                f"observed {receipt.get('route')!r}"
            ),
        }
    symbols = receipt.get("kernel_symbols")
    if (
        not isinstance(symbols, list)
        or not symbols
        or not all(isinstance(symbol, str) and symbol for symbol in symbols)
    ):
        return {
            "status": "skip",
            "note": "execution receipt has no kernel_symbols route proof",
        }
    if expected_route not in symbols:
        return {
            "status": "skip",
            "note": "expected route is absent from observed kernel_symbols",
        }
    expected_shapes = [shape.strip() for shape in grid.split(";") if shape.strip()]
    observed_shapes = receipt.get("executed_shapes")
    if not isinstance(observed_shapes, list) or not all(
        isinstance(shape, str) and shape for shape in observed_shapes
    ):
        return {
            "status": "skip",
            "note": "execution receipt has no executed_shapes list",
        }
    missing = sorted(set(expected_shapes) - set(observed_shapes))
    if missing:
        return {
            "status": "skip",
            "note": f"execution receipt is missing required shapes: {missing}",
        }
    result = {
        "status": "pass",
        "route": expected_route,
        "producer": receipt["producer"],
        "kernel_symbols": sorted(set(symbols)),
        "required_shapes": expected_shapes,
        "executed_shapes": observed_shapes,
        "receipt": str(path),
        "receipt_sha256": file_sha256(path),
    }
    # A pass here proves the route executed and nothing more. When the caller asked for shape
    # capture and none arrived, saying only "pass" publishes an absence as a confirmation:
    # every observed case was a route deriving its shapes in its body, where the requested
    # names never became locals the probe could see. Name the gap in the report so no consumer
    # can read the empty list as evidence that no shapes were needed.
    requested_vars = [name for name in receipt.get("shape_vars") or [] if name]
    if requested_vars and not observed_shapes:
        result["shape_capture"] = {
            "requested": requested_vars,
            "observed": 0,
            "note": (
                "shape capture was requested but produced no rows: the route never bound all "
                "of these names as locals the probe could read. This receipt asserts route "
                "execution only; it makes no claim about the shapes exercised."
            ),
        }
    return result


def loaded_native_artifacts(roots: list[pathlib.Path]) -> list[dict]:
    resolved_roots = [root.resolve() for root in roots if root.is_dir()]
    files = set()
    for module in tuple(sys.modules.values()):
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        path = pathlib.Path(module_file).resolve()
        if not (path.suffix == ".so" or ".so." in path.name):
            continue
        if any(root in path.parents for root in resolved_roots):
            files.add(path)
    return [
        {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(files)
    ]


def runtime_identity(
    module_name: str,
    expected_root: pathlib.Path,
    dependency_root: pathlib.Path | None,
) -> dict:
    module = importlib.import_module(module_name)
    module_path = pathlib.Path(module.__file__).resolve()
    root = expected_root.resolve()
    if root not in module_path.parents:
        raise RuntimeError(f"{module_name} resolved outside {root}: {module_path}")
    roots = [root]
    if dependency_root:
        roots.append(dependency_root.resolve())
    artifacts = loaded_native_artifacts(roots)
    return {
        "module": module_name,
        "module_path": str(module_path),
        "module_version": getattr(module, "__version__", None),
        "expected_root": str(root),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "source_sha": None,
        "native_artifacts": artifacts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    stats = subparsers.add_parser("pytest-stats")
    stats.add_argument("junit_xml", type=pathlib.Path)

    receipt = subparsers.add_parser("receipt")
    receipt.add_argument("path", type=pathlib.Path)
    receipt.add_argument("--expected-route", default="")
    receipt.add_argument("--grid", default="")

    runtime = subparsers.add_parser("runtime")
    runtime.add_argument("module")
    runtime.add_argument("expected_root", type=pathlib.Path)
    runtime.add_argument("--dependency-root", type=pathlib.Path)
    runtime.add_argument("--output", type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "pytest-stats":
        result = pytest_stats(args.junit_xml)
    elif args.command == "receipt":
        result = validate_receipt(args.path, args.expected_route, args.grid)
    else:
        result = runtime_identity(
            args.module,
            args.expected_root,
            args.dependency_root,
        )
    if args.command == "runtime" and args.output:
        args.output.write_text(json.dumps(result) + "\n")
    else:
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
