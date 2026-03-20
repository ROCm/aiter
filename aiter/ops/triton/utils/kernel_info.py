"""
Kernel info collection for Triton kernels (similar to hipBLASLt --print_kernel_info).

- AITER_PRINT_KERNEL_INFO=1: print "Kernel Information:" plus name and config per launch.
- AITER_PRINT_KERNEL_NAME=1: print "Kernel Information:" then unique module.qualname lines only (no per-launch config).
  If both are set, name-only mode is used for printing.

Production: leave both unset (default). Call sites should pass the raw @triton.jit / autotune object as the first
argument — not triton_kernel_label(fn) — so when printing is off, no unwrap work runs (see collect_kernel_info).

For collection and printing, AITER_PRINT_* are read at call time so they match the benchmark.
Set them before the process starts (normal use).
"""
import inspect
import os
from typing import Any, Dict, List, Optional, Set, Union

# Snapshot env at import (used for docs / rare fast paths only)
_PRINT_KERNEL_INFO_ENV: bool = os.environ.get("AITER_PRINT_KERNEL_INFO", "0") == "1"
_PRINT_KERNEL_NAME_ENV: bool = os.environ.get("AITER_PRINT_KERNEL_NAME", "0") == "1"
_KERNEL_INFO_ACTIVE: bool = _PRINT_KERNEL_INFO_ENV or _PRINT_KERNEL_NAME_ENV


def _kernel_info_active_runtime() -> bool:
    """Read env at call time so collection matches AITER_PRINT_* even if import order varies."""
    return (
        os.environ.get("AITER_PRINT_KERNEL_INFO", "0") == "1"
        or os.environ.get("AITER_PRINT_KERNEL_NAME", "0") == "1"
    )

# Global list to collect kernel info (deduplicated)
_kernel_info_list: List[Dict[str, Any]] = []
_seen_kernels: Set[str] = set()  # Track seen kernel+config combinations


def triton_kernel_label(fn: Any) -> str:
    """
    Resolve @triton.jit / @triton.autotune wrappers to the underlying Python function
    and return module.qualname (the stable Triton kernel identity in source).
    """
    base: Any = fn
    seen: Set[int] = set()
    while base is not None and id(base) not in seen:
        seen.add(id(base))
        if hasattr(base, "base_fn"):
            base = base.base_fn
            continue
        nxt = getattr(base, "fn", None)
        if nxt is not None and nxt is not base:
            base = nxt
            continue
        break
    if base is not None and not inspect.isfunction(base) and not inspect.ismethod(base):
        # Fallback: some wrappers may not unwrap fully
        return repr(fn)
    mod = getattr(base, "__module__", "") if base is not None else ""
    qual = (
        getattr(base, "__qualname__", None)
        or getattr(base, "__name__", None)
        or repr(base)
    )
    return f"{mod}.{qual}" if mod else str(qual)


def _kernel_info_enabled() -> bool:
    return _kernel_info_active_runtime()


def collect_kernel_info(kernel_ref: Union[str, Any], config: Optional[Dict[str, Any]] = None) -> None:
    """
    Record one launch. Pass the raw JIT/autotune object (e.g. _attn_fwd) or a precomputed string id.

    When kernel printing is disabled, this returns immediately without calling triton_kernel_label.
    """
    if not _kernel_info_active_runtime():
        return
    if config is None:
        config = {}
    kernel_name = kernel_ref if isinstance(kernel_ref, str) else triton_kernel_label(kernel_ref)
    # Format config params as string
    config_str = ", ".join([f"{k}={v}" for k, v in sorted(config.items())])
    # Create unique key for deduplication
    unique_key = f"{kernel_name}:{config_str}"

    # Only collect if we haven't seen this kernel+config combination before
    if unique_key not in _seen_kernels:
        _seen_kernels.add(unique_key)
        _kernel_info_list.append({
            "kernel_name": kernel_name,
            "config_params": config_str
        })


def print_kernel_info() -> None:
    """Print collected kernel information (similar to hipBLASLt --print_kernel_info)"""
    if not _kernel_info_active_runtime():
        return
    if not _kernel_info_list:
        if os.environ.get("AITER_PRINT_KERNEL_NAME", "0") == "1" or os.environ.get(
            "AITER_PRINT_KERNEL_INFO", "0"
        ) == "1":
            print("Kernel Information:")
            print("(no instrumented Triton launches recorded; check collect_kernel_info hooks)")
        return
    # Name-only: one line per unique Triton kernel (module.qualname), no config
    if os.environ.get("AITER_PRINT_KERNEL_NAME", "0") == "1":
        print("Kernel Information:")
        seen_names: Set[str] = set()
        for info in _kernel_info_list:
            name = info["kernel_name"]
            if name not in seen_names:
                seen_names.add(name)
                print(name)
        _kernel_info_list.clear()
        _seen_kernels.clear()
        return
    if os.environ.get("AITER_PRINT_KERNEL_INFO", "0") == "1":
        print("Kernel Information:")
        for info in _kernel_info_list:
            print(f"{info['kernel_name']}: {info['config_params']}")
        _kernel_info_list.clear()
        _seen_kernels.clear()


def write_kernel_names_to_csv(csv_path: str) -> None:
    """
    Add a kernel_names column (semicolon-separated unique ids) to an existing CSV.
    Intended for bench_mha.py -o together with AITER_PRINT_KERNEL_NAME=1.
    The CSV is written by Triton/bench (-o); this only augments it after the run.
    """
    import csv
    import glob
    import os

    if os.environ.get("AITER_PRINT_KERNEL_NAME", "0") != "1":
        return
    path = csv_path
    if not os.path.isfile(path):
        d = os.path.dirname(os.path.abspath(path)) or os.path.abspath(".")
        stem = os.path.splitext(os.path.basename(path))[0]
        cand = os.path.join(d, stem + ".csv")
        if os.path.isfile(cand):
            path = cand
        else:
            matches = sorted(
                glob.glob(os.path.join(d, "*.csv")),
                key=os.path.getmtime,
                reverse=True,
            )
            prefer = [p for p in matches if os.path.basename(p).startswith(stem)]
            path = (prefer[0] if prefer else (matches[0] if matches else path))
    if not os.path.isfile(path):
        return
    names: List[str] = []
    seen: Set[str] = set()
    for info in _kernel_info_list:
        n = info["kernel_name"]
        if n not in seen:
            seen.add(n)
            names.append(n)
    val = ";".join(names)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if "kernel_names" not in fieldnames:
            fieldnames.append("kernel_names")
        rows = []
        for row in reader:
            row["kernel_names"] = val
            rows.append(row)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def clear_kernel_info():
    """Clear collected kernel info"""
    _kernel_info_list.clear()
    _seen_kernels.clear()
