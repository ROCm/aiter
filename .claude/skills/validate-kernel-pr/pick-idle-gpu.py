#!/usr/bin/env python3
"""Select an AMD GPU that stays idle across a sampling window."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ACTIVITY_BASIS = "activity+vram"
VRAM_ONLY_BASIS = "vram-only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--max-busy", type=int, default=2)
    parser.add_argument("--max-used-gib", type=float, default=2.0)
    parser.add_argument("--min-free-gib", type=float, default=16.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--all",
        action="store_true",
        help="print every eligible HIP index in preference order, one per line, so a caller "
        "whose lock on the first choice is contended can fall through to the next instead of "
        "reporting no idle GPU while idle GPUs remain",
    )
    args = parser.parse_args()
    if args.samples < 1 or args.interval < 0:
        parser.error("samples must be positive and interval must be non-negative")
    if args.max_busy < 0 or args.max_used_gib < 0 or args.min_free_gib < 0:
        parser.error("thresholds must be non-negative")
    return args


def amdsmi_search_paths() -> list[Path]:
    """Directories that have shipped the amdsmi bindings across ROCm releases."""
    roots = [Path(os.environ["ROCM_PATH"])] if os.environ.get("ROCM_PATH") else []
    roots.append(Path("/opt/rocm"))
    roots.extend(sorted(Path("/opt").glob("rocm-*"), reverse=True))

    candidates = [
        Path("/usr/lib/python3/dist-packages"),
        Path(
            f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages"
        ),
    ]
    for root in roots:
        # ROCm >= 7.1 ships the bindings under share/amd_smi; older builds put
        # them next to the CLI. Probe both, newest ROCm first.
        candidates.append(root / "share" / "amd_smi")
        candidates.append(root / "libexec" / "amdsmi_cli")
    return candidates


def import_amdsmi():
    try:
        import amdsmi

        return amdsmi
    except ImportError:
        for candidate in amdsmi_search_paths():
            if candidate.is_dir() and str(candidate) not in sys.path:
                sys.path.append(str(candidate))
        import amdsmi

        return amdsmi


def read_activity(amdsmi, handle) -> tuple[int | None, int | None]:
    """Return (gfx, umc) busy percentages, or None for whichever is unknown.

    Some driver and amd-smi combinations fail this query outright (MI308X on
    ROCm 7.0 raises AMDSMI_STATUS_UNEXPECTED_DATA) or report "N/A". Both mean
    unknown, which must stay distinct from a measured 0 -- reporting unknown as
    idle would claim an idleness that was never observed.
    """
    try:
        activity = amdsmi.amdsmi_get_gpu_activity(handle)
    except (OSError, amdsmi.AmdSmiException):
        return None, None
    gfx = activity.get("gfx_activity")
    umc = activity.get("umc_activity")
    return (
        gfx if isinstance(gfx, int) else None,
        umc if isinstance(umc, int) else None,
    )


def sample(amdsmi, count: int, interval: float) -> tuple[list[dict], int]:
    gpus = []
    for smi_index, handle in enumerate(amdsmi.amdsmi_get_processor_handles()):
        enumeration = amdsmi.amdsmi_get_gpu_enumeration_info(handle)
        gpus.append(
            {
                "smi_index": smi_index,
                "hip_index": enumeration.get("hip_id"),
                "bdf": amdsmi.amdsmi_get_gpu_device_bdf(handle),
                "handle": handle,
                "gfx": [],
                "umc": [],
            }
        )
    peak_concurrent = 0
    for sample_index in range(count):
        if sample_index:
            time.sleep(interval)
        busy = 0
        for gpu in gpus:
            gfx, umc = read_activity(amdsmi, gpu["handle"])
            if gfx is not None:
                gpu["gfx"].append(gfx)
                busy += int(gfx > 5)
            if umc is not None:
                gpu["umc"].append(umc)
        peak_concurrent = max(peak_concurrent, busy)
    for gpu in gpus:
        memory = amdsmi.amdsmi_get_gpu_vram_usage(gpu["handle"])
        used = memory["vram_used"] / 1024
        total = memory["vram_total"] / 1024
        gpu.update(
            {
                "used_gib": used,
                "free_gib": total - used,
                "peak_gfx": max(gpu["gfx"]) if gpu["gfx"] else None,
                "mean_gfx": (sum(gpu["gfx"]) / len(gpu["gfx"]) if gpu["gfx"] else None),
                "peak_umc": max(gpu["umc"]) if gpu["umc"] else None,
            }
        )
        del gpu["handle"]
    return gpus, peak_concurrent


def main() -> int:
    args = parse_args()
    try:
        amdsmi = import_amdsmi()
    except ImportError as error:
        print(f"AMD SMI import failed: {error}", file=sys.stderr)
        return 2
    try:
        amdsmi.amdsmi_init()
        try:
            gpus, peak_concurrent = sample(amdsmi, args.samples, args.interval)
        finally:
            amdsmi.amdsmi_shut_down()
    except (OSError, amdsmi.AmdSmiException) as error:
        print(f"AMD SMI probe failed: {error}", file=sys.stderr)
        return 2

    eligible = [
        gpu
        for gpu in gpus
        if gpu["hip_index"] is not None
        and (gpu["peak_gfx"] is None or gpu["peak_gfx"] <= args.max_busy)
        and gpu["used_gib"] <= args.max_used_gib
        and gpu["free_gib"] >= args.min_free_gib
    ]
    # Prefer GPUs whose idleness was actually measured over ones where the
    # activity query failed and only VRAM could be checked.
    eligible.sort(
        key=lambda gpu: (
            gpu["peak_gfx"] is None,
            gpu["peak_gfx"] or 0,
            gpu["mean_gfx"] or 0.0,
            gpu["used_gib"],
            -gpu["free_gib"],
        )
    )

    if not args.quiet:
        print(
            f"Sampled {args.samples} times over {args.samples * args.interval:.0f}s",
            file=sys.stderr,
        )
        print(
            f"{'smi':>4} {'hip':>4} {'bdf':<14} {'peak%':>6} {'mean%':>6} "
            f"{'umc%':>5} {'used':>9} {'free':>9}  verdict",
            file=sys.stderr,
        )
        for gpu in sorted(gpus, key=lambda item: item["smi_index"]):
            if gpu["hip_index"] is None:
                verdict = "SKIP no hip_id"
            elif gpu["peak_gfx"] is not None and gpu["peak_gfx"] > args.max_busy:
                verdict = f"BUSY peaked {gpu['peak_gfx']}%"
            elif gpu["used_gib"] > args.max_used_gib:
                verdict = f"HELD {gpu['used_gib']:.1f} GiB used"
            elif gpu["free_gib"] < args.min_free_gib:
                verdict = f"FULL {gpu['free_gib']:.1f} GiB free"
            elif gpu["peak_gfx"] is None:
                verdict = "idle by VRAM only (activity unavailable)"
            else:
                verdict = "idle"
            hip_index = "-" if gpu["hip_index"] is None else gpu["hip_index"]
            peak_gfx = "n/a" if gpu["peak_gfx"] is None else str(gpu["peak_gfx"])
            mean_gfx = "n/a" if gpu["mean_gfx"] is None else f"{gpu['mean_gfx']:.1f}"
            peak_umc = "n/a" if gpu["peak_umc"] is None else str(gpu["peak_umc"])
            print(
                f"{gpu['smi_index']:>4} {hip_index:>4} {gpu['bdf']:<14} "
                f"{peak_gfx:>6} {mean_gfx:>6} "
                f"{peak_umc:>5} {gpu['used_gib']:>6.1f} GiB "
                f"{gpu['free_gib']:>6.1f} GiB  {verdict}",
                file=sys.stderr,
            )
        if gpus and peak_concurrent >= len(gpus) - 1:
            print(
                f"WARNING: {peak_concurrent}/{len(gpus)} GPUs were busy together; "
                "shared fabric/power may perturb the run.",
                file=sys.stderr,
            )

    if not gpus:
        # Exit 3, distinct from 1: an absent device is an environment fact, and
        # reporting it as "activity unavailable" would blame the validator instead.
        print("This host reports no GPUs.", file=sys.stderr)
        return 3
    if not eligible:
        if all(gpu["peak_gfx"] is None for gpu in gpus):
            print(
                "No GPU stayed below the resident-memory thresholds; GPU activity "
                "is unavailable on this host, so only VRAM was considered.",
                file=sys.stderr,
            )
        else:
            print(
                "No GPU stayed below the activity and resident-memory thresholds.",
                file=sys.stderr,
            )
        return 1
    selected = eligible[0]
    basis = ACTIVITY_BASIS if selected["peak_gfx"] is not None else VRAM_ONLY_BASIS
    # Machine-readable and deliberately outside the --quiet guard: callers record
    # this so a report never presents a VRAM-only claim as a measured-idle one.
    print(f"idleness-basis: {basis}", file=sys.stderr)
    if not args.quiet:
        print(
            f"Chose HIP index {selected['hip_index']} "
            f"(amd-smi {selected['smi_index']}, {selected['bdf']}).",
            file=sys.stderr,
        )
    if args.all:
        for gpu in eligible:
            print(gpu["hip_index"])
        return 0
    print(selected["hip_index"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
