"""Select an AMD GPU that stays idle across a sampling window."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--max-busy", type=int, default=2)
    parser.add_argument("--max-used-gib", type=float, default=2.0)
    parser.add_argument("--min-free-gib", type=float, default=16.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.interval < 0:
        parser.error("samples must be positive and interval must be non-negative")
    if args.max_busy < 0 or args.max_used_gib < 0 or args.min_free_gib < 0:
        parser.error("thresholds must be non-negative")
    return args


def import_amdsmi():
    try:
        import amdsmi

        return amdsmi
    except ImportError:
        for candidate in (
            Path("/usr/lib/python3/dist-packages"),
            Path(
                f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages"
            ),
            Path("/opt/rocm/libexec/amdsmi_cli"),
        ):
            if candidate.is_dir() and str(candidate) not in sys.path:
                sys.path.append(str(candidate))
        import amdsmi

        return amdsmi


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
            activity = amdsmi.amdsmi_get_gpu_activity(gpu["handle"])
            gfx = activity.get("gfx_activity")
            umc = activity.get("umc_activity")
            gfx = gfx if isinstance(gfx, int) else 0
            umc = umc if isinstance(umc, int) else 0
            gpu["gfx"].append(gfx)
            gpu["umc"].append(umc)
            busy += int(gfx > 5)
        peak_concurrent = max(peak_concurrent, busy)
    for gpu in gpus:
        memory = amdsmi.amdsmi_get_gpu_vram_usage(gpu["handle"])
        used = memory["vram_used"] / 1024
        total = memory["vram_total"] / 1024
        gpu.update(
            {
                "used_gib": used,
                "free_gib": total - used,
                "peak_gfx": max(gpu["gfx"]),
                "mean_gfx": sum(gpu["gfx"]) / len(gpu["gfx"]),
                "peak_umc": max(gpu["umc"]),
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
        and gpu["peak_gfx"] <= args.max_busy
        and gpu["used_gib"] <= args.max_used_gib
        and gpu["free_gib"] >= args.min_free_gib
    ]
    eligible.sort(
        key=lambda gpu: (
            gpu["peak_gfx"],
            gpu["mean_gfx"],
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
            elif gpu["peak_gfx"] > args.max_busy:
                verdict = f"BUSY peaked {gpu['peak_gfx']}%"
            elif gpu["used_gib"] > args.max_used_gib:
                verdict = f"HELD {gpu['used_gib']:.1f} GiB used"
            elif gpu["free_gib"] < args.min_free_gib:
                verdict = f"FULL {gpu['free_gib']:.1f} GiB free"
            else:
                verdict = "idle"
            hip_index = "-" if gpu["hip_index"] is None else gpu["hip_index"]
            print(
                f"{gpu['smi_index']:>4} {hip_index:>4} {gpu['bdf']:<14} "
                f"{gpu['peak_gfx']:>6} {gpu['mean_gfx']:>6.1f} "
                f"{gpu['peak_umc']:>5} {gpu['used_gib']:>6.1f} GiB "
                f"{gpu['free_gib']:>6.1f} GiB  {verdict}",
                file=sys.stderr,
            )
        if gpus and peak_concurrent >= len(gpus) - 1:
            print(
                f"WARNING: {peak_concurrent}/{len(gpus)} GPUs were busy together; "
                "shared fabric/power may perturb the run.",
                file=sys.stderr,
            )

    if not eligible:
        print(
            "No GPU stayed below the activity and resident-memory thresholds.",
            file=sys.stderr,
        )
        return 1
    selected = eligible[0]
    if not args.quiet:
        print(
            f"Chose HIP index {selected['hip_index']} "
            f"(amd-smi {selected['smi_index']}, {selected['bdf']}).",
            file=sys.stderr,
        )
    print(selected["hip_index"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
