import argparse
import json
import math
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare baseline kernel timings against new timings and generate a regression report."
    )
    parser.add_argument(
        "--baseline", required=True, type=str, help="Path to baseline JSON file"
    )
    parser.add_argument(
        "--new", required=True, type=str, help="Path to new JSON file"
    )
    parser.add_argument(
        "--output", required=True, type=str, help="Path to output report file"
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=0.03,
        help="Fractional threshold above which a slowdown is flagged as REGRESSION (default: 0.03)",
    )
    return parser.parse_args()


def compute_geomean(ratios):
    """Compute the geometric mean of a list of positive ratios."""
    if not ratios:
        return 1.0
    log_sum = sum(math.log(r) for r in ratios)
    return math.exp(log_sum / len(ratios))


def compare_kernel(baseline_shapes, new_shapes, threshold):
    """Compare timings for a single kernel across all shared shapes.

    Returns:
        rows: list of (shape, baseline_ns, new_ns, delta, status) tuples
        geomean: geometric mean of (baseline_time / new_time) across shapes
        warnings: list of warning strings
    """
    rows = []
    ratios = []
    warnings = []

    all_shapes = sorted(set(baseline_shapes.keys()) | set(new_shapes.keys()))

    for shape in all_shapes:
        if shape not in new_shapes:
            warnings.append(f"  WARNING: shape {shape} in baseline but not in new, skipping")
            continue
        if shape not in baseline_shapes:
            warnings.append(f"  WARNING: shape {shape} in new but not in baseline, skipping")
            continue

        baseline_ns = baseline_shapes[shape].get("time_ns", 0)
        new_ns = new_shapes[shape].get("time_ns", 0)

        if baseline_ns == 0:
            warnings.append(f"  WARNING: shape {shape} has zero baseline time, skipping")
            continue
        if new_ns == 0:
            warnings.append(f"  WARNING: shape {shape} has zero new time, skipping")
            continue

        delta = (new_ns - baseline_ns) / baseline_ns
        status = "REGRESSION" if delta > threshold else "OK"
        rows.append((shape, baseline_ns, new_ns, delta, status))
        ratios.append(baseline_ns / new_ns)

    geomean = compute_geomean(ratios)
    return rows, geomean, warnings


def format_report(baseline_data, new_data, threshold):
    """Build the full comparison report.

    Returns:
        report: the report as a string
        all_pass: True if every kernel passes
    """
    lines = []
    summary_lines = []
    all_pass = True

    all_kernels = sorted(set(baseline_data.keys()) | set(new_data.keys()))
    warnings_global = []

    for kernel in all_kernels:
        if kernel not in new_data:
            warnings_global.append(f"WARNING: kernel {kernel} in baseline but not in new, skipping")
            continue
        if kernel not in baseline_data:
            warnings_global.append(f"WARNING: kernel {kernel} in new but not in baseline, skipping")
            continue

        rows, geomean, warnings = compare_kernel(
            baseline_data[kernel], new_data[kernel], threshold
        )

        passed = geomean >= 1.0
        if not passed:
            all_pass = False

        pass_label = "PASS" if passed else "FAIL"

        lines.append(f"=== {kernel} ===")
        for w in warnings:
            lines.append(w)

        header = f"{'Shape':<20}{'Baseline(ns)':>14}{'New(ns)':>14}{'Delta':>10}{'Status':>14}"
        lines.append(header)
        for shape, b_ns, n_ns, delta, status in rows:
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta * 100:.1f}%"
            lines.append(
                f"{shape:<20}{b_ns:>14}{n_ns:>14}{delta_str:>10}{status:>14}"
            )

        lines.append(f"Geomean speedup: {geomean:.2f}x ({pass_label})")
        lines.append("")

        summary_lines.append(f"{kernel}: {pass_label} (geomean {geomean:.2f}x)")

    if warnings_global:
        for w in warnings_global:
            lines.insert(0, w)
        lines.insert(len(warnings_global), "")

    lines.append("=== SUMMARY ===")
    for s in summary_lines:
        lines.append(s)
    lines.append(f"Overall: {'PASS' if all_pass else 'FAIL'}")

    report = "\n".join(lines)
    return report, all_pass


def main():
    args = parse_args()

    with open(args.baseline, "r") as f:
        baseline_data = json.load(f)

    with open(args.new, "r") as f:
        new_data = json.load(f)

    report, all_pass = format_report(baseline_data, new_data, args.regression_threshold)

    print(report)

    with open(args.output, "w") as f:
        f.write(report)
        f.write("\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
