"""Write the fastest config per M from sweep logs into a GEMM config file.

    python3 write_best_configs.py <op> <DIM>=<int>... [--arch A] [--backend B]
                                  [--install]

Reads sweeps/<op>/<arch>-<backend>/<dims>/M=*.jsonl. Name and directory are
the ones get_gemm_config() looks up for the shape, taken from the lookup the
sweep recorded: the config family, the N/K (or B/N/K, or custom) suffix, the
arch and the backend. Each tuned M goes in the smallest of the lookup's M
bounds that covers it, and the largest tuned M becomes "any". Without
--install the file lands under tuned_configs/, mirroring the config tree; with
--install, in aiter/ops/triton/configs/ itself.
"""

import argparse
import json
import sys
from pathlib import Path

from _utils import (
    CONFIGS_ROOT,
    SWEEPS_DIR,
    TUNED_DIR,
    get_case,
    parse_dims,
    read_jsonl,
    shape_tag,
    tuned_filename,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("op", help="tuning case from gemm_cases.py, e.g. gemm_a8w8")
    parser.add_argument(
        "dims",
        nargs="+",
        metavar="DIM=INT",
        help="the op's dims besides M: N=7168 K=2048",
    )
    parser.add_argument("--arch", help="use the logs swept on this arch, e.g. gfx950")
    parser.add_argument(
        "--backend", choices=("triton", "gluon"), help="use the logs of this backend"
    )
    parser.add_argument(
        "--no-any",
        action="store_true",
        help='keep the largest M as M_LEQ_<bound> instead of making it "any"',
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="write into aiter/ops/triton/configs/ instead of tuned_configs/",
    )
    return parser.parse_args()


def find_logs(op, dims, arch, backend):
    tag = shape_tag(op, dims)
    dirs = [d for d in sorted((SWEEPS_DIR / op).glob(f"*/{tag}")) if d.is_dir()]
    if arch:
        dirs = [d for d in dirs if d.parent.name.startswith(f"{arch}-")]
    if backend:
        dirs = [d for d in dirs if d.parent.name.endswith(f"-{backend}")]
    if not dirs:
        sys.exit(f"no sweep logs match {SWEEPS_DIR / op}/<arch>-<backend>/{tag}")
    if len(dirs) > 1:
        found = ", ".join(d.parent.name for d in dirs)
        sys.exit(f"logs for {found}; pick one with --arch / --backend")
    return sorted(dirs[0].glob("M=*.jsonl"), key=lambda p: int(p.stem[2:]))


def buckets_for(best, bounds, last_any):
    """Bucket each tuned M's config under the smallest bound that covers it (the
    larger M wins a shared bucket); the largest M's bucket becomes "any"."""
    buckets, owner, notes = {}, {}, []
    top = max(best)
    for m in sorted(best):
        bound = next((b for b in bounds if b >= m), None)
        if bound is None:
            if not (last_any and m == top):
                notes.append(f'M={m} is above every M bound; only "any" can cover it')
            continue
        key = f"M_LEQ_{bound}"
        if key in owner:
            notes.append(f"M={owner[key]} and M={m} both fall in {key}; kept M={m}")
        buckets[key], owner[key] = best[m], m
    if last_any:
        for key, m in owner.items():
            if m == top:
                del buckets[key]
                break
        buckets["any"] = best[top]
    return buckets, notes


def main():
    args = parse_args()
    dims = parse_dims(args.dims)
    get_case(args.op, dims, None)

    first, best, rows = None, {}, []
    for path in find_logs(args.op, dims, args.arch, args.backend):
        header, *results = read_jsonl(path)
        if first is None:
            first = header
        elif (header["lookup"], header["arch"]) != (first["lookup"], first["arch"]):
            sys.exit(
                f"{path} was swept for a different config lookup than M={first['M']}"
            )
        ran = [r for r in results if "us" in r]
        if not ran:
            print(f"M={header['M']}: no config ran; skipped")
            continue
        winner = min(ran, key=lambda r: r["us"])
        best[header["M"]] = winner["config"]
        rows.append(
            (header["M"], winner["us"], header["baseline_us"], winner["config"])
        )
    if not best:
        sys.exit("no results to write")

    lookup, family = first["lookup"], first["family"]
    name = tuned_filename(lookup)
    if name is None:
        sys.exit(f"{args.op} looks up its config without N and K; nothing to name")
    buckets, notes = buckets_for(best, family["bounds"], not args.no_any)
    keys = family["keys"]
    content = {
        bucket: {key: config[key] for key in keys if key in config}
        for bucket, config in buckets.items()
    }

    tree_dir = Path(family["config_dir"]).relative_to(family["configs_root"])
    out_dir = (CONFIGS_ROOT if args.install else TUNED_DIR) / tree_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / name
    existed = out.exists()
    out.write_text(json.dumps(content, indent=4) + "\n")

    print("M\tbest (us)\tcurrent (us)\tconfig")
    for m, us, baseline, config in rows:
        current = f"{baseline:.3f}" if baseline is not None else "N/A"
        print(f"{m}\t{us:.3f}\t\t{current}\t\t{config}")
    for note in notes:
        print(f"Note: {note}")
    if "any" not in content:
        print('Warning: no "any" bucket; M above the last bucket will raise KeyError')
    print(f"{'Replaced' if existed else 'Wrote'} {out}")
    if not args.install:
        print(f"Install it with --install, or copy it into {CONFIGS_ROOT / tree_dir}")


if __name__ == "__main__":
    main()
