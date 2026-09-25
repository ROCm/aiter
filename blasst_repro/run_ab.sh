#!/bin/bash
# Patched vs unpatched Triton, side by side, in ONE container.
#
# The chain-dot patch is opt-in via TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF, so a
# single image produces both arms: run the benchmark twice and diff.
#
# HONEST CAVEAT. The FORCE=0 arm runs the PATCHED build with the patch inert,
# not a stock Triton build. isSameOrAcrossOneIfLevel() returns false immediately
# when the env var is unset, so codegen matches stock -- but if you need a
# literal stock comparison, build the base image without the patch and run the
# benchmark there instead.
#
#   ./run_ab.sh [extra args passed through to the benchmark]
#   ./run_ab.sh --shapes 32x8 --seqlens 16384
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-${HERE}/results}"
mkdir -p "$OUT"

# Separate Triton caches per arm.
for force in 0 1; do
  echo "############ TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF=$force ############"
  TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF=$force \
  TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}_force${force}" \
  python3 "${HERE}/bench_blasst_ua2d.py" \
      --csv "${OUT}/bench_force${force}.csv" "$@" \
    | tee "${OUT}/bench_force${force}.log"
  echo
done

echo "############ SIDE BY SIDE ############"
python3 - "$OUT/bench_force0.csv" "$OUT/bench_force1.csv" <<'PY'
import csv, sys

def load(p):
    with open(p) as fh:
        return {(r["case"], r["threshold"]): r for r in csv.DictReader(fh)}

stock, patched = load(sys.argv[1]), load(sys.argv[2])
keys = [k for k in patched if k in stock]
if not keys:
    print("no overlapping rows -- did the two arms run the same shapes?")
    raise SystemExit(1)

print(f"{'case':<26} {'lambda':>9} {'elide':>7} "
      f"{'stock ms':>9} {'patched':>9} {'patch worth':>12}")
graded = []  # (threshold, elide or None, patch_worth)
for case, thr in keys:
    s, p = stock[(case, thr)], patched[(case, thr)]
    worth = float(s["ms"]) / float(p["ms"])
    raw = p["elide"]
    el = float(raw) if raw not in ("", None) else None
    print(f"{case:<26} {thr:>9} "
          f"{'      -' if el is None else f'{100 * el:6.1f}%'} "
          f"{float(s['ms']):9.3f} {float(p['ms']):9.3f} {worth:11.4f}x")
    graded.append((float(thr), el, worth))

# Split on MEASURED elision, not on the threshold value. A low threshold can
# elide nothing at all, so bucketing by lambda would file genuine overhead-floor
# points under "skipping active" and overstate the gap.
have_elide = all(e is not None for _, e, _ in graded)
if have_elide:
    floor = [g for _, e, g in graded if e < 0.01]
    live = [g for _, e, g in graded if e >= 0.01]
    basis = "measured elision (<1% vs >=1% of tiles)"
else:
    floor = [g for t, _, g in graded if t <= 1e-6]
    live = [g for t, _, g in graded if t > 1e-6]
    basis = ("threshold value -- RERUN WITHOUT --skip-elision for an honest "
             "split; a low lambda that elides nothing lands in the wrong "
             "bucket here")
print()
print(f"  split by: {basis}")
if floor:
    print(f"  no elision      : patch worth {sum(floor)/len(floor):.4f}x  "
          f"({len(floor)} pts) -- this is the skip-CHECK overhead only")
if live:
    print(f"  elision active  : patch worth {sum(live)/len(live):.4f}x  "
          f"({len(live)} pts) -- the conditional dot is being taken")
print("  dense is unaffected by the patch -- the conditional folds away when")
print("  block skipping is off, so this gap applies only to the feature.")
PY
