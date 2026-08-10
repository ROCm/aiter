#!/bin/bash
# ISA A/B for a FlyDSL migration, with no GPU.
#   ./isa_ab.sh <git-ref> <path...> -- <compile_gate case>...
# Compiles the given cases with <path...> at <git-ref> (the "before"), then at
# the working-tree state ("after"), and diffs the final ISA.
set -u
REF="$1"; shift
PATHS=(); while [ "$1" != "--" ]; do PATHS+=("$1"); shift; done; shift
CASES=("$@")

run() {  # run <outdir>
  rm -rf "$1" ~/.flydsl/cache; mkdir -p "$1"
  PYTHONPATH=/tmp/envstub FLYDSL_GPU_ARCH=gfx950 COMPILE_ONLY=1 \
  FLYDSL_DEBUG_DUMP_ASM=1 FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR="$1" \
  timeout 1800 /opt/venv/bin/python compile_gate.py "${CASES[@]}" 2>/dev/null \
    | grep -E '^\[|^[0-9a-f]{16} '
}

# An empty path list swaps nothing, so the "before" and "after" runs compile the
# same tree and the verdict reads IDENTICAL while proving nothing. Same class of
# false pass as an empty hash list -- refuse it here too.
if [ "${#PATHS[@]}" -eq 0 ]; then
  echo "INVALID - no files to swap; the A/B would compare the tree to itself" >&2
  exit 2
fi
echo "swapping ${#PATHS[@]} file(s) at $REF"
TMP=$(mktemp -d)
for p in "${PATHS[@]}"; do mkdir -p "$TMP/$(dirname "$p")"; cp "$p" "$TMP/$p"; done

# This script overwrites working-tree files with their $REF content, so the
# restore MUST run on every exit path. Without the trap, a `git show` failure
# (e.g. a path that does not exist at $REF) left the tree holding $REF content
# and silently discarded uncommitted work.
restore() { for p in "${PATHS[@]}"; do [ -f "$TMP/$p" ] && cp "$TMP/$p" "$p"; done; }
trap 'restore; rm -rf "$TMP"' EXIT

for p in "${PATHS[@]}"; do
  git show "$REF:$p" > "$p" || { echo "INVALID - $p does not exist at $REF" >&2; exit 2; }
done
echo "=== BEFORE ($REF) ==="; run /tmp/isa_before > /tmp/ab_before.txt; cat /tmp/ab_before.txt
restore
echo "=== AFTER (working tree) ==="; run /tmp/isa_after > /tmp/ab_after.txt; cat /tmp/ab_after.txt

echo "=== VERDICT ==="
# A case that failed to compile, or produced no ISA at all, must never read as a
# pass: comparing two empty hash lists would otherwise report IDENTICAL.
if grep -q '^\[FAIL\]' /tmp/ab_before.txt /tmp/ab_after.txt; then
  echo "INVALID - a case failed to compile; fix the harness before trusting a verdict"
  grep -h '^\[FAIL\]' /tmp/ab_before.txt /tmp/ab_after.txt
  exit 2
fi
n_before=$(grep -cE '^[0-9a-f]{16} ' /tmp/ab_before.txt)
n_after=$(grep -cE '^[0-9a-f]{16} ' /tmp/ab_after.txt)
if [ "$n_before" -eq 0 ] || [ "$n_after" -eq 0 ]; then
  echo "INVALID - no ISA produced (before=$n_before after=$n_after)"; exit 2
fi
if diff -q <(grep -E '^[0-9a-f]{16} ' /tmp/ab_before.txt) \
           <(grep -E '^[0-9a-f]{16} ' /tmp/ab_after.txt) >/dev/null; then
  echo "ISA IDENTICAL across $n_after kernel(s) - provably behaviour- and perf-neutral"
else
  echo "ISA DIFFERS:"; diff /tmp/ab_before.txt /tmp/ab_after.txt
fi
