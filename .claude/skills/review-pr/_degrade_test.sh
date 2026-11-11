#!/usr/bin/env bash
# Drive the REAL consumer region of fetch.sh (bytes extracted by line markers, not a replica)
# against evidence that must be rejected, and assert the review survives it.
set -uo pipefail
S="$1"; T=$(mktemp -d); WORK="$T/work"; mkdir -p "$WORK"
START=$(grep -n '^if \[ -n "\$VALIDATION_REPORT" \]; then$' "$S/fetch.sh" | cut -d: -f1)
END=$(awk -v s="$START" 'NR>s && /^fi$/{print NR}' "$S/fetch.sh" | sed -n 2p)
{ echo 'set -euo pipefail'; sed -n "${START},${END}p" "$S/fetch.sh"; } > "$T/region.sh"

printf '{"headRefOid":"HEADSHA","author":{"login":"x"}}' > "$WORK/pr_meta.json"
echo BASESHA > "$WORK/base_head.txt"
printf 'diff --git a/x b/x\n' > "$WORK/pr.diff"
# A report for a DIFFERENT head: the integrity check must refuse it.
printf '{"repo":{"head":"OTHERSHA","base":"BASESHA","patch_sha256":"z"}}' > "$WORK/stale.json"
printf '{"required":true,"families":["runtime"],"blocking_reason":null}' > "$WORK/validation_requirement.json"

# The schema lives under a differently-named validator in each repo; find it rather than
# hard-code a path that is right in one checkout and silently wrong in the other.
SCHEMA=$(find "$S/.." -maxdepth 2 -name report_schema.json | head -1)
export WORK SCHEMA VALIDATION_REPORT="$WORK/stale.json"
out=$(bash "$T/region.sh" 2>&1); rc=$?
fails=0
[ "$rc" -eq 0 ] || { echo "  ❌ region exited $rc — a rejected report still aborts the run"; fails=1; }
[ -f "$WORK/validation_report.json" ] && { echo "  ❌ stale evidence was consumed anyway"; fails=1; }
grep -q "rejected before use" "$WORK/auto_validation_outcome.txt" 2>/dev/null \
  || { echo "  ❌ the rejection reason was not recorded"; fails=1; }
printf '%s\n' "$out" | grep -q "validation REQUIRED but not run" \
  || { echo "  ❌ the no-evidence branch did not run"; fails=1; }
rm -rf "$T"
[ "$fails" -eq 0 ] && echo "ok" || echo "FAILED"
