#!/usr/bin/env bash
# Execute check_citations.py's main() end to end. The 34 pytest cases exercised check();
# main() was never run by anything, so a NameError in it survived review, review of the
# review, and a green gate -- citations.txt carried a traceback and the pipeline read that
# as "nothing flagged". A function no test calls is a function that does not work yet.
set -uo pipefail
S="$1"; T=$(mktemp -d); mkdir -p "$T/head"
printf 'a\nb\nc\n' > "$T/head/real.py"
printf '+++ b/real.py\n' > "$T/pr.diff"
fails=0

printf '## x\n\360\237\224\264 real.py:2 something\n' > "$T/card.md"
out=$(python3 "$S/check_citations.py" "$T" 2>&1)
printf '%s\n' "$out" | grep -q "^CITATIONS: every cited" \
  || { echo "  ❌ a resolvable citation was not accepted -- got: $(printf '%s' "$out" | head -1)"; fails=1; }
printf '%s\n' "$out" | grep -qi "traceback" && { echo "  ❌ main() crashed"; fails=1; }

printf '## x\n\360\237\224\264 ghost.py:9 missing\n' > "$T/card.md"
out=$(python3 "$S/check_citations.py" "$T" 2>&1)
printf '%s\n' "$out" | grep -q "^CITATIONS BAD:" \
  || { echo "  ❌ an unresolvable citation was not flagged -- got: $(printf '%s' "$out" | head -1)"; fails=1; }
printf '%s\n' "$out" | grep -qi "traceback" && { echo "  ❌ main() crashed on the bad path"; fails=1; }

rm -rf "$T"
[ "$fails" -eq 0 ] && echo "ok" || echo "FAILED"
