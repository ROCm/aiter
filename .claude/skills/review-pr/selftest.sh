#!/usr/bin/env bash
# Guard the two run_one.sh behaviours a refactor breaks silently, because neither shows up until
# a review fails at 2am:
#
#   1. Failure triage. Every agent failure used to collapse into `fail glm`, so a review that
#      merely ran out of wall clock paged the model owner about a backend that was answering in
#      under a second. A timeout must route to flow; only a real backend fault routes to glm.
#   2. The refuter-timeout downgrade. The worker's card is finished work; a refuter timeout must
#      publish it rather than discard it -- but only by admitting it on the card, which is what
#      the independent gate enforces. If the card edit drifts, the downgrade would start hiding
#      unrefuted findings behind a normal-looking review line.
#
# Usage: bash .claude/skills/review-pr/selftest.sh
set -uo pipefail
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ok=0; bad=0
t() {  # <name> <got> <want>
  if [ "$2" = "$3" ]; then echo "  ✅ $1"; ok=$((ok + 1))
  else echo "  ❌ $1 — got '$2', want '$3'"; bad=$((bad + 1)); fi
}

echo "[failure triage]"
# Run agent_fail straight out of run_one.sh with fail() stubbed, so the test reads the shipping
# code rather than a copy of it.
route() {  # <rc> -> the failure class agent_fail picked
  { echo 'set -euo pipefail'
    echo 'fail() { echo "$1"; exit 0; }'
    sed -n '/^agent_fail()/,/^}/p' "$S/run_one.sh"
    echo 'agent_fail worker "$1" 2'
  } | bash -s -- "$1"
}
t "a timeout routes to flow" "$(route 124)" "flow"
t "a backend fault routes to glm" "$(route 1)" "glm"

echo "[run budget]"
# AGENT_TIMEOUT x RETRIES x (worker+refuter) can exceed the workflow's timeout-minutes, and a run
# killed at the job cap dies with no classified status at all. run_agent must refuse to start an
# attempt that cannot fit, and must report it as the timeout it is.
budget_guard() {
  { echo 'set -uo pipefail'
    echo 'say() { :; }'
    echo 'PROJ=/tmp'
    echo 'DEADLINE=0'                  # the whole budget is already spent
    echo 'AITER_AGENT_TIMEOUT=2400'
    sed -n '/^run_agent()/,/^}/p' "$S/run_one.sh"
    echo 'run_agent probe /dev/null /dev/null true; echo $?'
  } | bash
}
t "an attempt that cannot fit the budget reports a timeout" "$(budget_guard)" "124"

echo "[refuter-timeout downgrade]"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
printf 'NONE AVAILABLE -- the refuter did not finish within 2400s on this box\n' > "$tmp/independent.txt"

# The downgrade applies exactly the sed run_one.sh uses.
printf 'Review (advisory): \342\232\240 NEEDS WORK\n\342\232\240 a reported finding\n' > "$tmp/card.md"
sed -i '0,/^Review (advisory):/s//Review (advisory, not independently refuted):/' "$tmp/card.md"
rc=0; python3 "$S/triage.py" independent "$tmp/independent.txt" "$tmp/card.md" >/dev/null 2>&1 || rc=$?
t "downgraded card passes the independent gate" "$rc" "0"

# And the guard still bites: an unavailable refutation the card does not admit must stay red.
printf 'Review (advisory): \342\232\240 NEEDS WORK\n\342\232\240 a reported finding\n' > "$tmp/card.md"
rc=0; python3 "$S/triage.py" independent "$tmp/independent.txt" "$tmp/card.md" >/dev/null 2>&1 || rc=$?
t "a silent downgrade is rejected" "$rc" "1"

echo "=== $ok green / $bad red ==="
[ "$bad" -eq 0 ]
