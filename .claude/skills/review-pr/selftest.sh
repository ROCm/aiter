#!/usr/bin/env bash
# Guard the bot behaviours a refactor breaks silently, because none of them show up until a
# review fails at 2am:
#
#   1. Failure triage. Every agent failure used to collapse into `fail glm`, so a review that
#      merely ran out of wall clock paged the model owner about a backend that was answering in
#      under a second. A timeout must route to flow; only a real backend fault routes to glm.
#   2. The refuter-timeout downgrade. The worker's card is finished work; a refuter timeout must
#      publish it rather than discard it -- but only by admitting it on the card, which is what
#      the independent gate enforces. If the card edit drifts, the downgrade would start hiding
#      unrefuted findings behind a normal-looking review line.
#   3. The watchdog's alarm and, just as importantly, its two silences -- see [watchdog] below.
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

# The line above only proves agent_fail routes a 124 correctly. It says nothing about whether
# run_agent still PRODUCES a 124 on a timeout -- and if that regresses to a plain 1, every
# timeout silently becomes a backend fault again while these checks stay green. Drive a real
# timeout through run_agent instead of trusting its return path.
real_timeout() {
  local out; out=$(mktemp)
  { echo 'set -uo pipefail'
    echo 'say() { :; }'
    echo 'PROJ=/tmp'
    echo 'DEADLINE=$(( $(date +%s) + 3600 ))'    # plenty of budget; the agent itself times out
    echo 'AITER_AGENT_TIMEOUT=1'
    sed -n '/^run_agent()/,/^}/p' "$S/run_one.sh"
    printf 'run_agent probe /dev/null %s sh -c "sleep 5"; echo $?\n' "$out"
  } | bash
  rm -f "$out"
}
t "run_agent itself reports 124 on a real timeout" "$(real_timeout)" "124"

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

echo "[watchdog]"
# The watchdog is the one guard that cannot be tested by running it: `issue_comment` workflows
# only ever run from the default branch, so GitHub will not execute it until this merges. Drive
# the shipping script against a faked API and clock instead. It needs node, which the hosted
# runner it actually runs on has and the self-hosted box does not -- so say so out loud rather
# than counting an absent check as a passing one.
if command -v node >/dev/null 2>&1; then
  wd=$(node "$S/watchdog_test.js" 2>&1); wd_rc=$?
  printf '%s\n' "$wd" | grep -E '^  (✅|❌|⚠️)' || true
  wok=$(printf '%s\n' "$wd" | grep -c '✅' || true)
  wbad=$(printf '%s\n' "$wd" | grep -c '❌' || true)
  ok=$((ok + wok)); bad=$((bad + wbad))
  if [ "$wd_rc" -ne 0 ] && [ "$wbad" -eq 0 ]; then
    why=$(printf '%s\n' "$wd" | grep -m1 -E 'Error|MODULE_NOT_FOUND|No such file' | sed 's/^ *//' | cut -c1-110)
    echo "  ❌ watchdog_test.js did not run — ${why:-exit $wd_rc}"; bad=$((bad + 1))
  fi
else
  echo "  ⚠️  node not installed here — watchdog checks did NOT run"
fi

echo "=== $ok green / $bad red ==="
[ "$bad" -eq 0 ]
