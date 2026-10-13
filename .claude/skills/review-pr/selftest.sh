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

defaults() {  # emit run_one.sh's own default assignments, so a harness cannot drift from them
  sed -n '/^: "${AITER_RUN_BUDGET/p; /^: "${AITER_AGENT_TIMEOUT/p' "$S/run_one.sh"
}

echo "[failure triage]"
# Run agent_fail straight out of run_one.sh with fail() stubbed, so the test reads the shipping
# code rather than a copy of it.
route() {  # <rc> -> the failure class agent_fail picked
  { echo 'set -euo pipefail'
    defaults
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
    defaults
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
    defaults
    sed -n '/^run_agent()/,/^}/p' "$S/run_one.sh"
    echo 'run_agent probe /dev/null /dev/null true; echo $?'
  } | bash
}
t "an attempt that cannot fit the budget reports a timeout" "$(budget_guard)" "124"

# Out-of-budget and ran-out-of-clock both return 124 so neither gets retried, but they have
# opposite fixes. Telling someone to raise AITER_AGENT_TIMEOUT for a budget refusal makes the
# refusal fire sooner -- the message must name the knob that actually helps.
fail_message() {  # <DEADLINE> <AITER_AGENT_TIMEOUT> <cmd...> -> the text the owner is paged with
  { echo 'set -uo pipefail'
    echo 'say() { :; }'; echo 'PROJ=/tmp'
    printf 'DEADLINE=%s\n' "$1"; printf 'AITER_AGENT_TIMEOUT=%s\n' "$2"; shift 2
    defaults
    echo 'fail() { echo "$3"; exit 0; }'
    sed -n '/^run_agent()/,/^}/p' "$S/run_one.sh"
    sed -n '/^agent_fail()/,/^}/p' "$S/run_one.sh"
    printf 'rc=0; run_agent probe /dev/null /tmp/_st_out %s || rc=$?\n' "$*"
    echo 'agent_fail probe "$rc" 2'
  } | bash
}
budget_msg=$(fail_message 0 2400 true)
clock_msg=$(fail_message "$(( $(date +%s) + 3600 ))" 1 sh -c '"sleep 5"')
t "a budget refusal names the run budget" \
  "$(printf '%s' "$budget_msg" | grep -c AITER_RUN_BUDGET)" "1"
t "a budget refusal does not blame the agent timeout" \
  "$(printf '%s' "$budget_msg" | grep -c 'raise it in the runner .env')" "0"
t "a real timeout still names the agent timeout" \
  "$(printf '%s' "$clock_msg" | grep -c AITER_AGENT_TIMEOUT)" "1"

echo "[refuter call site]"
# The harnesses above run run_agent in isolation under `set -uo pipefail`. The shipping script
# runs under `set -euo pipefail`, where a bare `cmd; rc=$?` exits before rc is ever read -- so
# proving the function returns 124 says nothing about whether the caller survives to act on it.
# Drive the call site itself, errexit on, exactly as run_one.sh has it.
refuter_site() {  # <rc run_agent returns> [card's first line] -> what the call site actually did
  local W; W=$(mktemp -d)
  printf '%s\n\342\232\240 a finding\n' "${2:-Review (advisory): NEEDS WORK}" > "$W/card.md"
  { echo 'set -euo pipefail'
    echo 'say() { :; }'
    echo 'bash() { :; }'                       # stub out render.sh
    echo 'agent_fail() { echo "agent_fail $1 $2"; exit 0; }'
    printf 'run_agent() { return %s; }\n' "$1"
    printf 'SKILL=%s\nW=%s\n' "$S" "$W"
    echo 'REFUTER_CMD=(true)'
    defaults
    sed -n '/^  bash "\$SKILL\/render.sh" refuter/,/^  fi$/p' "$S/run_one.sh"
    echo 'grep -q "not independently refuted" "$W/card.md" && echo downgraded || echo no-downgrade'
  } | bash
  rm -rf "$W"
}
t "a refuter timeout reaches the downgrade" "$(refuter_site 124)" "downgraded"
# The card is model-generated and the downgrade is a sed anchored to one literal shape. If the
# worker bolds or indents that line the sed matches nothing, exits 0, and the independent gate
# then discards a card that was known-good -- the one case the downgrade exists to rescue.
t "a card the sed cannot match is still downgraded" \
  "$(refuter_site 124 '**Review (advisory):** NEEDS WORK')" "downgraded"
t "a refuter fault reaches the triage" "$(refuter_site 1)" "agent_fail refuter 1"

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
# only ever run from the default branch, so GitHub will not execute that job until this merges.
# Drive the shipping script against a fake API and a fake clock instead.
wd=$(python3 "$S/watchdog_test.py" 2>&1); wd_rc=$?
printf '%s
' "$wd" | grep -E '^  (✅|❌)' || true
wok=$(printf '%s
' "$wd" | grep -c '✅' || true)
wbad=$(printf '%s
' "$wd" | grep -c '❌' || true)
ok=$((ok + wok)); bad=$((bad + wbad))
if [ "$wd_rc" -ne 0 ] && [ "$wbad" -eq 0 ]; then
  why=$(printf '%s
' "$wd" | grep -m1 -E 'Error|Traceback|No such file' | sed 's/^ *//' | cut -c1-110)
  echo "  ❌ watchdog_test.py did not run — ${why:-exit $wd_rc}"; bad=$((bad + 1))
fi

echo "[queue gate]"
# Same reason as the watchdog: the gate only runs from the default branch, so GitHub will not
# execute this until it merges. Drive the shipping script against a fake API and a fixed clock.
qg=$(python3 "$S/queue_gate_test.py" 2>&1); qg_rc=$?
printf '%s\n' "$qg" | grep -E '^  (✅|❌)' || true
qok=$(printf '%s\n' "$qg" | grep -c '✅' || true)
qbad=$(printf '%s\n' "$qg" | grep -c '❌' || true)
ok=$((ok + qok)); bad=$((bad + qbad))
if [ "$qg_rc" -ne 0 ] && [ "$qbad" -eq 0 ]; then
  why=$(printf '%s\n' "$qg" | grep -m1 -E 'Error|Traceback|No such file' | sed 's/^ *//' | cut -c1-110)
  echo "  ❌ queue_gate_test.py did not run — ${why:-exit $qg_rc}"; bad=$((bad + 1))
fi

echo "[notify handoff]"
# The report job stays quiet when _notify.py says it already spoke, so that flag must mean a
# comment actually landed. _notify.py returns 0 without posting in three cases -- no status
# file, no token, a POST that threw -- and claiming "notified" in any of them would trade a
# visible failure for a silent one. Two of the three are testable without a network.
notified_flag() {  # <status file contents or empty> -> what _notify.py wrote to GITHUB_OUTPUT
  local w o; w=$(mktemp -d); o=$(mktemp)
  [ -n "$1" ] && printf '%s\n' "$1" > "$w/.aiter-review-status"
  ( cd "$w" && GITHUB_WORKSPACE="$w" GITHUB_OUTPUT="$o" AITER_BOT_TOKEN= GITHUB_REPOSITORY= \
      python3 "$S/_notify.py" 42 >/dev/null 2>&1 )
  local n; n=$(grep -c notified "$o" 2>/dev/null || true)   # grep -c exits 1 on zero matches
  echo "${n:-0}"
  rm -rf "$w" "$o"
}
t "no status to report claims nothing" "$(notified_flag '')" "0"
t "a report it could not send claims nothing" "$(notified_flag 'flow\tsomething broke')" "0"

echo "[lost review]"
# The other end of the watchdog: a review that started and then vanished. Same reason it cannot
# be exercised here -- the job only runs from the default branch.
lr=$(python3 "$S/lost_review_test.py" 2>&1); lr_rc=$?
printf '%s\n' "$lr" | grep -E '^  (✅|❌)' || true
lok=$(printf '%s\n' "$lr" | grep -c '✅' || true)
lbad=$(printf '%s\n' "$lr" | grep -c '❌' || true)
ok=$((ok + lok)); bad=$((bad + lbad))
if [ "$lr_rc" -ne 0 ] && [ "$lbad" -eq 0 ]; then
  why=$(printf '%s\n' "$lr" | grep -m1 -E 'Error|Traceback|No such file' | sed 's/^ *//' | cut -c1-110)
  echo "  ❌ lost_review_test.py did not run — ${why:-exit $lr_rc}"; bad=$((bad + 1))
fi

echo "=== $ok green / $bad red ==="
[ "$bad" -eq 0 ]
