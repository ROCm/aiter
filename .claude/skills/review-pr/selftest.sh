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

echo "[gates call site]"
# Same bug as the refuter had: `python3 _gates.py ...` bare under `set -euo pipefail` exits the
# script on a red gate, so the EXIT trap reports "exited unexpectedly with no classified
# failure" -- while the log says plainly which gate went red. A reader gets the useless half.
gates_site() {  # <rc from _gates.py> -> what the call site reported
  local W; W=$(mktemp -d)
  # the fixture must not be the file the code tees to: tee truncates it on open, and the
  # stub's cat then races against that -- a test that passes or fails by timing is worse than none
  printf '======== SEVEN GATES ========\n  \342\234\205 answers\n  \342\235\214 card         exit=1\n' > "$W/fixture.out"
  { echo 'set -euo pipefail'
    echo 'say() { :; }'
    echo 'fail() { echo "$1 | $3"; exit 0; }'
    printf 'W=%s\nSKILL=%s\nPROJ=/tmp\n' "$W" "$S"
    printf 'python3() { cat "%s/fixture.out"; return %s; }\n' "$W" "$1"
    sed -n '/^# 4) gates + collect/,/^python3 "\$SKILL\/_collect.py"/p' "$S/run_one.sh"
    echo 'echo "REACHED-END"'
  } | bash
  rm -rf "$W"
}
t "a red gate is reported as a classified failure" \
  "$(gates_site 1 | grep ' | ' | cut -d'|' -f1 | tr -d ' ')" "flow"
t "the report names which gate went red" \
  "$(gates_site 1 | grep ' | ' | grep -c card)" "1"
t "green gates fall through to collect" "$(gates_site 0 | tail -1)" "REACHED-END"

# A gate that prints more than one line carrying `exit=` must not be named twice: a reader
# shown "card,card,ledger" learns less than one shown "card,ledger".
gates_multi() {
  local W; W=$(mktemp -d)
  printf '   \342\235\214 card    exit=1\n   \342\235\214 card    exit=1\n   \342\235\214 ledger  exit=1\n' > "$W/fixture.out"
  { echo 'set -euo pipefail'
    echo 'say() { :; }'
    echo 'fail() { echo "$3"; exit 0; }'
    printf 'W=%s\nSKILL=%s\nPROJ=/tmp\n' "$W" "$S"
    printf 'python3() { cat "%s/fixture.out"; return 1; }\n' "$W"
    sed -n '/^# 4) gates + collect/,/^python3 "\$SKILL\/_collect.py"/p' "$S/run_one.sh"
  } | bash | grep ' gate'
  rm -rf "$W"
}
t "a gate red on several lines is named once" \
  "$(gates_multi | grep -oE 'by the [a-z,]+ gate' | head -1)" "by the card,ledger gate"

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

echo "[design gate]"
# Step 7.8's gate, driven like the independent gate above. The routing is the checkable
# half -- the person is on the roster and not the author, the size is the PR's real size,
# an over-bar PR carries a split -- and a routing gate that cannot go red is a rubber stamp,
# so each rule gets its negative case here.
design_gate() {  # <design.txt> <card.md> <reviewers.txt> <pr_meta.json> -> problem kinds, or "routed"
  local d out rc; d=$(mktemp -d)
  printf '%s\n' "$1" > "$d/design.txt"
  printf '%s\n' "$2" > "$d/card.md"
  printf '%s\n' "$3" > "$d/reviewers.txt"
  printf '%s' "$4" > "$d/pr_meta.json"
  out=$(python3 "$S/triage.py" design "$d/design.txt" "$d/card.md" \
        "$d/reviewers.txt" "$d/pr_meta.json" 2>/dev/null); rc=$?
  rm -rf "$d"
  if [ "$rc" -eq 0 ]; then echo routed
  else printf '%s\n' "$out" | grep -E '^[A-Z][A-Z -]+:' | sed 's/:.*//' | paste -sd, -; fi
}
dg_roster='# roster
bob (git-log, aiter/fused_moe.py)
carol
alice'
dg_meta_small='{"title":"fix the moe gate","author":{"login":"alice"},"files":[{"path":"aiter/fused_moe.py","additions":20,"deletions":5}]}'
dg_meta_wide='{"title":"fix the moe gate","author":{"login":"alice"},"files":[{"path":"aiter/fused_moe.py","additions":20,"deletions":5},{"path":"aiter/mla.py","additions":3,"deletions":1}]}'
dg_meta_big='{"title":"fix the moe gate","author":{"login":"alice"},"files":[{"path":"aiter/fused_moe.py","additions":1200,"deletions":0}]}'
dg_design='DESIGN: route the expert through the per-expert gate rather than the shared one
HUMAN: recommend bob review whether the per-expert gate in fused_moe.py holds at TP=8'
dg_card='Review (advisory): NEEDS WORK
Size: 25 lines / 1 files
Design (advisory): route the expert through the per-expert gate rather than the shared one
Human: recommend bob review whether the per-expert gate in fused_moe.py holds at TP=8'
t "a routed design passes" "$(design_gate "$dg_design" "$dg_card" "$dg_roster" "$dg_meta_small")" "routed"
# The card's header block spells its separators with an em-dash (Validation/Perf style) while
# design.txt's house style is `--` (refutations.txt style). A worker following the card
# template to the letter wrote `NONE — <reason>` and reded DESIGN-DISAGREES on a card that
# said exactly what design.txt said -- found by dry-running the documented formats, not by
# reading the regex. Pin both spellings.
t "a NONE design written in the card's em-dash style passes" \
  "$(design_gate 'DESIGN: NONE -- a tooling-only change inside the existing skill layout
HUMAN: N/A -- no design decision to route' \
    'Review (advisory): NEEDS WORK
Size: 25 lines / 1 files
Design (advisory): NONE — a tooling-only change inside the existing skill layout
Human: N/A — no design decision to route' \
    "$dg_roster" "$dg_meta_small")" "routed"
t "a card with no Size line is rejected" \
  "$(design_gate "$dg_design" "$(printf '%s\n' "$dg_card" | grep -v '^Size:')" "$dg_roster" "$dg_meta_small")" "UNREPORTED-SIZE"
t "an empty design.txt is a skipped step" \
  "$(design_gate '' "$dg_card" "$dg_roster" "$dg_meta_small")" "DESIGN MISSING"
t "a size the PR does not have is rejected" \
  "$(design_gate "$dg_design" "$dg_card" "$dg_roster" "$dg_meta_wide")" "SIZE-MISMATCH"
t "an off-roster person is rejected" \
  "$(design_gate 'DESIGN: route the expert through the per-expert gate rather than the shared one
HUMAN: recommend dave review whether the per-expert gate in fused_moe.py holds at TP=8' \
    'Review (advisory): NEEDS WORK
Size: 25 lines / 1 files
Design (advisory): route the expert through the per-expert gate rather than the shared one
Human: recommend dave review whether the per-expert gate in fused_moe.py holds at TP=8' \
    "$dg_roster" "$dg_meta_small")" "OFF-ROSTER"
t "routing back to the PR author is rejected" \
  "$(design_gate 'DESIGN: route the expert through the per-expert gate rather than the shared one
HUMAN: recommend alice review whether the per-expert gate in fused_moe.py holds at TP=8' \
    'Review (advisory): NEEDS WORK
Size: 25 lines / 1 files
Design (advisory): route the expert through the per-expert gate rather than the shared one
Human: recommend alice review whether the per-expert gate in fused_moe.py holds at TP=8' \
    "$dg_roster" "$dg_meta_small")" "SELF-ROUTED"
t "a question about nothing in the PR is rejected" \
  "$(design_gate 'DESIGN: route the expert through the per-expert gate rather than the shared one
HUMAN: recommend bob review whether this overall approach makes sense at all' \
    'Review (advisory): NEEDS WORK
Size: 25 lines / 1 files
Design (advisory): route the expert through the per-expert gate rather than the shared one
Human: recommend bob review whether this overall approach makes sense at all' \
    "$dg_roster" "$dg_meta_small")" "UNANCHORED"
dg_split_card='Review (advisory): NEEDS WORK
Size: 1200 lines / 1 files — over the review bar
Design (advisory): SPLIT -- the gate change and the cache layout change are separable stacks
Human: recommend bob review whether the fused_moe.py split keeps the gate change reviewable'
t "an over-bar PR that routes its design question passes" \
  "$(design_gate 'DESIGN: SPLIT -- the gate change and the cache layout change are separable stacks
HUMAN: recommend bob review whether the fused_moe.py split keeps the gate change reviewable' \
    "$dg_split_card" "$dg_roster" "$dg_meta_big")" "routed"
t "an over-bar PR with an unrouted design question is rejected" \
  "$(design_gate 'DESIGN: SPLIT -- the gate change and the cache layout change are separable stacks
HUMAN: N/A -- nobody else has touched these paths' \
    'Review (advisory): NEEDS WORK
Size: 1200 lines / 1 files — over the review bar
Design (advisory): SPLIT -- the gate change and the cache layout change are separable stacks
Human: N/A -- nobody else has touched these paths' \
    "$dg_roster" "$dg_meta_big")" "UNROUTED-DESIGN"
t "an over-bar PR without a split design is rejected" \
  "$(design_gate "$dg_design" 'Review (advisory): NEEDS WORK
Size: 1200 lines / 1 files — over the review bar
Design (advisory): route the expert through the per-expert gate rather than the shared one
Human: recommend bob review whether the per-expert gate in fused_moe.py holds at TP=8' \
    "$dg_roster" "$dg_meta_big")" "UNSPLIT"

echo "[reviewer roster]"
# The roster is the design gate's ground truth, and two defects found by dry-running 10
# over-bar PRs live here, not in the gate: copilot-pull-request-reviewer "reviews" PRs on
# this repo and rode 3 of those 10 rosters (routing a design question to a bot reads as
# routed and lands in a void), and a PR that is all brand-new files has no git-log
# authors, so it routed nowhere exactly when it was newest. Drive the shipping roster
# block against a fake PR and a fake worktree.
roster_of() {  # <pr_meta.json> <worktree> -> the roster fetch.sh would write
  { sed -n '/^  python3 - "\$WORK\/pr_meta.json" "\$TARGET_WT" <<.PYEOF./,/^PYEOF$/p' "$S/fetch.sh" \
      | sed "1s|.*|python3 - \"$1\" \"$2\" <<'PYEOF'|"
  } | bash
}
rr=$(mktemp -d)
git -C "$rr" init -q . && git -C "$rr" config user.email carol@amd.com && git -C "$rr" config user.name Carol
mkdir -p "$rr/sub" && echo x > "$rr/sub/existing.py" && git -C "$rr" add . && git -C "$rr" commit -qm one
git -C "$rr" config user.email 1234+dave@users.noreply.github.com && git -C "$rr" config user.name "Dave D"
echo y >> "$rr/sub/existing.py" && git -C "$rr" commit -qam two
printf '%s' '{"author":{"login":"alice"},"reviews":[{"author":{"login":"copilot-pull-request-reviewer"}},{"author":{"login":"bob"}}],"files":[{"path":"sub/existing.py","additions":1,"deletions":0}]}' > "$rr/meta.json"
t "a bot reviewer never reaches the roster" \
  "$(roster_of "$rr/meta.json" "$rr" | grep -c copilot)" "0"
t "the human reviewer and the path authors do" \
  "$(roster_of "$rr/meta.json" "$rr" | grep -cE '^(bob|Carol|dave)$')" "3"
printf '%s' '{"author":{"login":"alice"},"reviews":[],"files":[{"path":"sub/brand_new.py","additions":900,"deletions":0}]}' > "$rr/meta2.json"
t "an all-new-file PR falls back to the parent dir's authors" \
  "$(roster_of "$rr/meta2.json" "$rr" | grep -cE '^(Carol|dave)$')" "2"
rm -rf "$rr"

echo "[derive]"
# Families for a synthetic diff, run through the shipping parse() + derive().
fams() {  # <diff text> -> space-separated family names
  printf '%s' "$1" | python3 -c '
import sys; sys.path.insert(0, sys.argv[1]); import triage
print(" ".join(f for f, _ in triage.derive(triage.parse(sys.stdin.read()))))' "$S"
}
repin='diff --git a/3rdparty/composable_kernel b/3rdparty/composable_kernel
--- a/3rdparty/composable_kernel
+++ b/3rdparty/composable_kernel
@@ -1 +1 @@
-Subproject commit 1111111111111111111111111111111111111111
+Subproject commit 2222222222222222222222222222222222222222
'
docs='diff --git a/docs/ck.md b/docs/ck.md
--- a/docs/ck.md
+++ b/docs/ck.md
@@ -1 +1 @@
-old note
+bumped CK for the new kernel
'
# 3rdparty/ is INFRA: if the repin is not seen first, both of these return infra-only and HK20
# is never selected.
t "a standalone submodule repin selects HK20" "$(fams "$repin")" "submodule-repin"
t "a repin with only docs alongside selects HK20" "$(fams "$repin$docs")" "submodule-repin"
t "docs alone stay infra-only" "$(fams "$docs")" "infra-only"
rules_of() {  # <diff text> <family> -> that family's rules
  printf '%s' "$1" | python3 -c '
import sys; sys.path.insert(0, sys.argv[1]); import triage
print(dict(triage.derive(triage.parse(sys.stdin.read()))).get(sys.argv[2], ""))' "$S" "$2"
}
flydsl='diff --git a/aiter/ops/flydsl/kernels/moe_gemm.py b/aiter/ops/flydsl/kernels/moe_gemm.py
--- a/aiter/ops/flydsl/kernels/moe_gemm.py
+++ b/aiter/ops/flydsl/kernels/moe_gemm.py
@@ -1 +1 @@
-    scale = 1.0 / amax
+    scale = amax if store_divisor else 1.0 / amax
'
# is_kernel() skips FlyDSL, so flydsl-kernel is the only family that can carry A4 there.
t "a FlyDSL kernel edit selects A4" \
  "$(rules_of "$flydsl" flydsl-kernel | grep -ow A4)" "A4"
# Raw-pointer launches live in the wrappers (aiter/ops/flydsl/*.py), which derive `flydsl` but
# not `flydsl-kernel`, so D14 and D13 must reach a wrapper-only diff through `flydsl`.
flydsl_wrapper='diff --git a/aiter/ops/flydsl/qsa_sparse_decode.py b/aiter/ops/flydsl/qsa_sparse_decode.py
--- a/aiter/ops/flydsl/qsa_sparse_decode.py
+++ b/aiter/ops/flydsl/qsa_sparse_decode.py
@@ -1 +1 @@
-    _run_compiled(exe, ptr_arg(q), ptr_arg(out), stream)
+    _run_compiled(exe, ptr_arg(q), ptr_arg(kv_indices), ptr_arg(out), stream)
'
t "a FlyDSL wrapper-only edit selects D14" \
  "$(rules_of "$flydsl_wrapper" flydsl | grep -ow D14)" "D14"
t "a FlyDSL wrapper-only edit selects D13" \
  "$(rules_of "$flydsl_wrapper" flydsl | grep -ow D13)" "D13"

# derive -> verdict -> card: a submodule path has no extension, so unless audit_card counts it
# as a citation, a card that reports HK20 without the rule code (as SKILL.md requires) fails
# with UNREPORTED-FIRE.
card_problems() {  # <diff text> <verdict line> <finding line> -> problem kinds, or "none"
  printf '%s' "$1" | python3 -c '
import sys; sys.path.insert(0, sys.argv[1]); import triage
diff = sys.stdin.read()
_, probs = triage.audit_card(sys.argv[3], sys.argv[2], "", "", diff)
print(" ".join(k for k, _, _ in probs) or "none")' "$S" "$2" "$3"
}
v_red='HK20 FIRE -- 3rdparty/composable_kernel pinned to 2222222 which is on no ROCm/composable_kernel branch'
v_inc='HK20 FIRE -- 3rdparty/composable_kernel could not verify 2222222: fetch failed'
t "a red HK20 finding without the rule code reaches the card" \
  "$(card_problems "$repin" "$v_red" $'\360\237\224\264 3rdparty/composable_kernel is pinned to 2222222, on no ROCm/composable_kernel branch')" "none"
t "an inconclusive HK20 warning reaches the card" \
  "$(card_problems "$repin" "$v_inc" $'⚠️ could not verify 3rdparty/composable_kernel at 2222222 (fetch failed)')" "none"
t "an HK20 FIRE missing from the card is still caught" \
  "$(card_problems "$repin" "$v_red" '')" "UNREPORTED-FIRE"
# Mixed-hex commit SHA (like 1f66d862) must be recognized as concrete evidence
v_hex='HK20 FIRE -- 3rdparty/composable_kernel pinned to 1f66d862 which is on no ROCm/composable_kernel branch'
t "a mixed-hex commit SHA is concrete evidence" \
  "$(card_problems "$repin" "$v_hex" $'\360\237\224\264 3rdparty/composable_kernel is pinned to 1f66d862, on no ROCm/composable_kernel branch')" "none"

# A4 must be selected when a host-side dispatch wrapper adds an activation branch
dispatch_host='diff --git a/aiter/fused_moe.py b/aiter/fused_moe.py
--- a/aiter/fused_moe.py
+++ b/aiter/fused_moe.py
@@ -100,6 +100,8 @@
     if activation == ActivationType.Silu:
         return silu(x)
+    elif activation == ActivationType.Relu2:
+        return relu2(x)
'
t "a host-side activation branch selects A4 via dispatch-change" \
  "$(rules_of "$dispatch_host" dispatch-change | grep -ow A4)" "A4"

# Every rule id derive() can emit must be registered in ALL_RULES. It is read by the
# derivation-failed, diff-too-large and underivable fallbacks -- the moments the full
# catalogue matters most -- and by the rules=N/M denominator. B8 and D12 reached main
# unregistered, and HK4-HK11 had been missing for longer, because nothing compared the
# two lists. Ported from ROCm/ATOM, where this guard was added during the port.
_rule_ids_consistent() {
  python3 - "$S" <<'PYEOF'
import inspect, re, sys
sys.path.insert(0, sys.argv[1])
import triage
src = inspect.getsource(triage.derive)
emitted = set()
# hit("family-name", "A1 B2 ...") -- the rule list is the second argument, and rule ids
# are an uppercase letter run followed by digits, so the fragments a looser pattern
# picked up (IE, ule) cannot match.
for m in re.finditer(r"hit\([^)]*?,\s*.((?:[A-Z]+[0-9]+[a-z]?(?: |(?=.)))+).\)", src):
    emitted |= set(m.group(1).split())
missing = sorted(emitted - set(triage.ALL_RULES.split()))
print("ok" if emitted and not missing else ("empty" if not emitted else "missing:" + ",".join(missing)))
PYEOF
}
t "every id derive() emits is in ALL_RULES" "$(_rule_ids_consistent)" "ok"

echo "[mapping]"
# MAPPING.md is generated; a family or rule added to derive() without regenerating it drifts.
t "MAPPING.md matches \`triage.py mapping\`" \
  "$(python3 "$S/triage.py" mapping | diff -q - "$S/MAPPING.md" >/dev/null && echo same || echo drift)" "same"

# Rejected validation evidence must degrade the review, not delete it. The regression this
# pins happened for real on ATOM's first live auto-validation run: the consumer ran under
# errexit, so one missing optional import aborted fetch.sh and produced no card at all --
# indistinguishable, to the human waiting on it, from the bot being down. _degrade_test.sh
# extracts the real consumer region out of fetch.sh by line markers (not a replica of it)
# and drives it with a report for the wrong head, which must be refused AND survived.
t "rejected validation evidence degrades instead of aborting" \
  "$(bash "$S/_degrade_test.sh" "$S" 2>&1 | tail -1)" "ok"

# A fetch that dies must say why, in the log, in terms the runner can act on. The live run
# this pins happened on ATOM: a spent GitHub API quota ended the log mid-sentence after
# "fetch..." -- errexit aborted inside the pipeline before the classification could run, and
# the exit trap wrote a status file while printing nothing. "Check gh auth/token" would also
# have been the wrong advice: the token was fine and a quota was empty, which resets on a clock.
t "a dead fetch names the cause the runner can act on" \
  "$(bash "$S/_fetchfail_test.sh" "$S" 2>&1 | tail -1)" "ok"

# check_citations.py's main() is executed here. The 34 pytest cases covered check(); main()
# was called by nothing, so a NameError in it survived review and a green gate -- citations.txt
# carried a traceback, and the publish step read that as "nothing flagged" and anchored freely.
t "the citation checker runs end to end" \
  "$(bash "$S/_citations_test.sh" "$S" 2>&1 | tail -1)" "ok"

# check_prompts must work outside a git checkout. It asked git for the repo root, and
# `git rev-parse --show-toplevel` prints an empty string when there is no repo -- making
# the SKILL.md path RELATIVE to the caller's cwd, so the drift check died with
# FileNotFoundError on a `git archive HEAD` export. That export is exactly how a reviewer
# is told to verify, and every trial that passed had been inside a checkout.
_drift_outside_a_repo() {
  # The whole skill tree, at the same depth it sits in the repo, with no .git anywhere --
  # what `git archive HEAD | tar -x` produces. Checks whose inputs are absent skip and
  # still exit 0; what must not happen is the crash that an empty repo-root produced.
  # Reconstruct what the check actually needs -- the skill tree at its repo depth, plus
  # the workflows it reads (the whole review-bot family, not just the one it read first)
  # -- in a directory with no .git. Deliberately NOT `git archive`:
  # this suite is itself meant to run from an export, where there is no repo to archive
  # from, and a guard that only works inside a checkout cannot test leaving one.
  local t; t=$(mktemp -d)
  mkdir -p "$t/.claude/skills" "$t/.github/workflows"
  cp -r "$S" "$t/.claude/skills/review-pr"
  rm -rf "$t/.claude/skills/review-pr/reports"
  cp "$S"/../../../.github/workflows/*review-bot*.yml "$t/.github/workflows/" 2>/dev/null
  ( cd /tmp && python3 "$t/.claude/skills/review-pr/check_prompts.py" >/dev/null 2>&1; echo $? )
  rm -rf "$t"
}
t "the drift check runs outside a git checkout" "$(_drift_outside_a_repo)" "0"


# AITER_AGENT_TIMEOUT and AITER_RUN_BUDGET are coupled and nothing else enforces it. A review
# runs two agents in series and the budget gate refuses to START one that cannot finish, so
# raising the per-agent timeout alone leaves the refuter unable to start: the worker runs for
# an hour, the independent gate reds, the review is withheld, and the failure is reported as a
# timeout whose named fix makes it worse. Hit for real while re-reviewing ATOM#2762.
_budget_pair() {  # <agent> <budget> -> rc of run_one's precondition
  ( AITER_AGENT_TIMEOUT=$1 AITER_RUN_BUDGET=$2 \
    timeout 20 bash "$S/run_one.sh" 0 x/y >/dev/null 2>&1; echo $? )
}
t "a per-agent timeout that starves the refuter is refused" "$(_budget_pair 5400 7200)" "2"

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

echo "[gate watch]"
# Same reason once more: a scheduled workflow never runs on a PR branch, and the failure it
# watches for -- hosted runners starved -- cannot be reproduced on demand. Drive the shipping
# script against a fake API and a fixed clock.
gw=$(python3 "$S/gate_watch_test.py" 2>&1); gw_rc=$?
printf '%s\n' "$gw" | grep -E '^  (✅|❌)' || true
gok=$(printf '%s\n' "$gw" | grep -c '✅' || true)
gbad=$(printf '%s\n' "$gw" | grep -c '❌' || true)
ok=$((ok + gok)); bad=$((bad + gbad))
if [ "$gw_rc" -ne 0 ] && [ "$gbad" -eq 0 ]; then
  why=$(printf '%s\n' "$gw" | grep -m1 -E 'Error|Traceback|No such file' | sed 's/^ *//' | cut -c1-110)
  echo "  ❌ gate_watch_test.py did not run — ${why:-exit $gw_rc}"; bad=$((bad + 1))
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

echo "[publish]"
# The publish path's guards are pytest cases (_publish_test.py): the --post closure, the
# anchoring rule (span anchors and the permalinks for what cannot be anchored included), and
# the degrade-to-summary fallback. Run them from here so they are not a
# file nobody executes -- a test suite that only runs when someone remembers it is the same
# kind of silence this script exists to break. pytest is not part of the runner's baseline,
# so its absence is reported out loud rather than passed over: a check that did not run is
# not a check that passed, and it is counted red in neither direction here, it is SAID.
if python3 -c "import pytest" 2>/dev/null; then
  pt=$(cd "$S" && python3 -m pytest _publish_test.py -q 2>&1); pt_rc=$?
  n=$(printf '%s\n' "$pt" | grep -oE '[0-9]+ passed' | grep -oE '^[0-9]+' || echo 0)
  if [ "$pt_rc" -eq 0 ]; then
    echo "  ✅ _publish_test.py: ${n:-0} passed (--post closure, anchoring + spans + permalinks, degrade-to-summary)"
    ok=$((ok + 1))
  else
    printf '%s\n' "$pt" | grep -E '^(FAILED|ERROR)' | sed 's/^/  ❌ /' || true
    echo "  ❌ _publish_test.py failed — run: python3 -m pytest $S/_publish_test.py"
    bad=$((bad + 1))
  fi
else
  echo "  ⏭️  _publish_test.py NOT RUN — no pytest on this box. The --post closure, the"
  echo "      anchoring rule (spans, permalinks) and the degrade-to-summary fallback are"
  echo "      unverified here."
fi

echo "=== $ok green / $bad red ==="
[ "$bad" -eq 0 ]
