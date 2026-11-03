#!/usr/bin/env bash
# Drive the REAL fetch-failure classification out of run_one.sh (bytes extracted by markers,
# not a replica) and assert that a spent GitHub quota is named as such and said out loud.
set -uo pipefail
S="$1"; T=$(mktemp -d)
A=$(grep -n '^# errexit must not win here' "$S/run_one.sh" | cut -d: -f1)
B=$(awk -v a="$A" 'NR>a && /^rm -f "\$FL"$/{print NR; exit}' "$S/run_one.sh")
{
  # The real script runs `set -euo pipefail`. Testing the region under bare `set -e`
  # is what let an unreachable classification pass as a working one: pipefail turns a
  # no-match grep inside a command substitution into a fatal assignment, and none of
  # the code under test ran in production. Use the flags the script actually uses.
  echo 'set -euo pipefail'
  echo 'say() { echo "$*"; }'
  echo 'fail() { local c=$1 x=$2; shift 2; say "$*"; printf "%s\t%s\n" "$c" "$*" > "$STATUS"; exit "$x"; }'
  echo 'FL=$(mktemp)'
  sed -n "${A},${B}p" "$S/run_one.sh"
} > "$T/region.sh"

mkdir -p "$T/skill"; printf '#!/usr/bin/env bash\necho "GraphQL: API rate limit already exceeded for user ID 1."\nexit 1\n' > "$T/skill/fetch.sh"
export SKILL="$T/skill" PROJ="$T" GH_FALLBACK_TOK="" PR=1 REPO=x/y STATUS="$T/status"
fails=0

_run() { out=$(PATH="$1:$PATH" bash "$T/region.sh" 2>&1); rc=$?; }
_want() { printf '%s\n' "$out" | grep -q "$1" || { echo "  ❌ $2"; fails=1; }; }
_deny() { printf '%s\n' "$out" | grep -q "$1" && { echo "  ❌ $3"; fails=1; }; }

# 1) Quota spent, and the refill time is readable: say the minute, not "within the hour".
mkdir -p "$T/bin"
printf '#!/usr/bin/env bash\necho $(( $(date +%%s) + 900 ))\n' > "$T/bin/gh"; chmod +x "$T/bin/gh"
_run "$T/bin"
[ "$rc" -ne 0 ] || { echo "  ❌ a failed fetch was treated as success"; fails=1; }
_want "quota for this bot is spent" "the quota was not named"
_want "refills at .*UTC, in about [0-9]\+ min" "the refill time was not stated as a minute"
_want "Nothing about this PR was examined" "it did not say the PR went unexamined"
grep -q "^env" "$STATUS" 2>/dev/null || { echo "  ❌ no classified status was written"; fails=1; }

# 2) Quota spent and the refill time cannot be read: the vague form, never a wrong number.
printf '#!/usr/bin/env bash\nexit 1\n' > "$T/bin/gh"
_run "$T/bin"
_want "within the hour" "the fallback wording was lost"
_deny "in about" "" "an unreadable reset time still produced a number"

# 2b) A RESTRICTED account, not a throttled one: 60 core with GraphQL at 0 is what GitHub
# leaves on an account it has flagged. Telling that person to wait sends them to wait for
# something that never arrives, so it must be named as the different thing it is.
printf '#!/usr/bin/env bash\necho "60 0 $(( $(date +%%s) + 900 ))"\n' > "$T/bin/gh"
_run "$T/bin"
_want "restricted the account" "a restricted account was reported as throttling"
_deny "re-comment then" "" "a restricted account was told to simply retry later"
_want "Nothing about this PR was examined" "it did not say the PR went unexamined"

# 3) A non-quota failure must keep its generic cause, not be mislabelled a quota.
printf '#!/usr/bin/env bash\necho "fatal: could not read from remote"\nexit 1\n' > "$T/skill/fetch.sh"
_run "$T/bin"
_want "check gh auth/token" "a non-quota failure lost its generic cause"
_deny "quota for this bot is spent" "" "a non-quota failure was mislabelled a quota"

rm -rf "$T"
[ "$fails" -eq 0 ] && echo "ok" || echo "FAILED"
