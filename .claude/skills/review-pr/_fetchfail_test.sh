#!/usr/bin/env bash
# Drive the REAL fetch-failure classification out of run_one.sh (bytes extracted by markers,
# not a replica) and assert that a spent GitHub quota is named as such and said out loud.
set -uo pipefail
S="$1"; T=$(mktemp -d)
A=$(grep -n '^# errexit must not win here' "$S/run_one.sh" | cut -d: -f1)
B=$(awk -v a="$A" 'NR>a && /^rm -f "\$FL"$/{print NR; exit}' "$S/run_one.sh")
{
  echo 'set -e'
  echo 'say() { echo "$*"; }'
  echo 'fail() { local c=$1 x=$2; shift 2; say "$*"; printf "%s\t%s\n" "$c" "$*" > "$STATUS"; exit "$x"; }'
  echo 'FL=$(mktemp)'
  sed -n "${A},${B}p" "$S/run_one.sh"
} > "$T/region.sh"

mkdir -p "$T/skill"; printf '#!/usr/bin/env bash\necho "GraphQL: API rate limit already exceeded for user ID 1."\nexit 1\n' > "$T/skill/fetch.sh"
export SKILL="$T/skill" PROJ="$T" GH_FALLBACK_TOK="" PR=1 REPO=x/y STATUS="$T/status"
out=$(bash "$T/region.sh" 2>&1); rc=$?
fails=0
[ "$rc" -ne 0 ] || { echo "  ❌ a failed fetch was treated as success"; fails=1; }
printf '%s\n' "$out" | grep -q "spent API quota" \
  || { echo "  ❌ the quota was not named — got: $(printf '%s' "$out" | head -1)"; fails=1; }
printf '%s\n' "$out" | grep -q "Nothing about this PR was examined" \
  || { echo "  ❌ it did not say the PR went unexamined"; fails=1; }
grep -q "^env" "$STATUS" 2>/dev/null || { echo "  ❌ no classified status was written"; fails=1; }

# A non-quota failure must still fall through to the generic cause, not be mislabelled a quota.
printf '#!/usr/bin/env bash\necho "fatal: could not read from remote"\nexit 1\n' > "$T/skill/fetch.sh"
out2=$(bash "$T/region.sh" 2>&1) || true
printf '%s\n' "$out2" | grep -q "check gh auth/token" \
  || { echo "  ❌ a non-quota failure lost its generic cause"; fails=1; }
printf '%s\n' "$out2" | grep -q "spent API quota" && { echo "  ❌ a non-quota failure was mislabelled a quota"; fails=1; }
rm -rf "$T"
[ "$fails" -eq 0 ] && echo "ok" || echo "FAILED"
