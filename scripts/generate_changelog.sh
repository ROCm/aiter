#!/bin/bash
# Generate categorized changelog from git log between two refs
# Usage: ./scripts/generate_changelog.sh <from_ref> <to_ref> [output_file]
# Example: ./scripts/generate_changelog.sh v0.1.11.post1 release/v0.1.12

set -euo pipefail

FROM_REF="${1:?Usage: $0 <from_ref> <to_ref> [output_file]}"
TO_REF="${2:?Usage: $0 <from_ref> <to_ref> [output_file]}"
OUTPUT="${3:-RELEASE_NOTES.md}"
REPO_URL="https://github.com/ROCm/aiter"

echo "Generating changelog: ${FROM_REF}..${TO_REF}"

# Get all commit subjects with PR numbers
COMMITS=$(git log "${FROM_REF}..${TO_REF}" --format="%s" --reverse)
TOTAL=$(echo "$COMMITS" | wc -l)

# Temp files for categories
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

touch "$TMP/features" "$TMP/performance" "$TMP/fixes" "$TMP/refactor" "$TMP/ci" "$TMP/other"

while IFS= read -r line; do
    # Extract PR number if present
    PR_NUM=$(echo "$line" | grep -oP '#\d+' | tail -1 || true)
    PR_LINK=""
    if [ -n "$PR_NUM" ]; then
        PR_LINK=" (${REPO_URL}/pull/${PR_NUM#\#})"
    fi

    # Clean up subject (remove trailing PR reference for display)
    SUBJECT=$(echo "$line" | sed 's/ (#[0-9]*)$//')

    ENTRY="- ${SUBJECT}${PR_LINK}"

    # Categorize by prefix/keywords
    case "$line" in
        *"[feat]"*|*"feat("*|*"feat:"*|"Add "*|"add "*|"support"*|"Support"*|"Enable "*|"enable "*|"Introduce "*|"new "*|"New "*)
            echo "$ENTRY" >> "$TMP/features" ;;
        *"[Perf]"*|*"tune"*|*"Tune"*|*"tuned"*|*"Retune"*|*"retune"*|*"optim"*|*"Optim"*|*"perf"*|*"speed"*)
            echo "$ENTRY" >> "$TMP/performance" ;;
        *"fix"*|*"Fix"*|*"FIX"*|*"bug"*|*"Bug"*|*"hotfix"*|*"Revert"*|*"revert"*|*"accuracy"*)
            echo "$ENTRY" >> "$TMP/fixes" ;;
        *"refactor"*|*"Refactor"*|*"replace"*|*"Replace"*|*"remove"*|*"Remove"*|*"rm "*|*"[OPUS]"*|*"opus"*|*"migrate"*|*"clean"*)
            echo "$ENTRY" >> "$TMP/refactor" ;;
        "CI:"*|"CI "*|*"[CI]"*|*"test"*|*"Test"*|*"build"*|*"Build"*)
            echo "$ENTRY" >> "$TMP/ci" ;;
        *)
            echo "$ENTRY" >> "$TMP/other" ;;
    esac
done <<< "$COMMITS"

# Count per category
N_FEAT=$(wc -l < "$TMP/features")
N_PERF=$(wc -l < "$TMP/performance")
N_FIX=$(wc -l < "$TMP/fixes")
N_REF=$(wc -l < "$TMP/refactor")
N_CI=$(wc -l < "$TMP/ci")
N_OTHER=$(wc -l < "$TMP/other")

# Generate markdown
cat > "$OUTPUT" <<EOF
# AITER Release Notes

**Version:** $(basename "$TO_REF" | sed 's/release\///')
**Date:** $(date +%Y-%m-%d)
**Base:** \`${TO_REF}\` (${TOTAL} commits since ${FROM_REF})

---

## Highlights

<!-- Fill in key highlights manually -->

---

## New Features (${N_FEAT})

$(cat "$TMP/features")

## Performance (${N_PERF})

$(cat "$TMP/performance")

## Bug Fixes (${N_FIX})

$(cat "$TMP/fixes")

## Refactoring (${N_REF})

$(cat "$TMP/refactor")

## CI & Infrastructure (${N_CI})

$(cat "$TMP/ci")

## Other (${N_OTHER})

$(cat "$TMP/other")

---

## Known Issues

- Issue #2656: DeepSeek-R1-MXFP4 accuracy regression from Triton GEMM config retune (PR #2434). Partial fix applied on release branch.

## Compatibility

- **GPU Architectures:** gfx942 (MI300X), gfx950 (MI355X)
- **Python:** 3.10, 3.12
- **ROCm:** 7.0+
- **Triton:** 3.6.0
EOF

echo "Generated ${OUTPUT} (${TOTAL} commits categorized)"
echo "  Features: ${N_FEAT}, Performance: ${N_PERF}, Fixes: ${N_FIX}"
echo "  Refactoring: ${N_REF}, CI: ${N_CI}, Other: ${N_OTHER}"
