---
name: format-code
description: >
 Format and lint AITER code before committing. Matches the pre-commit pipeline
 in `.githooks/pre-commit` and `CONTRIBUTE.md`: runs autoflake / black 26.3.0 /
 ruff 0.15.7 on Python, clang-format-18 (Google style via `.clang-format`) on
 C/C++/HIP, strips trailing whitespace, converts UTF-8 to ASCII//TRANSLIT,
 ensures trailing newlines, and bumps the copyright year at the top of modified
 files. Operates only on files changed in `git diff` / `git diff --cached`.
 Use this skill before any AITER commit or when the user says "format",
 "lint", "clean up code", or "pre-commit".
allowed-tools: Bash Read Edit Grep Glob
---

# Format AITER Code Before Committing

AITER uses a specific combination of formatters pinned in `CONTRIBUTE.md`:

| Tool | Version | Applies to |
|------|---------|------------|
| `black` | 26.3.0 | `.py` |
| `ruff` | 0.15.7 | `.py` lint |
| `autoflake` | any | `.py` unused imports / vars |
| `clang-format-18` | 18 | `.c .cc .cpp .cxx .cu .cuh .h .hpp .hxx` |

The shipping pre-commit hook (`.githooks/pre-commit`) also:
- Bumps the copyright year in the top 10 lines of every modified source file.
- Strips trailing whitespace.
- Ensures every file ends in a newline.
- Converts UTF-8 non-ASCII characters to ASCII via `iconv -t ascii//TRANSLIT`.

## Step 1: Install tools (once per environment)

```bash
pip install "black==26.3.0" "ruff==0.15.7" autoflake
# clang-format-18 (Ubuntu/Debian):
sudo apt-get install -y clang-format-18
# Then install the repo's git hook:
bash ./.githooks/install
```

Verify:

```bash
black --version       # should print 26.3.0
ruff --version        # should print 0.15.7
clang-format-18 --version
```

## Step 2: Collect changed files

Only touch files the user actually changed (staged OR unstaged):

```bash
changed=$( (git diff --name-only --cached; git diff --name-only) | sort -u )
echo "$changed"
```

If empty, tell the user there's nothing to format and stop.

## Step 3: Run the pipeline per file

### 3.1 Python files (`.py`)

```bash
for f in $(echo "$changed" | grep -E '\.py$' || true); do
  [ -f "$f" ] || continue
  autoflake --in-place --remove-all-unused-imports --remove-unused-variables "$f"
  black "$f"
  ruff check --fix "$f" || true   # auto-fixes what it can; manual review for the rest
done
```

AITER's `CONTRIBUTE.md` says PEP 8 + 88-char line width (`black` default) +
type hints on function signatures. `ruff` config (if present in
`pyproject.toml`) is respected automatically.

### 3.2 C/C++/HIP files

The repo root carries a `.clang-format` (Google style with AITER tweaks).
`clang-format-18` honors it by default — **don't pass `--style=Google`**,
that would override the local file.

```bash
for f in $(echo "$changed" | grep -E '\.(c|cc|cpp|cxx|cu|cuh|h|hpp|hxx|cl|h\.in|hpp\.in|cpp\.in)$' || true); do
  [ -f "$f" ] || continue
  clang-format-18 -i "$f"
done
```

### 3.3 Whitespace / encoding / newline (matches `.githooks/pre-commit`)

```bash
for f in $changed; do
  [ -f "$f" ] || continue
  case "$f" in
    *.py|*.c|*.cc|*.cpp|*.cxx|*.cu|*.cuh|*.h|*.hpp|*.hxx|*.cl)
      sed -i -e 's/[[:space:]]*$//' "$f"
      sed -i -e '$a\' "$f"
      tmp=$(mktemp) && iconv -s -f utf-8 -t ascii//TRANSLIT "$f" > "$tmp" && \
        chmod --reference="$f" "$tmp" && mv -f "$tmp" "$f"
      ;;
  esac
done
```

On Windows / PowerShell without `sed`/`iconv`, let `black` / `clang-format`
handle whitespace and skip the `iconv` step (it's for CJK → ASCII cleanup).

### 3.4 Copyright year bump

```bash
year=$(date +%Y)
for f in $changed; do
  [ -f "$f" ] || continue
  /usr/bin/perl -pi -e 'INIT { $year = "'"$year"'" }
      s/^([*\/#[:space:]]*)Copyright\s+(?:\(C\)\s*)?(\d+)(?:\s*-\s*\d+)?/qq($1Copyright (C) $2@{[$year != $2 ? "-$year" : ""]})/ie
      if $. < 10' "$f"
done
```

(This is a direct port of the perl snippet in `.githooks/pre-commit`.)

## Step 4: Re-stage files

`black`, `clang-format`, and the in-place edits above leave the files as
"modified" again even if they were already staged. Re-add:

```bash
git add -u
```

Don't `git add .` blindly — it would pull in unrelated new files.

## Step 5: Summary

Report to the user:

- # Python files formatted
- # C/C++/HIP files formatted
- Any `ruff` warnings that couldn't be auto-fixed (these need manual review)
- Any `clang-format` diffs that look semantically suspicious (usually none)

If the user ran `git commit` and the hook modified files, remind them to
re-stage and amend:

```bash
git add -u
git commit --amend --no-edit      # only if the original commit was the last one
```

## Full pipeline script (for copy-paste)

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

changed=$( (git diff --name-only --cached; git diff --name-only) | sort -u )
[ -z "$changed" ] && { echo "Nothing to format."; exit 0; }

py=$(echo "$changed" | grep -E '\.py$' || true)
cc=$(echo "$changed" | grep -E '\.(c|cc|cpp|cxx|cu|cuh|h|hpp|hxx|cl)$' || true)

for f in $py; do
  [ -f "$f" ] || continue
  autoflake --in-place --remove-all-unused-imports --remove-unused-variables "$f"
  black "$f"
  ruff check --fix "$f" || true
done

for f in $cc; do
  [ -f "$f" ] || continue
  clang-format-18 -i "$f"
done

# Whitespace + newline + copyright
year=$(date +%Y)
for f in $changed; do
  [ -f "$f" ] || continue
  case "$f" in
    *.py|*.c|*.cc|*.cpp|*.cxx|*.cu|*.cuh|*.h|*.hpp|*.hxx|*.cl)
      sed -i -e 's/[[:space:]]*$//' "$f"
      sed -i -e '$a\' "$f"
      /usr/bin/perl -pi -e 'INIT { $year = "'"$year"'" }
        s/^([*\/#[:space:]]*)Copyright\s+(?:\(C\)\s*)?(\d+)(?:\s*-\s*\d+)?/qq($1Copyright (C) $2@{[$year != $2 ? "-$year" : ""]})/ie
        if $. < 10' "$f"
      ;;
  esac
done

echo "Python:  $(echo "$py" | grep -c '^' || true) files"
echo "C/C++:   $(echo "$cc" | grep -c '^' || true) files"
echo ""
echo "Remember to re-stage formatted files: git add -u"
```

## Common pitfalls

| Symptom | Fix |
|---------|-----|
| `black` says version mismatch in CI | Pin locally: `pip install "black==26.3.0"`. |
| `clang-format` changes look wrong | You're using `clang-format` v14/v17 — install v18 (`apt-get install clang-format-18`). |
| `ruff` removes an import you actually use | Add `# noqa: F401` to that line or add the symbol to `__all__`. |
| Pre-commit hook blocks the commit | Run this skill first, then `git add -u && git commit`. |
| Copyright year stays at old value | The perl regex only runs in the first 10 lines of the file. Make sure the `Copyright (C)` header is near the top. |
| `iconv: TRANSLIT not supported` | Rare; on Alpine, install `glibc-compat`. Otherwise skip the `iconv` step. |
