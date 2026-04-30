#!/usr/bin/env bash
# Resolve AITER wheel URL for pip install.
# Handles broken wheels where filename version != metadata version.
#
# Usage: source this file after setting AITER_INDEX_URL and AITER_VERSION
# Optional: set AITER_PYTHON_TAG (e.g. cp310, cp312) before sourcing (default: cp312)
# Output: sets AITER_INSTALL_CMD (full pip install command for AITER)

AITER_PYTHON_TAG="${AITER_PYTHON_TAG:-cp312}"

if [ -n "${AITER_INDEX_URL:-}" ]; then
  # Export so the heredoc Python script can read them via os.environ
  export AITER_INDEX_URL AITER_PYTHON_TAG
  export AITER_VERSION="${AITER_VERSION:-}"

  # Find the direct wheel URL from the index.
  # Some S3 buckets serve wheels under an /amd-aiter/ subdirectory (PEP 503 style),
  # others serve them at the root of the index URL.  Try both locations.
  WHEEL_URL=$(python3 << 'PYEOF'
import os, re, sys, urllib.parse, urllib.request

index = os.environ.get("AITER_INDEX_URL", "").rstrip("/")
version_filter = os.environ.get("AITER_VERSION", "")
pytag = os.environ.get("AITER_PYTHON_TAG", "cp312")

def list_wheels(page_url):
    try:
        with urllib.request.urlopen(page_url, timeout=30) as resp:
            html = urllib.parse.unquote(resp.read().decode())
    except Exception:
        return []
    pat = r'href="([^"]*amd_aiter-[^"]*' + pytag + r'-' + pytag + r'-[^"]+\.whl)"'
    found = re.findall(pat, html)
    if not found:
        pat2 = r'(amd_aiter-[^\s"<>]*' + pytag + r'-' + pytag + r'-[^\s"<>]+\.whl)'
        found = re.findall(pat2, html)
    return [os.path.basename(urllib.parse.unquote(w)) for w in found]

def is_downloadable(url):
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200
    except Exception:
        return False

if index.endswith("amd-aiter"):
    bases = [index + "/"]
else:
    bases = [index + "/amd-aiter/", index + "/"]

filenames = []
for base in bases:
    filenames = list_wheels(base)
    if filenames:
        break

if not filenames:
    sys.exit(0)

if version_filter:
    encoded = version_filter.replace("+", "%2B")
    filenames = [f for f in filenames if version_filter in f or encoded in f]

if not filenames:
    sys.exit(0)

best_name = sorted(filenames)[-1]
best_encoded = best_name.replace("+", "%2B")

for base in bases:
    candidate = base.rstrip("/") + "/" + best_encoded
    if is_downloadable(candidate):
        print(candidate)
        sys.exit(0)

print(bases[0].rstrip("/") + "/" + best_encoded)
PYEOF
  ) || true

  if [ -n "${WHEEL_URL}" ]; then
    AITER_INSTALL_CMD="pip install --force-reinstall '${WHEEL_URL}'"
    echo "AITER wheel: ${WHEEL_URL}"
  else
    echo "WARNING: Could not resolve ${AITER_PYTHON_TAG} wheel URL, falling back to pip install amd-aiter"
    AITER_INSTALL_CMD="pip install --force-reinstall --extra-index-url '${AITER_INDEX_URL}' amd-aiter"
  fi
else
  AITER_INSTALL_CMD="pip install amd-aiter"
fi

echo "AITER install: ${AITER_INSTALL_CMD}"
