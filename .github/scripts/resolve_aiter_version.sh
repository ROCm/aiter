#!/usr/bin/env bash
# Resolve AITER wheel URL for pip install.
# Handles broken wheels where filename version != metadata version.
#
# Usage: source this file after setting AITER_INDEX_URL and AITER_VERSION
# Optional: set AITER_PYTHON_TAG (e.g. cp310, cp312) before sourcing (default: cp312)
# Output: sets AITER_INSTALL_CMD (full pip install command for AITER)

AITER_PYTHON_TAG="${AITER_PYTHON_TAG:-cp312}"

if [ -n "${AITER_INDEX_URL:-}" ]; then
  # Find the direct wheel URL from the index.
  # Some S3 buckets serve wheels under an /amd-aiter/ subdirectory (PEP 503 style),
  # others serve them at the root of the index URL.  Try both locations.
  WHEEL_URL=$(python3 -c "
import os, re, sys, urllib.parse, urllib.request

index = '${AITER_INDEX_URL}'.rstrip('/')
version_filter = '${AITER_VERSION:-}'
pytag = '${AITER_PYTHON_TAG}'

def list_wheels(page_url):
    \"\"\"Return wheel filenames found at page_url.\"\"\"
    try:
        with urllib.request.urlopen(page_url, timeout=30) as resp:
            html = urllib.parse.unquote(resp.read().decode())
    except Exception:
        return []
    pat = r'href=\"([^\"]*amd_aiter-[^\"]*' + pytag + r'-' + pytag + r'-[^\"]+\.whl)\"'
    found = re.findall(pat, html)
    if not found:
        pat2 = r'(amd_aiter-[^\s\"<>]*' + pytag + r'-' + pytag + r'-[^\s\"<>]+\.whl)'
        found = re.findall(pat2, html)
    return [os.path.basename(urllib.parse.unquote(w)) for w in found]

def is_downloadable(url):
    \"\"\"HEAD-check whether the URL returns 200.\"\"\"
    req = urllib.request.Request(url, method='HEAD')
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200
    except Exception:
        return False

# Build list of base URLs to try (for both listing and downloading)
if index.rstrip('/').endswith('amd-aiter'):
    bases = [index + '/']
else:
    bases = [index + '/amd-aiter/', index + '/']

# Collect wheel filenames from whichever listing succeeds
filenames = []
for base in bases:
    filenames = list_wheels(base)
    if filenames:
        break

if not filenames:
    raise SystemExit(f'No {pytag} wheels found at any of: {bases}')

# Filter by version
if version_filter:
    encoded = version_filter.replace('+', '%2B')
    filenames = [f for f in filenames if version_filter in f or encoded in f]

if not filenames:
    raise SystemExit('No matching wheel found for version filter')

# Pick latest
best_name = sorted(filenames)[-1]
best_encoded = best_name.replace('+', '%2B')

# Try each base URL for actual download (listing may succeed at one path
# while downloads only work at another)
for base in bases:
    candidate = base.rstrip('/') + '/' + best_encoded
    if is_downloadable(candidate):
        print(candidate)
        sys.exit(0)

# None passed HEAD check — return the first candidate and let pip error clearly
print(bases[0].rstrip('/') + '/' + best_encoded)
" 2>/dev/null)

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
