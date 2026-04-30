#!/usr/bin/env bash
# Resolve AITER wheel URL for pip install.
# Handles broken wheels where filename version != metadata version.
#
# Usage: source this file after setting AITER_INDEX_URL and AITER_VERSION
# Optional: set AITER_PYTHON_TAG (e.g. cp310, cp312) before sourcing (default: cp312)
# Output: sets AITER_INSTALL_CMD (full pip install command for AITER)

AITER_PYTHON_TAG="${AITER_PYTHON_TAG:-cp312}"

if [ -n "${AITER_INDEX_URL:-}" ]; then
  # Find the direct wheel URL from the index
  WHEEL_URL=$(python3 -c "
import re, urllib.parse, urllib.request
index = '${AITER_INDEX_URL}'.rstrip('/')
version_filter = '${AITER_VERSION:-}'
pytag = '${AITER_PYTHON_TAG}'
# If the URL already ends with a package path, use it directly; otherwise append /amd-aiter/
if index.rstrip('/').endswith('amd-aiter'):
    url = index + '/'
else:
    url = index + '/amd-aiter/'
with urllib.request.urlopen(url, timeout=30) as resp:
    html = urllib.parse.unquote(resp.read().decode())
# Find all wheels matching the requested Python tag (any platform tag)
pattern = r'href=\"([^\"]*amd_aiter-[^\"]*' + pytag + r'-' + pytag + r'-[^\"]+\.whl)\"'
wheels = re.findall(pattern, html)
if not wheels:
    # Try without href quotes (some indices use different formatting)
    pattern2 = r'(amd_aiter-[^\s\"<>]*' + pytag + r'-' + pytag + r'-[^\s\"<>]+\.whl)'
    wheels = re.findall(pattern2, html)
if not wheels:
    raise SystemExit(f'No {pytag} wheels found at {url}')
# Resolve relative URLs
resolved = []
for w in wheels:
    if w.startswith('http'):
        resolved.append(w)
    elif w.startswith('../'):
        resolved.append(index + '/' + w.lstrip('./'))
    else:
        resolved.append(url + w)
# Filter by version if specified
if version_filter:
    encoded = version_filter.replace('+', '%2B')
    resolved = [w for w in resolved if version_filter in w or encoded in w]
# Pick latest (last alphabetically)
if resolved:
    print(sorted(resolved)[-1])
else:
    raise SystemExit('No matching wheel found')
" 2>/dev/null)

  if [ -n "${WHEEL_URL}" ]; then
    # Try direct URL first; if 403, fall back to pip index install
    AITER_INSTALL_CMD="pip install --force-reinstall '${WHEEL_URL}' || pip install --force-reinstall --extra-index-url '${AITER_INDEX_URL}' 'amd-aiter==${AITER_VERSION:-}'"
    echo "AITER wheel: ${WHEEL_URL}"
  else
    echo "WARNING: Could not resolve ${AITER_PYTHON_TAG} wheel URL, falling back to pip install amd-aiter"
    AITER_INSTALL_CMD="pip install --extra-index-url '${AITER_INDEX_URL}' amd-aiter"
  fi
else
  AITER_INSTALL_CMD="pip install amd-aiter"
fi

echo "AITER install: ${AITER_INSTALL_CMD}"
