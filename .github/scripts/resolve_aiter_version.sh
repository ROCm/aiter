#!/usr/bin/env bash
# Resolve AITER wheel URL for pip install.
# Handles broken wheels where filename version != metadata version.
#
# Usage: source this file after setting AITER_INDEX_URL and AITER_VERSION
# Output: sets AITER_INSTALL_CMD (full pip install command for AITER)

if [ -n "${AITER_INDEX_URL:-}" ]; then
  # Find the direct wheel URL from the index
  WHEEL_URL=$(python3 -c "
import re, urllib.request
index = '${AITER_INDEX_URL}'.rstrip('/')
version_filter = '${AITER_VERSION:-}'
url = index + '/amd-aiter/'
with urllib.request.urlopen(url, timeout=30) as resp:
    html = resp.read().decode()
# Find all cp312 wheel hrefs
wheels = re.findall(r'href=\"([^\"]*amd_aiter-[^\"]*cp312-cp312-linux_x86_64\.whl)\"', html)
if not wheels:
    raise SystemExit('No cp312 wheels found')
# Resolve relative URLs
resolved = []
for w in wheels:
    if w.startswith('http'):
        resolved.append(w)
    elif w.startswith('../'):
        resolved.append(index + '/' + w.lstrip('./'))
    else:
        resolved.append(index + '/' + w)
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
    AITER_INSTALL_CMD="pip install --force-reinstall '${WHEEL_URL}'"
    echo "AITER wheel: ${WHEEL_URL}"
  else
    echo "WARNING: Could not resolve wheel URL, falling back to pip install amd-aiter"
    AITER_INSTALL_CMD="pip install --extra-index-url '${AITER_INDEX_URL}' amd-aiter"
  fi
else
  AITER_INSTALL_CMD="pip install amd-aiter"
fi

echo "AITER install: ${AITER_INSTALL_CMD}"
