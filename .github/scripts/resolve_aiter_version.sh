#!/usr/bin/env bash
# Resolve AITER package spec for pip install.
# Handles PEP 440 local versions (e.g. +rocm7.2.1) that pip won't auto-select.
#
# Usage: source this file after setting AITER_INDEX_URL and AITER_VERSION
# Output: sets AITER_PKG (e.g. "amd-aiter==0.1.12.post1+rocm7.2.1")

if [ -n "${AITER_VERSION:-}" ]; then
  AITER_PKG="amd-aiter==${AITER_VERSION}"
elif [ -n "${AITER_INDEX_URL:-}" ]; then
  AITER_PKG=$(python3 -c "
import re, urllib.request
url = '${AITER_INDEX_URL}'.rstrip('/') + '/amd-aiter/'
with urllib.request.urlopen(url, timeout=30) as resp:
    html = resp.read().decode()
versions = re.findall(r'amd_aiter-([^-]+)-cp312-cp312-linux_x86_64\.whl', html)
# Deduplicate (URL-encoded and plain), keep only decoded versions
clean = sorted(set(v.replace('%2B', '+') for v in versions))
if clean:
    print(f'amd-aiter=={clean[-1]}')
else:
    print('amd-aiter')
" 2>/dev/null || echo "amd-aiter")
else
  AITER_PKG="amd-aiter"
fi

echo "AITER package: ${AITER_PKG}"
