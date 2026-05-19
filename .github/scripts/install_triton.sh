#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
REQ_FILE=${TRITON_REQUIREMENTS_FILE:-"${REPO_ROOT}/.github/requirements/triton-test.txt"}

if [[ ! -f "${REQ_FILE}" ]]; then
    echo "Triton requirements file not found: ${REQ_FILE}" >&2
    exit 1
fi

TRITON_INDEX_URL=$(awk '$1 == "--extra-index-url" { print $2; exit }' "${REQ_FILE}")
TRITON_SPEC=$(awk '/^triton==/ { print; exit }' "${REQ_FILE}")

if [[ -z "${TRITON_INDEX_URL}" || -z "${TRITON_SPEC}" ]]; then
    echo "Could not find Triton index URL and pin in ${REQ_FILE}" >&2
    exit 1
fi

python3 -m pip uninstall -y triton pytorch-triton pytorch-triton-rocm triton-rocm amd-triton || true

echo "Installing ${TRITON_SPEC} from ${TRITON_INDEX_URL}"
python3 -m pip install --extra-index-url "${TRITON_INDEX_URL}" "${TRITON_SPEC}"

python3 - <<'PY'
import triton
from packaging.version import Version

if Version(triton.__version__) < Version("3.6.0"):
    raise SystemExit(f"triton>=3.6.0 is required, found {triton.__version__}")

print(f"Installed triton {triton.__version__}")
PY
