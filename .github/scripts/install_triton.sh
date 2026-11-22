#!/bin/bash
set -euo pipefail

# --resolve: print the AMD Triton index URL and exit (used by download_triton_wheel.sh).
RESOLVE_ONLY=0
[[ "${1:-}" == "--resolve" ]] && RESOLVE_ONLY=1

if [[ ${RESOLVE_ONLY} -eq 0 ]] && python3 - <<'PYCHECK' 2>/dev/null
import torch
from packaging.version import Version
exit(0 if Version(torch.__version__.split("+")[0].split("dev")[0]) < Version("2.9.1") else 1)
PYCHECK
then
    TRITON_INFO=$(python3 - <<'PYINFO' 2>/dev/null
try:
    import importlib.metadata as m
    for n in ["triton", "triton-rocm", "pytorch-triton-rocm", "pytorch-triton", "amd-triton"]:
        try:
            print(f"(keeping {n}=={m.version(n)})")
            break
        except Exception:
            pass
except Exception:
    pass
PYINFO
)
    echo "[aiter] torch < 2.9.1 detected, triton reinstall skipped for compatibility${TRITON_INFO:+ ${TRITON_INFO}}."
    echo "To use aiter-compatible triton, please upgrade torch to 2.9.1 or later."
    exit 0
fi

TRITON_VERSION=3.9.0
if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
    echo "[aiter] triton ${TRITON_VERSION} needs Python 3.11 or newer; keeping the existing triton (if any)." >&2
    exit "${RESOLVE_ONLY}"
fi
# Full wheel tag, e.g. cp312-cp312-linux_x86_64 (cp313t when free-threaded).
TRITON_WHEEL_TAG=$(python3 -c 'import platform, sys, sysconfig
p = "cp%d%d" % sys.version_info[:2]
print(p, p + ("t" if sysconfig.get_config_var("Py_GIL_DISABLED") else ""), "linux_" + platform.machine(), sep="-")')

install_triton_from_wheelhouse() {
    local wheel_dir="$1"

    if [[ -z "${wheel_dir}" || ! -d "${wheel_dir}" ]]; then
        return 1
    fi

    local wheels=()
    shopt -s nullglob
    wheels=("${wheel_dir}"/triton*.whl)
    shopt -u nullglob

    if [[ "${#wheels[@]}" -eq 0 ]]; then
        echo "No triton wheels found in ${wheel_dir}"
        return 1
    fi

    echo "Installing triton from local wheelhouse: ${wheel_dir}"
    if ! python3 -m pip install --no-index --find-links "${wheel_dir}" "triton==${TRITON_VERSION}"; then
        echo "Local triton wheel install failed; falling back to the AMD index."
        return 1
    fi

    echo "Installing triton-kernels from local wheelhouse: ${wheel_dir}"
    if ! python3 -m pip install --no-index --find-links "${wheel_dir}" triton-kernels; then
        echo "Local triton-kernels wheel install failed; falling back to the AMD index."
        return 1
    fi

    return 0
}

# 0: the index lists this triton wheel, 1: it does not (or HTTP 403/404), 2: unreachable.
has_triton_wheel() {
    python3 - "$1" "triton-${TRITON_VERSION}+" "-${TRITON_WHEEL_TAG}.whl" <<'PY'
import re, sys, time, urllib.error, urllib.request
url, prefix, suffix = sys.argv[1:]
for attempt in range(3):
    try:
        page = urllib.request.urlopen(url, timeout=30).read().decode()
        break
    except urllib.error.HTTPError:
        sys.exit(1)
    except OSError:
        if attempt == 2:
            sys.exit(2)
        time.sleep(2 * (attempt + 1))
wheels = re.findall(r'triton-[^"#</]+\.whl', page)
sys.exit(0 if any(w.startswith(prefix) and w.endswith(suffix) for w in wheels) else 1)
PY
}

# Set TRITON_INDEX_URL to the first index with a matching wheel, or exit before touching triton.
resolve_triton_index() {
    local candidates=() rocm url rc
    TRITON_DEFAULT_ROCM_VERSION="${TRITON_DEFAULT_ROCM_VERSION:-7.2.0}"
    ROCM_VERSION=$(dpkg -l rocm-core 2>/dev/null | awk '/^ii/{print $3}' || true)
    if [[ -z "$ROCM_VERSION" ]]; then
        # RPM-based systems (e.g. rocm-core-7.2.0.70200-43.el8.x86_64 -> 7.2.0.70200)
        ROCM_VERSION=$(rpm -q --queryformat '%{VERSION}' rocm-core 2>/dev/null) || ROCM_VERSION=""
    fi
    if [[ -z "$ROCM_VERSION" ]]; then
        # pip ROCm SDK images have no rocm-core package
        ROCM_VERSION=$(python3 -m pip list --format=freeze 2>/dev/null | sed -n 's/^rocm==\([0-9][0-9.]*\).*/\1/p' || true)
    fi
    if [[ -n "$ROCM_VERSION" ]]; then
        ROCM_MAJOR_MINOR=$(echo "$ROCM_VERSION" | cut -d. -f1,2)
        candidates+=("${ROCM_MAJOR_MINOR}.0")
        # A ROCm release without its own index uses the newest one of its major (7.12 -> 7.2).
        case "${ROCM_MAJOR_MINOR%%.*}" in 7) candidates+=(7.2.0) ;; 10) candidates+=(10.2.0) ;; esac
    else
        echo "rocm-core not found; using default ROCm version ${TRITON_DEFAULT_ROCM_VERSION}" >&2
    fi
    candidates+=("${TRITON_DEFAULT_ROCM_VERSION}")
    for rocm in "${candidates[@]}"; do
        url="https://pypi.amd.com/triton/release-1031/rocm-${rocm}/simple/"
        rc=0
        has_triton_wheel "${url}triton/" || rc=$?
        if [[ ${rc} -eq 0 ]]; then
            TRITON_INDEX_URL="${url}"
            return
        elif [[ ${rc} -eq 2 ]]; then
            echo "[aiter] Cannot reach ${url}; triton not changed." >&2
            exit 1
        fi
    done
    echo "[aiter] No triton ${TRITON_VERSION} ${TRITON_WHEEL_TAG} wheel (tried ROCm ${candidates[*]}); triton not changed." >&2
    exit 1
}

TRITON_INDEX_URL=""
TRITON_WHEEL_DIR=${TRITON_WHEEL_DIR:-}
if [[ ${RESOLVE_ONLY} -eq 1 ]]; then
    resolve_triton_index
    echo "${TRITON_INDEX_URL}"
    exit 0
fi
# Use the wheelhouse only if it has a matching wheel; otherwise resolve the index before uninstalling.
if ! compgen -G "${TRITON_WHEEL_DIR:-/nonexistent}/triton-${TRITON_VERSION}+*-${TRITON_WHEEL_TAG}.whl" >/dev/null; then
    TRITON_WHEEL_DIR=""
    resolve_triton_index
fi

python3 -m pip uninstall -y triton pytorch-triton pytorch-triton-rocm triton-rocm amd-triton || true
python3 -m pip uninstall -y triton-kernels || true

if ! install_triton_from_wheelhouse "${TRITON_WHEEL_DIR}"; then
    [[ -n "${TRITON_INDEX_URL}" ]] || resolve_triton_index
    echo "Installing triton from $TRITON_INDEX_URL"
    python3 -m pip install --extra-index-url "$TRITON_INDEX_URL" "triton==${TRITON_VERSION}"

    echo "Installing triton-kernels from $TRITON_INDEX_URL"
    python3 -m pip install --extra-index-url "$TRITON_INDEX_URL" "triton-kernels==1.0.0"
fi

python3 - <<'PY'
import triton
from packaging.version import Version

if Version(triton.__version__) < Version("3.6.0"):
    raise SystemExit(f"triton>=3.6.0 is required, found {triton.__version__}")

print(f"Installed triton {triton.__version__}")
PY
