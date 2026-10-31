#!/bin/bash
# Python environment helper for the aiter swap.
#
# Usage: env.sh backup  <backup_dir>   save the current aiter editable install, flydsl and pip freeze
#        env.sh install <aiter_dir>    pip install -e <aiter_dir> (keeps the system triton), check the import path
#        env.sh restore <backup_dir>   put the backed-up aiter/flydsl back, compare pip freeze
#        env.sh status                 show installed versions and import paths
set -euo pipefail
SP=$(python3 -c "import site; print(site.getsitepackages()[0])")
CMD=${1:-status}

case "$CMD" in
  backup)
    B=$2
    if [ -f "$B/pip_freeze_before.txt" ]; then
      echo "backup already exists in $B, not overwriting"
      exit 0
    fi
    mkdir -p "$B/aiter_editable" "$B/flydsl_pkg"
    pip freeze 2>/dev/null > "$B/pip_freeze_before.txt"
    cp -a "$SP"/__editable__.amd_aiter-*.pth "$SP"/__editable___amd_aiter_*_finder.py "$SP"/amd_aiter-*.dist-info "$B/aiter_editable/"
    cp -a "$SP"/flydsl "$SP"/flydsl-*.dist-info "$SP"/flydsl.libs "$B/flydsl_pkg/"
    echo "backed up to $B"
    ;;
  install)
    A=$(cd "$2" && pwd)
    # aiter JIT needs the CK headers; clone them from /app/aiter when the submodule is empty.
    if [ -z "$(ls -A "$A/3rdparty/composable_kernel" 2>/dev/null)" ]; then
      git -C "$A" -c protocol.file.allow=always \
        -c submodule.3rdparty/composable_kernel.url=/app/aiter/3rdparty/composable_kernel \
        submodule update --init 3rdparty/composable_kernel \
        || git -C "$A" submodule update --init 3rdparty/composable_kernel
    fi
    # AITER_USE_SYSTEM_TRITON=1 keeps /app/triton-mi450; without it setup.py swaps in PyPI triton.
    LOG=${TMPDIR:-/tmp}/aiter_install_$$.log
    if ! (cd "$A" && AITER_USE_SYSTEM_TRITON=1 python3 -m pip install --break-system-packages -e . > "$LOG" 2>&1); then
      tail -n 20 "$LOG"
      exit 1
    fi
    grep -E "^Successfully installed" "$LOG" || true
    got=$(cd /tmp && PYTHONPATH=$A python -c "import aiter; print(aiter.__file__)" 2>/dev/null | tail -1)
    if [ "$got" != "$A/aiter/__init__.py" ]; then
      echo "import check failed: aiter resolves to '$got'"
      exit 1
    fi
    echo "installed $A (import ok with PYTHONPATH=$A)"
    ;;
  restore)
    B=$2
    if pgrep -f "^python -m atom.entrypoints.openai_server" > /dev/null; then
      echo "an ATOM server is still running; stop it before restoring"
      exit 1
    fi
    python3 -m pip uninstall -y --break-system-packages amd-aiter flydsl 2>&1 | grep -i "successfully" || true
    cp -a "$B/aiter_editable/." "$SP/"
    if [ -d "$B/flydsl_pkg" ]; then cp -a "$B/flydsl_pkg/." "$SP/"; else cp -a "$B"/flydsl_dev*/. "$SP/"; fi
    if pip freeze 2>/dev/null | diff -q "$B/pip_freeze_before.txt" - > /dev/null; then
      echo "restored: pip freeze identical to $B/pip_freeze_before.txt"
    else
      echo "restored, but pip freeze differs from the backup:"
      pip freeze 2>/dev/null | diff "$B/pip_freeze_before.txt" - || true
    fi
    ;;
  status)
    pip list 2>/dev/null | grep -E "^(amd-aiter|flydsl|triton|atom|torch) " || true
    echo "PYTHONPATH=${PYTHONPATH:-}"
    (cd /tmp && python -c "import aiter, atom; print('aiter ->', aiter.__file__); print('atom  ->', atom.__file__)" 2>/dev/null | grep -- "->")
    ;;
  *)
    sed -n '2,8p' "$0"
    exit 1
    ;;
esac
