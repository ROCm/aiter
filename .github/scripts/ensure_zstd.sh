#!/bin/bash

set -euo pipefail

if command -v zstd >/dev/null 2>&1 && command -v unzstd >/dev/null 2>&1; then
    zstd --version
    exit 0
fi

install_zstd() {
    if command -v apt-get >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1; then
        sudo apt-get update || return $?
        sudo apt-get install -y zstd || return $?
    elif command -v apt-get >/dev/null 2>&1; then
        apt-get update || return $?
        apt-get install -y zstd || return $?
    else
        return 127
    fi
}

install_status=0
install_zstd || install_status=$?

if [ "${install_status}" -ne 0 ]; then
    echo "::warning::Installing zstd failed with exit code ${install_status}; Triton wheel cache restore may miss"
fi

if command -v zstd >/dev/null 2>&1 && command -v unzstd >/dev/null 2>&1; then
    zstd --version
else
    echo "::warning::zstd/unzstd are still unavailable; Triton wheel cache restore may miss"
fi
