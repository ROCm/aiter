#!/bin/bash

set -u

if command -v zstd >/dev/null 2>&1 && command -v unzstd >/dev/null 2>&1; then
    zstd --version
    exit 0
fi

if command -v apt-get >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y zstd
elif command -v apt-get >/dev/null 2>&1; then
    apt-get update && apt-get install -y zstd
else
    echo "::warning::apt-get is unavailable; Triton wheel cache restore may miss if this runner lacks zstd"
fi

if command -v zstd >/dev/null 2>&1 && command -v unzstd >/dev/null 2>&1; then
    zstd --version
else
    echo "::warning::zstd/unzstd are still unavailable; Triton wheel cache restore may miss"
fi
