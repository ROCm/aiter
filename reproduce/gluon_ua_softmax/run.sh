#!/bin/bash
# Run both paths in separate processes; an LLVM abort kills the process.
cd "$(dirname "$0")"
export TRITON_CACHE_DIR=$(mktemp -d)
for mode in --fixed --broken; do
    python repro.py $mode
    echo "  $mode -> rc $?"
done
rm -rf "$TRITON_CACHE_DIR"
