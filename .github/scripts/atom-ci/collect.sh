#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
set -uo pipefail

mkdir -p atom_diagnostics
for file in /tmp/atom_server.log /tmp/atom_client.log; do
    docker cp "atom_aiter_test:$file" "atom_diagnostics/$(basename "$file")" || true
done
docker exec atom_aiter_test bash -lc '
    cd /app/aiter-test/aiter/jit/build || exit
    find . -type f \( -name .ninja_log -o -name build.ninja -o -name "*.log" \) -print0 |
        tar --null -T - -czf /workspace/atom_diagnostics/jit-build-logs.tar.gz
' || true
docker exec atom_aiter_test bash -lc 'cd /app/aiter-test && git rev-parse HEAD && git submodule status' > atom_diagnostics/aiter-revisions.txt 2>&1 || true
docker exec atom_aiter_test bash -lc 'pip list --format=json' > atom_diagnostics/packages.json 2>&1 || true
docker image inspect "${BASE_IMAGE:-rocm/atom-dev:latest}" --format '{{json .RepoDigests}}' > atom_diagnostics/image-digests.json 2>&1 || true
docker stats --no-stream atom_aiter_test > atom_diagnostics/docker-stats.txt 2>&1 || true
docker inspect --format '{{json .State}}' atom_aiter_test > atom_diagnostics/container-state.json 2>&1 || true
if [ -d accuracy_test_results ]; then
    cp -r accuracy_test_results atom_diagnostics/
fi
if [ -f atom_accuracy_output.txt ]; then
    cp atom_accuracy_output.txt atom_diagnostics/
fi
exit 0
