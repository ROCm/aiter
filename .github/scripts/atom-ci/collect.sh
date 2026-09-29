#!/bin/bash
set -uo pipefail

mkdir -p atom_diagnostics
touch atom_diagnostics/stop-monitor
for file in /tmp/atom_server.log /tmp/atom_client.log; do
    docker cp "atom_aiter_test:$file" "atom_diagnostics/$(basename "$file")" || true
done
docker exec atom_aiter_test bash -lc '
    cd /app/aiter-test/aiter/jit/build || exit
    find . -type f \( -name .ninja_log -o -name build.ninja -o -name "*.log" \) -print0 |
        tar --null -T - -czf /workspace/atom_diagnostics/jit-build-logs.tar.gz
' || true
docker stats --no-stream atom_aiter_test > atom_diagnostics/docker-stats.txt 2>&1 || true
docker inspect --format '{{json .State}}' atom_aiter_test > atom_diagnostics/container-state.json 2>&1 || true
if [ -d accuracy_test_results ]; then
    cp -r accuracy_test_results atom_diagnostics/
fi
if [ -f atom_accuracy_output.txt ]; then
    cp atom_accuracy_output.txt atom_diagnostics/
fi
