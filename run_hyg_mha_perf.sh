#!/usr/bin/env bash
set -euo pipefail

cd /data/yanguahe/code/wk_sp1/aiter

git fetch origin hyg_dev
git reset --hard origin/hyg_dev

mkdir -p perf_logs

set +e
docker exec -i hyg_fyd1 bash -lc 'cd /data/yanguahe/code/wk_sp1/aiter && bash ./run-mha-perf.sh' 2>&1 | tee perf_logs/run-mha-perf.log
test_status=${PIPESTATUS[0]}
set -e

git add -f perf_logs/run-mha-perf.log
if ! git diff --cached --quiet; then
    git -c user.name=yanguahe -c user.email=yanguahe@amd.com commit --amend --author='yanguahe <yanguahe@amd.com>' -m Update
fi
git push -f origin hyg_dev

exit "$test_status"
