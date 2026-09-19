#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# One-command gfx950 correctness and performance acceptance for native FlashKDA.
# The benchmark is the caller-side dispatcher: it invokes flash_kda_fwd for HIP
# and chunk_kimi_delta_attn for Triton as two independent library interfaces.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
python_bin=${PYTHON:-python3}
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
result_dir=$(realpath -m -- "${1:-${TMPDIR:-/tmp}/flash-kda-gfx950-${timestamp}}")

case "${result_dir}" in
    "${repo_root}"|"${repo_root}"/*)
        echo "ERROR: result directory must be outside the checkout" >&2
        exit 2
        ;;
esac
if [[ -e ${result_dir} ]]; then
    echo "ERROR: refusing to reuse an existing result path: ${result_dir}" >&2
    exit 2
fi

cd "${repo_root}"
if ! git diff --quiet -- || ! git diff --cached --quiet -- ||
    [[ -n $(git ls-files --others --exclude-standard) ]]; then
    echo "ERROR: acceptance requires a clean checkout" >&2
    git status --short --untracked-files=all >&2
    exit 2
fi
submodule_status=$(git submodule status --recursive)
if grep -Eq '^[-+U]' <<<"${submodule_status}"; then
    echo "ERROR: acceptance requires clean, initialized submodules" >&2
    printf '%s\n' "${submodule_status}" >&2
    exit 2
fi
if ! git submodule foreach --recursive --quiet \
    'test -z "$(git status --porcelain --untracked-files=all)"'; then
    echo "ERROR: acceptance requires clean submodule worktrees" >&2
    exit 2
fi

mkdir -p "${result_dir}"

# Remove inherited route and tuning controls before the first Python import so
# the recorded result always represents this checkout's default policy.
while IFS='=' read -r name _; do
    case "${name}" in
        AITER_FDA_*|AITER_REBUILD|AITER_TRITON_ONLY|FLASH_KDA_*|CHUNK_DELTA_ATTN_*|KDA_*|FLA_*)
            unset "${name}"
            ;;
    esac
done < <(env)
unset CK_DIR HIP_KITTENS_DIR OPUS_GEN_CO_DIR PYTHONOPTIMIZE

export PYTHONPATH="${repo_root}"
export AITER_AOT_IMPORT=1
export AITER_JIT_DIR="${result_dir}/jit"
export AITER_META_DIR="${repo_root}"
export GPU_ARCHS=gfx950
export MAX_JOBS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export TRITON_CACHE_DIR="${result_dir}/triton-cache"
mkdir -p "${AITER_JIT_DIR}" "${TRITON_CACHE_DIR}"

{
    git rev-parse HEAD
    git status --short --branch
    "${python_bin}" --version
    "${python_bin}" - <<'PY'
import torch

props = torch.cuda.get_device_properties(0)
arch_detail = getattr(props, "gcnArchName", "unknown")
arch = arch_detail.split(":", 1)[0]
if arch != "gfx950" or props.multi_processor_count != 256:
    raise SystemExit(
        "acceptance requires the 256-CU gfx950 reference target; "
        f"got {arch}/{props.multi_processor_count} CU"
    )
print(f"torch={torch.__version__}")
print(f"rocm={torch.version.hip}")
print(f"device={props.name}")
print(f"arch={arch_detail}")
print(f"compute_units={props.multi_processor_count}")
PY
} | tee "${result_dir}/environment.txt"

git diff --check

"${python_bin}" -m pytest -q \
    op_tests/triton_tests/chunk_delta_attn/test_flash_kda_native_python_api.py \
    | tee "${result_dir}/python-api.log"

# Build once through the descriptor entry, then let subsequent tests exercise
# raw-v3 as well. The test matrix covers dense/packed, fresh/resume, empty
# sequences, graph replay, GVA ratios 2/4, FP32 state, and BF16 state.
export AITER_REBUILD=1
"${python_bin}" op_tests/op_benchmarks/triton/validate_flash_kda_raw_path.py \
    --tokens 2048 \
    --heads 12 \
    | tee "${result_dir}/raw-abi-and-routing.log"
unset AITER_REBUILD

"${python_bin}" -m pytest -q \
    op_tests/triton_tests/chunk_delta_attn/test_flash_kda_native.py \
    | tee "${result_dir}/correctness.log"

common_bench_args=(
    --execution graph
    --backend native
    --backend triton
    --omit-max-seqlen-hint
    --warmup 10
    --repeat 50
    --require-arch gfx950
    --require-compute-units 256
    --min-speedup 1.0
    --min-geomean-speedup 1.0
    --min-paired-win-fraction 0.5
)

# Reproduce the primary 11-case Hq=2, HV=4 cross-coverage table.
"${python_bin}" op_tests/op_benchmarks/triton/bench_flash_kda_native.py \
    --suite core \
    --heads 2 \
    --value-heads 4 \
    "${common_bench_args[@]}" \
    --csv "${result_dir}/core-hq2-hv4.csv" \
    --raw-csv "${result_dir}/core-hq2-hv4-raw.csv" \
    --json "${result_dir}/core-hq2-hv4.json" \
    | tee "${result_dir}/core-hq2-hv4.log"

# Reproduce the higher-ratio GVA acceptance cells.
"${python_bin}" op_tests/op_benchmarks/triton/bench_flash_kda_native.py \
    --case single-2k \
    --case ragged-16k \
    --case resume-4x4k \
    --heads 2 \
    --value-heads 8 \
    "${common_bench_args[@]}" \
    --csv "${result_dir}/ratio4-hq2-hv8.csv" \
    --raw-csv "${result_dir}/ratio4-hq2-hv8-raw.csv" \
    --json "${result_dir}/ratio4-hq2-hv8.json" \
    | tee "${result_dir}/ratio4-hq2-hv8.log"

echo "FlashKDA gfx950 acceptance passed. Results: ${result_dir}"
