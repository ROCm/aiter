#!/usr/bin/env bash
set -euo pipefail

echo
echo "============================================================"
echo "Aiter GPU visibility check"
echo "============================================================"

find_rocm_smi() {
    for candidate in rocm-smi rocmsmi /opt/rocm/bin/rocm-smi /opt/rocm/bin/rocmsmi; do
        if command -v "${candidate}" >/dev/null 2>&1; then
            command -v "${candidate}"
            return 0
        fi
        if [[ -x "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

count_rocm_smi_gpus() {
    local rocm_smi="$1"
    local output_file
    output_file="$(mktemp)"

    set +e
    "${rocm_smi}" --showid 2>&1 | tee "${output_file}"
    local status=${PIPESTATUS[0]}
    set -e

    local count
    count="$(awk '/GPU\[[0-9]+\]/ && !seen[$1]++ { count++ } END { print count + 0 }' "${output_file}")"
    rm -f "${output_file}"

    echo "rocm-smi/rocmsmi command: ${rocm_smi}"
    echo "rocm-smi/rocmsmi visible GPU count: ${count}"
    if [[ "${status}" -ne 0 ]]; then
        echo "::warning::${rocm_smi} --showid exited with status ${status}"
    fi
}

if rocm_smi="$(find_rocm_smi)"; then
    count_rocm_smi_gpus "${rocm_smi}"
else
    echo "::warning::rocm-smi/rocmsmi was not found in the test container"
fi

python3 - <<'PY'
try:
    import torch

    print("HIP runtime GPU visibility:")
    count = torch.cuda.device_count()
    print(f"  torch.cuda.is_available={torch.cuda.is_available()}")
    print(f"  torch.cuda.device_count={count}")
    for idx in range(count):
        print(f"  torch.cuda.device[{idx}]={torch.cuda.get_device_name(idx)}")
except Exception as exc:
    print(f"::warning::Failed to query HIP devices through torch: {exc}")
PY

echo "============================================================"
echo
