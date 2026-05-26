#!/bin/bash

set -euo pipefail

TRITON_WHEEL_DIR="${1:-triton_wheels}"
ARTIFACT_PREFIX="${TRITON_WHEEL_ARTIFACT_PREFIX:-triton_wheelhouse}"

write_outputs() {
    if [[ -z "${GITHUB_OUTPUT:-}" ]]; then
        return 0
    fi

    {
        echo "artifact_name=${artifact_name}"
        echo "artifact_run_id=${artifact_run_id}"
        echo "upload_artifact=${upload_artifact}"
        echo "wheel_sha256=${wheel_sha256}"
        echo "rocm_version=${rocm_major_minor}"
        echo "python_tag=${python_tag}"
    } >> "${GITHUB_OUTPUT}"
}

artifact_name="${ARTIFACT_PREFIX}_missing"
artifact_run_id="${GITHUB_RUN_ID:-}"
upload_artifact="false"
wheel_sha256=""
rocm_major_minor="unknown"
python_tag="py_unknown"

shopt -s nullglob
wheels=("${TRITON_WHEEL_DIR}"/triton*.whl)
shopt -u nullglob

if [[ "${#wheels[@]}" -ne 1 ]]; then
    echo "::warning::Expected exactly one Triton wheel in ${TRITON_WHEEL_DIR}, found ${#wheels[@]}."
    write_outputs
    exit 0
fi

metadata_file="${TRITON_WHEEL_DIR}/triton-wheel-metadata.env"
if [[ -f "${metadata_file}" ]]; then
    # shellcheck disable=SC1090
    source "${metadata_file}"
    rocm_major_minor="${ROCM_MAJOR_MINOR:-unknown}"
    python_tag="${PYTHON_TAG:-py_unknown}"
fi

wheel_sha256=$(sha256sum "${wheels[0]}" | awk '{print $1}')
safe_rocm="${rocm_major_minor//./_}"
artifact_name="${ARTIFACT_PREFIX}_rocm_${safe_rocm}_${python_tag}_${wheel_sha256:0:16}"
artifact_run_id="${GITHUB_RUN_ID:-}"
upload_artifact="true"

if [[ -n "${GITHUB_REPOSITORY:-}" && -n "${GH_TOKEN:-${GITHUB_TOKEN:-}}" ]]; then
    echo "Checking for existing GitHub artifact ${artifact_name}"
    if artifacts_json=$(gh api "/repos/${GITHUB_REPOSITORY}/actions/artifacts?name=${artifact_name}&per_page=20"); then
        existing_run_id=$(printf '%s' "${artifacts_json}" | python3 -c '
import json
import sys

payload = json.load(sys.stdin)
artifacts = [
    artifact for artifact in payload.get("artifacts", [])
    if not artifact.get("expired")
]
artifacts.sort(key=lambda artifact: artifact.get("created_at", ""), reverse=True)
if artifacts:
    print(artifacts[0].get("workflow_run", {}).get("id", ""))
')
        if [[ -n "${existing_run_id}" ]]; then
            echo "Found existing Triton wheel artifact ${artifact_name} in run ${existing_run_id}; skipping upload."
            artifact_run_id="${existing_run_id}"
            upload_artifact="false"
        fi
    else
        echo "::warning::Unable to query GitHub artifacts; uploading current Triton wheel artifact."
    fi
else
    echo "GitHub repository/token is unavailable; uploading current Triton wheel artifact."
fi

echo "Resolved Triton wheel artifact: ${artifact_name}"
echo "Artifact run id: ${artifact_run_id}"
echo "Upload artifact: ${upload_artifact}"
write_outputs
