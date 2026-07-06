#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/data/yanguahe/code/wk_sp1}"
AITER_DIR="${AITER_DIR:-${WORKSPACE_DIR}/aiter}"
LLVM_OBJDUMP="${LLVM_OBJDUMP:-${WORKSPACE_DIR}/llvm-project/buildmlir/bin/llvm-objdump}"
MCPU="${MCPU:-gfx1250}"

SRC_DIR="${SRC_DIR:-${AITER_DIR}/hsa/${MCPU}}"
OUT_DIR="${OUT_DIR:-${AITER_DIR}/dump_asm/hsa/${MCPU}}"
SRC_DIR="${SRC_DIR%/}"
OUT_DIR="${OUT_DIR%/}"

if [[ ! -x "${LLVM_OBJDUMP}" ]]; then
    echo "ERROR: llvm-objdump is not executable: ${LLVM_OBJDUMP}" >&2
    exit 1
fi

if [[ ! -d "${SRC_DIR}" ]]; then
    echo "ERROR: source directory does not exist: ${SRC_DIR}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

dumped=0
failed=0

while IFS= read -r -d '' co_file; do
    rel_path="${co_file#${SRC_DIR}/}"
    out_file="${OUT_DIR}/${rel_path%.co}.s"
    mkdir -p "$(dirname "${out_file}")"

    echo "Dumping ${rel_path} -> ${out_file#${OUT_DIR}/}"
    if "${LLVM_OBJDUMP}" -d --mcpu="${MCPU}" "${co_file}" > "${out_file}"; then
        dumped=$((dumped + 1))
    else
        failed=$((failed + 1))
        echo "ERROR: failed to dump ${co_file}" >&2
        rm -f "${out_file}"
    fi
done < <(find "${SRC_DIR}" -type f -name '*.co' -print0 | sort -z)

echo "Done. dumped=${dumped}, failed=${failed}, output=${OUT_DIR}"

if [[ "${dumped}" -eq 0 ]]; then
    echo "WARNING: no .co files found under ${SRC_DIR}" >&2
fi

if [[ "${failed}" -ne 0 ]]; then
    exit 1
fi
