#!/usr/bin/env bash
# SM-SK2-RAN-r2 HOST build (STAMPS=1 adds the debug buffer; device variant must match) (source-only; NOT executed). Plain clang++ -x c++ (never define __HIPCC__).
# Needs gen/sm_expected_hashes.h from build_device.sh. Image gate uses in-tree host/sha256_image.hpp (no OpenSSL).
set -euo pipefail
CAND="${CAND:-/work/candidates/SM-SK2-RAN-r1}"
GEN_DIR="${GEN_DIR:?set GEN_DIR to the gen directory of the device build}"
MORI_ROOT="${MORI_ROOT:-/reference-mori}"
MORI_SO="${MORI_SO:-/reference-mori/build/src/cco/libmori_cco.so}"
CLANGXX="${CLANGXX:-/opt/rocm/llvm/bin/clang++}"
BUILD_DIR="${BUILD_DIR:?set BUILD_DIR to a fresh, non-existent output directory}"
[ -e "${BUILD_DIR}" ] && { echo "refuse: BUILD_DIR exists (${BUILD_DIR})" >&2; exit 1; }
bash -n "${BASH_SOURCE[0]}"
( cd "${CAND}" && sha256sum -c SOURCE_SHA256.txt )
[ -f "${MORI_SO}" ] || { echo "MORI_SO_MISSING: ${MORI_SO}"; exit 2; }
mkdir -p "${BUILD_DIR}"
STAMPS="${STAMPS:-0}"; HDENSE=0; case "${MODE:-indexer}" in indexer) HIDX=1;; reuse) HIDX=0;; dense) HIDX=0; HDENSE=1;; *) echo "MODE must be indexer, reuse or dense" >&2; exit 2;; esac; HOSTDEFS=( -DSM_R5X_INDEXER=${HIDX} -DSM_DENSE=${HDENSE} -DSM_T1_B_HEADS=${B_HEADS:-2} ); [ "${STAMPS}" = "1" ] && HOSTDEFS+=( -DSM_STAMPS_HOST=1 )   # must match the device variant
FLAGS=( "${HOSTDEFS[@]}" -x c++ -D__HIP_PLATFORM_AMD__=1 -I/opt/rocm/include -I"${GEN_DIR}" -I"${CAND}/interface" -I"${CAND}/host" -I"${MORI_ROOT}/include"
        -std=c++20 -O3 -fno-fast-math -fPIC -pthread )
LIBS=( -x none "${MORI_SO}" -L/opt/rocm/lib -lamdhip64 -ldl -Wl,--no-undefined )
{ echo "# SM-SK2-RAN-r1 host build"; "${CLANGXX}" --version; echo "FLAGS: ${FLAGS[*]}"; } | tee "${BUILD_DIR}/build-command.txt"
"${CLANGXX}" "${FLAGS[@]}" -shared "${CAND}/host/driver_sm.cpp" -o "${BUILD_DIR}/sm_driver.so" "${LIBS[@]}"
echo "HOST_BUILD_OK"
