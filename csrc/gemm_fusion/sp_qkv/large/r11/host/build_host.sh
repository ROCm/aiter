#!/usr/bin/env bash
# LG-QSDMA-r11 HOST driver build (source-only recipe): libmori_cco-linked shared lib for ctypes (bridge_comm.py).
# Plain clang++ -x c++ (never define __HIPCC__/__CUDACC__), as PR695-M4/host/build_host.sh.
set -euo pipefail
LG="${LG:-/work/candidates/LG-QSDMA-r11}"
MORI_ROOT="${MORI_ROOT:-/reference-mori}"
MORI_SO="${MORI_SO:-/reference-mori/build/src/cco/libmori_cco.so}"
CLANGXX="${CLANGXX:-/opt/rocm/llvm/bin/clang++}"
LG_REMOTE_HALF="${LG_REMOTE_HALF:-0}"   # must equal the device build
BUILD_DIR="${BUILD_DIR:?set BUILD_DIR to a fresh, non-existent output directory}"
[ -e "${BUILD_DIR}" ] && { echo "refuse: BUILD_DIR exists (${BUILD_DIR})" >&2; exit 1; }
bash -n "${BASH_SOURCE[0]}"
( cd "${LG}" && sha256sum -c "${LG}/SOURCE_SHA256.txt" )
[ -f "${MORI_SO}" ] || { echo "MORI_SO_MISSING: ${MORI_SO}"; exit 2; }
mkdir -p "${BUILD_DIR}"
FLAGS=( -x c++ -D__HIP_PLATFORM_AMD__=1 -I/opt/rocm/include -I"${LG}/interface" -I"${MORI_ROOT}/include"
        -DLG_REMOTE_HALF=${LG_REMOTE_HALF} -DLG_R6_POSTER=${LG_R6_POSTER:-1} -DLG_X_INDEXER=${LG_X_INDEXER:-1} -DLG_R8_FOLD=${LG_R8_FOLD:-0} -DLG_DEBUG_POSTCOUNT=${LG_DEBUG_POSTCOUNT:-0} -std=c++20 -O2 -fno-fast-math -fPIC -pthread )
LIBS=( -x none "${MORI_SO}" -L/opt/rocm/lib -lamdhip64 -ldl -Wl,--no-undefined )
{ echo "# LG-QSDMA-r11 host build"; "${CLANGXX}" --version; echo "FLAGS: ${FLAGS[*]}"; } | tee "${BUILD_DIR}/build-command.txt"
"${CLANGXX}" "${FLAGS[@]}" -shared "${LG}/host/lg_comm_driver.cpp" -o "${BUILD_DIR}/liblg_comm.so" "${LIBS[@]}"
echo "LG_HOST_BUILD_OK"