# Tools for the ISA-level kernel optimization guide (docs/isa_kernel_optimization.md):
# ROCm LLVM (llvm-objdump / llvm-readelf / llvm-objcopy / clang / ld.lld) and rocprofv3.
#
# This image is enough to inspect, extract, edit and reassemble .co files.  To *run*
# a modified kernel you additionally need PyTorch + AITER; either start from a
# rocm/pytorch image instead and install AITER from source as described in the
# top-level README, or mount an existing environment into this container.
ARG ROCM_IMAGE=rocm/dev-ubuntu-24.04:7.2.1-complete
FROM ${ROCM_IMAGE}

# rocprof-trace-decoder (needed by `rocprofv3 --att`) is not part of the ROCm 7.2
# packages.  Build it from rocm-systems; pin ROCM_SYSTEMS_REF to a tag or branch for
# reproducible images.  Pre-built binaries are also published at
# https://github.com/ROCm/rocprof-trace-decoder/releases.
ARG ROCM_SYSTEMS_REF=develop
RUN git clone --depth 1 --branch "${ROCM_SYSTEMS_REF}" --filter=blob:none --sparse \
        https://github.com/ROCm/rocm-systems.git /tmp/rocm-systems && \
    cd /tmp/rocm-systems && git sparse-checkout set projects/rocprof-trace-decoder && \
    cd projects/rocprof-trace-decoder && \
    cmake -B build -DCMAKE_INSTALL_PREFIX=/opt/rocm -DLLVM_DIR=/opt/rocm/llvm/lib/cmake/llvm && \
    cmake --build build -j"$(nproc)" && \
    cmake --install build && \
    rm -rf /tmp/rocm-systems

COPY extract_asm.py analyze_kernel.py roundtrip.sh /workspace/
RUN chmod +x /workspace/roundtrip.sh
WORKDIR /workspace

# Sanity check: the tools the guide relies on
RUN /opt/rocm/llvm/bin/llvm-objdump --version | grep -m1 -i version && \
    /opt/rocm/llvm/bin/clang --version | head -1 && \
    ls /opt/rocm/llvm/bin/ld.lld /opt/rocm/llvm/bin/llvm-readelf /opt/rocm/llvm/bin/llvm-objcopy && \
    (rocprofv3 --help >/dev/null 2>&1 && echo "rocprofv3 OK" || echo "rocprofv3 not found")

CMD ["/bin/bash"]
