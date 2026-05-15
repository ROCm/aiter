FROM rocm/dev-ubuntu-24.04:7.2.1-complete

# LLVM tools are already at /opt/rocm/llvm/bin/ in this image
# rocprofv3 is already at /opt/rocm/bin/

# Install Python deps and AITER
RUN pip install --no-cache-dir aiter && \
    pip install --no-cache-dir pandas matplotlib

# Build ATT trace decoder from source (not shipped as pre-built binary)
RUN git clone --depth 1 --branch develop \
        https://github.com/ROCm/rocm-systems.git /tmp/rocm-systems && \
    cd /tmp/rocm-systems/projects/rocprof-trace-decoder && \
    cmake -B build -DCMAKE_INSTALL_PREFIX=/opt/rocm && \
    cmake --build build -j$(nproc) && \
    cmake --install build && \
    rm -rf /tmp/rocm-systems

# Copy example scripts
COPY extract_asm.py analyze_kernel.py roundtrip.sh /workspace/
RUN chmod +x /workspace/roundtrip.sh

WORKDIR /workspace

# Verify tools are available
RUN /opt/rocm/llvm/bin/llvm-objdump --version | head -1 && \
    /opt/rocm/llvm/bin/clang++ --version | head -1 && \
    rocprofv3 --version 2>&1 | head -1 || true

CMD ["/bin/bash"]
