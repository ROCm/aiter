FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
ENV DEBIAN_FRONTEND=noninteractive
ARG ROCM_VERSION=10.0.0
ARG ROCM_WHEEL_INDEX=https://stable.repo.amd.com/rocm/whl-next/
# Ensure video and render groups exist
RUN groupadd -g 44 video || true
RUN groupadd -g 993 render || true
ARG USERNAME=aiter
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${USERNAME} && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME} && usermod -aG video,render ${USERNAME}
RUN usermod -aG sudo "${USERNAME}" && \
    echo "%sudo ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/devcontainer && \
    chmod 0440 /etc/sudoers.d/devcontainer
# Install system dependencies for aiter, Composable Kernel, and vLLM development.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    git-lfs \
    openssh-server \
    vim \
    nano \
    clang-format-18 \
    dos2unix \
    cmake \
    ninja-build \
    wget \
    curl \
    build-essential \
    ccache \
    ca-certificates \
    cppcheck \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libboost-dev \
    libmsgpack-cxx-dev \
    libgtest-dev \
    pkg-config \
    poppler-utils \
    qpdf \
    gh \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/clang-format-18 /usr/bin/clang-format \
    && git lfs install --system
# Trust the corporate CA before downloading developer tools.
#COPY AMD_RootCA.crt /usr/local/share/ca-certificates/AMD_RootCA.crt
RUN update-ca-certificates
# Install VS Code CLI for tunneling
RUN curl -Lf 'https://update.code.visualstudio.com/latest/cli-linux-x64/stable' --output vscode_cli.tar.gz \
    && tar -xf vscode_cli.tar.gz -C /usr/local/bin \
    && rm vscode_cli.tar.gz
# Install Cursor CLI for tunneling
ENV CURSOR_DIR=/opt/cursor
RUN mkdir -p $CURSOR_DIR && curl -Lf 'https://api2.cursor.sh/updates/download-latest?os=cli-alpine-x64' | tar -C $CURSOR_DIR -xzf -
# Create virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Add the development payload (headers, CMake packages, LLVM tools, and device
# libs) plus ROCm Compute and Systems Profiler.
RUN python -m pip install --no-cache-dir \
        --index-url "${ROCM_WHEEL_INDEX}" \
        "rocm[devel,profiler]==${ROCM_VERSION}" \
    && rocm-sdk init
# rocm-sdk init expands the wheel into a normal ROCm prefix under site-packages.
ENV ROCM_PATH="${VIRTUAL_ENV}/lib/python3.14/site-packages/_rocm_sdk_devel"
ENV ROCM_HOME="${ROCM_PATH}" \
    ROCM_ROOT="${ROCM_PATH}" \
    HIP_PATH="${ROCM_PATH}" \
    HIP_HOME="${ROCM_PATH}" \
    LLVM_PATH="${ROCM_PATH}/lib/llvm" \
    HIP_DEVICE_LIB_PATH="${ROCM_PATH}/amdgcn/bitcode" \
    CMAKE_PREFIX_PATH="${ROCM_PATH}" \
    CMAKE_HIP_COMPILER_ROCM_ROOT="${ROCM_PATH}" \
    TRITON_LIBHIP_PATH="${ROCM_PATH}/lib/libamdhip64.so" \
    PKG_CONFIG_PATH="${ROCM_PATH}/lib/pkgconfig"
ENV PATH="${ROCM_PATH}/bin:${ROCM_PATH}/lib/llvm/bin:${PATH}"
ENV LD_LIBRARY_PATH="${ROCM_PATH}/lib"
# Bundle the ATT decoder with rocprofv3 from the same ROCm payload, and fail the
# image build if it is absent.
RUN test -r "${ROCM_PATH}/lib/librocprof-trace-decoder.so"
# Common aiter, FlyDSL, CK, and vLLM development/test requirements.
COPY requirements.txt /tmp/devcontainer/
RUN python -m pip install --no-cache-dir -r /tmp/devcontainer/requirements.txt \
    && python -m pip install --no-cache-dir tabulate==0.9.0 pyzmq==27.1.0 \
    && rm -rf /tmp/devcontainer ~/.cache/pip
# Directory write permission is sufficient for pip to add, replace, or remove
# packages. Do not recursively chown/chmod files: Docker's OverlayFS copy-up
# would split the hardlinks that assemble the ROCm wheel prefix.
RUN find "${VIRTUAL_ENV}" -type d -exec chmod a+rwx {} +
USER ${USERNAME}
WORKDIR /home/${USERNAME}
