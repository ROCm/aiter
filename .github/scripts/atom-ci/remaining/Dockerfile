ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG AITER_SHA
RUN pip uninstall -y amd-aiter
RUN pip install --upgrade pybind11==3.1.0
RUN rm -rf /app/aiter-test
RUN git clone --no-checkout https://github.com/ROCm/aiter.git /app/aiter-test && \
    cd /app/aiter-test && \
    git checkout --detach ${AITER_SHA} && \
    git submodule sync && git submodule update --init --recursive && \
    MAX_JOBS=64 PREBUILD_KERNELS=0 GPU_ARCHS=gfx950 pip install -e . && \
    ./.github/scripts/install_triton.sh
