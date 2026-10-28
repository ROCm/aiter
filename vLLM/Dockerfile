# syntax=docker/dockerfile:1
#
# Standalone aiter + vLLM for MI350P. Build context unused.
#
# A/B: build twice from this file,
#   docker build -t vllm-aiter-ar   .                         # main + FlyDSL comm branch
#   docker build -t vllm-aiter-main --build-arg WITH_ALLREDUCE=0 .   # plain main

# Pinned: nightly of vLLM 49cb2bbf71 (2026-10-16), the vLLM the existing images
# were built on. Dockerfile.vllm-fork overlays a fork branch cut from this commit.
ARG BASE=vllm/vllm-openai-rocm:nightly-49cb2bbf71cfd36ee5aa33d1e184b1cddff55294
FROM ${BASE}

# main as of 2026-10-21: the newest main the comm branch below contains, so the
# baseline image and the all-reduce image differ only by the comm work.
ARG AITER_MAIN=d9b256b44f2b10e6dddc9c7ab325514386c25f5d
ARG AITER_UPSTREAM=https://github.com/ROCm/aiter.git
ARG AITER_DIR=/opt/aiter

# 1 = merge the FlyDSL comm branch, 0 = baseline.
ARG WITH_ALLREDUCE=1
# Tip of users/vpietila/flydsl-fused-comm-kernels (PR 5670's branch): one-shot and
# fused AR+RMSNorm, plus FlyDSLQuickAllReduce (aiter.dist...flydsl_backends),
# which the vLLM fork needs. The older PR head 30a0caf5d lacks the latter.
ARG B_ALLREDUCE=3c4adc5ea70af0b60e4d942bbb8b7ee19da3f7bf
ARG B_ALLREDUCE_BRANCH=users/vpietila/flydsl-fused-comm-kernels

# MI350P branches. Re-resolve with a forcing refspec: these get force-pushed,
# and a plain fetch keeps the old tip and still exits 0.
#   git ls-remote https://github.com/ROCm/aiter.git 'refs/heads/<branch>'
ARG B_FLYDSLXCD=0a330263d61d6eea7e49b466361f00e4decce6fc  # relbers/gfx950_flydsl_xcd_count
ARG B_TRITONXCD=7ad35acd2d6affe795b251b73ca8c567340dca8c  # relbers/gfx950_triton_xcd_count
ARG B_OPUSXCD=08de374c34f50a25717e0821f0aad668f128c1ba    # relbers/gfx950_opus_xcd_count
ARG B_HIPXCD=af390d69f43baa858b3c6edaad8adff1bdd34400     # relbers/gfx950_hip_xcd_count
ARG B_ARCHCU=9b65132fa091fb7e59300b0541d343a0afd964c2     # relbers/arch-all-cu-nums
ARG B_AOTFILTER=5193cf94a2a76c3b10a96f2036a6fd4a824a24a1  # relbers/flydsl-gemm-aot-arch-filter
ARG B_FMOEACC=abbef9d36ee0614ce80c9f4c8db1f079b1a851d4    # relbers/fmoe-tuner-accuracy
ARG B_TUNING=5e0e5998a86d7a4649d72c521feff39dbc9e9c4e     # mi350p-tuning

ENV MAX_JOBS=6

# Bare "gfx950" means 256 CU -> 8 XCDs, baking the wrong die count into the
# AOT kernels. MI350P is 128 CU -> 4.
ENV AITER_GPU_TARGETS=gfx950:128

# Merge commits need an identity; a clean image has none.
ENV GIT_AUTHOR_NAME="aiter mi350p build" \
    GIT_AUTHOR_EMAIL="aiter-mi350p@localhost" \
    GIT_COMMITTER_NAME="aiter mi350p build" \
    GIT_COMMITTER_EMAIL="aiter-mi350p@localhost"

# One remote: upstream ROCm.
RUN git clone --no-checkout "${AITER_UPSTREAM}" "${AITER_DIR}" \
    && cd "${AITER_DIR}" \
    && git fetch -q origin "+refs/heads/${B_ALLREDUCE_BRANCH}:refs/remotes/origin/allreduce" \
      '+refs/heads/relbers/*:refs/remotes/origin/relbers/*' \
      '+refs/heads/mi350p-tuning:refs/remotes/origin/mi350p-tuning' \
    && git checkout -q -B mi350p "${AITER_MAIN}"

RUN cd "${AITER_DIR}" \
    && if [ "${WITH_ALLREDUCE}" = "1" ]; then \
         git merge --no-edit "${B_ALLREDUCE}" ; \
       else echo "baseline: aiter main only"; fi

# The four xcd_count branches first, then the rest.
#
# With WITH_ALLREDUCE=1 these steps stop on a modify/delete. The comm branch
# deletes aiter/ops/flydsl/{,kernels/}quick_allreduce_int4.py and their
# op_test, having rewritten that path under kernels/comm/; main modifies the
# same three files in #6049. The branches carry main past that commit, so the
# deletion and the edit meet. The three files are the comm branch's to
# reconcile. Onto AITER_MAIN alone the eight merge with no conflict.
RUN cd "${AITER_DIR}" && git merge --no-edit "${B_FLYDSLXCD}"
RUN cd "${AITER_DIR}" && git merge --no-edit "${B_TRITONXCD}"
RUN cd "${AITER_DIR}" && git merge --no-edit "${B_OPUSXCD}"
RUN cd "${AITER_DIR}" && git merge --no-edit "${B_HIPXCD}"
RUN cd "${AITER_DIR}" && git merge --no-edit "${B_ARCHCU}"
RUN cd "${AITER_DIR}" && git merge --no-edit "${B_AOTFILTER}"
RUN cd "${AITER_DIR}" && git merge --no-edit "${B_FMOEACC}"
RUN cd "${AITER_DIR}" && git merge --no-edit "${B_TUNING}"

# ck_tile comes from this submodule (jit/core.py points CK_DIR at it). Without
# it the build falls back to /opt/rocm's older copy and fails to compile.
RUN cd "${AITER_DIR}" && git submodule update --init --recursive \
    && test -d 3rdparty/composable_kernel/include/ck_tile

RUN cd "${AITER_DIR}" && pip install -r requirements.txt && pip install pyyaml

# Installing as a wheel, the way vLLM's docker/Dockerfile.rocm_base does.
# PYTHONPATH over the pre-installed aiter would leave two trees, one shadowing
# the other. PREBUILD_KERNELS compiles kernels in rather than leaving them to
# JIT on first request.
RUN cd "${AITER_DIR}" \
    && PREBUILD_KERNELS=1 AITER_USE_SYSTEM_TRITON=1 GPU_ARCHS=gfx950 \
    python3 setup.py bdist_wheel --dist-dir=dist \
    && ls -la dist/*.whl

# AITER_AOT_IMPORT, or importing aiter without a GPU dies in triton's arch_info.
RUN pip install --force-reinstall --no-deps "${AITER_DIR}"/dist/*.whl \
    && AITER_AOT_IMPORT=1 python3 -c "import aiter; print('aiter:', aiter.__file__)" \
    && if [ "${WITH_ALLREDUCE}" = "1" ]; then \
         AITER_AOT_IMPORT=1 python3 -c "from aiter.ops.flydsl import allreduce_policy; from aiter.dist.device_communicators.flydsl_backends import FlyDSLQuickAllReduce, FlyDSLAllReduceRMSNorm; print('flydsl allreduce + quick allreduce ok')"; \
       fi

# vLLM defaults this to False and then never calls aiter.
ENV VLLM_ROCM_USE_AITER=1 \
    VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 \
    HSA_NO_SCRATCH_RECLAIM=1 \
    SAFETENSORS_FAST_GPU=1

# AITER_FLY_AR follows WITH_ALLREDUCE, so the baseline image runs the stock
# all-reduce path. The quantization setting is the same in both images.
# 7168 is the DeepSeek V4 hidden size: the fused engines must be built before
# graph capture, or the fused path declines during capture.
ENV AITER_FLY_AR=${WITH_ALLREDUCE} \
    AITER_FLY_AR_LINK=pcie \
    AITER_FLY_AR_FUSED_HIDDENS=7168 \
    AITER_QUICK_REDUCE_QUANTIZATION=INT4

# Base image entrypoint is "vllm serve", which would swallow any command.
ENTRYPOINT []
WORKDIR /work
CMD ["vllm", "--help"]
