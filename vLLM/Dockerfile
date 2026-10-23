# syntax=docker/dockerfile:1
#
# Standalone aiter + vLLM for MI350P. Build context unused.
#
# A/B: build twice from this file,
#   docker build -t vllm-aiter-ar   .                         # main + PR 5670
#   docker build -t vllm-aiter-main --build-arg WITH_ALLREDUCE=0 .   # plain main

# Pinned: nightly of vLLM 49cb2bbf71 (2026-10-16), the vLLM the existing images
# were built on. Dockerfile.vllm-fork overlays a fork branch cut from this commit.
ARG BASE=vllm/vllm-openai-rocm:nightly-49cb2bbf71cfd36ee5aa33d1e184b1cddff55294
FROM ${BASE}

# main as of 2026-10-16.
ARG AITER_MAIN=2bfc305e6058f6094173cf04a3318448edcaeca7
ARG AITER_UPSTREAM=https://github.com/ROCm/aiter.git
ARG AITER_DIR=/opt/aiter

# 1 = merge PR 5670, 0 = baseline.
ARG WITH_ALLREDUCE=1
# PR 5670 head (rebased to contain PR 5461).
ARG B_ALLREDUCE=30a0caf5d22e466cdd670d6c9370cef23869f766

ENV MAX_JOBS=6

# Merge commits need an identity; a clean image has none.
ENV GIT_AUTHOR_NAME="aiter mi350p build" \
    GIT_AUTHOR_EMAIL="aiter-mi350p@localhost" \
    GIT_COMMITTER_NAME="aiter mi350p build" \
    GIT_COMMITTER_EMAIL="aiter-mi350p@localhost"

# One remote: upstream ROCm. PR heads come from refs/pull, so forks are fine.
RUN git clone --no-checkout "${AITER_UPSTREAM}" "${AITER_DIR}" \
    && cd "${AITER_DIR}" \
    && git fetch -q origin '+refs/pull/5670/head:refs/remotes/origin/pr/5670' \
    && git checkout -q -B mi350p "${AITER_MAIN}"

RUN cd "${AITER_DIR}" \
    && if [ "${WITH_ALLREDUCE}" = "1" ]; then \
         git merge --no-edit "${B_ALLREDUCE}" ; \
       else echo "baseline: aiter main only"; fi

# ---- Parked: MI350P tuning PRs (5182, 5340, 5504, 5505, 5848, 5668) ----------
# Re-enable once they merge conflict-free with main + PR 5670.
# ENV AITER_GPU_TARGETS=gfx950:128 and the mi350p config-count check.
# -------------------------------------------------------------------------

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
         AITER_AOT_IMPORT=1 python3 -c "from aiter.ops.flydsl import allreduce_policy; print('flydsl allreduce policy ok')"; \
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
