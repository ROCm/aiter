# Triton with 0001-force-chain-dot-across-scf-if.patch, for reproducing the
# block-skip numbers in this PR.
#
# WHY THIS EXISTS. Block skipping wraps the P@V dot in a conditional so a
# skipped tile never loads V:
#
#     if not (ENABLE_BLOCK_SKIP and all_skip):
#         acc = tl.dot(P.to(V.dtype), V, acc=acc)
#
# That puts P@V one scf.if deeper than the QK dot. The AMD backend decides warp
# layout with isChainDotHead/isChainDotTail, which require both dots in the SAME
# MLIR region, so the flash-attention-tuned warpsPerCTA=[num_warps, 1] degrades
# to a generic [2, 2]. The patch adds an opt-in env var,
# TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF, letting the detectors see across exactly
# one level of scf.if nesting -- recovering the layout without restructuring the
# kernel and without giving up the V-load skip, which is where the bandwidth
# saving comes from.
#
# The patch is inert unless the env var is set, so ONE image serves both arms of
# the comparison. See ../README.md.
#
# PINNED ON PURPOSE. The patch is tied to the commit it applies to. Building
# against a moving `main` would make the numbers unreproducible and the patch
# liable to silently fail to apply.
FROM rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0

ENV MAX_JOBS=16

# transformers/accelerate must be installed BEFORE the Triton swap below.
# torch's install metadata pins an exact Triton version; installing them after
# that pin is broken makes pip's resolver silently re-resolve torch+triton to
# non-ROCm wheels, and you end up benchmarking a CPU build.
RUN pip install --no-cache-dir transformers accelerate \
    && python3 -c "import torch; assert torch.version.hip is not None, 'lost ROCm torch build'; print('torch (ROCm):', torch.__version__)"

COPY 0001-force-chain-dot-across-scf-if.patch /opt/0001-force-chain-dot-across-scf-if.patch

# --no-build-isolation skips pyproject.toml's [build-system].requires, so
# ninja/cmake/nanobind must be installed explicitly first (nanobind pinned to
# match triton's own pyproject.toml requirement).
RUN pip uninstall -y triton pytorch-triton pytorch-triton-rocm triton-rocm amd-triton || true \
    && pip install --no-cache-dir ninja cmake 'nanobind==2.10.2' \
    && git clone https://github.com/triton-lang/triton.git /opt/triton-src \
    && cd /opt/triton-src \
    && git checkout 71d121b0690ad2615f19c85a1c29b08b55a9f80e \
    && git apply /opt/0001-force-chain-dot-across-scf-if.patch \
    && git rev-parse HEAD > /opt/triton-src-commit.txt \
    && echo "patched: 0001-force-chain-dot-across-scf-if.patch" >> /opt/triton-src-commit.txt \
    && pip install --no-cache-dir -e . --no-build-isolation \
    && python3 -c "import triton; print('Triton built from source:', triton.__version__)" \
    && python3 -c "import torch; assert torch.version.hip is not None, 'lost ROCm torch build after triton swap'; print('torch (ROCm, post-triton-swap):', torch.__version__)"

WORKDIR /workspace
