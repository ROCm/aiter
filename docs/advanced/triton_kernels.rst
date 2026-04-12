Triton Kernels
==============

AITER uses `Triton <https://triton-lang.org/>`_ as one of its backend
implementations for GPU kernels. Triton kernels are written in Python and
compiled to GPU machine code at runtime.

Source Locations
-----------------

Triton kernel sources are located in two directories:

- ``aiter/ops/triton/`` -- Triton-based operator implementations that are
  called from the main ops API.
- ``aiter/_triton_kernels/`` -- Lower-level Triton kernel definitions used
  internally by the operator layer.

Benefits
--------

- **Portability**: Triton kernels work across GPU architectures without
  per-target assembly code.
- **Maintainability**: Written in Python with Triton DSL, making them easier
  to read and modify than hand-written assembly.
- **gfx1250 support**: On MI450 (CDNA 4), where hand-tuned ASM kernels are
  not available, Triton is the primary compute backend alongside HIP.

Other Backends
--------------

AITER supports multiple kernel backends depending on the GPU architecture and
operation:

- **ASM** -- Hand-tuned assembly for peak performance on CDNA 3/3.5.
- **Composable Kernel (CK)** -- C++ template library for fused kernels.
- **CK Tile** -- Tile-based CK backend for structured operations.
- **FlyDSL** -- Domain-specific language for peak-performance kernel
  generation (Meta donation).

The backend selection is automatic based on the GPU architecture and operation
type. On gfx1250 (MI450), AITER uses Triton+HIP exclusively (no ASM, no CK).
