AITER Documentation
===================

**AITER (AI Tensor Engine for ROCm)** provides optimized GPU operators for
inference and training. Its HIP, Composable Kernel (CK), assembly, Triton and
FlyDSL implementations cover attention, GEMM, MoE, normalization, quantization
and communication.

This site follows **main**; it can describe changes newer than a published
release. Source: |source_revision|. Built: |build_date|. For a released wheel,
consult its `release notes <https://github.com/ROCm/aiter/releases>`_ and matching
source tag. A source signature is not a claim that every backend, dtype and
GPU combination has been validated.

Start here
----------

* :doc:`installation`: choose a release wheel or a development checkout.
* :doc:`quickstart`: compare attention, RMSNorm and MoE with reference results.
* :doc:`tutorials/add_new_op`: trace an operator from Python through its HIP binding.
* :doc:`autotuning_pipeline`: tune and validate configurations for your workload.

Hardware
--------

The project lists CDNA3 (gfx942: MI300X/MI325X) and CDNA4 (gfx950:
MI350/MI355X) support. RDNA3 (gfx1100), RDNA3.5 (gfx1151) and RDNA4
(gfx1201) support is experimental: Triton and most FlyDSL kernels run there,
along with many HIP kernels; most CK and assembly kernels are CDNA-only.
Check the `maintained hardware table
<https://github.com/ROCm/aiter#supported-hardware>`_ and the specific operator's
tests before choosing a backend. Installing a package does not make all its
operators portable to every target.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/attention
   api/gemm
   api/moe
   api/normalization
   api/operators

.. toctree::
   :maxdepth: 1
   :caption: Development and Operations

   jit_cache
   autotuning_pipeline
   triton_comms
   isa_kernel_optimization
   examples/isa_optimization/README
   aiter_container_nonroot_setup
   newsletter/2026-05

Related projects
----------------

`ATOM <https://rocm.github.io/ATOM/>`_ uses AITER for model serving;
`FlyDSL <https://rocm.github.io/FlyDSL/>`_ supports kernel authoring;
`MORI <https://rocm.github.io/mori/>`_ provides communication primitives.
Use the dependency versions selected by your framework's tested stack.
