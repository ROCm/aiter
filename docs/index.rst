.. AITER documentation master file

AITER Documentation
===================

**AITER** (AI Tensor Engine for ROCm) is AMD's high-performance AI operator library for ROCm, providing optimized kernels for inference and training workloads.

.. image:: https://img.shields.io/badge/ROCm-Compatible-red
   :target: https://rocm.docs.amd.com/
   :alt: ROCm Compatible

.. image:: https://img.shields.io/github/license/ROCm/aiter
   :target: https://github.com/ROCm/aiter/blob/main/LICENSE
   :alt: License

Why AITER?
----------

* **High Performance**: Optimized kernels using Triton, Composable Kernel (CK), ASM, and FlyDSL
* **Comprehensive**: Supports both inference and training workloads
* **Flexible**: C++ and Python APIs for easy integration
* **AMD Optimized**: Built specifically for AMD Instinct GPUs and the ROCm platform

Quick Start
-----------

Installation
^^^^^^^^^^^^

.. code-block:: bash

   # From GitHub Release (recommended)
   pip install amd-aiter --find-links https://github.com/ROCm/aiter/releases/latest

   # From source
   git clone --recursive https://github.com/ROCm/aiter.git
   cd aiter
   pip install -e .

Quick Example
^^^^^^^^^^^^^

.. code-block:: python

   import aiter
   import torch

   # RMS Normalization
   x = torch.randn(2, 4096, dtype=torch.bfloat16, device="cuda")
   weight = torch.ones(4096, dtype=torch.bfloat16, device="cuda")
   out = aiter.rms_norm(x, weight, 1e-6)

   # Fused MoE
   # See API Reference for full function signatures

Core Features
-------------

Attention Kernels
^^^^^^^^^^^^^^^^^

* **Multi-Head Attention (MHA)**: Flash attention forward and backward passes
* **Multi-Latent Attention (MLA)**: DeepSeek-style latent attention for decode and prefill
* **Paged Attention**: Efficient KV-cache management for serving (v1, v2, ragged, ASM)

GEMM Operations
^^^^^^^^^^^^^^^

* **FP8 GEMM (A8W8)**: Multiple backends -- CK, CK Tile, ASM, FlyDSL
* **BF16/FP16 GEMM (A16W16)**: ASM-optimized with auto-tuning
* **FP4 GEMM (A4W4)**: FP4 precision with block-scale support
* **Batched GEMM**: FP8 and BF16 batched operations
* **DeepGEMM**: Specialized deep GEMM kernels
* **Auto-Tuned GEMM**: Pre-tuned configurations for common model shapes

Mixture of Experts (MoE)
^^^^^^^^^^^^^^^^^^^^^^^^^

* **Fused MoE**: Optimized expert routing and computation (``fmoe``, ``fmoe_g1u1``)
* **Quantized MoE**: FP8 block-scale and INT8 expert weights
* **2-Stage MoE**: Sorting + compute pipeline for large expert counts

Normalization & Activation
^^^^^^^^^^^^^^^^^^^^^^^^^^

* **RMSNorm / LayerNorm**: With fused variants (residual add, quantization)
* **Activation**: SiLU, GELU, GELU-tanh (fused with multiply)

Other Operators
^^^^^^^^^^^^^^^

* **RoPE**: Rotary position embeddings (forward, backward, cached)
* **Quantization**: Per-token, per-tensor, per-group, FP4/FP8 conversion
* **KV Cache**: reshape_and_cache with optional quantization
* **Sampling**: Greedy, random, mixed sampling kernels
* **Communication**: Custom AllReduce, fused AllReduce+RMSNorm+Quant

GPU Support
-----------

.. list-table::
   :header-rows: 1
   :widths: 20 15 25 20 20

   * - Architecture
     - gfx Target
     - GPUs
     - ROCm Version
     - Status
   * - CDNA 3
     - gfx942
     - MI300A, MI300X, MI325X
     - ROCm 7.0+
     - Fully supported
   * - CDNA 3.5
     - gfx950
     - MI355X
     - ROCm 7.0+
     - Fully supported
   * - CDNA 4
     - gfx1250
     - MI450
     - ROCm 7.2+
     - Experimental

Quick Links
-----------

* :doc:`quickstart` - Get started in 5 minutes
* :doc:`compatibility` - ROCm version matrix and installation options
* :doc:`models` - Supported model architectures
* :doc:`tutorials/add_new_op` - How to add a new operator
* :doc:`gemm_tuning` - GEMM performance tuning guide

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   compatibility
   models

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/attention
   api/gemm
   api/moe
   api/normalization
   api/operators

.. toctree::
   :maxdepth: 2
   :caption: Guides

   tutorials/index
   gemm_tuning
   advanced/triton_kernels
   performance/benchmarks

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
