.. AITER documentation master file

AITER Documentation
===================

**AITER** (AMD Inference and Training Enhanced Repository) is AMD's high-performance AI operator library for ROCm, providing optimized kernels for inference and training workloads.

.. image:: https://img.shields.io/badge/ROCm-Compatible-red
   :target: https://rocm.docs.amd.com/
   :alt: ROCm Compatible

.. image:: https://img.shields.io/github/license/ROCm/aiter
   :target: https://github.com/ROCm/aiter/blob/main/LICENSE
   :alt: License

Why AITER?
----------

* **High Performance**: Optimized kernels using Triton, Composable Kernel (CK), and hand-written assembly
* **Comprehensive**: Supports both inference and training workloads
* **Flexible**: C++ and Python APIs for easy integration
* **AMD Optimized**: Built specifically for AMD GPUs and the ROCm platform

Quick Start
-----------

Installation
^^^^^^^^^^^^

.. code-block:: bash

   pip install aiter  # Coming soon!

   # For now, install from source:
   git clone --recursive https://github.com/ROCm/aiter.git
   cd aiter
   python3 setup.py develop

Quick Example
^^^^^^^^^^^^^

.. code-block:: python

   import aiter
   import torch

   # Example: Flash Attention
   # TODO: Add actual example code

Core Features
-------------

Attention Kernels
^^^^^^^^^^^^^^^^^

* **Multi-Head Attention (MHA)**: Standard attention with optimized implementations
* **Multi-Latent Attention (MLA)**: DeepSeek-style latent attention
* **Paged Attention**: Efficient KV-cache management for serving

GEMM Operations
^^^^^^^^^^^^^^^

* **Mixed Precision GEMM**: FP16, BF16, FP8, INT4 support
* **Tuned GEMM**: Pre-tuned configurations for common shapes
* **Fused Operations**: GEMM with activation fusion

Mixture of Experts (MoE)
^^^^^^^^^^^^^^^^^^^^^^^^^

* **Fused MoE**: Optimized expert routing and computation
* **Multiple Routing**: Support for various routing strategies
* **Quantized Experts**: FP8 and INT4 expert weights

Normalization
^^^^^^^^^^^^^

* **RMSNorm**: Root mean square normalization
* **LayerNorm**: Standard layer normalization
* **Fused Variants**: Combined with other operations

Other Operators
^^^^^^^^^^^^^^^

* **RoPE**: Rotary position embeddings
* **Quantization**: BF16/FP16 → FP8/INT4 conversion
* **Element-wise**: Optimized basic operations
* **Communication**: AllReduce and collective operations via Triton/Iris

GPU Support
-----------

AITER supports AMD GPUs with the following architectures:

.. list-table::
   :header-rows: 1
   :widths: 20 20 30 30

   * - Architecture
     - gfx Target
     - Example GPUs
     - ROCm Version
   * - CDNA 2
     - gfx90a
     - MI210, MI250, MI250X
     - ROCm 5.0+
   * - CDNA 3
     - gfx942
     - MI300A, MI300X
     - ROCm 6.0+
   * - CDNA 3.5
     - gfx950
     - MI350X (upcoming)
     - ROCm 6.3+

Quick Links
-----------

* 🚀 :doc:`quickstart` - Get started in 5 minutes
* 📖 :doc:`tutorials/add_new_op` - **How to add a new operator** (step-by-step)
* 🔧 :doc:`api/attention` - Flash Attention API
* 💡 :doc:`tutorials/basic_usage` - Basic usage examples

Table of Contents
-----------------

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
   :maxdepth: 2
   :caption: Advanced Topics

   performance/benchmarks
   performance/profiling
   advanced/triton_kernels
   advanced/ck_integration

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
