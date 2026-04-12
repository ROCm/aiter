Performance Benchmarks
======================

Up-to-date performance results for AITER kernels on AMD Instinct GPUs are
published on the `AMD AI Frameworks Performance Dashboard
<https://rocm.github.io/oss-dashboard/>`_.

The dashboard includes:

- Operator-level throughput comparisons (GEMM, MoE, Attention, Norm)
- End-to-end model serving throughput and latency
- Cross-platform comparisons (MI300X, MI325X, MI355X vs. NVIDIA B200, B300)

Running Kernel Benchmarks Locally
----------------------------------

AITER includes kernel-level benchmarks in its test suite. To run them:

.. code-block:: bash

   pytest tests/ -k "benchmark"

For GEMM-specific benchmarking, use the GEMM tuner in benchmark mode:

.. code-block:: bash

   python3 gradlib/gradlib/gemm_tuner.py \
       --tuned_file results.csv \
       --input_file shapes.csv

See :doc:`../gemm_tuning` for details on shape file format and tuning
workflow.
