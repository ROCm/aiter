GEMM Tuning Guide
=================

AITER ships pre-tuned GEMM configurations for popular models. When serving a
new model or using untested shapes, you may see warnings like
``"not found tuned config"`` in server logs. This guide walks through the
tuning process.

When to Tune
-------------

Tune GEMM when:

- Server logs show ``"not found tuned config"`` warnings for specific shapes.
- You are deploying a model not listed in :doc:`models`.
- You want to optimize for a specific batch size / concurrency profile.

Step 1: Identify Missing Shapes
---------------------------------

Check server logs for lines like::

   [WARNING] not found tuned config for M=128, N=4096, K=14336

Collect all unique ``(M, N, K)`` triples that need tuning.

Step 2: Create an Untuned CSV
------------------------------

Create a CSV file listing the shapes to tune. The required columns are:

.. list-table::
   :header-rows: 1
   :widths: 15 50

   * - Column
     - Description
   * - M
     - Batch dimension (number of tokens)
   * - N
     - Output dimension
   * - K
     - Input / reduction dimension
   * - dtype
     - Data type (e.g., ``bf16``, ``fp8``, ``a8w8``)

Example ``untuned.csv``::

   M,N,K,dtype
   128,4096,14336,bf16
   256,4096,14336,bf16
   512,14336,4096,bf16

**Recommended M values** for serving workloads: 1, 2, 4, 8, 16, 32, 64, 128,
256, 512, 1024, 2048, 4096. These cover typical decode (small M) and prefill
(large M) batch sizes.

Step 3: Run the Tuner
----------------------

.. code-block:: bash

   python3 gradlib/gradlib/gemm_tuner.py \
       --tuned_file output.csv \
       --input_file untuned.csv

The tuner benchmarks multiple kernel implementations (ASM, CK, Triton) for
each shape and records the fastest configuration. This process runs on GPU
and may take several minutes per shape.

Step 4: Register the Tuned Config
-----------------------------------

Copy the output CSV to the model configs directory:

.. code-block:: bash

   cp output.csv aiter/configs/model_configs/<model>_tuned_gemm.csv

The inference engine will automatically load tuned configs from this directory
at startup.

Tips
----

- Run tuning on the same GPU architecture you will deploy on. Tuned configs
  are architecture-specific.
- For MoE models, the expert GEMM shapes may differ from the attention GEMM
  shapes. Make sure to include both.
- Re-tune when upgrading ROCm or AITER versions, as kernel implementations
  may change.
