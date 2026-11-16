Tutorials
=========

Learn AITER through hands-on examples.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   basic_usage
   attention_tutorial
   variable_length

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   add_new_op
   moe_tutorial
   custom_kernels
   quantization
   triton_comms

.. toctree::
   :maxdepth: 2
   :caption: Integration

   vllm_integration
   pytorch_lightning
   deepspeed

Tutorial Overview
-----------------

Basic Tutorials
^^^^^^^^^^^^^^^

* :doc:`basic_usage` - Your first AITER program
* :doc:`attention_tutorial` - Understanding attention kernels
* :doc:`variable_length` - Handling variable-length sequences

Advanced Topics
^^^^^^^^^^^^^^^

* :doc:`add_new_op` - **How to add a new operator** (step-by-step guide)
* :doc:`moe_tutorial` - Mixture of Experts optimization
* :doc:`custom_kernels` - Writing custom ROCm kernels
* :doc:`quantization` - INT8 quantization for inference
* :doc:`triton_comms` - Triton-based communication primitives

Integration Guides
^^^^^^^^^^^^^^^^^^

* :doc:`vllm_integration` - Using AITER with vLLM
* :doc:`pytorch_lightning` - PyTorch Lightning integration
* :doc:`deepspeed` - DeepSpeed integration

Prerequisites
-------------

All tutorials assume:

* Python 3.8+
* PyTorch 2.0+ with ROCm support
* AITER installed (see :doc:`../installation`)
* AMD GPU (gfx90a, gfx942, or gfx950)

Example Data
------------

Some tutorials use sample data. Download with:

.. code-block:: bash

   # Coming soon: test data downloader
   bash scripts/download_test_data.sh

Jupyter Notebooks
-----------------

Interactive notebooks are available in the ``examples/`` directory:

.. code-block:: bash

   # Install Jupyter
   pip install jupyter

   # Launch notebooks
   cd examples
   jupyter notebook

Running Examples
----------------

All tutorial code can be run directly:

.. code-block:: bash

   # Clone repository
   git clone https://github.com/ROCm/aiter.git
   cd aiter

   # Run tutorial script
   python examples/basic_usage.py

Community Examples
------------------

Check out community-contributed examples:

* **Llama 2 inference** - Optimized inference with AITER
* **Mixtral 8x7B** - MoE model acceleration
* **GPT-style models** - Training and inference

Contributing Tutorials
----------------------

We welcome tutorial contributions! See :doc:`../contributing` for guidelines.

Tips for following tutorials:

1. **Start with basics** - Don't skip the fundamentals
2. **Run the code** - Type it out, don't just copy-paste
3. **Experiment** - Modify parameters and observe changes
4. **Profile** - Use ROCm profiler to understand performance
5. **Ask questions** - Open issues or discussions on GitHub
