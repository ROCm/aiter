Installation
============

Requirements
------------

System Requirements
^^^^^^^^^^^^^^^^^^^

* **Operating System**: Linux (Ubuntu 20.04+, RHEL 8+, or SLES 15+)
* **Python**: 3.8 or later
* **ROCm**: 5.7 or later (6.0+ recommended)
* **GPU**: AMD GPU with gfx90a, gfx942, or gfx950 architecture

Software Dependencies
^^^^^^^^^^^^^^^^^^^^^

* PyTorch 2.0+ with ROCm support
* ROCm libraries (hipBLAS, rocBLAS, MIOpen)
* Optional: Triton for Triton-based kernels

Installation Methods
--------------------

Method 1: From PyPI (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
   PyPI package is coming soon!

.. code-block:: bash

   pip install amd-aiter

Method 2: From Source
^^^^^^^^^^^^^^^^^^^^^

Basic Installation
""""""""""""""""""

.. code-block:: bash

   # Clone the repository
   git clone --recursive https://github.com/ROCm/aiter.git
   cd aiter

   # Install in development mode
   python3 setup.py develop

Development Mode (JIT)
""""""""""""""""""""""

Kernels are compiled on first use:

.. code-block:: bash

   python3 setup.py develop

Precompiled Installation
""""""""""""""""""""""""

Precompile kernels at install time:

.. code-block:: bash

   PREBUILD_KERNELS=2 GPU_ARCHS="gfx942" python3 setup.py install

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 60 20

   * - Variable
     - Description
     - Default
   * - ``GPU_ARCHS``
     - Target GPU architecture(s), semicolon-separated. Use ``native`` to auto-detect.
     - ``native``
   * - ``PREBUILD_KERNELS``
     - ``0`` = JIT only, ``1`` = core kernels, ``2`` = inference kernels, ``3`` = MHA only
     - ``0``
   * - ``MAX_JOBS``
     - Max parallel compilation threads
     - Auto-calculated

Example Configurations
""""""""""""""""""""""

.. code-block:: bash

   # For MI300X with full precompilation
   PREBUILD_KERNELS=2 GPU_ARCHS="gfx942" python3 setup.py install

   # For MI250X + MI300X multi-arch
   GPU_ARCHS="gfx90a;gfx942" python3 setup.py install

   # Auto-detect current GPU
   GPU_ARCHS="native" python3 setup.py install

Method 3: Docker
^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Coming soon
   docker pull amd/aiter:latest
   docker run --device=/dev/kfd --device=/dev/dri amd/aiter:latest

Verifying Installation
-----------------------

.. code-block:: python

   import aiter
   import torch

   # Check ROCm availability
   print(f"PyTorch version: {torch.__version__}")
   print(f"ROCm available: {torch.cuda.is_available()}")
   print(f"ROCm version: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")

   # Verify AITER can import key operators
   from aiter import flash_attn_with_kvcache, rmsnorm
   print("AITER operators loaded successfully!")

Optional: Triton Communication Support
---------------------------------------

For Triton-based communication primitives:

.. code-block:: bash

   pip install -r requirements-triton-comms.txt

See :doc:`tutorials/triton_comms` for more details.

Troubleshooting
---------------

ROCm Not Found
^^^^^^^^^^^^^^

If ROCm is not detected:

.. code-block:: bash

   export ROCM_PATH=/opt/rocm
   export PATH=$ROCM_PATH/bin:$PATH

Compilation Errors
^^^^^^^^^^^^^^^^^^

For compilation issues:

1. Ensure ROCm is properly installed: ``rocm-smi``
2. Check Python version: ``python3 --version``
3. Verify GPU architecture: ``rocminfo | grep gfx``

Import Errors
^^^^^^^^^^^^^

If you get import errors:

.. code-block:: bash

   # Ensure ROCm libraries are in library path
   export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

Next Steps
----------

* :doc:`quickstart` - Get started with your first AITER program
* :doc:`tutorials/index` - Learn through examples
* :doc:`api/attention` - Explore the API reference
