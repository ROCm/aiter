ROCm Compatibility Matrix
=========================

Supported GPU Architectures
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 30 15 40

   * - Architecture
     - GPU
     - gfx Target
     - Status
   * - CDNA 3
     - MI300A, MI300X, MI325X
     - gfx942
     - Fully supported, pre-built wheels
   * - CDNA 3.5
     - MI355X
     - gfx950
     - Fully supported, pre-built wheels
   * - CDNA 4
     - MI450
     - gfx1250
     - Experimental, Triton+HIP only

ROCm Version Matrix (pre-built wheels)
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 20

   * - ROCm
     - Python 3.10
     - Python 3.12
     - PyTorch
   * - 7.2.1
     - yes
     - yes
     - 2.9.1
   * - 7.1.1
     - yes
     - yes
     - 2.10.0
   * - 7.0.2
     - yes
     - yes
     - 2.9.1

Installation
------------

From GitHub Release (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install amd-aiter --find-links https://github.com/ROCm/aiter/releases/latest

From source
^^^^^^^^^^^

.. code-block:: bash

   git clone --recursive https://github.com/ROCm/aiter.git
   cd aiter && pip install -e .

.. note::

   Building from source requires a ROCm installation matching one of the
   supported versions above, along with ``ninja`` and ``cmake``.
