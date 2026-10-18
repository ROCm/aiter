Installation
============

Use Linux, Python **3.10 or later**, a ROCm-enabled PyTorch installation and an
operator-compatible AMD GPU. PyTorch continues to use the ``torch.cuda`` API
and ``device="cuda"`` on ROCm. Check ``torch.version.hip`` as well as GPU
availability; a CUDA PyTorch build is not a ROCm build.

Release wheels
--------------

The distribution is **amd-aiter**; the Python import is **aiter**. Choose an asset
from `GitHub Releases <https://github.com/ROCm/aiter/releases>`_ that matches your
Python ABI, ROCm version, CPU architecture and platform's glibc baseline.
The current release workflow builds six ``manylinux_2_28`` combinations:
ROCm 7.0/7.1/7.2 with Python 3.10/3.12. This is an artifact matrix, not a
promise of compatibility with arbitrary PyTorch versions.

After downloading the matching wheel, install its local filename with
``python -m pip install /path/to/downloaded-wheel.whl``.

Preserve the framework's PyTorch, Triton and FlyDSL constraints. Record the wheel
filename and release tag with benchmark and correctness results. Source builds
are available when no published wheel matches your environment.

Development checkout
--------------------

Start in a working ROCm/PyTorch environment with the ROCm compiler toolchain.

.. code-block:: bash

   git clone --recursive https://github.com/ROCm/aiter.git
   cd aiter
   python3 setup.py develop

For an existing clone, initialize its dependencies before building:

.. code-block:: bash

   git submodule sync
   git submodule update --init --recursive

``setup.py develop`` installs required FlyDSL and selects AMD Triton for the
ROCm installation. The exact Python requirements from this checkout are:

.. literalinclude:: ../requirements.txt
   :language: text

Use those constraints rather than copying the version displayed by FlyDSL's
main-branch documentation. To retain an already matched Triton installation,
set ``AITER_USE_SYSTEM_TRITON=1`` when running ``setup.py develop``. With an
editable pip installation, run the Triton installer explicitly:

.. code-block:: bash

   python -m pip install -e .
   ./.github/scripts/install_triton.sh

JIT and prebuilding
-------------------

The default ``PREBUILD_KERNELS=0`` compiles kernels on demand. First use can be
slow; warm up the actual workload before benchmarking. See :doc:`jit_cache` for
cache behavior. Prebuilding is an installation choice, not a runtime speed switch.

.. code-block:: bash

   PREBUILD_KERNELS=2 GPU_ARCHS="gfx942" python3 setup.py install

``GPU_ARCHS`` accepts semicolon-separated targets or ``native``. The current
``setup.py`` selects modules as follows; disabling CK further changes the set:

* ``1`` excludes tuning modules and retains only FMHA v3 forward MHA modules.
* ``2`` excludes backward and tuning modules.
* ``3`` retains only ``module_fmha_v3*`` modules.

Use ``MAX_JOBS`` to limit compilation parallelism when build memory is limited.

Containers and communication
----------------------------

:doc:`aiter_container_nonroot_setup` describes building AITER in a ROCm/PyTorch
container and exposing GPU devices to a non-root user. Pin your base image by
tag or digest and record it with the installed package versions; the recipe's
``latest`` example is not a tested-stack guarantee.

For optional Iris/Triton communication dependencies, run from the checkout:

.. code-block:: bash

   python -m pip install -r requirements-triton-comms.txt

See :doc:`triton_comms` for launch requirements.

Verify the environment
----------------------

.. literalinclude:: examples/quickstart.py
   :language: python
   :start-after: # BEGIN environment
   :end-before: # END environment

Then run :doc:`quickstart` to check numerical results. If import fails, first
confirm that installation and execution use the same Python interpreter. If
HIP libraries cannot be loaded, verify ``ROCM_PATH`` and the runtime library
path. For JIT errors, retain the compiler output, target reported by
``rocminfo``, package versions and source revision in the issue report.
