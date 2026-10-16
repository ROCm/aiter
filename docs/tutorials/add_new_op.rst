Adding an Operator
==================

Follow an existing backend path so registration, stream handling and testing
match the rest of AITER. This walkthrough traces the small HIP
``silu_and_mul`` operator. It computes ``silu(gate) * up`` from concatenated
inputs; the caller owns the output buffer. No standalone ``aiter._C`` extension
or NVIDIA ``sm_*`` build flags are required.

1. Define the Python contract
-----------------------------

.. literalinclude:: ../../aiter/ops/activation.py
   :language: python
   :start-at: @compile_ops("module_activation", develop=True)
   :end-at: def silu_and_mul(out: Tensor, input: Tensor, limit: float = 0.0) -> None: ...

The decorator identifies the JIT module. The function's argument order and
mutation behavior must agree with the binding. ``aiter/__init__.py`` exports
the activation module in the normal ROCm import path. Add new public exports
there only when they can participate in that import contract.

2. Register the build inputs
----------------------------

``aiter/jit/optCompilerConfig.json`` defines ``module_activation`` with
``csrc/pybind/activation_pybind.cu`` and
``csrc/kernels/activation_kernels.cu`` as source files. It also declares compiler
flags, include paths and whether the module excludes the torch C++ dependency.
For a new module, add its recipe alongside comparable modules and use the same
module name in ``compile_ops``. Do not add a separate build system to the tutorial.

AITER selects AMD targets through ``GPU_ARCHS`` and its JIT compiler machinery.
Use targets such as ``gfx942`` or ``gfx950`` where the implementation supports
them. Consult :doc:`../installation` and :doc:`../jit_cache` for build/cache
behavior; initialize recursive submodules before compiling CK-dependent code.

3. Match the native binding and implementation
----------------------------------------------

.. literalinclude:: ../../csrc/pybind/activation_pybind.cu
   :language: cpp

The ``ACTIVATION_PYBIND`` macro in ``csrc/include/rocm_ops.hpp`` registers
``silu_and_mul`` and its arguments. The declaration is in
``csrc/include/activation.h``. The implementation in
``csrc/kernels/activation_kernels.cu`` receives ``aiter_tensor_t`` views,
checks its clipping parameter and launches the activation kernel. Follow the
existing tensor/dtype dispatch and stream macros instead of importing CUDA-only
types. Keep the Python, binding and native signatures synchronized.

4. Validate against an independent reference
--------------------------------------------

The existing ``op_tests/test_activation.py`` defines the PyTorch reference and
benchmarks. A small output-buffer check is also included in the quickstart:

.. literalinclude:: ../examples/quickstart.py
   :language: python
   :start-after: # BEGIN activation
   :end-before: # END activation

For a new operator, cover meaningful boundary shapes, supported dtypes, output
mutation, non-default parameters, invalid inputs and target restrictions.
Use the repository's operator-test conventions and correctness tolerances for
the precision involved. Run the numerical test on a supported ROCm stack before
claiming support; a CPU documentation build cannot validate a HIP launch.

5. Benchmark and document the public API
----------------------------------------

Reuse ``aiter.test_common.run_perftest`` and the maintained activation benchmark
rather than timing a single cold launch. Include warmup, stack versions, tensor
shapes, dtype and hardware with results. If you introduce a public entry point,
add an ``aiter-function`` directive with its full source module path to the API
page; the CPU docs build checks that it exists and renders its current signature.

Other backend paths
-------------------

* **CK / CK Tile:** start from the matching ``csrc/py_itfs_ck`` binding and JIT
  recipe; code generation and CK submodules may be required.
* **Triton:** follow an existing operator in ``aiter/ops/triton`` and its matching
  tests under ``op_tests/triton_tests``. Use AITER's selected Triton build.
* **FlyDSL:** follow ``aiter/ops/flydsl`` and ``op_tests/flydsl_tests`` with the
  FlyDSL dependency pinned by this checkout. See the
  `FlyDSL documentation <https://rocm.github.io/FlyDSL/>`_ for kernel authoring.
* **Assembly:** see :doc:`../isa_kernel_optimization` and preserve the ABI,
  metadata and supported-target requirements of the dispatching wrapper.
