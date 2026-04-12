How to Add a New Operator
==========================

This tutorial shows how to add a custom operator to AITER using the JIT
compilation system. AITER kernels are written in HIP C++ or Triton and are
JIT-compiled at first use via ``ninja``.

Overview
--------

1. Write the kernel (HIP C++ in ``csrc/`` or Triton in ``aiter/ops/triton/``)
2. Register the build config in ``aiter/jit/optCompilerConfig.json``
3. Create the Python op in ``aiter/ops/``
4. Provide a fake-tensor implementation for ``torch.compile``
5. Add tests in ``op_tests/``

Option A: HIP C++ Kernel
--------------------------

Step 1: Write the HIP Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a kernel file in ``csrc/kernels/``. AITER targets AMD GPUs, so use HIP
APIs and ``__hip_bfloat16`` (not ``__nv_bfloat16``). Source files use the
``.cu`` extension but are compiled with ``hipcc``.

.. code-block:: cpp

   // csrc/kernels/my_op_kernels.cu
   #include <hip/hip_runtime.h>
   #include <hip/hip_fp16.h>
   #include <hip/hip_bfloat16.h>

   template <typename T>
   __global__ void my_op_kernel(
       const T* __restrict__ input,
       T* __restrict__ output,
       int n
   ) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx < n) {
           output[idx] = input[idx];  // your computation here
       }
   }

   void launch_my_op(
       const void* input, void* output, int n, hipStream_t stream
   ) {
       int threads = 256;
       int blocks = (n + threads - 1) / threads;
       my_op_kernel<__half><<<blocks, threads, 0, stream>>>(
           static_cast<const __half*>(input),
           static_cast<__half*>(output),
           n
       );
   }

GPU architecture targets are ``gfx942`` (MI300X), ``gfx950`` (MI355X), and
``gfx1250`` (MI450). The JIT system detects the current GPU and compiles for
the correct target automatically.

Step 2: Write the PyBind Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a pybind wrapper in ``csrc/pybind/``:

.. code-block:: cpp

   // csrc/pybind/my_op_pybind.cu
   #include <torch/extension.h>

   // Forward declaration
   void launch_my_op(const void* input, void* output, int n, hipStream_t stream);

   void my_op_fwd(torch::Tensor input, torch::Tensor output) {
       int n = input.numel();
       auto stream = at::cuda::getCurrentHIPStream().stream();
       launch_my_op(input.data_ptr(), output.data_ptr(), n, stream);
   }

   PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
       m.def("my_op_fwd", &my_op_fwd, "My custom op forward");
   }

Step 3: Register the Build Config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add an entry to ``aiter/jit/optCompilerConfig.json``:

.. code-block:: json

   {
       "module_my_op": {
           "srcs": [
               "f'{AITER_CSRC_DIR}/pybind/my_op_pybind.cu'",
               "f'{AITER_CSRC_DIR}/kernels/my_op_kernels.cu'"
           ],
           "flags_extra_cc": [],
           "flags_extra_hip": [],
           "extra_ldflags": "None",
           "extra_include": [],
           "verbose": "False",
           "blob_gen_cmd": "''"
       }
   }

The ``srcs`` entries are f-string expressions evaluated at build time.
``AITER_CSRC_DIR`` points to the ``csrc/`` directory.

Step 4: Create the Python Op
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create ``aiter/ops/my_op.py`` using the ``@compile_ops`` decorator. This
decorator handles JIT compilation, module caching, and ``torch.compile``
registration.

.. code-block:: python

   # aiter/ops/my_op.py
   import torch
   from torch import Tensor
   from ..jit.core import compile_ops


   def gen_my_op_fake(input: Tensor) -> Tensor:
       """Fake tensor impl for torch.compile tracing."""
       return torch.empty_like(input)


   @compile_ops("module_my_op", gen_fake=gen_my_op_fake)
   def my_op_fwd(input: Tensor) -> Tensor:
       """My custom operator."""
       ...

Key points:

- The first argument to ``@compile_ops`` is the module name matching the key
  in ``optCompilerConfig.json``.
- The function body is ``...`` (ellipsis). The decorator replaces it with the
  JIT-compiled C++ implementation at runtime.
- The function name must match the pybind function name. Use ``fc_name`` if
  they differ: ``@compile_ops("module_my_op", fc_name="my_op_fwd")``.
- The ``gen_fake`` callable returns tensors with the correct shape/dtype for
  ``torch.compile`` tracing without running the real kernel.

Option B: Triton Kernel
--------------------------

Triton kernels live under ``aiter/ops/triton/`` and do not need
``optCompilerConfig.json`` entries. They are compiled by the Triton JIT
compiler directly.

Step 1: Write the Triton Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create the kernel in ``aiter/ops/triton/_triton_kernels/``:

.. code-block:: python

   # aiter/ops/triton/_triton_kernels/my_triton_op.py
   import triton
   import triton.language as tl

   @triton.jit
   def _my_triton_kernel(
       input_ptr, output_ptr,
       n_elements,
       BLOCK_SIZE: tl.constexpr,
   ):
       pid = tl.program_id(0)
       offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       mask = offsets < n_elements

       x = tl.load(input_ptr + offsets, mask=mask)
       # Your computation here
       tl.store(output_ptr + offsets, x, mask=mask)

Step 2: Write the Python Wrapper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create the wrapper in ``aiter/ops/triton/``:

.. code-block:: python

   # aiter/ops/triton/my_triton_op.py
   import torch
   import triton
   from aiter.ops.triton._triton_kernels.my_triton_op import _my_triton_kernel

   def my_triton_op(output: torch.Tensor, input: torch.Tensor):
       n_elements = input.numel()
       BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 1024))
       grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
       _my_triton_kernel[grid](
           input, output, n_elements, BLOCK_SIZE=BLOCK_SIZE,
       )

Adding Tests
------------

Create a test file in ``op_tests/``. Follow the existing pattern using
``aiter.test_common``:

.. code-block:: python

   # op_tests/test_my_op.py
   import torch
   import aiter
   from aiter.test_common import checkAllclose, benchmark

   @benchmark()
   def test_my_op(m, n, dtype):
       ret = {}
       input = torch.randn(m, n, dtype=dtype, device="cuda")

       # Reference (PyTorch)
       ref_output = input.clone()  # replace with actual reference

       # AITER op
       output = torch.empty_like(input)
       aiter.my_op_fwd(output, input)

       err = checkAllclose(ref_output, output)
       ret["M"] = m
       ret["N"] = n
       ret["err"] = err
       return ret

   if __name__ == "__main__":
       for dtype in [torch.float16, torch.bfloat16]:
           for m in [1, 32, 512]:
               test_my_op(m, 4096, dtype)

Run with:

.. code-block:: bash

   python op_tests/test_my_op.py

torch.compile Compatibility
----------------------------

The ``gen_fake`` function passed to ``@compile_ops`` is registered as the
fake-tensor implementation via ``torch_compile_guard`` in
``aiter/jit/utils/torch_guard.py``. This allows ``torch.compile`` to trace
through the op without executing the real kernel.

For HIP ops, the decorator handles this automatically. For Triton ops, if you
need ``torch.compile`` support, register the op manually:

.. code-block:: python

   import torch

   @torch.library.custom_op("aiter::my_triton_op", mutates_args=["output"])
   def my_triton_op(output: torch.Tensor, input: torch.Tensor) -> None:
       # call the Triton kernel
       ...

   @my_triton_op.register_fake
   def _(output: torch.Tensor, input: torch.Tensor) -> None:
       pass  # output is mutated in-place, nothing to return

Best Practices
--------------

1. **Match existing patterns.** Study ``aiter/ops/activation.py`` (simple HIP
   ops) or ``aiter/ops/triton/quant/quant.py`` (Triton ops) as templates.
2. **Test correctness first.** Compare against a PyTorch reference
   implementation with ``checkAllclose``.
3. **Use in-place output tensors.** Most AITER ops take a pre-allocated ``out``
   tensor as the first argument and return ``None``.
4. **Profile with rocm-trace-lite.** Measure kernel duration and memory
   bandwidth to verify performance.
5. **Run ruff and pytest before committing.** Lint and test locally before
   pushing.

See Also
--------

* :doc:`../api/operators` - Existing operator reference
* :doc:`../api/gemm` - GEMM operator reference
