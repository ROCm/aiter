How to Add a New Operator
==========================

This tutorial shows you how to add a custom operator to AITER.

Overview
--------

Adding a new operator involves:

1. **Define the operator interface** (Python)
2. **Implement the kernel** (ROCm/HIP C++)
3. **Create Python bindings** (PyBind11)
4. **Add tests**
5. **Register the operator**

Step 1: Define the Operator Interface
--------------------------------------

Create your operator's Python interface in ``aiter/ops/``:

.. code-block:: python

   # aiter/ops/my_custom_op.py
   import torch
   from typing import Optional

   def my_custom_op(
       input: torch.Tensor,
       weight: torch.Tensor,
       bias: Optional[torch.Tensor] = None,
       activation: str = "gelu"
   ) -> torch.Tensor:
       """
       Custom operator that does something awesome.

       Args:
           input: Input tensor (batch, seq_len, hidden_dim)
           weight: Weight tensor (hidden_dim, output_dim)
           bias: Optional bias tensor (output_dim,)
           activation: Activation function ('gelu', 'relu', 'none')

       Returns:
           Output tensor (batch, seq_len, output_dim)
       """
       # Import the C++ extension
       from aiter._C import my_custom_op_impl

       # Input validation
       assert input.is_cuda, "Input must be on CUDA device"
       assert input.dtype in [torch.float16, torch.bfloat16], \
           "Only FP16/BF16 supported"

       # Call C++ implementation
       return my_custom_op_impl(input, weight, bias, activation)

Step 2: Implement the ROCm Kernel
----------------------------------

Create the kernel implementation in ``csrc/``:

.. code-block:: cpp

   // csrc/my_custom_op.hip
   #include <hip/hip_runtime.h>
   #include <torch/extension.h>

   // Kernel implementation
   template<typename T>
   __global__ void my_custom_kernel(
       const T* input,
       const T* weight,
       const T* bias,
       T* output,
       int batch_size,
       int seq_len,
       int hidden_dim,
       int output_dim
   ) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       int total_elements = batch_size * seq_len * output_dim;

       if (idx < total_elements) {
           int b = idx / (seq_len * output_dim);
           int s = (idx / output_dim) % seq_len;
           int o = idx % output_dim;

           // Your computation here
           T sum = 0;
           for (int h = 0; h < hidden_dim; h++) {
               int input_idx = b * seq_len * hidden_dim + s * hidden_dim + h;
               int weight_idx = h * output_dim + o;
               sum += input[input_idx] * weight[weight_idx];
           }

           if (bias != nullptr) {
               sum += bias[o];
           }

           // Apply activation
           // (GELU, ReLU, etc.)
           output[idx] = sum;
       }
   }

   // Host function
   torch::Tensor my_custom_op_cuda(
       torch::Tensor input,
       torch::Tensor weight,
       torch::Tensor bias,
       std::string activation
   ) {
       // Get dimensions
       auto batch_size = input.size(0);
       auto seq_len = input.size(1);
       auto hidden_dim = input.size(2);
       auto output_dim = weight.size(1);

       // Allocate output
       auto output = torch::empty(
           {batch_size, seq_len, output_dim},
           input.options()
       );

       // Launch kernel
       int total_elements = batch_size * seq_len * output_dim;
       int threads = 256;
       int blocks = (total_elements + threads - 1) / threads;

       if (input.dtype() == torch::kFloat16) {
           my_custom_kernel<__half><<<blocks, threads>>>(
               reinterpret_cast<__half*>(input.data_ptr()),
               reinterpret_cast<__half*>(weight.data_ptr()),
               bias.defined() ? reinterpret_cast<__half*>(bias.data_ptr()) : nullptr,
               reinterpret_cast<__half*>(output.data_ptr()),
               batch_size, seq_len, hidden_dim, output_dim
           );
       } else {
           // BF16 case
           my_custom_kernel<__nv_bfloat16><<<blocks, threads>>>(
               reinterpret_cast<__nv_bfloat16*>(input.data_ptr()),
               reinterpret_cast<__nv_bfloat16*>(weight.data_ptr()),
               bias.defined() ? reinterpret_cast<__nv_bfloat16*>(bias.data_ptr()) : nullptr,
               reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
               batch_size, seq_len, hidden_dim, output_dim
           );
       }

       return output;
   }

Step 3: Create Python Bindings
-------------------------------

Add PyBind11 bindings in ``csrc/my_custom_op_bindings.cpp``:

.. code-block:: cpp

   #include <torch/extension.h>

   // Forward declare CUDA function
   torch::Tensor my_custom_op_cuda(
       torch::Tensor input,
       torch::Tensor weight,
       torch::Tensor bias,
       std::string activation
   );

   // Wrapper for Python
   torch::Tensor my_custom_op_impl(
       torch::Tensor input,
       torch::Tensor weight,
       torch::Tensor bias,
       std::string activation
   ) {
       TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
       return my_custom_op_cuda(input, weight, bias, activation);
   }

   PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
       m.def("my_custom_op_impl", &my_custom_op_impl,
             "My custom operator (CUDA)",
             py::arg("input"),
             py::arg("weight"),
             py::arg("bias"),
             py::arg("activation"));
   }

Step 4: Update Build Configuration
-----------------------------------

Add your operator to ``setup.py``:

.. code-block:: python

   # setup.py
   from setuptools import setup
   from torch.utils.cpp_extension import BuildExtension, CUDAExtension

   setup(
       name='aiter',
       ext_modules=[
           CUDAExtension(
               name='aiter._C',
               sources=[
                   'csrc/my_custom_op.hip',
                   'csrc/my_custom_op_bindings.cpp',
                   # ... other sources
               ],
               extra_compile_args={
                   'cxx': ['-O3', '-std=c++17'],
                   'nvcc': [
                       '-O3',
                       '--use_fast_math',
                       '-gencode', 'arch=compute_90a,code=sm_90a',  # MI250X
                       '-gencode', 'arch=compute_942,code=sm_942',  # MI300X
                   ]
               }
           ),
       ],
       cmdclass={'build_ext': BuildExtension}
   )

Step 5: Add Tests
-----------------

Create tests in ``tests/test_my_custom_op.py``:

.. code-block:: python

   import torch
   import pytest
   from aiter.ops import my_custom_op

   @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
   @pytest.mark.parametrize("batch_size", [1, 4, 16])
   @pytest.mark.parametrize("seq_len", [128, 512, 2048])
   def test_my_custom_op_correctness(dtype, batch_size, seq_len):
       hidden_dim = 512
       output_dim = 2048

       # Create inputs
       input = torch.randn(batch_size, seq_len, hidden_dim,
                          device='cuda', dtype=dtype)
       weight = torch.randn(hidden_dim, output_dim,
                           device='cuda', dtype=dtype)
       bias = torch.randn(output_dim, device='cuda', dtype=dtype)

       # Run custom op
       output = my_custom_op(input, weight, bias, activation='gelu')

       # Reference implementation (PyTorch)
       ref_output = torch.matmul(input, weight)
       if bias is not None:
           ref_output = ref_output + bias
       ref_output = torch.nn.functional.gelu(ref_output)

       # Check correctness
       torch.testing.assert_close(
           output, ref_output,
           rtol=1e-2, atol=1e-2  # FP16/BF16 tolerance
       )

   def test_my_custom_op_performance():
       batch_size, seq_len = 16, 2048
       hidden_dim, output_dim = 4096, 4096

       input = torch.randn(batch_size, seq_len, hidden_dim,
                          device='cuda', dtype=torch.float16)
       weight = torch.randn(hidden_dim, output_dim,
                           device='cuda', dtype=torch.float16)
       bias = torch.randn(output_dim, device='cuda', dtype=torch.float16)

       # Warmup
       for _ in range(10):
           _ = my_custom_op(input, weight, bias)
       torch.cuda.synchronize()

       # Benchmark
       import time
       start = time.time()
       for _ in range(100):
           output = my_custom_op(input, weight, bias)
       torch.cuda.synchronize()
       elapsed = time.time() - start

       print(f"Average time: {elapsed/100*1000:.2f} ms")
       print(f"Throughput: {batch_size*seq_len*100/elapsed:.2f} tokens/sec")

Step 6: Build and Install
--------------------------

Build your extension:

.. code-block:: bash

   # Clean build
   python setup.py clean
   rm -rf build/

   # Build and install
   python setup.py develop

   # Or for production
   python setup.py install

Step 7: Register in Main Module
--------------------------------

Add to ``aiter/__init__.py``:

.. code-block:: python

   # aiter/__init__.py
   from aiter.ops.my_custom_op import my_custom_op

   __all__ = [
       'my_custom_op',
       # ... other exports
   ]

Advanced: Optimizations
-----------------------

Use CK (Composable Kernel) for Better Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   #include "ck/tensor_operation/gpu/device/device_gemm.hpp"

   // Use CK's optimized GEMM
   using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemm<
       /* ... template parameters ... */
   >;

Use Triton for Easier Kernel Development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import triton
   import triton.language as tl

   @triton.jit
   def my_custom_kernel(
       input_ptr, weight_ptr, output_ptr,
       M, N, K,
       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
   ):
       # Triton kernel implementation
       # (Much easier than raw HIP/CUDA!)
       pass

Common Patterns
---------------

Pattern 1: Fused Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Combine multiple ops into one kernel:

.. code-block:: python

   def fused_linear_gelu(input, weight, bias):
       """
       Fuses: output = GELU(input @ weight + bias)
       Faster than separate ops!
       """
       pass

Pattern 2: In-Place Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Modify tensors in-place to save memory:

.. code-block:: python

   def inplace_rmsnorm_(input, weight, eps=1e-6):
       """
       In-place RMSNorm (modifies input)
       Note the trailing underscore!
       """
       pass

Pattern 3: Autograd Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add backward pass for training:

.. code-block:: python

   class MyCustomOpFunction(torch.autograd.Function):
       @staticmethod
       def forward(ctx, input, weight, bias):
           ctx.save_for_backward(input, weight, bias)
           return my_custom_op_impl(input, weight, bias)

       @staticmethod
       def backward(ctx, grad_output):
           input, weight, bias = ctx.saved_tensors
           # Compute gradients
           grad_input = ...
           grad_weight = ...
           grad_bias = ...
           return grad_input, grad_weight, grad_bias

Best Practices
--------------

1. **Start Simple**: Get it working first, optimize later
2. **Test Correctness**: Always compare with PyTorch reference
3. **Profile First**: Use ``rocprof`` to find bottlenecks
4. **Use CK/Triton**: Don't write raw kernels unless necessary
5. **Document Everything**: Add docstrings and comments
6. **Add Type Hints**: Makes the API clearer
7. **Handle Edge Cases**: Check for invalid inputs

Debugging Tips
--------------

Print Kernel Launches
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   export HIP_VISIBLE_DEVICES=0
   export AMD_LOG_LEVEL=3  # Verbose logging

Check for Memory Errors
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Use compute-sanitizer (if available)
   rocm-compute-sanitizer python test_my_op.py

Profile Your Operator
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   rocprof --stats python benchmark_my_op.py

Example: Complete RMSNorm Implementation
-----------------------------------------

Here's a complete example you can use as a template:

**Python Interface** (``aiter/ops/rmsnorm.py``):

.. code-block:: python

   import torch
   from aiter._C import rmsnorm_forward

   def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
       """
       Root Mean Square Layer Normalization.

       Args:
           x: Input tensor (..., hidden_dim)
           weight: Scaling weights (hidden_dim,)
           eps: Epsilon for numerical stability

       Returns:
           Normalized tensor with same shape as input
       """
       assert x.is_cuda and weight.is_cuda
       assert x.dtype in [torch.float16, torch.bfloat16]
       return rmsnorm_forward(x, weight, eps)

**See Full Code**: Check ``csrc/`` directory for complete implementations!

Next Steps
----------

* :doc:`../api/operators` - See existing operator implementations
* :doc:`../benchmarks` - Learn how to benchmark your operator
* :doc:`profiling` - Profile and optimize performance

Contributing
------------

Want to contribute your operator to AITER?

1. Follow the coding style
2. Add comprehensive tests
3. Benchmark vs existing solutions
4. Submit a PR with clear description

See ``CONTRIBUTING.md`` for details!
