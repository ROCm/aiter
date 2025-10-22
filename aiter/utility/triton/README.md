# Triton Metadata Redirect Module

A Python module that provides decorators and utilities for customizing Triton kernel metadata file paths during compilation. This allows you to redirect .json and .hsaco files to custom directories for better performance.

## Features

* Thread-safe implementation using thread-local storage to avoid race conditions

* Dual usage patterns supporting both JIT and AOT compilation

* Non-intrusive design that doesn't interfere with existing @triton.jit functionality

* Automatic path replacement for both JSON metadata and HSACO binary files

## Usage

### Dump Custom *.json/*.hsaco Files to a Custom Directory

First you need to dump the metadata files to a custom directory.

### For JIT Compilation (Using Decorator)

```python
from aiter.utility.triton.triton_metadata_redirect import with_custom_metadata_path
import triton
import triton.language as tl

@with_custom_metadata_path("/custom/path")
@triton.jit
def my_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Your kernel implementation
    pid = tl.program_id(0)
    # ... kernel logic
```

### For AOT Compilation (Using Context Manager)

```python
# kernel.py
import triton
import triton.language as tl

@triton.jit
def kr_example(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
  return


# aot_compile.py
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.tools.compile import compile_kernel, CompileArgs

from aiter.utility.triton.triton_metadata_redirect import AOTMetadataContext

target = GPUTarget("hip", 'gfx942', 64)

################################################################################
## compile aot kernel0:f32
compile_args = CompileArgs(
    path=f"/path/to/kernel.py",
    kernel_name="kr_example",
    signature=f"*fp32:16,*fp32:16,*fp32:16,i32,64",
    grid="bs,nheads,max_seqlen_q",
    num_warps=4,
    num_stages=2,
    out_name="f32_kernel",
)

with AOTMetadataContext("kr_example", "/path/to/aot0"):
    triton_kernel0, output_files0 = compile_kernel(compile_args)

################################################################################
## compile aot kernel1:f16
compile_args = CompileArgs(
    path=f"/path/to/kernel.py",
    kernel_name="kr_example",
    signature=f"*fp16:16,*fp16:16,*fp16:16,i32,32",
    grid="bs,nheads,max_seqlen_q",
    num_warps=2,
    num_stages=2,
    out_name="f16_kernel",
)

with AOTMetadataContext("kr_example", "/path/to/aot1"):
    triton_kernel1, output_files1 = compile_kernel(compile_args)
```

### For Separate Compile And Run

```python
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.tools.compile import compile_kernel, CompileArgs

from aiter.utility.triton.triton_metadata_redirect import AOTMetadataContext

@triton.jit
def kernel(x_ptr, SIZE: tl.constexpr):
    ...

# compile
target = GPUTarget("hip", 'gfx942', 64)
src = triton.compiler.ASTSource(
    fn=kernel
    signature = {
        "x_ptr": "*fp32",
    },
    constexprs={"SIZE": 64}
)
with AOTMetadataContext("kernel", "/custom/path"):
    kernel = triton.compile(src, target=target)

# run
x = torch.zeros((1, 1, 2), dtype=torch.float32, device='cuda')
kernel[(1, 1, 1)](x)
```

## How It Works

The module patches the `CompiledKernel.__init__` method to intercept metadata file path assignments during Triton kernel compilation. When a kernel is registered (either via decorator or context manager), the module redirects the output paths for:

* {kernel_name}.json - Kernel metadata file

* {kernel_name}.hsaco - Compiled kernel binary

## Thread Safety

The implementation uses thread-local storage to ensure complete thread safety:

* Each thread maintains its own registry of kernel-to-directory mappings

* No global locks or race conditions

* Perfect for multi-threaded compilation environments

## NOTE

**Critical:** For the same kernel name with different template parameters, you must store the generated .json and .hsaco files in separate directories. The module does not differentiate between kernel variants based on template parameters - it only uses the kernel name.

This module implements a simple kernel name to directory mapping without considering compilation parameter variations. In order to handle multiple instantiations, we need to include information about the compilation parameters in the path. However, due to the instability of the hash value (which depends on the environment such as `TRITON_PATH`, `ENV`, etc.), we haven't found a good way to deal with it for now.

* Hash Instability: Triton's hash values depend on TRITON_PATH and environment variables, making them unreliable for matching

* Functional Equivalence: Even kernels with identical functionality may have different hashes due to external factors

* Simplicity: The name-based approach provides a straightforward mapping mechanism

### JIT Compilation

When using the decorator approach for JIT compilation:

* Single Mapping: Each kernel name maps to exactly one directory

* Manual Management: You are responsible for ensuring kernel compatibility

* File Naming: The module looks for {kernel_name}.json and {kernel_name}.hsaco in the specified directory

Example workflow:

```python
# Kernel will use /path/to/kernel/kr_example.json and /path/to/kernel/kr_example.hsaco
@with_custom_metadata_path("/path/to/kernel")
@triton.jit
def kr_example(x_ptr, BLOCK_SIZE: tl.constexpr):
    pass
```

### AOT Compilation

For AOT compilation with multiple kernel variants:

* Sequential Processing: You must compile different variants sequentially

Correct approach for multiple variants:

```python
# Compile first variant
with AOTMetadataContext("kernel_name", "/path/variant1"):
    kernel1, output_files1 = compile_kernel(compile_args_variant1)

# Then compile second variant
with AOTMetadataContext("kernel_name", "/path/variant2"):
    kernel2, output_files2 = compile_kernel(compile_args_variant2)

# And so on for additional variants...
```
