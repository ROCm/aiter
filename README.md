# aiter
![image](https://github.com/user-attachments/assets/9457804f-77cd-44b0-a088-992e4b9971c6)


AITER is AMD’s centralized repository that support various of high performance AI operators for AI workloads acceleration, where a good unified place for all the customer operator-level requests, which can match different customers' needs. Developers can focus on operators, and let the customers integrate this op collection into their own private/public/whatever framework.
 

Some summary of the features:
* C++ level API
* Python level API
* The underneath kernel could come from triton/ck/asm
* Not just inference kernels, but also training kernels and GEMM+communication kernels—allowing for workarounds in any kernel-framework combination for any architecture limitation.



## Installation

### Basic Installation
```bash
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
python3 setup.py develop
```

If you happen to forget the `--recursive` during `clone`, you can use the following command after `cd aiter`
```bash
git submodule sync && git submodule update --init --recursive
```

### Optional: Triton-based Communication (Iris)

AITER supports GPU-initiated communication using the [Iris library](https://github.com/ROCm/iris). This enables high-performance Triton-based communication primitives like reduce-scatter and all-gather.

**Install with Triton communication support:**
```bash
# Option 1: Install via extras
pip install -e ".[triton_comms]"

# Option 2: Install all optional dependencies
pip install -e ".[all]"
```

**Basic usage:**
```python
from aiter import IrisCommContext, reduce_scatter, all_gather
import torch.distributed as dist

# Initialize PyTorch distributed
dist.init_process_group(backend="nccl")

# Use Iris-based communication
with IrisCommContext(heap_size=2**30) as ctx:  # 1GB heap
    input_tensor = ctx.iris_ctx.empty((4096, 4096), dtype=torch.float32)
    output = reduce_scatter(input_tensor, ctx)
    result = all_gather(output, ctx)
```

**Automatic heap size calculation:**
```python
from aiter import IrisCommContext, calculate_heap_size
import torch

# Automatically calculate required heap size for your operations
M, N = 8192, 7168  # Your tensor dimensions
heap_size = calculate_heap_size(
    M=M,
    N=N,
    dtype=torch.float32,
    world_size=2,  # Number of GPUs
    quant_mode="fp8_per_token",  # "none", "fp8_per_token", or "fp4_per_token"
    all_gather=True,
)

# Guaranteed to have enough memory
with IrisCommContext(heap_size=heap_size) as ctx:
    # Your operations here
    pass
```

For more details, see [docs/iris_communication.md](docs/iris_communication.md)

## Run operators supported by aiter

There are number of op test, you can run them with: `python3 op_tests/test_layernorm2d.py`
|  **Ops**                      | **Description**                                                                                                                                                   |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|ELEMENT WISE                   | ops: + - * /                                                                                                                                                      |
|SIGMOID                        | (x) = 1 / (1 + e^-x)                                                                                                                                              |
|AllREDUCE                      | Reduce + Broadcast                                                                                                                                                |
|KVCACHE                        | W_K W_V                                                                                                                                                           |
|MHA                            | Multi-Head Attention                                                                                                                                              |
|MLA                            | Multi-head Latent Attention with [KV-Cache layout](https://docs.flashinfer.ai/tutorials/kv_layout.html#page-table-layout )                                        |
|PA                             | Paged Attention                                                                                                                                                   |
|FusedMoe                       | Mixture of Experts                                                                                                                                                |
|QUANT                          | BF16/FP16 -> FP8/INT4                                                                                                                                             |
|RMSNORM                        | root mean square                                                                                                                                                  |
|LAYERNORM                      | x = (x - u) / (σ2 + ϵ) e*0.5                                                                                                                                      |
|ROPE                           | Rotary Position Embedding                                                                                                                                         |
|GEMM                           | D=αAβB+C                                                                                                                                                          |
