# Grouped GEMM API

Standalone `grouped_gemm_fprop` / `grouped_gemm_dgrad` / `grouped_gemm_wgrad` built
on top of [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) CK + hipBLASLt
grouped GEMM kernels, designed for integration with pytorch training framework.

## Files

| File | Description |
|------|-------------|
| `grouped_gemm_ops.py` | API implementation — `grouped_gemm_fprop`, `grouped_gemm_dgrad`, `grouped_gemm_wgrad` |
| `test_grouped_gemm_api.py` | Correctness tests (437 cases) — compares each API against a pure-PyTorch reference |
| `test_determinism.py` | Determinism tests (72 cases) — runs each API twice with identical inputs, asserts bit-exact match |

## Prerequisites

- Docker image: `rocm/primus:v26.1`
- AMD Instinct GPU (MI300X / MI355X)

Pull the image if you haven't already:

```bash
docker pull rocm/primus:v26.1
```

## API Signatures

```python
def grouped_gemm_fprop(
    x: torch.Tensor,           # [GM, K]
    w: torch.Tensor,           # [G, N, K]
    split_sizes: torch.Tensor, # [G], int32 or int64
) -> torch.Tensor:             # [GM, N]

def grouped_gemm_dgrad(
    dy: torch.Tensor,          # [GM, N]
    w: torch.Tensor,           # [G, N, K]
    split_sizes: torch.Tensor, # [G]
) -> torch.Tensor:             # [GM, K]

def grouped_gemm_wgrad(
    dy: torch.Tensor,                      # [GM, N]
    x: torch.Tensor,                       # [GM, K]
    split_sizes: torch.Tensor,             # [G]
    wgrad: Optional[torch.Tensor] = None,  # [G, N, K]
    output_accum: bool = False,
) -> torch.Tensor:                         # [G, N, K]
```

## Running with Docker

> **Note:** The mount target must NOT be `/workspace` — the Docker image uses
> `/workspace/aiter` for an editable install. Use `/test_dir` (or any other path) instead.

### Set the source directory

Point `GROUPED_GEMM_API_DIR` to the directory containing the Python files:

```bash
# Adjust the path to wherever grouped_gemm_ops.py lives
export GROUPED_GEMM_API_DIR=/path/to/grouped_gemm_api
```

### Run all correctness tests

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v "${GROUPED_GEMM_API_DIR}":/test_dir \
  --workdir /test_dir \
  -e PYTHONPATH=/test_dir \
  rocm/primus:v26.1 \
  python3 -m pytest test_grouped_gemm_api.py -v --tb=short
```

### Run all determinism tests

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v "${GROUPED_GEMM_API_DIR}":/test_dir \
  --workdir /test_dir \
  -e PYTHONPATH=/test_dir \
  rocm/primus:v26.1 \
  python3 -m pytest test_determinism.py -v --tb=short -s
```

### Run a specific test class or test

```bash
# Only fprop correctness tests
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v "${GROUPED_GEMM_API_DIR}":/test_dir \
  --workdir /test_dir \
  -e PYTHONPATH=/test_dir \
  rocm/primus:v26.1 \
  python3 -m pytest test_grouped_gemm_api.py::TestGroupedGemmAPI::test_fprop -v --tb=short

# Only wgrad output_accum tests
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v "${GROUPED_GEMM_API_DIR}":/test_dir \
  --workdir /test_dir \
  -e PYTHONPATH=/test_dir \
  rocm/primus:v26.1 \
  python3 -m pytest test_grouped_gemm_api.py::TestWgradOutputAccum -v --tb=short

# Only edge-case tests
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v "${GROUPED_GEMM_API_DIR}":/test_dir \
  --workdir /test_dir \
  -e PYTHONPATH=/test_dir \
  rocm/primus:v26.1 \
  python3 -m pytest test_grouped_gemm_api.py::TestEdgeCases -v --tb=short
```

### Run an interactive session (for debugging)

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v "${GROUPED_GEMM_API_DIR}":/test_dir \
  --workdir /test_dir \
  -e PYTHONPATH=/test_dir \
  rocm/primus:v26.1 \
  bash
```

Then inside the container:

```python
python3
>>> from grouped_gemm_ops import *
>>> import torch
>>> G, M, N, K = 8, 256, 4096, 2048
>>> split_sizes = torch.full((G,), M, dtype=torch.int64, device="cuda")
>>> x = torch.randn(G*M, K, device="cuda", dtype=torch.bfloat16)
>>> w = torch.randn(G, N, K, device="cuda", dtype=torch.bfloat16)
>>> out = grouped_gemm_fprop(x, w, split_sizes)
>>> out.shape
torch.Size([2048, 4096])
```

## Test Matrix

### Correctness tests (`test_grouped_gemm_api.py`) — 437 cases

| Parameter | Values |
|-----------|--------|
| G (groups) | 1, 2, 4, 8 |
| M (tokens per group) | 128, 256, 512 |
| (N, K) | (2048, 1536), (4096, 4096), (1024, 2048) |
| dtype | bfloat16, float16 |
| balanced | True, False |

Additional tests: output_accum, single group, int32 split_sizes, autograd consistency.

### Determinism tests (`test_determinism.py`) — 72 cases

| Parameter | Values |
|-----------|--------|
| G (groups) | 1, 4, 8 |
| M (tokens per group) | 128, 512 |
| (N, K) | (2048, 1536), (4096, 4096) |
| dtype | bfloat16, float16 |
