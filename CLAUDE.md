# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AITER (AMD Inference and Training Engine Repository) is AMD's centralized high-performance operator library for AI workloads on ROCm GPUs. It provides optimized implementations of operators for LLM inference and training, with kernels implemented in HIP/CK (Composable Kernel), Triton, and assembly.

Key characteristics:
- Multi-backend kernel support: Composable Kernel (CK), Triton, HIP/CUDA, and assembly
- JIT (Just-In-Time) compilation system for dynamic kernel generation
- Auto-tuning infrastructure with pre-tuned configurations per GPU architecture
- Supports MI200, MI300, MI350 series AMD GPUs
- Both Python and C++ APIs available

## Build and Installation

### Basic Installation

```bash
# Clone with submodules (Composable Kernel is required)
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter

# If you forgot --recursive during clone
git submodule sync && git submodule update --init --recursive

# Development mode installation (recommended for development)
python3 setup.py develop

# Production installation
pip install -e .
```

### Build with Pre-built Kernels

To speed up compilation by pre-building common kernels:

```bash
# Pre-build common kernels for specific GPU architectures
PREBUILD_KERNELS=1 GPU_ARCHS="gfx942;gfx950" python setup.py develop
```

PREBUILD_KERNELS modes:
- `0`: No pre-build (default), excludes tune modules
- `1`: Pre-build inference kernels only (excludes backward and tune)
- `2`: Pre-build all forward kernels (excludes backward and tune)
- `3`: Pre-build only FMHA v3 and enum modules

### Installation with Triton Communication Support

```bash
pip install -e .
pip install -r requirements-triton-comms.txt
```

## Development Commands

### Code Formatting and Linting

```bash
# Install pre-commit hooks (REQUIRED before committing)
pip install black==25.1.0 ruff==0.11.11
apt install clang-format
bash ./.githooks/install

# Manual formatting
black aiter/ op_tests/          # Python formatting
ruff check aiter/ op_tests/     # Python linting
find csrc/ -name "*.cu" -o -name "*.h" -o -name "*.cpp" | xargs clang-format -i  # C++/HIP
```

### Testing

```bash
# Run all tests
bash .github/scripts/aiter_test.sh

# Run specific operator test
python op_tests/test_rmsnorm2d.py

# Run test with specific parameters
python op_tests/test_rmsnorm2d.py --dtype bf16 --m 1024 --n 4096

# Run Triton-specific tests
python op_tests/triton_tests/normalization/test_rmsnorm.py

# Run multi-GPU tests
MULTIGPU=TRUE bash .github/scripts/aiter_test.sh
```

### Debugging and Profiling

```bash
# Enable verbose logging
AITER_LOG_MORE=1 python op_tests/test_gemm_a8w8.py

# Set log level (DEBUG, INFO, WARNING, ERROR)
AITER_LOG_LEVEL=DEBUG python op_tests/test_rmsnorm2d.py

# Log tuned configuration selection
AITER_LOG_TUNED_CONFIG=1 python op_tests/test_gemm_a8w8.py

# Force rebuild of JIT modules
AITER_REBUILD=1 python op_tests/test_rmsnorm2d.py
```

### Single Test Development

To run a single test file efficiently during development:

```bash
# Direct execution
python op_tests/test_<operator_name>.py

# With specific dtype
python op_tests/test_rmsnorm2d.py -d bf16

# With specific dimensions
python op_tests/test_gemm_a8w8.py -m 1024 -n 4096 -k 4096
```

## Architecture Overview

### JIT Compilation System

AITER uses a custom JIT compilation system that dynamically compiles and loads operator kernels:

1. **Module Registry** (`aiter/jit/optCompilerConfig.json`): Defines all compilable modules with their source files, flags, and code generation commands
2. **JIT Core** (`aiter/jit/core.py`): Handles compilation, caching, and loading of modules
3. **Decorator-based API** (`@compile_ops`): Wraps Python functions to trigger JIT compilation on first call

**Key JIT workflow:**
- On first call to an operator, check if `.so` module exists in JIT cache (`~/.cache/aiter/jit/` by default)
- If not found, run code generation (if needed), compile C++/HIP sources with `hipcc`, and cache the module
- Load the compiled `.so` and bind it to the Python function
- Subsequent calls use the cached module directly

**Code Generation (`blob_gen_cmd`):**
Many modules use code generators to create kernel instances:
- CK tile generation: `generate.py` scripts in Composable Kernel examples
- Assembly codegen: `hsa/codegen.py` for hand-written assembly kernels
- Dynamic kernel generation: Python scripts that generate C++/HIP code based on parameters

### Operator Structure

Operators are organized in `aiter/ops/`:
- Each operator has a Python file (e.g., `rmsnorm.py`, `gemm_op_a8w8.py`)
- Operators use `@compile_ops` decorator to link to C++/HIP implementations
- Backend implementations are in `csrc/`:
  - `csrc/kernels/`: Core HIP/CK tile kernel implementations
  - `csrc/py_itfs_ck/`: Python interface for CK-based kernels
  - `csrc/py_itfs_cu/`: Python interface for CUDA/HIP kernels
  - `csrc/pybind/`: PyBind11 bindings for C++ functions
  - `csrc/ck_*`: CK-specific GEMM implementations
  - `csrc/include/`: Header files including CK tile headers

### Auto-tuning System

AITER uses pre-tuned configurations for optimal performance:

1. **Tuning Scripts**: Located in operator-specific directories (e.g., `csrc/ck_batched_gemm_a8w8/`)
2. **Configuration Files**: Stored in `aiter/configs/*.csv` (e.g., `a8w8_tuned_gemm.csv`)
3. **Runtime Selection**: Operators read CSV configs to select best kernel for given shape/GPU
4. **Environment Override**: Use `AITER_CONFIG_*` environment variables to point to custom configs

**Example auto-tuning workflow:**
```bash
# Run manual tuning pipeline
# See docs/autotuning_pipeline.md for details
# GitHub Actions: https://github.com/ROCm/aiter/actions/workflows/operators-tuning.yaml
```

### Testing Framework

All tests follow a consistent pattern (see `aiter/test_common.py`):

1. **@perftest() decorator**: Wraps test functions for performance measurement
2. **Reference implementation**: PyTorch or other baseline (e.g., `run_torch`)
3. **AITER implementation**: Optimized kernel (e.g., `run_ck`)
4. **Correctness check**: `checkAllclose()` compares outputs
5. **Performance reporting**: Automatically prints latency comparisons

**Test file structure:**
```python
@perftest()
def run_reference(input, ...):
    # PyTorch/baseline implementation
    return output

@perftest()
def run_aiter(input, ...):
    # AITER optimized implementation
    return output

def test_operator(dtype, m, n, ...):
    # Setup inputs
    # Run both implementations
    # Check correctness with checkAllclose()
    # Report performance
```

### Multi-Backend Support

AITER supports multiple kernel backends with runtime selection:

- **CK (Composable Kernel)**: Primary backend for GEMM-like ops, tile-based, highly optimized
- **Triton**: For rapid prototyping and operators like normalization, attention
- **HIP/CUDA**: Direct kernel implementations for fine control
- **Assembly (ASM)**: Hand-optimized kernels for critical paths (in `hsa/`)

**Backend selection logic:**
- Usually automatic based on auto-tuned configs
- Some operators expose backend parameter (e.g., `backend="ck"`)
- CK tiles are code-generated based on data types and shapes

## Common Development Workflows

### Adding a New Operator

1. Create Python operator file in `aiter/ops/new_operator.py`
2. Add C++/HIP implementation in `csrc/kernels/new_operator.cu`
3. Add PyBind11 binding in `csrc/pybind/new_operator_pybind.cu`
4. Register module in `aiter/jit/optCompilerConfig.json`
5. Use `@compile_ops` decorator in Python to link to JIT module
6. Add test in `op_tests/test_new_operator.py`
7. Export from `aiter/__init__.py`

**Important**: Minimize external dependencies when adding new operators:
- Prefer using existing dependencies (PyTorch, ROCm/HIP, Composable Kernel, Triton)
- Avoid introducing new third-party libraries unless absolutely necessary
- If a new dependency is required, provide strong justification in the PR
- Consider implementing functionality from scratch if the dependency is simple

### Modifying an Existing Kernel

1. Locate the kernel source in `csrc/kernels/` or `csrc/ck_*/`
2. Modify the implementation
3. If using JIT, set `AITER_REBUILD=1` to force recompilation
4. Run corresponding test to verify correctness
5. Check performance impact with `AITER_LOG_MORE=1`

### Working with Composable Kernel (CK)

When integrating or modifying CK tile-based operators:

1. CK source is in submodule: `3rdparty/composable_kernel/`
2. CK tile headers: `csrc/include/ck_tile/` (symlinked to CK)
3. Generate CK instances:
   ```bash
   cd 3rdparty/composable_kernel/example/ck_tile/<op>/
   python generate.py -d fwd --receipt 200 --filter "*bf16*" --output_dir /tmp/ck_gen
   ```
4. Update `blob_gen_cmd` in `optCompilerConfig.json` to use generator
5. CK instances are code-generated at JIT compile time

### GPU Architecture Handling

AITER detects GPU architecture at runtime:
- `get_gfx()`: Returns current GPU architecture (e.g., "gfx942" for MI300X)
- `get_gfx_list()`: Returns list of all visible GPUs
- Auto-tuned configs are architecture-specific
- Some operators have arch-specific code paths

**Supported architectures:**
- gfx90a: MI200 series
- gfx942: MI300X
- gfx950: MI350X

## Important Configuration Files

- `aiter/jit/optCompilerConfig.json`: JIT module registry
- `aiter/configs/*.csv`: Auto-tuned operator configurations
- `setup.py`: Build system with PREBUILD_KERNELS logic
- `.githooks/pre-commit`: Automatic code formatting on commit
- `pyproject.toml`: Build system configuration

## Environment Variables

**Logging:**
- `AITER_LOG_LEVEL`: Set log level (DEBUG, INFO, WARNING, ERROR)
- `AITER_LOG_MORE`: Enable detailed logging with timestamps and file locations
- `AITER_LOG_TUNED_CONFIG`: Log which tuned config is selected

**Build:**
- `PREBUILD_KERNELS`: Pre-build mode (0, 1, 2, 3)
- `GPU_ARCHS`: Target GPU architectures (e.g., "gfx942;gfx950")
- `BUILD_TARGET`: Build target (auto, rocm)
- `MAX_JOBS`: Parallel compilation jobs
- `AITER_REBUILD`: Force rebuild of JIT modules

**JIT:**
- `CK_DIR`: Path to Composable Kernel (default: `./3rdparty/composable_kernel`)

**Auto-tuning Config Overrides:**
- `AITER_CONFIG_GEMM_A8W8`: Path to A8W8 GEMM config CSV
- `AITER_CONFIG_GEMM_A4W4`: Path to A4W4 GEMM config CSV
- `AITER_CONFIG_FMOE`: Path to FusedMoE config CSV
- (See `aiter/jit/core.py` for full list)

## Common Pitfalls

1. **Missing submodules**: Always clone with `--recursive` or run `git submodule update --init --recursive`
2. **JIT cache staleness**: Use `AITER_REBUILD=1` after modifying C++/HIP sources
3. **GPU architecture mismatch**: Ensure tuned configs exist for your GPU (check `aiter/configs/`)
4. **MAX_JOBS too high**: Can cause OOM during compilation; setup.py auto-calculates based on memory
5. **Missing pre-commit hooks**: Run `bash ./.githooks/install` before committing

## Kernel Performance Guidelines

When developing or optimizing kernels:

1. **Memory-bound operators** (most operators):
   - Use vectorized loads/stores (vec4, vec8, vec16)
   - Ensure coalesced memory access
   - Minimize global memory accesses

2. **Benchmark against reference**:
   - Use `@perftest()` decorator for timing
   - Compare with PyTorch baseline
   - Report bandwidth utilization (% of peak)

3. **Profile with rocprof** for detailed analysis:
   ```bash
   rocprof --stats python op_tests/test_operator.py
   ```

4. **Roofline analysis**:
   - Calculate arithmetic intensity (FLOPs/Byte)
   - Compare to hardware specs (MI300X: ~380 TFLOPS FP16, 3.2 TB/s)
   - Document if compute-bound or memory-bound

## Contributing and Pull Requests

### PR Title Format

Use one of these prefixes for PR titles:
- `[Bugfix]` - Bug fixes
- `[Feature]` - New features or operators
- `[Kernel]` - Kernel optimizations or new kernels
- `[HIP]` - HIP-specific changes
- `[CK]` - Composable Kernel integration
- `[Triton]` - Triton kernel changes
- `[Perf]` - Performance optimizations
- `[Doc]` - Documentation improvements
- `[Test]` - Test additions or fixes

Examples:
- `[Kernel][Perf] Optimize RMSNorm for MI300X using vec16 loads`
- `[Feature] Add PagedAttention operator with CK backend`
- `[Bugfix] Fix numerical instability in FP16 softmax`

### Developer Certificate of Origin (DCO)

All commits must be signed off with the Developer Certificate of Origin:

```bash
# Sign off commits
git commit -s -m "Your commit message"

# Enable automatic sign-off globally
git config --global format.signoff true
```

This adds a `Signed-off-by` line to your commit message certifying that you have the right to submit the contribution under the MIT License.
