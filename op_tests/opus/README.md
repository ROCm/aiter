# OPUS Unit Tests

This directory contains unit tests for **OPUS** (AI Operator Micro Std), a lightweight templated C++ DSL for AMD GPU kernel development used in AITER.

## Location

All OPUS tests are organized under:
```
aiter/op_tests/opus/
```

## Files

### C++ Unit Tests

- **`test_opus_basic.cpp`** - Core OPUS functionality tests
  - Number and sequence operations
  - Array and tuple containers  
  - Static for loops
  - Type traits
  - Layout abstractions

### Python Tests

- **`test_integration.py`** - Integration tests for OPUS-based kernels
  - RMSNorm + Quantization kernels
  - Cache operations
  - Quantization kernels
  - Vectorized operations

- **`test_opus_types.py`** - Python unit tests for OPUS data types
  - Vectorized load/store operations
  - Memory layout tests
  - Data type conversion tests

### Build Files

- **`CMakeLists.txt`** - CMake build configuration for C++ tests
- **`Makefile`** - Makefile for easy building
- **`__init__.py`** - Python package marker

## Prerequisites

- **HIP/ROCm** - Required for C++ tests (OPUS uses GPU-specific extensions)
- **CMake 3.16+** - Optional, for CMake builds
- **Python 3.8+** with PyTorch - For Python integration tests

## Building and Running Tests

### C++ Tests (requires hipcc)

```bash
cd aiter/op_tests/opus

# Using Make
make all
make test

# Using CMake
mkdir build && cd build
cmake ..
make
ctest

# Manual compilation with hipcc
hipcc -std=c++17 -O2 -I../../csrc/include test_opus_basic.cpp -o test_opus_basic
./test_opus_basic
```

### Python Tests

```bash
cd aiter/op_tests/opus

# Run all Python tests
python test_integration.py

# Run with specific parameters
python test_integration.py --dtype bf16 -m 1024 -n 4096

# Run specific test category
python test_integration.py --test rmsnorm
python test_integration.py --test cache
python test_integration.py --test quant

# Run type tests
python test_opus_types.py

# Or use pytest (if installed)
pytest test_opus_types.py -v
```

## Test Categories

### C++ Tests (`test_opus_basic.cpp`)

| Component | Tests |
|-----------|-------|
| Number | Literals (`10_I`), arithmetic, comparison |
| Sequence | Creation, indexing, reduction |
| Array | Access, fill, clear, concat |
| Tuple | Creation, get, concat, repeated |
| Static For | Unrolled loops, nested loops |
| Type Traits | is_constant, is_seq, is_array, is_tuple |
| Layout | Basic layout creation |

### Python Integration Tests (`test_integration.py`)

| Test Function | OPUS Features Tested |
|---------------|---------------------|
| `test_rmsnorm_quant()` | `gmem`, `vector_t`, `static_for` |
| `test_cache_operations()` | Buffer load/store, vectorized access |
| `test_quant_kernels()` | Vector types, unrolled loops |
| `test_vectorized_operations()` | Various vector sizes and types |

### Python Type Tests (`test_opus_types.py`)

| Test Class | Tests |
|------------|-------|
| `TestOpusDataTypes` | FP16, BF16, FP32, INT8 operations |
| `TestOpusVectorSizes` | Vector sizes 4, 8, 16 |
| `TestOpusMemoryLayout` | Contiguous and strided memory |
| `TestOpusConstants` | Numeric limits and constants |

## OPUS Usage in AITER

OPUS is used in the following AITER kernels:

- `csrc/kernels/rmsnorm_quant_kernels.cu` - RMSNorm with quantization
- `csrc/kernels/quant_kernels.cu` - Quantization operations
- `csrc/kernels/cache_kernels.cu` - KV-cache operations
- `csrc/kernels/topk_plain_kernels.cu` - Top-k operations

See `csrc/include/aiter_opus_plus.h` for AITER-specific OPUS extensions.

## Adding New Tests

### C++ Tests

1. Add test function to `test_opus_basic.cpp`:

```cpp
bool test_my_feature() {
    // Test code here
    TEST_ASSERT_EQ(actual, expected, "description");
    return true;
}
```

2. Register test in `main()`:

```cpp
RUN_TEST(test_my_feature);
```

### Python Tests

1. Create a new test file or add to existing:

```python
def test_my_kernel():
    print("\n--- My Kernel Test ---")
    # Test code here
    return True
```

2. Or use pytest style:

```python
class TestMyFeature:
    def test_something(self):
        assert True
```

## References

- OPUS README: `csrc/include/opus/README.md`
- OPUS Header: `csrc/include/opus/opus.hpp`
- AITER OPUS Extensions: `csrc/include/aiter_opus_plus.h`
