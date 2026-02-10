# OPUS MFMA PyTorch extension

All sources for the 32×32×8 fp16 MFMA test and PyTorch extension live here. The extension is built with **BuildExtension** and **CUDAExtension** (`torch.utils.cpp_extension`).

- **test_opus_mfma.cu** / **test_opus_mfma.h** — kernel and `run_mfma_32x32x8_f16()` (C API); `.cu` ensures HIP/device compiler is used
- **opus_mfma_ext.cpp** — pybind11 wrapper for PyTorch
- **setup.py** — builds the extension via `CUDAExtension` + `BuildExtension`
- **test_opus_mfma.py** — builds (if needed) and runs test (random A/B vs `torch` GEMM)

**Run test** (from `op_tests/opus`): `python3 mfma/test_opus_mfma.py` — builds in-place on first run if needed.

**Build only** (from `mfma/`): `python setup.py build_ext --inplace`
