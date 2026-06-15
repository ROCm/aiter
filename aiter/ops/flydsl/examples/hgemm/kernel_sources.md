# HGEMM kernel sources

Authoritative FlyDSL kernel implementation for this op. Paths are repo-relative.
These files are the source of truth; this example does not duplicate them.

| Path | Role | Entry symbol |
| ---- | ---- | ------------ |
| `aiter/ops/flydsl/gemm_kernels.py` | Public wrapper: validates shapes/tiling, allocates output, compiles + launches. | `flydsl_hgemm` |
| `aiter/ops/flydsl/kernels/hgemm_dispatch.py` | Dispatches a tiling config to the right kernel family and builds it. | `compile_flydsl_hgemm_kernel` |
| `aiter/ops/flydsl/kernels/splitk_hgemm.py` | Kernel body for the default `hgemm` family (split-K capable GEMM). | `compile_hgemm_kernel` |
| `aiter/ops/flydsl/kernels/small_m_hgemm.py` | Alternate kernel body for the `small_m` family (selected via `kernel_family="small_m"`). | `compile_small_m_hgemm_kernel` |

The example's default path (`kernel_family=None`) builds the `hgemm` family, so
`splitk_hgemm.py` is the active kernel body; `small_m_hgemm.py` is only used when
the small-M family is requested.
