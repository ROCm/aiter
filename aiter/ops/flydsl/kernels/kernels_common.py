"""Common helpers shared by kernel modules.

Keep helper naming consistent with other kernel helpers (e.g. `mfma_preshuffle_pipeline.py`),
but this module is intentionally small and MLIR-dialect facing.
"""

import functools

from flydsl._mlir import ir
from flydsl.expr.typing import T
from flydsl._mlir.dialects import (
    arith as _std_arith,
    builtin,
    gpu as _gpu,
    llvm as _llvm,
)
from flydsl.expr import buffer_ops
from flydsl.runtime.device import get_rocm_arch, is_rdna_arch


def get_warp_size(arch=None):
    """Return the wavefront/warp size for the given GPU architecture.

    CDNA (gfx9xx) uses wave64, RDNA (gfx10xx/gfx11xx/gfx12xx) uses wave32.
    """
    if arch is None:
        arch = get_rocm_arch()
    return 32 if is_rdna_arch(arch) else 64


# Per-CU LDS budget used by the MoE-GEMM `waves_per_eu` occupancy padding.
# Values come from the AMD CDNA ISA reference:
#   gfx942 (MI300X): 64 KB per CU
#   gfx950 (MI355X): 160 KB per CU
# Anything else falls back to the safer (smaller) value so the padding never
# over-reserves and silently kills occupancy.
_PER_CU_LDS_BYTES_BY_ARCH = {
    "gfx942": 64 * 1024,
    "gfx950": 160 * 1024,
}


@functools.lru_cache(maxsize=4)
def per_cu_lds_bytes(arch=None) -> int:
    """Return the per-CU LDS budget in bytes for the given gfx target.

    Used by `moe_gemm_2stage.py` / `mixed_moe_gemm_2stage.py` (and the
    candidate filter in `moe_kernels._lds_within_limit`) to size the
    `waves_per_eu` LDS reservation against the **device's actual** LDS, not
    a hard-coded gfx950 number.  Without this, gfx942 kernels would inherit
    the gfx950 80 KB reservation and collapse to 1 workgroup per CU.
    """
    if arch is None:
        arch = get_rocm_arch()
    key = str(arch).split(":", 1)[0].lower()
    return _PER_CU_LDS_BYTES_BY_ARCH.get(key, 64 * 1024)


def _create_llvm_ptr(value, address_space: int = 1):
    value = buffer_ops._unwrap_value(value)
    if isinstance(value.type, ir.IndexType):
        i64_type = T.i64
        value = buffer_ops._unwrap_value(_std_arith.IndexCastOp(i64_type, value).result)
    ptr_type = ir.Type.parse(f"!llvm.ptr<{address_space}>")
    return _llvm.IntToPtrOp(ptr_type, value).result


def stream_ptr_to_async_token(stream_ptr_value, loc=None, ip=None):
    stream_llvm_ptr = _create_llvm_ptr(stream_ptr_value)

    async_token_type = _gpu.AsyncTokenType.get()
    cast_op = builtin.UnrealizedConversionCastOp(
        [async_token_type], [stream_llvm_ptr], loc=loc, ip=ip
    )
    return cast_op.results[0]
