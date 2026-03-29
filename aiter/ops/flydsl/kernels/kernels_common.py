"""Common helpers shared by kernel modules.

Keep helper naming consistent with other kernel helpers (e.g. `mfma_preshuffle_pipeline.py`),
but this module is intentionally small and MLIR-dialect facing.
"""

from flydsl._mlir import ir
from flydsl.expr.typing import T
from flydsl._mlir.dialects import (
    arith as _std_arith,
    builtin,
    gpu as _gpu,
    llvm as _llvm,
)
from flydsl.expr import buffer_ops


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
