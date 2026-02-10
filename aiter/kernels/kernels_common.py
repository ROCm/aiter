from _mlir.dialects import builtin, gpu as _gpu
from flydsl.dialects.ext import buffer_ops


def stream_ptr_to_async_token(stream_ptr_value, loc=None, ip=None):
    stream_llvm_ptr = buffer_ops.create_llvm_ptr(stream_ptr_value)
    
    async_token_type = _gpu.AsyncTokenType.get()
    cast_op = builtin.UnrealizedConversionCastOp(
        [async_token_type], [stream_llvm_ptr], loc=loc, ip=ip
    )
    return cast_op.results[0]