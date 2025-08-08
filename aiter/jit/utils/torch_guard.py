from torch.library import Library

aiter_lib = Library("aiter", "FRAGMENT")


def torch_compile_guard(
    mutates_args: list[str] = [],
):
    def decorator(func):
        import torch

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        def abstract_impl(*args, **kwargs):
            return func(*args, **kwargs)

        def custom_wrapper(*args, **kwargs):
            return getattr(torch.ops.aiter, func.__name__)(*args, **kwargs)

        if hasattr(torch.ops.aiter, func.__name__):
            return custom_wrapper
        if hasattr(torch.library, "infer_schema"):
            schema_str = torch.library.infer_schema(func, mutates_args=mutates_args)
        else:
            # for pytorch 2.4
            import torch._custom_op.impl

            schema_str = torch._custom_op.impl.infer_schema(
                func, mutates_args=mutates_args
            )

        op_name = func.__name__
        my_lib = aiter_lib
        my_lib.define(op_name + schema_str, tags=())
        my_lib.impl(op_name, wrapper, dispatch_key="CUDA")
        my_lib._register_fake(op_name, abstract_impl)

        return custom_wrapper

    return decorator
