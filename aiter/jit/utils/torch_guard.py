aiter_lib = None


def torch_compile_guard(mutates_args: list[str] = [], device: str = "cpu"):
    def decorator(func):
        try:
            import torch
            from torch.library import Library
            import inspect
        except ImportError:

            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        global aiter_lib
        aiter_lib = Library("aiter", "FRAGMENT") if aiter_lib is None else aiter_lib
        op_name = func.__name__

        def custom_impl(dummy_tensor, *args, **kwargs):
            return func(*args, **kwargs)

        def outer_wrapper(*args, **kwargs):
            dummy = torch.empty(1, device=device)
            return getattr(torch.ops.aiter, op_name)(dummy, *args, **kwargs)

        if hasattr(torch.ops.aiter, func.__name__):
            return outer_wrapper
        if hasattr(torch.library, "infer_schema"):
            schema_str = torch.library.infer_schema(func, mutates_args=mutates_args)
        else:
            # for pytorch 2.4
            import torch._custom_op.impl

            schema_str = torch._custom_op.impl.infer_schema(
                func, mutates_args=mutates_args
            )

        sig = inspect.signature(func)
        input_part, output_part = schema_str.split("->", 1)
        if not sig.parameters:
            new_input = "(Tensor dummy)"
        else:
            new_input = "(Tensor dummy, " + input_part[1:]

        schema_str = f"{new_input} -> {output_part}".strip()

        my_lib = aiter_lib
        my_lib.define(op_name + schema_str, tags=())
        my_lib.impl(op_name, custom_impl, dispatch_key="CUDA")
        my_lib.impl(op_name, custom_impl, dispatch_key="CPU")
        my_lib._register_fake(op_name, custom_impl)

        return outer_wrapper

    return decorator
