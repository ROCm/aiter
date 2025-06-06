import functools
import triton

AUTOTUNE_OPS = dict()


def aiter_register(module, kernels):
    def decorator(op_func):
        """Register a op so tuning script can discover it"""
        AUTOTUNE_OPS[op_func.__name__] = {
            "module": module,
            "op_func": op_func,
            "kernels": kernels,
        }
        return op_func

    return decorator


def aiter_register_input_generator(op_func_name):
    def decorator(input_generator_func):
        AUTOTUNE_OPS[op_func_name]["input_generator_func"] = input_generator_func
        return input_generator_func

    return decorator
