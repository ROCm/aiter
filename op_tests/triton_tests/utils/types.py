import contextlib

import torch

str_to_torch_dtype = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@contextlib.contextmanager
def with_default_dtype(dtype):
    """Set the torch default dtype and restore the previous one on exit.

    Use it only to test that library code does not depend on the default dtype.
    """
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev_dtype)
