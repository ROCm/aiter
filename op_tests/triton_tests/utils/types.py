import torch

__all__ = ["str_to_torch_dtype"]

str_to_torch_dtype = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
