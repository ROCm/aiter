# gfx1250 compat shim: TE's dot_product_attention/backends.py imports
#   from aiter.ops.triton.mha import flash_attn_func, _flash_attn_forward,
#       flash_attn_onekernel_backward, flash_attn_varlen_func
# but in this aiter revision those live under aiter.ops.triton.attention.mha.
# Re-export them here so NVTE_FLASH_ATTN_AITER=1 can find them.
from aiter.ops.triton.attention.mha import (  # noqa: F401
    flash_attn_func,
    flash_attn_varlen_func,
    _flash_attn_forward,
    flash_attn_onekernel_backward,
)
