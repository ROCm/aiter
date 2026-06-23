import torch
import triton

from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.triton._triton_kernels.fusions.lm_head_argmax import (
    _local_argmax_pack_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()
_MAX_BLOCK_M = 131072


def _local_argmax_pack_fake(logits: torch.Tensor, vocab_start_idx: int) -> torch.Tensor:
    return torch.empty((logits.shape[0], 2), dtype=torch.float32, device=logits.device)


@torch_compile_guard(gen_fake=_local_argmax_pack_fake)
def local_argmax_pack(logits: torch.Tensor, vocab_start_idx: int) -> torch.Tensor:
    """Reduce local LM-head logits and pack ``(max_val, global_idx)`` as fp32."""
    _LOGGER.info(
        f"LOCAL_ARGMAX_PACK: logits={tuple(logits.shape)} vocab_start_idx={vocab_start_idx}"
    )
    if logits.dim() != 2:
        raise ValueError("local_argmax_pack expects a 2-D logits tensor")

    N, M = logits.shape
    if N == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=logits.device)
    if M > _MAX_BLOCK_M:
        local_max_val, local_idx = logits.max(dim=-1)
        global_idx = local_idx + vocab_start_idx
        return torch.stack([local_max_val.float(), global_idx.float()], dim=-1)

    packed = torch.empty((N, 2), dtype=torch.float32, device=logits.device)
    block_m = triton.next_power_of_2(M)
    num_warps = 8 if block_m >= 2048 else 4

    _local_argmax_pack_kernel[(N,)](
        logits,
        packed,
        vocab_start_idx,
        N=N,
        M=M,
        stride_logits_n=logits.stride(0),
        stride_logits_m=logits.stride(1),
        stride_packed_n=packed.stride(0),
        BLOCK_M=block_m,
        num_warps=num_warps,
        num_stages=2,
    )
    return packed
