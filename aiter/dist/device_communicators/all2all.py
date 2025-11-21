import torch
import importlib.util
from .base_device_communicator import All2AllManagerBase, Cache
from functools import cache
from aiter import logger


@cache
def _has_module(module_name: str) -> bool:
    """Return True if *module_name* can be found in the current environment.
    The result is cached so that subsequent queries for the same module incur
    no additional overhead.
    """
    return importlib.util.find_spec(module_name) is not None


def has_mori() -> bool:
    """Whether the optional `mori` package is available."""
    return _has_module("mori")


class MoriAll2AllManager(All2AllManagerBase):
    def __init__(self, cpu_group):
        assert has_mori(), (
            "MoRI kernels not found. Please follow https://github.com/ROCm/mori/blob/main/README.md"
            " to install MoRI kernels."
        )  # noqa
        import mori

        super().__init__(cpu_group)
        self.handle_cache = Cache()

        torch._C._distributed_c10d._register_process_group("mori", cpu_group)
        mori.shmem.shmem_torch_process_group_init("mori")

    def _make_all2all_kwargs(
        self,
        rank: int,
        num_ep_ranks: int,
        input_dtype: torch.dtype,
        quant_dtype: torch.dtype,
        token_hidden_size: int,
        scale_dim: int,
        scale_type_size: int,
        max_num_tokens_per_dp_rank: int,
        num_local_experts: int,
        num_experts_per_token: int,
        gpu_per_node: int,
    ):
        import mori  # type: ignore[import-not-found]

        if not self.internode:
            # single node
            kernel_type = mori.ops.EpDispatchCombineKernelType.IntraNode
            warp_num_per_block = 16
            block_num = 80
            rdma_block_num = 0
        else:
            # multi node
            kernel_type = mori.ops.EpDispatchCombineKernelType.InterNodeV1
            warp_num_per_block = 16
            block_num = 32
            rdma_block_num = 16

        return dict(
            rank=rank,
            world_size=num_ep_ranks,
            data_type=quant_dtype,
            hidden_dim=token_hidden_size,
            scale_dim=scale_dim,
            scale_type_size=scale_type_size,
            max_token_type_size=input_dtype.itemsize,
            max_num_inp_token_per_rank=max_num_tokens_per_dp_rank,
            num_experts_per_rank=num_local_experts,
            num_experts_per_token=num_experts_per_token,
            warp_num_per_block=warp_num_per_block,
            block_num=block_num,
            kernel_type=kernel_type,
            rdma_block_num=rdma_block_num,
            gpu_per_node=gpu_per_node,
        )

    def _make_handle(self, **kwargs):
        import mori  # type: ignore[import-not-found]

        mori_config = mori.ops.EpDispatchCombineConfig(**kwargs)
        handle = mori.ops.EpDispatchCombineOp(mori_config)
        return handle

    def get_handle(self, kwargs):
        import mori  # type: ignore[import-not-found]

        mori_kwargs = self._make_all2all_kwargs(**kwargs)
        logger.debug("MoRI all2all args %s", mori_kwargs)
        handle: mori.ops.EpDispatchCombineOp = self.handle_cache.get_or_create(
            mori_kwargs, self._make_handle
        )
        return handle
