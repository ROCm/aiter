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
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def has_mori() -> bool:
    """Whether the optional `mori` package is available."""
    return _has_module("mori")


class MoriAll2AllManager(All2AllManagerBase):
    @staticmethod
    def _init_mori_shmem(cpu_group) -> None:
        """Register *cpu_group* with mori's shmem heap and run the barrier."""
        import mori

        torch._C._distributed_c10d._register_process_group("mori", cpu_group)
        mori.shmem.shmem_torch_process_group_init("mori")

    def __init__(self, cpu_group):
        assert has_mori(), (
            "MoRI kernels not found. Please follow https://github.com/ROCm/mori/blob/main/README.md"
            " to install MoRI kernels."
        )  # noqa

        super().__init__(cpu_group)
        self.handle_cache = Cache()
        self._init_mori_shmem(cpu_group)

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


class FlyDSLAll2AllManager(All2AllManagerBase):
    """
    EP all2all backend backed by FlyDSL intranode dispatch/combine kernels.

    FlyDSL still uses mori's shmem heap for P2P buffer allocation, so mori
    must be installed alongside flydsl. The dispatch/combine *kernels* however
    are entirely FlyDSL-generated, replacing mori's comm primitives.

    TBO multi-instance ops are created via ``create_handle`` (non-cached) so
    the two ubatch ops are guaranteed to be distinct, independent objects.
    """

    @staticmethod
    def _init_mori_shmem(cpu_group) -> None:
        """Register *cpu_group* with mori's shmem heap and run the barrier."""
        import mori

        torch._C._distributed_c10d._register_process_group("mori", cpu_group)
        mori.shmem.shmem_torch_process_group_init("mori")

    @staticmethod
    def _make_quant_type(input_dtype: torch.dtype, quant_dtype: torch.dtype) -> str:
        fp8_dtypes = {
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        }
        if input_dtype == torch.bfloat16 and quant_dtype in fp8_dtypes:
            return "fp8_direct_cast"
        return "none"

    def __init__(self, cpu_group):
        try:
            from aiter.ops.flydsl.kernels.flydsl_dispatch_combine_intranode_op import (
                FlyDSLDispatchCombineConfig,
                FlyDSLDispatchCombineIntraNodeOp,
            )
        except ImportError as e:
            raise ImportError(
                "FlyDSL dispatch/combine module not found at "
                "'aiter.ops.flydsl.kernels.flydsl_dispatch_combine_intranode_op'."
            ) from e

        assert has_mori(), (
            "mori is required alongside FlyDSL for shmem buffer allocation. "
            "Please install mori."
        )
        super().__init__(cpu_group)
        self._flydsl_dispatch_config_cls = FlyDSLDispatchCombineConfig
        self._flydsl_dispatch_op_cls = FlyDSLDispatchCombineIntraNodeOp
        if self.internode:
            raise NotImplementedError(
                "FlyDSLAll2AllManager currently supports only intranode EP "
                "dispatch/combine. For inter-node EP, please use the mori "
                "backend."
            )

        # FlyDSL uses mori.shmem for P2P buffer allocation internally.
        # Keep shmem init behavior aligned with MoriAll2AllManager.
        self._init_mori_shmem(cpu_group)
        self.handle_cache = Cache()

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
    ):

        return dict(
            rank=rank,
            world_size=num_ep_ranks,
            # Buffer sized for input dtype (bf16) so one op handles both
            # bf16 and quantized input without reallocation; the dispatch
            # kernel specialisation is selected at call time by input.dtype.
            data_type=input_dtype,
            hidden_dim=token_hidden_size,
            scale_dim=scale_dim,
            scale_type_size=scale_type_size,
            max_token_type_size=input_dtype.itemsize,
            max_num_inp_token_per_rank=max_num_tokens_per_dp_rank,
            num_experts_per_rank=num_local_experts,
            num_experts_per_token=num_experts_per_token,
            quant_type=self._make_quant_type(input_dtype, quant_dtype),
        )

    def _make_handle(self, **kwargs):
        cfg = self._flydsl_dispatch_config_cls(**kwargs)
        return self._flydsl_dispatch_op_cls(cfg)

    def get_handle(self, kwargs):
        flydsl_kwargs = self._make_all2all_kwargs(**kwargs)
        logger.debug("FlyDSL all2all args %s", flydsl_kwargs)
        return self.handle_cache.get_or_create(flydsl_kwargs, self._make_handle)

    def create_handle(self, kwargs):
        """Create a fresh, uncached FlyDSL op instance.

        Unlike ``get_handle`` (which caches one op per config), every call
        returns a new independent op. Callers that need multiple distinct ops
        for the same config (e.g. ATOM for TBO ubatches) should call this
        and manage the instances themselves.
        """
        flydsl_kwargs = self._make_all2all_kwargs(**kwargs)
        logger.debug("FlyDSL all2all (uncached) args %s", flydsl_kwargs)
        return self._make_handle(**flydsl_kwargs)
