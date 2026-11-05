import importlib.util
import os
from functools import cache
from typing import Any

import torch

from aiter import logger

from .base_device_communicator import All2AllManagerBase, Cache


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


def _is_gfx125x() -> bool:
    """Runtime arch check for the EP backend.

    Not custom_all_reduce's _detect_gfx1250: that one honours
    AITER_CUSTOM_AR_DISABLE_GFX1250, and forcing the old custom-AR path should
    not change which backend mori's EP handle picks.
    """
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return get_gfx_runtime().startswith("gfx125")


class MoriAll2AllManager(All2AllManagerBase):
    @staticmethod
    def _init_mori_shmem(cpu_group) -> None:
        """Register *cpu_group* with mori's shmem heap and run the barrier.

        Skipped on gfx125x: that arch has no shmem support, so mori's EP
        backend there is CCO. Running this init would be useless and can hang.
        """
        if _is_gfx125x():
            return
        import mori

        torch._C._distributed_c10d._register_process_group("mori", cpu_group)
        mori.shmem.shmem_torch_process_group_init("mori")

    def __init__(self, cpu_group):
        assert has_mori(), (
            "MoRI kernels not found. Please follow https://github.com/ROCm/mori/blob/main/README.md"
            " to install MoRI kernels."
        )

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
        quant_type: str = "none",
        low_latency: bool = False,
    ):
        import mori  # type: ignore[import-not-found]

        # `low_latency` is the caller's ask; `self.internode` is measured from
        # the process group, never inferred from world size -- 2 nodes x 4 GPUs
        # would read as intra-node and pick kernels that assume P2P.
        num_qp_per_pe = None
        if low_latency:
            if self.internode:
                kernel_type = mori.ops.EpDispatchCombineKernelType.InterNodeV1LL
            else:
                # AsyncLL has no RDMA path.
                kernel_type = mori.ops.EpDispatchCombineKernelType.AsyncLL
            # Wide-EP reference values; overridable, neither has been swept.
            warp_num_per_block = int(os.environ.get("MORI_EP_WARP_PER_BLOCK", "8"))
            block_num = int(os.environ.get("MORI_EP_BLOCK_NUM", "96"))
            rdma_block_num = int(os.environ.get("MORI_EP_RDMA_BLOCK_NUM", "64"))
            num_qp_per_pe = 2
        elif self.internode:
            # multi node
            kernel_type = mori.ops.EpDispatchCombineKernelType.InterNodeV1
            warp_num_per_block = 16
            block_num = 32
            rdma_block_num = 16
        else:
            # single node
            kernel_type = mori.ops.EpDispatchCombineKernelType.IntraNode
            warp_num_per_block = 16
            block_num = 80
            rdma_block_num = 0

        kwargs = {
            "rank": rank,
            "world_size": num_ep_ranks,
            "data_type": quant_dtype,
            "hidden_dim": token_hidden_size,
            "scale_dim": scale_dim,
            "scale_type_size": scale_type_size,
            "max_token_type_size": input_dtype.itemsize,
            "max_num_inp_token_per_rank": max_num_tokens_per_dp_rank,
            "num_experts_per_rank": num_local_experts,
            "num_experts_per_token": num_experts_per_token,
            "warp_num_per_block": warp_num_per_block,
            "block_num": block_num,
            "kernel_type": kernel_type,
            "rdma_block_num": rdma_block_num,
            "gpu_per_node": gpu_per_node,
            # Combine-side codec. MoRI defaults this to "none" (bf16 on the wire);
            # "fp8_blockwise" picks the EpCombineIntraNodeKernel_*_fp8bwq_* kernels.
            "quant_type": quant_type,
        }
        if num_qp_per_pe is not None:
            # Omitted otherwise, so the other kernels keep the handle_cache key
            # and MoRI config they have always had.
            kwargs["num_qp_per_pe"] = num_qp_per_pe
        return kwargs

    def _make_handle(self, **kwargs):
        import mori  # type: ignore[import-not-found]

        mori_config = mori.ops.EpDispatchCombineConfig(**kwargs)
        handle = mori.ops.EpDispatchCombineOp(mori_config)
        return handle

    def get_handle(self, kwargs, index: int = 0):
        """Cached op for one config and ``index``. Always a single handle.

        ``index=0`` (default) is the shared singleton. Callers with several
        operations in flight use a distinct index for each concurrent slot.
        All indexes use ``handle_cache``.
        """
        mori_kwargs = self._make_all2all_kwargs(**kwargs)
        logger.debug("MoRI all2all index=%d args %s", index, mori_kwargs)
        return self.handle_cache.get_or_create(
            mori_kwargs, self._make_handle, index=index
        )


class MoriV2All2AllManager(All2AllManagerBase):
    """MORI v2 EP backend backed by a manager-owned CCO communicator.

    The communicator is created lazily because its symmetric VMM reservation is
    model/transport dependent.  MORI's regular v2 dispatch/combine op is cached
    through :meth:`get_handle`; higher-level users such as ``MegaMoEGfx1250``
    may borrow the same communicator through
    :meth:`get_or_create_cco_communicator`, but remain responsible for
    constructing and caching their own compute object.

    CCO lifecycle calls are collective and must be issued in the same order by
    one control thread per rank. Concurrent creation and teardown are not
    supported.
    """

    def __init__(self, cpu_group):
        try:
            cco_module = importlib.import_module("mori.cco")
            dispatch_module = importlib.import_module("mori.ops.dispatch_combine_v2")
            communicator_cls = cco_module.Communicator
            config_cls = dispatch_module.EpDispatchCombineConfig
            op_cls = dispatch_module.EpDispatchCombineOp
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "MoRI v2 requires 'mori.cco' and 'mori.ops.dispatch_combine_v2'."
            ) from exc

        super().__init__(cpu_group)
        self._cco_communicator_cls = communicator_cls
        self._dispatch_config_cls = config_cls
        self._dispatch_op_cls = op_cls
        self.handle_cache = Cache()
        # CCO window registration/deregistration is collective, so every rank
        # must keep the same handles alive and close them in the same order.
        self._live_handles: list[Any] = []
        self._cco_communicator: Any | None = None
        self._cco_per_rank_vmm = 0
        self._destroyed = False

    def _check_not_destroyed(self):
        if self._destroyed:
            raise RuntimeError("MORI v2 all2all manager has been destroyed")

    def _gather_group_values(self, value):
        values = [None] * self.world_size
        torch.distributed.all_gather_object(
            values,
            value,
            group=self.cpu_group,
        )
        return values

    def _create_cco_communicator(self, per_rank_vmm: int):
        uid = None
        uid_exception = None
        uid_error = None
        if self.rank == 0:
            try:
                uid = self._cco_communicator_cls.get_unique_id()
            except Exception as exc:  # noqa: BLE001
                uid_exception = exc
                uid_error = f"{type(exc).__name__}: {exc}"

        # Broadcast the status with the UID so a root-side failure does not leave
        # the remaining ranks waiting for a rendezvous token that never arrives.
        objects = [(uid, uid_error)]
        root_global_rank = torch.distributed.get_global_rank(self.cpu_group, 0)
        torch.distributed.broadcast_object_list(
            objects,
            src=root_global_rank,
            group=self.cpu_group,
        )
        uid, uid_error = objects[0]
        if uid_error is not None:
            error = RuntimeError(
                f"MORI v2 CCO unique ID creation failed on rank 0: {uid_error}"
            )
            if uid_exception is not None:
                raise error from uid_exception
            raise error

        comm = self._cco_communicator_cls.init(
            self.world_size,
            self.rank,
            uid,
            per_rank_vmm=per_rank_vmm,
        )
        try:
            comm.barrier()
        except Exception:
            # Do not publish a communicator that failed its collective setup.
            try:
                comm.destroy()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to destroy a MORI v2 CCO communicator after "
                    "collective initialization failed"
                )
            raise
        return comm

    def get_or_create_cco_communicator(self, per_rank_vmm: int):
        """Return the manager-owned CCO communicator with at least this capacity.

        CCO reserves its symmetric virtual address range at communicator
        creation, so an existing communicator cannot be grown in place.  Callers
        must request their maximum requirement before constructing any object
        that retains the communicator.
        """
        if not isinstance(per_rank_vmm, int) or isinstance(per_rank_vmm, bool):
            raise TypeError(
                f"per_rank_vmm must be an integer, got {type(per_rank_vmm).__name__}"
            )
        if per_rank_vmm <= 0:
            raise ValueError(f"per_rank_vmm must be > 0, got {per_rank_vmm}")

        self._check_not_destroyed()
        comm = self._cco_communicator
        if comm is not None:
            if per_rank_vmm > self._cco_per_rank_vmm:
                raise RuntimeError(
                    "MORI v2 CCO communicator is already initialized with "
                    f"per_rank_vmm={self._cco_per_rank_vmm}, which cannot "
                    f"satisfy the requested {per_rank_vmm}; request the "
                    "maximum capacity before creating any v2 handles"
                )
            return comm

        per_rank_capacities = self._gather_group_values(per_rank_vmm)
        if any(capacity != per_rank_vmm for capacity in per_rank_capacities):
            raise RuntimeError(
                "MORI v2 per_rank_vmm must match across the EP group; "
                f"got {per_rank_capacities}"
            )

        # Publish only after communicator creation and its barrier succeed.
        comm = self._create_cco_communicator(per_rank_vmm)
        self._cco_communicator = comm
        self._cco_per_rank_vmm = per_rank_vmm
        if self.rank == 0:
            logger.info(
                "MORI v2 CCO communicator ready: world_size=%d "
                "per_rank_vmm=%.2f GiB",
                self.world_size,
                per_rank_vmm / (1 << 30),
            )
        return comm

    def _make_handle(self, comm, **kwargs):
        config = self._dispatch_config_cls(**kwargs)
        return self._dispatch_op_cls(config, comm)

    def get_handle(self, kwargs, index: int = 0):
        """Return a cached MORI v2 dispatch/combine op.

        ``per_rank_vmm`` is a manager resource requirement and is deliberately
        removed before constructing ``EpDispatchCombineConfig``.  Rank and world
        size come from the process group; callers may provide matching values for
        compatibility, but a mismatch is rejected before creating collective
        resources.
        """
        op_kwargs = dict(kwargs)
        try:
            per_rank_vmm = op_kwargs.pop("per_rank_vmm")
        except KeyError as exc:
            raise ValueError(
                "MORI v2 all2all arguments must include per_rank_vmm"
            ) from exc

        for name, expected in (("rank", self.rank), ("world_size", self.world_size)):
            supplied = op_kwargs.pop(name, expected)
            if supplied != expected:
                raise ValueError(
                    f"{name}={supplied} does not match the EP group value {expected}"
                )
            op_kwargs[name] = expected

        comm = self.get_or_create_cco_communicator(per_rank_vmm)

        def create_handle(**config_kwargs):
            handle = self._make_handle(comm, **config_kwargs)
            self._live_handles.append(handle)
            return handle

        return self.handle_cache.get_or_create(
            op_kwargs,
            create_handle,
            index=index,
        )

    def destroy(self):
        """Close live v2 ops before releasing their shared CCO communicator."""
        self._destroyed = True
        for handle in reversed(self._live_handles):
            close = getattr(handle, "close", None)
            if close is None:
                continue
            try:
                close()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to close a MORI v2 all2all handle")
        self._live_handles.clear()
        self.handle_cache = Cache()

        if self._cco_communicator is not None:
            comm = self._cco_communicator
            try:
                comm.destroy()
            finally:
                self._cco_communicator = None
                self._cco_per_rank_vmm = 0


class FlyDSLAll2AllManager(All2AllManagerBase):
    """
    EP all2all backend backed by FlyDSL intranode dispatch/combine kernels.

    Self-contained: the symmetric P2P buffers are HIP IPC allocations exchanged
    over this group's CPU group (``aiter.ops.flydsl.ipc_symm_heap``); mori is
    not imported. Intra-node only.

    ``get_handle`` takes the same arguments as :class:`MoriAll2AllManager`
    (the mori-only ones -- ``gpu_per_node``, ``low_latency`` -- are accepted
    and ignored) so a caller can switch backends without changing its call.
    TBO multi-instance ops use ``get_handle(..., index=)`` so ubatches get
    distinct cached ops. ``create_handle`` remains for a one-off uncached op.
    """

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

        super().__init__(cpu_group)
        self._flydsl_dispatch_config_cls = FlyDSLDispatchCombineConfig
        self._flydsl_dispatch_op_cls = FlyDSLDispatchCombineIntraNodeOp
        if self.internode:
            raise NotImplementedError(
                "FlyDSLAll2AllManager currently supports only intranode EP "
                "dispatch/combine. For inter-node EP, please use the mori "
                "backend."
            )
        self.handle_cache = Cache()
        self._live_handles: list[Any] = []

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
        gpu_per_node: int | None = None,
        low_latency: bool = False,
        quant_type: str = "none",
    ):
        del gpu_per_node, low_latency  # mori-only knobs
        if quant_type not in ("none", "fp8_direct_cast"):
            raise ValueError(
                f"FlyDSL combine quant_type={quant_type!r} unsupported "
                "(none | fp8_direct_cast)"
            )
        return {
            "rank": rank,
            "world_size": num_ep_ranks,
            # Dispatch moves rows in the wire dtype (e.g. MXFP8 + e8m0
            # scales); combine returns the experts' output dtype.
            "dispatch_dtype": quant_dtype,
            "combine_dtype": input_dtype,
            "hidden_dim": token_hidden_size,
            "scale_dim": scale_dim,
            "scale_type_size": scale_type_size,
            "max_num_inp_token_per_rank": max_num_tokens_per_dp_rank,
            "num_experts_per_rank": num_local_experts,
            "num_experts_per_token": num_experts_per_token,
            "quant_type": quant_type,
        }

    def _make_handle(self, **kwargs):
        cfg = self._flydsl_dispatch_config_cls(**kwargs)
        handle = self._flydsl_dispatch_op_cls(cfg, group=self.cpu_group)
        self._live_handles.append(handle)
        return handle

    def get_handle(self, kwargs, index: int = 0):
        flydsl_kwargs = self._make_all2all_kwargs(**kwargs)
        logger.debug("FlyDSL all2all index=%d args %s", index, flydsl_kwargs)
        return self.handle_cache.get_or_create(
            flydsl_kwargs, self._make_handle, index=index
        )

    def create_handle(self, kwargs):
        """Create a fresh, uncached FlyDSL op instance.

        Prefer ``get_handle(kwargs, index=)`` when the op should be reused.
        Use this only if you need a throwaway instance that is not cached.
        """
        flydsl_kwargs = self._make_all2all_kwargs(**kwargs)
        logger.debug("FlyDSL all2all (uncached) args %s", flydsl_kwargs)
        return self._make_handle(**flydsl_kwargs)

    def destroy(self):
        for handle in reversed(self._live_handles):
            heap = getattr(handle, "_heap", None)
            if heap is not None:
                try:
                    heap.close()
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to free a FlyDSL all2all IPC heap")
        self._live_handles.clear()
        self.handle_cache = Cache()
