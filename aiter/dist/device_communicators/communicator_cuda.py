# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import os
import torch
from torch.distributed import ProcessGroup

from aiter import logger, get_hip_quant
from aiter.dist.parallel_state import is_global_first_rank
from aiter.ops.enum import QuantType
from aiter.utility.dtypes import fp8
from .base_device_communicator import DeviceCommunicatorBase

should_nccl_symm_mem_allreduce = False


class CudaCommunicator(DeviceCommunicatorBase):
    # AITER_AR_1STAGE=1 forces 1stage, =0 forces non-1stage, unset uses auto
    _ar_1stage_override = {"1": True, "0": False}.get(
        os.environ.get("AITER_AR_1STAGE", "")
    )
    _ar_1stage_max_kb = int(os.environ.get("AITER_AR_1STAGE_MAX_KB", -1))
    _ar_quant_max_bytes = int(os.environ.get("AITER_AR_QUANT_MAX_BYTES", -1))
    _ar_quant_no_prefill_max_bytes = int(
        os.environ.get("AITER_AR_QUANT_NO_PREFILL_MAX_BYTES", -1)
    )

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
    ):
        self._all2all_manager = None
        self._all2all_manager_created = False

        super().__init__(cpu_group, device, device_group, unique_name)
        from aiter.dist.parallel_state import _ENABLE_CUSTOM_ALL_REDUCE

        self.use_custom_allreduce = _ENABLE_CUSTOM_ALL_REDUCE
        self.use_torch_symm_mem = False

        # lazy import to avoid documentation build error
        from aiter.dist.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        # from aiter.dist.device_communicators.symm_mem import SymmMemCommunicator

        self.pynccl_comm = None
        if self.world_size > 1:
            from aiter.dist.device_communicators.communicator_pynccl import (
                PyNcclCommunicator,
            )

            try:
                self.pynccl_comm = PyNcclCommunicator(
                    group=self.cpu_group,
                    device=self.device,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize PyNcclCommunicator for group "
                    f"{self.unique_name}. Exception: {e}"
                )
            # if is_symmetric_memory_enabled():
            #     register_nccl_symmetric_ops(self.pynccl_comm)

        self.ca_comm: CustomAllreduce | None = None
        self.qr_comm = None
        self.symm_mem_comm = None
        # if use_torch_symm_mem and current_platform.is_cuda():
        #     self.symm_mem_comm = SymmMemCommunicator(
        #         group=self.cpu_group,
        #         device=self.device,
        #     )

        if self.use_custom_allreduce and self.world_size > 1:
            # Initialize a custom fast all-reduce implementation.
            self.ca_comm = CustomAllreduce(
                group=self.cpu_group,
                device=self.device,
                # symm_mem_enabled=(
                #     self.symm_mem_comm is not None and not self.symm_mem_comm.disabled
                # ),
            )

        if self.world_size > 1:
            from aiter.dist.device_communicators.quick_all_reduce import (
                QuickAllReduce,
            )

            #     # Initialize a custom quick all-reduce implementation for AMD.
            #     # Quick reduce is designed as a complement to custom allreduce.
            #     # Based on quickreduce (https://github.com/mk1-project/quickreduce).
            #     # If it's a rocm, 'use_custom_allreduce==True' means it must
            #     # currently be an MI300 series.
            self.qr_comm = QuickAllReduce(group=self.cpu_group, device=self.device)

    @property
    def all2all_manager(self):
        # Lazily create all2all_manager to avoid tp/dp/ep group haven't been created yet
        if not self._all2all_manager_created and self.use_all2all:
            self._all2all_manager_created = True

            if self.all2all_backend == "naive":
                from .all2all import NaiveAll2AllManager

                self._all2all_manager = NaiveAll2AllManager(self.cpu_group)
            elif self.all2all_backend == "allgather_reducescatter":
                from .all2all import AgRsAll2AllManager

                self._all2all_manager = AgRsAll2AllManager(self.cpu_group)
            elif self.all2all_backend == "pplx":
                from .all2all import PPLXAll2AllManager

                self._all2all_manager = PPLXAll2AllManager(self.cpu_group)
            elif self.all2all_backend == "deepep_high_throughput":
                from .all2all import DeepEPHTAll2AllManager

                self._all2all_manager = DeepEPHTAll2AllManager(self.cpu_group)
            elif self.all2all_backend == "deepep_low_latency":
                from .all2all import DeepEPLLAll2AllManager

                self._all2all_manager = DeepEPLLAll2AllManager(self.cpu_group)
            elif self.all2all_backend == "mori":
                from .all2all import MoriAll2AllManager

                self._all2all_manager = MoriAll2AllManager(self.cpu_group)
            elif self.all2all_backend == "flashinfer_all2allv":
                from .all2all import FlashInferAllToAllManager

                self._all2all_manager = FlashInferAllToAllManager(self.cpu_group)
            else:
                raise ValueError(f"Unknown all2all backend: {self.all2all_backend}")

            if is_global_first_rank():
                logger.info(
                    "Using %s all2all manager.",
                    self._all2all_manager.__class__.__name__,
                )
        # if self._all2all_manager is None:
        #     raise ValueError(f"all2all_manager is None for {self.unique_name}")
        return self._all2all_manager

    @all2all_manager.setter
    def all2all_manager(self, value):
        self._all2all_manager = value
        if value is not None:
            self._all2all_manager_created = True

    def all_reduce(
        self,
        input_,
        use_new: bool = True,
        ca_fp8_quant: bool = False,
        prefill_support: bool = False,
    ) -> torch.Tensor:
        # always try quick reduce first, then custom allreduce,
        # and then pynccl. (quick reduce just for ROCM MI3*)
        qr_comm = self.qr_comm
        if (
            qr_comm is not None
            and not qr_comm.disabled
            and qr_comm.should_quick_allreduce(input_)
            # input shape estimated at 2 * max concurrency for now. if performance issues, subject to change
        ):
            out = qr_comm.quick_all_reduce(input_)
            assert out is not None
            return out

        ca_comm = self.ca_comm
        if (
            ca_comm is not None
            and not ca_comm.disabled
            and ca_comm.should_custom_ar(input_, prefill_support)
        ):
            out = ca_comm.custom_all_reduce(input_, use_new, ca_fp8_quant)
            assert out is not None
            return out
        symm_mem_comm = self.symm_mem_comm
        if symm_mem_comm is not None and symm_mem_comm.should_use_symm_mem(input_):
            out = symm_mem_comm.all_reduce(input_)
            assert out is not None
            return out
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            out = pynccl_comm.all_reduce(input_)
            assert out is not None
            return out
        # fall back to the default all-reduce using PyTorch.
        # this usually happens during testing.
        # when we run the model, allreduce only happens for the TP
        # group, where we always have either custom allreduce or pynccl.
        out = input_.clone()
        torch.distributed.all_reduce(out, group=self.device_group)
        return out

    def fused_allreduce_rmsnorm(
        self,
        input_,
        res_inp_,
        weight_,
        eps,
        prefill_support: bool = False,
        x_pad_to_multiple: int = 0,
        gemma_norm: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from aiter.dist.device_communicators.custom_all_reduce import (
            can_pack_2d_last_dim_slice,
            is_weak_contiguous,
        )

        input_is_weak_contiguous = is_weak_contiguous(input_)
        residual_is_weak_contiguous = is_weak_contiguous(res_inp_)
        use_general_path = (
            not input_is_weak_contiguous or not residual_is_weak_contiguous
        )
        input_n = input_.shape[-1]
        residual_n = res_inp_.shape[-1]
        n = weight_.numel()
        if input_n < n:
            raise RuntimeError(
                "fused_allreduce_rmsnorm requires input width >= weight width, "
                f"got input_width={input_n}, weight_width={n}"
            )
        if residual_n != n:
            raise RuntimeError(
                "fused_allreduce_rmsnorm requires residual width == weight width, "
                f"got residual_width={residual_n}, weight_width={n}"
            )
        out_n = n
        if x_pad_to_multiple > 0:
            out_n = (n + x_pad_to_multiple - 1) // x_pad_to_multiple * x_pad_to_multiple
        total_bytes = input_.numel() * input_.element_size()
        can_use_fuse_ar_rms = (
            n <= 16384 and total_bytes < 8 * 1024 * 8192 and self.world_size != 6
        )
        ca_comm = self.ca_comm
        can_use_custom_ar = (
            ca_comm is not None and not ca_comm.disabled and can_use_fuse_ar_rms
        )
        total_bytes_limit = (
            self._ar_1stage_max_kb
            if self._ar_1stage_max_kb >= 0
            else 128 * 7168 * 2 // self.world_size
        )
        use_1stage = (
            self._ar_1stage_override
            if self._ar_1stage_override is not None
            else (total_bytes <= total_bytes_limit)
        )
        if (
            not use_general_path
            and can_use_custom_ar
            and ca_comm.should_custom_ar(input_, prefill_support)
        ):
            out, res_out = ca_comm.custom_fused_ar_rms(
                input_,
                res_inp_,
                weight_,
                eps,
                use_1stage,
                out_hidden_dim=out_n,
                gemma_norm=gemma_norm,
            )
            assert out is not None
            assert res_out is not None
            return out, res_out

        if (
            can_use_custom_ar
            and not input_is_weak_contiguous
            and residual_is_weak_contiguous
            and can_pack_2d_last_dim_slice(input_)
            and ca_comm.should_custom_ar_bytes(input_, prefill_support)
        ):
            out, res_out = ca_comm.custom_fused_ar_rms_packed_input(
                input_,
                res_inp_,
                weight_,
                eps,
                use_1stage,
                out_hidden_dim=out_n,
                prefill_support=prefill_support,
                gemma_norm=gemma_norm,
            )
            assert out is not None
            assert res_out is not None
            return out, res_out

        input_for_ar = input_ if input_is_weak_contiguous else input_.contiguous()
        ar_out = self.all_reduce(input_for_ar, prefill_support=prefill_support)
        if input_n != n:
            # The padded tail is semantically zero for the current MoE path, so
            # the fallback path only needs the valid hidden region for RMSNorm.
            ar_out = ar_out[..., :n].contiguous()

        if use_general_path or x_pad_to_multiple > 0 or input_n != n:
            # The custom fused AR+RMS kernel still falls back here for strided rows
            # or when custom all-reduce is unavailable for padded outputs.
            # Fall back to all-reduce + Triton RMSNorm so callers can pass strided
            # inputs/residuals and optionally request a padded output width.
            # The Triton kernel is 2-D, so flatten leading dims before launch and
            # restore the original batch shape on return.
            from aiter.ops.triton.normalization.fused_add_rmsnorm_pad import (
                fused_add_rmsnorm_pad,
            )

            ar_out_2d = ar_out.reshape(-1, ar_out.shape[-1])
            res_inp_2d = res_inp_.reshape(-1, res_inp_.shape[-1])
            norm_weight = weight_ + 1.0 if gemma_norm else weight_
            out_2d, residual_out_2d = fused_add_rmsnorm_pad(
                ar_out_2d,
                norm_weight,
                eps,
                res_inp_2d,
                x_pad_to_multiple=x_pad_to_multiple,
            )
            out = out_2d.reshape(input_.shape[:-1] + (out_2d.shape[-1],))
            residual_out = residual_out_2d.reshape(res_inp_.shape)
            return out, residual_out

        # call split kernel
        out = torch.empty_like(ar_out)
        residual_out = torch.empty_like(ar_out)
        from aiter import rmsnorm2d_fwd_with_add

        rmsnorm2d_fwd_with_add(
            out,
            ar_out,
            res_inp_,
            residual_out,
            weight_,
            eps,
            gemma_norm,
            0,
        )
        return out, residual_out

    def fused_allreduce_rmsnorm_quant(
        self,
        input_,
        res_inp_,
        weight_,
        eps,
        prefill_support: bool = False,
        quant_type="per_token",
        group_size=128,
        emit_bf16: bool = False,
        transpose_scale: bool = False,
        gemma_norm: bool = False,
    ):
        # quant_type arrives already canonicalized to a string ("per_token"/
        # "per_group"/"mxfp4") from the public API.
        if gemma_norm and quant_type != "per_token":
            raise NotImplementedError(
                "gemma_norm fused quant currently supports per-token FP8 only"
            )
        if quant_type == "per_group":
            return self.fused_allreduce_rmsnorm_quant_per_group(
                input_,
                res_inp_,
                weight_,
                eps,
                group_size=group_size,
                prefill_support=prefill_support,
                emit_bf16=emit_bf16,
                transpose_scale=transpose_scale,
            )
        if quant_type == "mxfp4":
            return self.fused_allreduce_rmsnorm_mxfp4_quant(
                input_,
                res_inp_,
                weight_,
                eps,
                prefill_support=prefill_support,
                emit_bf16=emit_bf16,
            )
        if emit_bf16:
            raise ValueError("emit_bf16 is not supported for per-token FP8 quant")
        hidden_dim = int(input_.shape[-1])
        element_size = input_.element_size()
        total_bytes = input_.numel() * element_size
        fused_quant_bytes_limit = (
            self._ar_quant_max_bytes if self._ar_quant_max_bytes >= 0 else 4096 * 1024
        )
        no_prefill_quant_bytes_limit = (
            self._ar_quant_no_prefill_max_bytes
            if self._ar_quant_no_prefill_max_bytes >= 0
            else 64 * 1024 * 1024
        )
        if (
            (
                hidden_dim in [512, 1024, 2048, 4096]
                or (
                    hidden_dim in [7168, 6144]
                    and input_.dtype in (torch.float16, torch.bfloat16)
                )
            )
            and total_bytes <= fused_quant_bytes_limit
            and (prefill_support or total_bytes <= no_prefill_quant_bytes_limit)
        ):
            total_bytes_limit = (
                self._ar_1stage_max_kb if self._ar_1stage_max_kb >= 0 else 128 * 1024
            )
            use_1stage = (
                self._ar_1stage_override
                if self._ar_1stage_override is not None
                else (total_bytes <= total_bytes_limit)
            )
            out, res_out, scale_out = self.ca_comm.custom_fused_ar_rms_quant(
                input_,
                res_inp_,
                weight_,
                eps,
                use_1stage,
                gemma_norm=gemma_norm,
            )
        else:
            out_, res_out = self.fused_allreduce_rmsnorm(
                input_,
                res_inp_,
                weight_,
                eps,
                prefill_support,
                gemma_norm=gemma_norm,
            )
            hip_quant = get_hip_quant(QuantType.per_Token)
            out, scale_out = hip_quant(out_, quant_dtype=fp8)
        assert out is not None
        assert res_out is not None
        assert scale_out is not None
        return out, res_out, scale_out

    def fused_allreduce_rmsnorm_quant_per_group(
        self,
        input_,
        res_inp_,
        weight_,
        eps,
        group_size=128,
        prefill_support: bool = False,
        emit_bf16: bool = False,
        transpose_scale: bool = False,
    ):
        """Fused AR+RMSNorm+per-group FP8 quant, optionally also emitting the
        pre-quantization bf16/fp16 normed output.

        When ``emit_bf16=False`` returns ``(fp8, residual_out, scale)``.
        When ``emit_bf16=True`` returns ``(fp8, residual_out, scale, bf16)`` —
        used by GDN-style layers that have both an FP8 projection and a bf16
        gating projection consuming the same normed activation, so they can
        skip the separate per-group quant kernel entirely (see Qwen3.5).
        """
        total_bytes = input_.numel() * input_.element_size()
        K = input_.shape[-1]
        fused_ok = False
        out = res_out = scale_out = bf16_out = None
        if (
            K % group_size == 0
            and K <= 16384
            and total_bytes < 8 * 1024 * 8192
            and self.world_size != 6
            and (prefill_support or total_bytes <= 64 * 1024 * 1024)
        ):
            total_bytes_limit = (
                self._ar_1stage_max_kb if self._ar_1stage_max_kb >= 0 else 128 * 1024
            )
            use_1stage = (
                self._ar_1stage_override
                if self._ar_1stage_override is not None
                else (total_bytes <= total_bytes_limit)
            )
            try:
                result = self.ca_comm.custom_fused_ar_rms_per_group_quant(
                    input_,
                    res_inp_,
                    weight_,
                    eps,
                    group_size,
                    use_1stage,
                    emit_bf16=emit_bf16,
                    transpose_scale=transpose_scale,
                )
                if result is not None:
                    if emit_bf16:
                        out, res_out, scale_out, bf16_out = result
                    else:
                        out, res_out, scale_out = result
                    fused_ok = True
            except Exception:
                pass
        if not fused_ok:
            out_, res_out = self.fused_allreduce_rmsnorm(
                input_, res_inp_, weight_, eps, prefill_support
            )
            hip_quant = get_hip_quant(QuantType.per_1x128)
            # The fused path and the registered op's fake return the per-group
            # scale in column-major (1, M) layout (stride (1, M)) when
            # transpose_scale=True. per_group_quant_hip cannot produce that
            # stride directly (with transpose_scale=True it returns a contiguous
            # (M, num_groups) buffer with SHUFFLED bytes -- a different physical
            # arrangement). So compute the plain row-major scale and, when
            # transpose_scale is requested, copy its values into a genuinely
            # column-major (1, M)-strided tensor so the runtime stride/values
            # match the fake (otherwise torch.compile's assert_size_stride fails
            # or the GEMM reads the wrong layout).
            out, scale_row = hip_quant(out_, quant_dtype=fp8, transpose_scale=False)
            if transpose_scale:
                M, num_groups = scale_row.shape
                scale_out = torch.empty(
                    (num_groups, M), dtype=scale_row.dtype, device=scale_row.device
                ).transpose(0, 1)
                scale_out.copy_(scale_row)
            else:
                scale_out = scale_row
            if emit_bf16:
                bf16_out = out_
        assert out is not None
        assert res_out is not None
        assert scale_out is not None
        if emit_bf16:
            assert bf16_out is not None
            return out, res_out, scale_out, bf16_out
        return out, res_out, scale_out

    def fused_qknorm_allreduce(
        self,
        qkv_in,
        q_w,
        k_w,
        eps,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_out, k_out, v_out = self.ca_comm.custom_fused_qknorm_ar(qkv_in, q_w, k_w, eps)
        assert q_out is not None
        assert k_out is not None
        assert v_out is not None
        return q_out, k_out, v_out

    def fused_qknorm_allreduce_rope(
        self,
        qkv_in,
        q_w,
        k_w,
        cos_sin_cache,
        position_ids,
        head_dim,
        rotary_dim,
        eps,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_out, k_out, v_out = self.ca_comm.custom_fused_qknorm_ar_rope(
            qkv_in,
            q_w,
            k_w,
            cos_sin_cache,
            position_ids,
            head_dim,
            rotary_dim,
            eps,
        )
        assert q_out is not None
        assert k_out is not None
        assert v_out is not None
        return q_out, k_out, v_out

    def fused_allreduce_rmsnorm_mxfp4_quant(
        self,
        input_,
        res_inp_,
        weight_,
        eps,
        prefill_support: bool = False,
        emit_bf16: bool = False,
    ):
        """Fused AR+RMSNorm with an MXFP4 quantization epilogue when supported.

        Selects the 1-stage decode-shape kernel when the shape qualifies,
        otherwise the 2-stage kernel for larger shapes that still fit in the
        512 KiB shared-memory reduce-scatter budget. Falls back to fused
        AR+RMSNorm + ``dynamic_mxfp4_quant`` for any shape neither kernel
        supports.

        ``AITER_AR_1STAGE``:
            * ``"1"``  -> only attempt the 1-stage kernel
            * ``"0"``  -> only attempt the 2-stage kernel
            * unset    -> auto: prefer 1-stage when eligible, else 2-stage
        """
        total_bytes = input_.numel() * input_.element_size()
        K = input_.shape[-1]
        token_num = input_.numel() // K
        element_size = input_.element_size()
        pack_size = 16 // element_size if element_size > 0 else 0
        block_size = K // pack_size if pack_size > 0 else 0

        # 1-stage gate: direct decode shapes only (matches kernel constraints).
        use_direct_mxfp4 = (
            token_num <= 4
            or (K <= 4096 and token_num <= 32)
            or (K <= 6144 and token_num <= 16)
            or (K == 8192 and token_num <= 8)
        )
        override = self._ar_1stage_override
        can_1stage = (
            override is not False
            and K % 32 == 0
            and K <= 16384
            and token_num <= 80
            and use_direct_mxfp4
        )

        # 2-stage gate: larger prefill shapes that still fit the 512 KiB
        # shared-memory reduce-scatter budget and split evenly across ranks.
        can_2stage = (
            override is not True
            and K % 32 == 0
            and pack_size > 0
            and K <= 8192
            and block_size % self.world_size == 0
            and (not emit_bf16 or block_size % 32 == 0)
            and total_bytes <= 512 * 1024
        )
        if override is None:
            prefer_2stage = (
                can_2stage and self.world_size == 8 and token_num >= 16 and K <= 6144
            )
            if prefer_2stage:
                can_1stage = False

        out_fp4 = res_out = scale_out = bf16_out = None
        ca_comm = self.ca_comm
        use_kernel = (
            ca_comm is not None
            and not ca_comm.disabled
            and ca_comm.should_custom_ar(input_, prefill_support)
            and self.world_size != 6
            and (prefill_support or total_bytes <= 64 * 1024 * 1024)
            and (can_1stage or can_2stage)
        )
        if use_kernel:
            result = ca_comm.custom_fused_ar_rms_mxfp4_quant(
                input_,
                res_inp_,
                weight_,
                eps,
                use_1stage=can_1stage,
                emit_bf16=emit_bf16,
            )
            assert result is not None
            if emit_bf16:
                out_fp4, res_out, scale_out, bf16_out = result
            else:
                out_fp4, res_out, scale_out = result
        else:
            normed, res_out = self.fused_allreduce_rmsnorm(
                input_, res_inp_, weight_, eps, prefill_support
            )
            from aiter.ops.triton.quant import dynamic_mxfp4_quant

            out_fp4, scale_out = dynamic_mxfp4_quant(normed)
            if emit_bf16:
                bf16_out = normed
        assert out_fp4 is not None
        assert res_out is not None
        assert scale_out is not None
        if emit_bf16:
            assert bf16_out is not None
            return out_fp4, res_out, scale_out, bf16_out
        return out_fp4, res_out, scale_out

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            dim += input_.dim()
        input_size = input_.size()
        world_size = self.world_size

        is_last_dim = dim == input_.dim() - 1
        ca_comm = self.ca_comm
        if (
            ca_comm is not None
            and not ca_comm.disabled
            and ca_comm.should_custom_ag(input_)
            and (
                dim == 0
                or (is_last_dim and input_size[-1] * input_.element_size() % 16 == 0)
            )
        ):
            out = ca_comm.custom_all_gather(input_, dim)
            assert out is not None
            return out

        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            output_size = (input_size[0] * world_size,) + input_size[1:]
            output_tensor = torch.empty(
                output_size, dtype=input_.dtype, device=input_.device
            )
            pynccl_comm.all_gather(output_tensor, input_)
            output_tensor = output_tensor.reshape((world_size,) + input_size)
            output_tensor = output_tensor.movedim(0, dim)
            output_tensor = output_tensor.reshape(
                input_size[:dim]
                + (world_size * input_size[dim],)
                + input_size[dim + 1 :]
            )
            return output_tensor

        # fall back to the default all-gather using PyTorch
        output_tensor = torch.empty(
            (world_size,) + input_size, dtype=input_.dtype, device=input_.device
        )
        torch.distributed.all_gather_into_tensor(
            output_tensor, input_, group=self.device_group
        )
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim] + (world_size * input_size[dim],) + input_size[dim + 1 :]
        )
        return output_tensor

    def reduce_scatter(
        self, input_: torch.Tensor, output_: torch.Tensor, dim: int = -1
    ):
        world_size = self.world_size
        ca_comm = self.ca_comm
        # Custom kernel supports scatter on first/last/mid dims; gate via
        # should_custom_rs which also rejects first-dim-non-vectorizable
        # shapes (no naive fallback exists for that case, see C++ dispatch).
        if (
            ca_comm is not None
            and not ca_comm.disabled
            and ca_comm.should_custom_rs(input_, dim)
        ):
            ca_comm.custom_reduce_scatter(input_, output_, dim)
        else:
            pynccl_comm = self.pynccl_comm
            assert pynccl_comm is not None
            if dim < 0:
                # Convert negative dim to positive.
                dim += input_.dim()

            # Note: This will produce an incorrect answer if we don't make
            # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
            input_tensor = input_.movedim(0, dim).contiguous()

            assert input_tensor.shape[0] % world_size == 0
            chunk_size = input_tensor.shape[0] // world_size
            output_shape = (chunk_size,) + input_tensor.shape[1:]
            output_.reshape(output_shape)

            pynccl_comm.reduce_scatter(output_, input_tensor)

            # Reshape before returning
            output_.movedim(0, dim).contiguous()

    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ):
        world_size = self.world_size
        pynccl_comm = self.pynccl_comm
        assert pynccl_comm is not None
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        if sizes is not None:
            assert len(sizes) == world_size
            assert input_tensor.shape[0] == sum(sizes)
            chunk_size = sizes[self.rank_in_group]
        else:
            assert input_tensor.shape[0] % world_size == 0
            chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        output = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )

        if sizes is not None:
            pynccl_comm.reduce_scatterv(output, input_tensor, sizes=sizes)
        else:
            pynccl_comm.reduce_scatter(output, input_tensor)

        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        """Sends a tensor to the destination rank in a blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size

        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.send(tensor, dst)
        else:
            torch.distributed.send(tensor, self.ranks[dst], self.device_group)

    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: int | None = None
    ) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.recv(tensor, src)
        else:
            torch.distributed.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def destroy(self):
        if self.pynccl_comm is not None:
            self.pynccl_comm = None
        if self.qr_comm is not None:
            self.qr_comm = None
        if self.ca_comm is not None:
            self.ca_comm = None
        if self._all2all_manager is not None:
            self._all2all_manager.destroy()
            self._all2all_manager = None

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ):
        if dim != 0:
            raise NotImplementedError("only dim 0 all-gatherv is supported")
        world_size = self.world_size
        pynccl_comm = self.pynccl_comm
        assert pynccl_comm is not None and not pynccl_comm.disabled

        # 'sizes' is not needed if all inputs in the same group have the same
        # shape
        if sizes is not None and all(s == sizes[0] for s in sizes):
            sizes = None

        def _all_gather_single(input_: torch.Tensor, sizes: list[int] | None = None):
            input_size = input_.size()
            if sizes is not None:
                assert len(sizes) == world_size
                assert (
                    input_.shape[dim] == sizes[self.rank_in_group]
                ), f"{input_.shape[dim]} != {sizes[self.rank_in_group]}"
                output_size = (sum(sizes),) + input_size[1:]
            else:
                output_size = (input_size[0] * world_size,) + input_size[1:]
            # Allocate output tensor.
            output_tensor = torch.empty(
                output_size, dtype=input_.dtype, device=input_.device
            )
            if sizes is not None:
                pynccl_comm.all_gatherv(output_tensor, input_, sizes=sizes)
            else:
                pynccl_comm.all_gather(output_tensor, input_)
            return output_tensor

        if isinstance(input_, torch.Tensor):
            return _all_gather_single(input_, sizes)

        output_list = []
        pynccl_comm.group_start()
        for inp in input_:
            output_list.append(_all_gather_single(inp, sizes=sizes))
        pynccl_comm.group_end()

        return output_list

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.all2all_manager is not None
        hidden_states, router_logits = self.all2all_manager.dispatch(
            hidden_states, router_logits, is_sequence_parallel
        )
        return hidden_states, router_logits

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        assert self.all2all_manager is not None
        hidden_states = self.all2all_manager.combine(
            hidden_states, is_sequence_parallel
        )
        return hidden_states
