import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
import aiter


class TRTLLMAllreduceFusion:
    _SUPPORTED_WORLD_SIZES = [2, 4, 8]

    # max_size: max supported allreduce size
    def __init__(
        self,
        group: ProcessGroup = None,
        max_size_in_bytes = 8192 * 16384,
    ) -> None:
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self.group = group
        rank = dist.get_rank(group=self.group)
        torch.cuda.set_device(rank)
        self.rank = rank
        self.fptr = None
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1:
            return

        if world_size not in TRTLLMAllreduceFusion._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "Custom allreduce fusion is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly.",
                world_size,
                str(TRTLLMAllreduceFusion._SUPPORTED_WORLD_SIZES),
            )
            return

        torch.cuda.set_device(rank)
        self.fptr = aiter.init_trtllm_ar_fusion(rank, world_size, max_size_in_bytes)
        handle = aiter.get_trtllm_ar_fusion_handle(self.fptr)
        handle_list = [None] * world_size
        dist.all_gather_object(handle_list, handle, group=group)
        aiter.open_trtllm_ar_fusion_handles(self.fptr, handle_list)
        torch.cuda.synchronize(rank)
        dist.barrier(group=group)
        # print(f"init TRTLLMAllreduceFusion at rank:{rank}", flush=True)

    def get_workspace(self, ref: torch.Tensor):
        return aiter.get_trtllm_ar_fusion_workspace(self.fptr, ref)
    
    def __del__(self):
        if self.fptr:
            aiter.destroy_trtllm_ar_fusion(self.fptr)


envs = {  
    "HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
}
for k,v in envs.items():
    os.environ[k] = v


class RMSNorm(nn.Module):
    def __init__(self, dim, norm_eps=1e-6, dtype=torch.float):
        super().__init__()
        self.eps = norm_eps
        self.weight = nn.Parameter(torch.randn(dim, dtype=dtype), requires_grad=False)

    def forward(self, x):
        input_dtype = x.dtype
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x.to(input_dtype)
        return self.weight * x


def setup(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:23459',
        rank=rank,
        world_size=world_size)


def worker(rank, world_size, allreduce_in, residual_in, rms, ref_residual_out, ref_norm_out, eps, use_fused=True):
    setup(rank, world_size)
    trtllm_instance = TRTLLMAllreduceFusion()
    num_tokens, hidden_dim = residual_in.shape
    local_allreduce_in = allreduce_in[rank].cuda(rank)
    local_residual_in = residual_in.cuda(rank)
    local_rms = rms.cuda(rank)
    prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ])
    with prof:
        if not use_fused:
            dist.all_reduce(local_allreduce_in)    
            local_norm_out = local_rms(local_allreduce_in + local_residual_in)
        else:
            local_residual_out = torch.empty_like(residual_in)
            local_norm_out = torch.empty_like(allreduce_in)
            aiter.trtllm_allreduce_rms(rank, world_size, local_allreduce_in, local_residual_in, 
                local_rms.weight.data, local_residual_out, local_norm_out, eps, trtllm_instance.get_workspace(local_allreduce_in))
    maxdiff = (local_norm_out.cpu() - ref_norm_out).abs().max()
    print(f"rank:{rank}, maxdiff:{maxdiff}")
    # assert torch.allclose(local_norm_out.cpu(), ref_norm_out)
    if rank == 0:
        print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10000))
    dist.destroy_process_group()


def main():
    def testcase(world_size=4, num_tokens=128, hidden_dim=1024, eps=1e-6, dtype=torch.float):
        allreduce_in = torch.randn(world_size, num_tokens, hidden_dim, dtype=dtype)
        residual_in = torch.randn(num_tokens, hidden_dim, dtype=dtype)
        rms = RMSNorm(hidden_dim, dtype=dtype)
        ref_residual_out = allreduce_in.sum(dim=0) + residual_in
        ref_norm_out = rms(ref_residual_out)
        mp.spawn(worker, args=(world_size, allreduce_in, residual_in, rms, ref_residual_out, ref_norm_out, eps), nprocs=world_size, join=True)
    # num_tokens = 129
    # testcase(num_tokens=num_tokens, dtype=torch.float)
    # testcase(num_tokens=num_tokens, dtype=torch.float)
    # testcase(num_tokens=num_tokens, dtype=torch.bfloat16)
    # testcase(num_tokens=num_tokens, dtype=torch.half)
    num_tokens = 128
    testcase(num_tokens=num_tokens, dtype=torch.float)
    testcase(num_tokens=num_tokens, dtype=torch.float)
    testcase(num_tokens=num_tokens, dtype=torch.bfloat16)
    testcase(num_tokens=num_tokens, dtype=torch.half)


if __name__ == '__main__':
    main()
