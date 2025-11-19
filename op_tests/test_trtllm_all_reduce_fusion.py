import os
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose


envs = {
    "HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
}
for k, v in envs.items():
    os.environ[k] = v


def rms_norm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float):
    input_dtype = x.dtype
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(input_dtype)
    return weight * x


class DistributedEnv:
    def __init__(self, rank, world_size):
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:22229",
            rank=rank,
            world_size=world_size,
        )
        self.rank = rank
        self.world_size = world_size
        self.group = dist.group.WORLD
        self.trtllm_instance = aiter.TRTLLMAllreduceFusion(group=self.group)
        self.barrier()

    def __del__(self):
        dist.destroy_process_group(self.group)

    def barrier(self):
        torch.cuda.set_device(self.rank)
        dist.barrier(self.group)
        torch.cuda.synchronize()

    def allreduce_add_rms_native(
        self, allreduce_in, residual_in, rms_weight, eps, fp8_out=True
    ):
        dist.all_reduce(allreduce_in)
        residual_out = allreduce_in + residual_in
        norm_out = rms_norm_forward(residual_out, rms_weight, eps)
        if fp8_out:
            norm_out_scale, _ = norm_out.float().abs().max(dim=-1, keepdim=True)
            norm_out_scale = norm_out_scale / 240
            norm_out = (norm_out / norm_out_scale).to(dtypes.fp8)
            return residual_out, norm_out, norm_out_scale
        else:
            return residual_out, norm_out

    def allreduce_add_rms_fused(
        self, allreduce_in, residual_in, rms_weight, eps, fp8_out=True
    ):
        residual_out = torch.empty_like(residual_in)
        norm_out = torch.empty_like(allreduce_in)
        if fp8_out:
            norm_out = norm_out.to(dtypes.fp8)
            scale_out = torch.empty(
                allreduce_in.shape[0],
                1,
                dtype=torch.float32,
                device=allreduce_in.device,
            )
        else:
            scale_out = torch.empty(1, dtype=torch.float32, device=allreduce_in.device)
        aiter.trtllm_allreduce_rms(
            self.rank,
            self.world_size,
            allreduce_in,
            residual_in,
            rms_weight,
            residual_out,
            norm_out,
            scale_out,
            eps,
            fp8_out,
            self.trtllm_instance.get_workspace(allreduce_in),
        )
        if fp8_out:
            return residual_out, norm_out, scale_out
        else:
            return residual_out, norm_out


def worker(
    rank, world_size, allreduce_in_, residual_in_, rms_weight_, eps, show_profile=False
):
    dist_env = DistributedEnv(rank, world_size)
    for i in range(len(allreduce_in_)):
        local_allreduce_in = allreduce_in_[i][rank].cuda(rank)
        local_residual_in = residual_in_[i].cuda(rank)
        local_rms_weight = rms_weight_[i].cuda(rank)
        num_tokens, hidden_dim = local_allreduce_in.shape
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        )
        with prof:
            dist_env.barrier()
            start_native = time.time()
            ref_residual_out, ref_norm_out, ref_scale_out = (
                dist_env.allreduce_add_rms_native(
                    local_allreduce_in.clone(),
                    local_residual_in,
                    local_rms_weight,
                    eps,
                    True,
                )
            )
            dist_env.barrier()
            start_fused = time.time()
            residual_out, norm_out, scale_out = dist_env.allreduce_add_rms_fused(
                local_allreduce_in.clone(),
                local_residual_in,
                local_rms_weight,
                eps,
                True,
            )
            dist_env.barrier()
            end = time.time()
        dur_native = start_fused - start_native
        dur_fused = end - start_fused
        speedup = dur_native / dur_fused
        print(f"dur_native:{dur_native}, dur_fused:{dur_fused}, speedup:{speedup}")
        if rank == 0 and show_profile:
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10000))

        ref_norm_out = ref_norm_out.float() * ref_scale_out
        norm_out = norm_out.float() * scale_out
        checkAllclose(
            residual_out.float(),
            ref_residual_out.float(),
            rtol=1e-2,
            atol=1e-2,
            msg="residual_out",
        )
        checkAllclose(
            norm_out.float(), ref_norm_out.float(), rtol=1e-2, atol=1e-2, msg="norm_out"
        )
        checkAllclose(
            scale_out.float(),
            ref_scale_out.float(),
            rtol=1e-2,
            atol=1e-2,
            msg="scale_out",
        )
        # residual_out_maxdiff = (residual_out.cpu().float() - ref_residual_out.cpu().float()).abs().max()
        # norm_out_maxdiff = (norm_out.cpu().float() - ref_norm_out.cpu().float()).abs().max()
        # print(f"rank:{rank}, residual_out_maxdiff:{residual_out_maxdiff}, norm_out_maxdiff:{norm_out_maxdiff}")


def testcase(
    world_size=4,
    num_tokens=128,
    hidden_dim=1024,
    eps=1e-6,
    dtype=torch.float,
    nsamples=5,
):
    print(
        f"\n============ world_size:{world_size}, num_tokens:{num_tokens}, hidden_dim:{hidden_dim}, eps:{eps}, dtype:{dtype}, nsamples:{nsamples} ============\n"
    )
    allreduce_in_ = []
    residual_in_ = []
    rms_weight_ = []
    for i in range(nsamples):
        allreduce_in_.append(
            torch.randn(world_size, num_tokens, hidden_dim, dtype=dtype).uniform_(-1, 1)
        )
        residual_in_.append(
            torch.randn(num_tokens, hidden_dim, dtype=dtype).uniform_(-1, 1)
        )
        rms_weight_.append(torch.randn(hidden_dim, dtype=dtype).uniform_(-1, 1))
    mp.spawn(
        worker,
        args=(
            world_size,
            allreduce_in_,
            residual_in_,
            rms_weight_,
            eps,
        ),
        nprocs=world_size,
        join=True,
    )


def main(world_size=4):
    # num_tokens = 1
    # testcase(world_size=world_size, num_tokens=num_tokens, hidden_dim=1024, dtype=torch.bfloat16)

    num_tokens = 129
    testcase(
        world_size=world_size, num_tokens=num_tokens, hidden_dim=1024, dtype=torch.float
    )
    testcase(
        world_size=world_size,
        num_tokens=num_tokens,
        hidden_dim=1024,
        dtype=torch.bfloat16,
    )
    testcase(
        world_size=world_size, num_tokens=num_tokens, hidden_dim=1024, dtype=torch.half
    )

    num_tokens = 128
    testcase(
        world_size=world_size, num_tokens=num_tokens, hidden_dim=1024, dtype=torch.float
    )
    testcase(
        world_size=world_size, num_tokens=num_tokens, hidden_dim=1024, dtype=torch.half
    )
    testcase(
        world_size=world_size,
        num_tokens=num_tokens,
        hidden_dim=1024,
        dtype=torch.bfloat16,
    )

    testcase(
        world_size=world_size, num_tokens=32768, hidden_dim=4096, dtype=torch.bfloat16
    )


if __name__ == "__main__":
    main()
