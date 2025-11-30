import os
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import aiter
from aiter.test_common import checkAllclose


def worker(
    rank, world_size, dtype, allreduce_in_, residual_in_, rms_weight_, eps, show_profile=False
):
    dist_env = aiter.TRTLLMDistEnv(rank, world_size, dtype=dtype, init_process_group=True)
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
        residual_out_maxdiff = (residual_out.cpu().float() - ref_residual_out.cpu().float()).abs().max()
        norm_out_maxdiff = (norm_out.cpu().float() - ref_norm_out.cpu().float()).abs().max()
        scale_out_maxdiff = (scale_out.cpu().float() - ref_scale_out.cpu().float()).abs().max()
        # print(f"ref_norm_out:{ref_norm_out.float().cpu()}, norm_out:{norm_out.float().cpu()}")
        # print(f"rank:{rank}, residual_out_maxdiff:{residual_out_maxdiff}, norm_out_maxdiff:{norm_out_maxdiff}, scale_out_maxdiff:{scale_out_maxdiff}")
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


def testcase(
    world_size=8,
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
            torch.randn(world_size, num_tokens, hidden_dim, dtype=dtype, device='cuda').uniform_(-1, 1).cpu()
        )
        residual_in_.append(
            torch.randn(num_tokens, hidden_dim, dtype=dtype, device='cuda').uniform_(-1, 1).cpu()
        )
        rms_weight_.append(torch.randn(hidden_dim, dtype=dtype, device='cuda').uniform_(-2/hidden_dim, 2/hidden_dim).cpu())
    mp.spawn(
        worker,
        args=(
            world_size,
            dtype,
            allreduce_in_,
            residual_in_,
            rms_weight_,
            eps,
        ),
        nprocs=world_size,
        join=True,
    )


def main(world_size=4):
    num_tokens = 1
    testcase(
        world_size=world_size,
        num_tokens=num_tokens,
        hidden_dim=4096,
        dtype=torch.bfloat16,
    )

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
