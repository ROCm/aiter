# torchrun --nproc_per_node=8 dist-torch-rms-mixed.py
#
def main():
    import time
    import torch
    import torch.distributed as dist
    import os

    # torch
    dist.init_process_group(backend='gloo')
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)

    import rms

    ngpu=8
    rows=4096
    cols=7168

    shape=(rows,cols)
    shape2=(rows)
    type=torch.bfloat16

    device = torch.device(f"cuda:{rank}")
    input_tensor = torch.ones(shape, dtype=type, device=device)
    output_tensor = torch.zeros(shape, dtype=type, device=device)
    weight_tensor = torch.ones(cols, dtype=type, device=device)
    weight_tensor.mul_(5)
    rsigma_tensor = torch.zeros(shape2, dtype=type, device=device)

#print(input_tensors[0].numel())
    epsilon = 17.0
    exec = rms.create_executor("bf16")
#    default_pg = dist.distributed_c10d._get_default_group()
    process_group = dist.group.WORLD
    start_time = time.time()
    for i in range(10000):
        exec.compute_with_distributed_tensors(input_tensor, output_tensor, weight_tensor, rsigma_tensor, epsilon, process_group)
    reusable_time = time.time() - start_time

    print(f"Reusable executor: {reusable_time: .3f}s")

    print(f'output_data = ', output_tensor[4000][0])

    exec.release()

    output_type=torch.float8_e4m3fn
    output_shape=(rows)
    output_tensor = torch.zeros(shape, dtype=output_type, device=device)
    rsigma_tensor = torch.empty(rows, dtype=output_type, device=device)


    exec_mixed = rms.create_executor("bf16", "fp8")
    start_time = time.time()

    for i in range(100):
        exec_mixed.compute_with_distributed_tensors(input_tensor, output_tensor, weight_tensor, rsigma_tensor, epsilon, process_group)
    reusable_time = time.time() - start_time

    print(f"Reusable executor: {reusable_time: .3f}s")


    print(f'output_data = ', output_tensor[4000][0])


#
# rsigma stores the reciprocal of the RMSNorm value. It can be ignored if a tensor with zero item is 
# passed in the function
#

    rsigma_tensor = torch.empty(0, dtype=output_type, device=device)

    exec_no_rsigma_mixed = rms.create_executor("bf16", "fp8")

    start_time = time.time()

    for i in range(100):
        exec_no_rsigma_mixed.compute_with_distributed_tensors(input_tensor, output_tensor, weight_tensor, rsigma_tensor, epsilon, process_group)
    reusable_time = time.time() - start_time

    print(f"Reusable executor: {reusable_time: .3f}s")
    print(f'output_data = ', output_tensor[39][1038])

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

