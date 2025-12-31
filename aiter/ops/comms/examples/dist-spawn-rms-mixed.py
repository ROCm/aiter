# python3 dist-spawn-rms-mixed.py
# 

import torch.multiprocessing as mp

def worker(rank, world_size):
    import time
    import torch
    import torch.distributed as dist
    import os
    import rms

    # torch
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
#    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)

    rows=4096
    cols=7168

    shape=(rows,cols)
    type=torch.bfloat16

    device = torch.device(f"cuda:{rank}")
    pg = dist.group.WORLD
#    pg = dist.distributed_c10d._get_default_group()
    input_tensor = torch.ones(shape, dtype=type, device=device)
    output_tensor = torch.zeros(shape, dtype=type, device=device)
    weight_tensor = torch.ones((cols), dtype=type, device=device)
    weight_tensor.mul_(5)
    rsigma_tensor = torch.zeros((rows), dtype=type, device=device)

#print(input_tensors[0].numel())
    epsilon = 17.0
    exec = rms.create_executor("bf16")
    
#    exec.compute_with_distributed_tensors(rsigma_tensor, epsilon, pg)
    exec.compute_with_distributed_tensors(input_tensor, output_tensor, weight_tensor, rsigma_tensor, epsilon, pg)

#    print(f'input_data = ', input_tensor[4000][0])
#    print(f'output_data = ', output_tensor[4000][0])
#    print(f'rsigma = ', rsigma_tensor[4095])

    print(f'input_data = ', input_tensor[40][0])
    print(f'output_data = ', output_tensor[40][0])
    print(f'rsigma = ', rsigma_tensor[39])


    output_type=torch.float8_e4m3fn
    output_shape=(rows)
    output_tensor = torch.zeros(shape, dtype=output_type, device=device)
    rsigma_tensor = torch.empty(rows, dtype=output_type, device=device)


    exec_mixed = rms.create_executor("bf16", "fp8")
    start_time = time.time()

    for i in range(100):
        exec_mixed.compute_with_distributed_tensors(input_tensor, output_tensor, weight_tensor, rsigma_tensor, epsilon, pg)
    reusable_time = time.time() - start_time

    print(f"Reusable executor: {reusable_time: .3f}s")

    print(f'output_data = ', output_tensor[30][1024])

    exec_mixed.release()
#
# rsigma stores the reciprocal of the RMSNorm value. It can be ignored if a tensor with zero item is 
# passed in the function
#

    rsigma_tensor = torch.empty(0, dtype=output_type, device=device)

    exec_no_rsigma_mixed = rms.create_executor("bf16", "fp8")

    start_time = time.time()

    for i in range(100):
        exec_no_rsigma_mixed.compute_with_distributed_tensors(input_tensor, output_tensor, weight_tensor, rsigma_tensor, epsilon, pg)
    reusable_time = time.time() - start_time

    print(f"Reusable executor: {reusable_time: .3f}s")
    print(f'output_data = ', output_tensor[39][1038])

    dist.destroy_process_group()

if __name__ == "__main__":
#    mp.set_start_method("spawn", force=True)
    world_size = 8
    mp.spawn(worker, args=(world_size,), nprocs=8, join=True)

