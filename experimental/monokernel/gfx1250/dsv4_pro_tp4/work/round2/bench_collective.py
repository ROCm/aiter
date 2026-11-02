"""TP4 graph benchmark for AITER LL versus existing semaphore-signaled pull AR."""
import os,json,statistics
from pathlib import Path
os.environ.setdefault('ENABLE_CK','0')
import torch,torch.distributed as dist
from aiter.dist.parallel_state import init_distributed_environment,ensure_model_parallel_initialized,set_custom_all_reduce,get_tp_group,graph_capture,destroy_model_parallel,destroy_distributed_environment
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
set_custom_all_reduce(True)
init_distributed_environment(world_size=4,rank=rank,distributed_init_method='env://')
ensure_model_parallel_initialized(4,1)
group=get_tp_group();dist.all_reduce(torch.zeros(1,device='cuda'),group=group.device_group)
graphs=[];rows=[]
for m in (1,8,16,32,64,128):
 gen=torch.Generator(device='cuda').manual_seed(m)
 inputs=[torch.randn(m,7168,device='cuda',dtype=torch.bfloat16,generator=gen) for _ in range(4)]
 x=inputs[rank].clone();ref=inputs[rank].float()
 for i in range(1,4):ref=ref+inputs[(rank+i)%4].float()
 ref=ref.bfloat16()
 for _ in range(3):out=tensor_model_parallel_all_reduce(x)
 torch.cuda.synchronize();torch.testing.assert_close(out,ref,rtol=0,atol=0)
 g=torch.cuda.CUDAGraph()
 with graph_capture() as gc,torch.cuda.graph(g,stream=gc.stream):
  for _ in range(32):out=tensor_model_parallel_all_reduce(x)
 graphs.append((g,x,out))
 for _ in range(20):g.replay()
 torch.cuda.synchronize();torch.testing.assert_close(out,ref,rtol=0,atol=0)
 times=[]
 for sample in range(7):
  group.barrier();torch.cuda.synchronize()
  start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
  start.record()
  for _ in range(200):g.replay()
  end.record();end.synchronize();times.append(start.elapsed_time(end)*1000/(200*32))
 # Change the payload and exercise the same captured pointers/epochs.
 x.mul_(2);ref=ref*2
 for _ in range(100):g.replay()
 torch.cuda.synchronize();torch.testing.assert_close(out,ref,rtol=0,atol=0)
 rows.append(dict(m=m,rank=rank,median_us=statistics.median(times),samples_us=times))
 if rank==0:print(rows[-1],flush=True)
mode='pull' if os.environ.get('AITER_CUSTOM_AR_DISABLE_LL')=='1' else 'll'
Path(f'/work/round2/collective_{mode}_rank{rank}.json').write_text(json.dumps(rows,indent=2))
group.barrier();torch.cuda.synchronize();destroy_model_parallel();destroy_distributed_environment()
