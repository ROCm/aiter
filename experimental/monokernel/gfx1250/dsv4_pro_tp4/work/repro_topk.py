import torch
from aiter.ops.triton.moe.moe_routing.topk import topk
for m in (16,128,256,512):
    for bias in (None,torch.zeros(384,device='cuda')):
        print('testing',m,'bias',bias is not None,flush=True)
        x=torch.randn(m,384,device='cuda',dtype=torch.float32)
        y=topk(x,6,apply_softmax=False,score_mode='sqrtsoftplus',bias=bias,renorm=True,routed_scaling_factor=2.5)
        torch.cuda.synchronize(); print('passed',flush=True)
