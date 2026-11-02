"""Validated C1 Q-A/KV + first weighted RMSNorm region; no serving integration.

FP8 weights/inputs and E8M0 scales; BF16 WMMA, FP32 atomic split-K.
One state per stream/in-flight execution. Calls sharing a state must not overlap.
"""
import torch
from fused import project_fused

class QAKVNorm:
    def __init__(self, device='cuda', split=16):
        if split not in (16,32):raise ValueError('Validated split choices: 16 or 32')
        self.split=split
        self.acc=torch.zeros(1536,device=device,dtype=torch.float32)
        self.count=torch.zeros(129,device=device,dtype=torch.int32)
        self.output=torch.empty(1536,device=device,dtype=torch.float32)
        self.kernel=None
    def __call__(self,X,XS,W,WS,NW,out=None):
        if out is None:out=self.output
        self.kernel=project_fused[(12*self.split,)](
            X,XS,W,WS,self.acc,NW,self.acc,self.count,out,
            BN=128,SPLIT=self.split,DIRECT=True,WAVES=4,BK=128,UNROLL=1,
            UPSCALE=True,REPLICATE=True,CACHE='.ca',SCALAR=True,PACKED=False,
            MODE=1,OUTLAY=1,NVEC=1,COUNTSTRIDE=128,NATIVE=0,NORMCACHE='.ca',
            num_warps=4,enable_fp_fusion=False)
        return out
