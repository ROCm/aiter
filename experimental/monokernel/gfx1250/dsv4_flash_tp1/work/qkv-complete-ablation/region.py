"""Isolated Flash C1 Q/KV preparation. Not an ATOM serving integration.

One state per stream; invocations sharing state must not overlap.
Input BF16[4096]; output BF16 Q[64,512], KV[512].
RoPE tables FP32[32], interleaved pairs in final64 channels.
Weights: FP8 E4M3 W[1536,4096], QB[32768,1024]; E8M0 uint8
scales[12,32]/[256,8]; weighted norm gamma FP32[1536].
"""
import torch
from qa import project_fused
from qb import qb_kernel
from combined import complete

class QKVRegion:
    def __init__(self,W,WS,NW,QB,QBS,mode='two'):
        if mode not in ('two','single'):raise ValueError('mode must be two or single')
        prop=torch.cuda.get_device_properties(W.device)
        if not prop.gcnArchName.startswith('gfx1250'):raise ValueError('gfx1250 required')
        if mode=='single' and prop.multi_processor_count<256:raise ValueError('single requires256 available CUs')
        assert W.shape==(1536,4096) and QB.shape==(32768,1024)
        assert WS.shape==(12,32) and QBS.shape==(256,8) and NW.numel()==1536
        assert W.dtype==QB.dtype==torch.float8_e4m3fn and WS.dtype==QBS.dtype==torch.uint8
        assert NW.dtype==torch.float32
        assert all(t.device==W.device and t.is_contiguous() for t in [W,WS,NW,QB,QBS])
        self.W,self.WS,self.NW,self.QB,self.QBS=W,WS,NW,QB,QBS
        self.mode=mode;self.device=W.device
        def z(shape,dtype=torch.float32):return torch.zeros(shape,device=W.device,dtype=dtype)
        self.acc=z(1536);self.count=z(129,torch.int32);self.y=z(1536)
        self.lp=z(1024,torch.float8_e4m3fn);self.ls=z(8,torch.uint8)
        self.qp=z(32768);self.qc=z(64,torch.int32)
        self.q=z((64,512),torch.bfloat16);self.kv=z(512,torch.bfloat16)
        self.counter=z(1,torch.int32);self.health=z(1,torch.int32)
        self.kw=dict(num_warps=4,enable_fp_fusion=False)
    def __call__(self,X,Cos,Sin):
        assert X.numel()==4096 and X.dtype==torch.bfloat16
        assert Cos.numel()==Sin.numel()==32 and Cos.dtype==Sin.dtype==torch.float32
        assert all(t.device==self.device and t.is_contiguous() for t in [X,Cos,Sin])
        if self.mode=='single':
            complete[(256,)](X,self.ls,self.W,self.WS,self.acc,self.NW,self.acc,self.count,self.y,self.lp,self.ls,Cos,Sin,self.kv,self.QB,self.QBS,self.qp,self.qc,self.q,self.counter,self.health,BN=128,GRID=256,**self.kw)
        else:
            project_fused[(192,)](X,self.ls,self.W,self.WS,self.acc,self.NW,self.acc,self.count,self.y,self.lp,self.ls,Cos,Sin,self.kv,INLINE=True,TAIL=True,MODE=1,BN=128,SPLIT=16,DIRECT=True,WAVES=4,BK=128,UNROLL=1,UPSCALE=True,REPLICATE=True,CACHE='.ca',SCALAR=True,PACKED=False,OUTLAY=1,NVEC=1,COUNTSTRIDE=128,NATIVE=0,NORMCACHE='.ca',**self.kw)
            qb_kernel[(512,)](self.lp,self.ls,self.QB,self.QBS,self.qp,self.qc,Cos,Sin,self.q,TAIL=True,BN=64,**self.kw)
        return self.q,self.kv
