"""Single-launch Flash C1 QKV region; all GEMM weight tiles use TDM.

Default mode='single', early=True, prefetch=True: non-speculative L2 prefetch. early=False delays the identical QB
pipeline until after the grid poll. mode='global_control' is a measurement
control preserving the old direct weight loads. No serving integration.
BF16 input[4096], FP32 cos/sin[32]; BF16 Q[64,512], KV[512].
Weights FP8 W[1536,4096], QB[32768,1024], uint8 scales[12,32]/[256,8],
FP32 norm gamma[1536]. One state per nonoverlapping stream. Check health
is zero after synchronization. Requires256 available gfx1250 CUs.
"""
import torch
from combined import complete

class QKVRegion:
    def __init__(self,W,WS,NW,QB,QBS,mode='single',early=True,prefetch=True):
        if mode not in ('single','global_control'):raise ValueError('mode must be single or global_control')
        prop=torch.cuda.get_device_properties(W.device)
        if not prop.gcnArchName.startswith('gfx1250'):raise ValueError('gfx1250 required')
        if prop.multi_processor_count<256:raise ValueError('single requires256 available CUs')
        assert W.shape==(1536,4096) and QB.shape==(32768,1024)
        assert WS.shape==(12,32) and QBS.shape==(256,8) and NW.numel()==1536
        assert W.dtype==QB.dtype==torch.float8_e4m3fn and WS.dtype==QBS.dtype==torch.uint8
        assert NW.dtype==torch.float32
        assert all(t.device==W.device and t.is_contiguous() for t in [W,WS,NW,QB,QBS])
        self.W,self.WS,self.NW,self.QB,self.QBS=W,WS,NW,QB,QBS
        self.mode=mode;self.early=bool(early);self.prefetch=bool(prefetch);self.device=W.device
        def z(shape,dtype=torch.float32):return torch.zeros(shape,device=W.device,dtype=dtype)
        self.p=z((16,1536));self.acc=z(1536);self.count=z(129,torch.int32);self.y=z(1536)
        self.lp=z(1024,torch.float8_e4m3fn);self.ls=z(8,torch.uint8)
        self.qp=z(32768);self.qc=z(64,torch.int32)
        self.q=z((64,512),torch.bfloat16);self.kv=z(512,torch.bfloat16)
        self.counter=z(1,torch.int32);self.health=z(1,torch.int32)
        self.kw=dict(num_warps=4,enable_fp_fusion=False)
    def __call__(self,X,Cos,Sin):
        assert X.numel()==4096 and X.dtype==torch.bfloat16
        assert Cos.numel()==Sin.numel()==32 and Cos.dtype==Sin.dtype==torch.float32
        assert all(t.device==self.device and t.is_contiguous() for t in [X,Cos,Sin])
        complete[(256,)](X,self.ls,self.W,self.WS,self.p,self.NW,self.acc,self.count,self.y,self.lp,self.ls,Cos,Sin,self.kv,self.QB,self.QBS,self.qp,self.qc,self.q,self.counter,self.health,BN=128,GRID=256,STYLE=3,U=4,QAMODE=0,ALLTDM=self.mode=='single',EARLY=self.early,PREFETCH=2 if self.prefetch and self.mode=="single" else 0,**self.kw)
        return self.q,self.kv
