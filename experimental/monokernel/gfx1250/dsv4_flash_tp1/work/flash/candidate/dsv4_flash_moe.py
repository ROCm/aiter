"""Experimental single-token gfx1250 routed FP4 MoE dispatch."""
import os
import torch
from atom.model_ops.dsv4_flash_moe_kernel import fused
_ENABLED=os.environ.get('ATOM_DSV4_FLASH_MOE','0')=='1'
_CHECK=int(os.environ.get('ATOM_DSV4_FLASH_MOE_CHECK_LIMIT','16'))
_state={}

def try_moe(x,xs,w1,w2,s1,s2,routing,gather,scatter,limit,eligible):
    if not _ENABLED or not eligible or x.shape!=(1,4096):return None,False
    if routing.block_m!=16 or routing.n_expts_act!=6 or gather.numel()!=6 or scatter.numel()!=6:return None,False
    if w1.shape[-1]*16!=4096 or w2.shape[-1]*16!=4096:return None,False
    if w1.dtype!=torch.uint8 or w2.dtype!=torch.uint8:return None,False
    key=w1.data_ptr()
    if key not in _state:
        _state[key]=dict(i=torch.empty((6,2048),dtype=x.dtype,device=x.device),is_=torch.empty((6,64),dtype=torch.uint8,device=x.device),y=torch.empty((6,4096),dtype=torch.bfloat16,device=x.device),counter=torch.zeros(6,dtype=torch.uint32,device=x.device),health=torch.zeros(1,dtype=torch.int32,device=x.device),ready=torch.zeros(32,dtype=torch.int32,device=x.device),checks=0)
    st=_state[key];z=torch.empty((1,4096),dtype=torch.bfloat16,device=x.device)
    a,b,c,d=[t.transpose(1,2) for t in (w1,s1,w2,s2)]
    e=routing.expt_data
    fused[(192,)](x,xs,a,b,c,d,st['i'],st['is_'],st['y'],gather,e.hist,e.token_offs_raw,e.token_offs_pad[-1:],e.block_pid_map,routing.gate_scal,st['counter'],st['health'],z,scatter,st['ready'],a.stride(),b.stride(),c.stride(),d.stride(),limit,BUFFERS=4,num_warps=4)
    check=st['checks']<_CHECK and not torch.cuda.is_current_stream_capturing()
    if check:
        assert st['health'].item()==0,'MoE bounded wait failed'
        assert not st['ready'].any().item(),'MoE reduction did not complete'
    return z,check

def validate(candidate,reference,w1):
    torch.testing.assert_close(candidate,reference,rtol=0,atol=0)
    st=_state[w1.data_ptr()];st['checks']+=1
    if st['checks'] in (2,16) or st['checks']%128==0:
        print(f"DSV4 Flash fused MoE weight={w1.data_ptr()}: {st['checks']} exact live checks passed",flush=True)
