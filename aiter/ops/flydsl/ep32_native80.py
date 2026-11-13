"""The target EP32 shape: BM80 atomic baseline or variable-width native output."""
from functools import cache
import os
import torch
from .moe_kernels import _run_compiled
from .mxmoe_native_reduce import ep32_native_reduce

@cache
def _kernel(device,valid,native):
    if native:
        from .kernels.mxmoe_gemm2_ep32_bm80var import compile_gemm2_ep32
    else:
        from .kernels.mxmoe_gemm2_ep32_bm80base import compile_gemm2_ep32
    return compile_gemm2_ep32(HIDDEN_MAX=3584,INTER_MAX=3072,BM=80,B_NT=True,
        GRID_WGS=512,use_valid_token_count=valid,
        **({'REDUCE':True,'NATIVE_C':True} if native else {}))

_buffers={}
def matmul(aq,ascale,bq,bscale,eids,cumsum,stids,weights,target,n_tokens,max_sorted,valid,native):
    mb=(max_sorted+79)//80
    exe=_kernel(target.device,bool(valid),bool(native))
    _run_compiled(exe,(aq.data_ptr(),ascale.data_ptr(),bq.data_ptr(),bscale.data_ptr(),
        eids.data_ptr(),cumsum.data_ptr(),stids.data_ptr(),weights.data_ptr(),target.data_ptr(),
        n_tokens,mb,mb,3072,3584,target.data_ptr(),target.data_ptr(),torch.cuda.current_stream(target.device)))

def dispatch(aq,ascale,bq,bscale,eids,cumsum,stids,weights,reverse,mask,ids,out,n_tokens,max_sorted,valid,native=None):
    native=os.environ.get('EP32_G2_NATIVE','1')=='1' if native is None else native
    target=out
    if native:
        key=(out.device,torch.cuda.current_stream(out.device).cuda_stream,max_sorted)
        if key not in _buffers:
            _buffers[key]=torch.empty((max_sorted,3584),dtype=torch.float32,device=out.device)
        target=_buffers[key]
    matmul(aq,ascale,bq,bscale,eids,cumsum,stids,weights,target,n_tokens,max_sorted,valid,native)
    if native:
        ep32_native_reduce(target,reverse,weights,mask,ids,cumsum,out)
    return out
