import sys
sys.path.insert(0,'/work/atom-actual-microbench/attention')
from graphbatch import FlatGraphBatch as GraphBatch
import sys,json,statistics
from pathlib import Path
sys.path[:0]=['/work/atom-actual-microbench/qkv','/work/atom-actual-microbench/mhc','/work/mhc-optimized','/app/aiter','/app/ATOM']
from native import NativeQKV
from native_api import NativeMHC
sys.path.insert(0,'/work/mhc-optimized')
from setup import State,R,HAF,HAS,HAB,AN,err
import aiter,torch
from atom.model_ops.v4_kernels.paged_decode import sparse_attn_v4_paged_decode
outdir=Path(__file__).resolve().parent
m=NativeMHC(HAF,HAS,HAB,AN)
s=State(128);r=aiter.mhc_res_shuffle(R.view(1,4,4096))
n=NativeQKV(cache=s.attn.cache.clone(),dest=127)
import setup as candidate_setup
candidate_setup.Cos=n.cos[1024].flatten().float().contiguous();candidate_setup.Sin=n.sin[1024].flatten().float().contiguous()
idx=torch.zeros(128*128,device='cuda',dtype=torch.int32);idx[:128]=torch.arange(128,device='cuda',dtype=torch.int32)
ptr=torch.tensor([0,128],device='cuda',dtype=torch.int32)
def actual():
 x,post,comb=m(r);q,kv=n(x)
 return sparse_attn_v4_paged_decode(q,n.cache,idx,ptr,s.attn.sink,512**-.5)
def candidate():return s.run(aiter.mhc_res_unshuffle(r),'opt_both')
def candidate_core():return s.run(R,'opt_both')
functions={'actual_atom':actual,'fused_with_input_adapter':candidate,'fused_core_plain_input':candidate_core}

checks={}
for name,fn in functions.items():
 for _ in range(3):fn()
 torch.cuda.synchronize();g=GraphBatch(fn)
 reference=fn().clone();torch.cuda.synchronize()
 g.out.fill_(float('nan'));g.replay();torch.cuda.synchronize()
 checks[name]={'finite':bool(torch.isfinite(g.out).all()),'bitwise_equal':bool(torch.equal(reference,g.out)),'max_abs':float((reference.float()-g.out.float()).abs().max()),'kernel_count':g.kernel_count,'node_types':g.node_types}
(outdir/'poison-graph-validation.json').write_text(json.dumps(checks,indent=2));print(checks)
