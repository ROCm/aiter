"""Capture this workspace's actual amd-bench BM80 call, then compare outputs."""
import sys,json,functools
from pathlib import Path
import torch
from aiter.ops.flydsl import ep32_native80 as impl
from aiter.ops.flydsl.mxmoe_native_reduce import ep32_native_reduce
sys.path.insert(0,'/root/workspace/workflow_34/moonshot_ep32_moe_v2/amd_moe_bench_share')
import bench_moe

original=impl.dispatch
done=False
@functools.wraps(original)
def check(*a,**kw):
    global done
    if done:return original(*a,**kw)
    done=True
    aq,sa,b,sb,eids,cs,sids,weight,rev,mask,ids,out,nt,ms,valid=a[:15]
    rows,pad=cs.cpu().tolist();p=sids[:rows].cpu().long();tok=p&0xffffff
    live=tok<pad;count=torch.bincount(tok[live],minlength=nt)
    small=((count>0)&(count<=2)).to(out.device)
    x=torch.full((ms,3584),float('nan'),dtype=torch.float32,device=out.device)
    def run(native,st=sids,m=mask):
        y=torch.zeros_like(out) if not native else torch.full_like(out,float('nan'))
        impl.matmul(aq,sa,b,sb,eids,cs,st,weight,x if native else y,nt,ms,valid,native)
        if native:ep32_native_reduce(x,rev,weight,m,ids,cs,y)
        torch.cuda.synchronize();return y
    ref=run(False)
    for replay in range(3):
        y=run(True);assert torch.equal(y[small],ref[small]),replay
        assert torch.isfinite(y).all()
    eid=eids[:rows//80].cpu().repeat_interleave(80)
    padding=((p&0xff000000)|pad).to(torch.int32)
    st=sids.clone();active=mask.nonzero().flatten().cpu().tolist()
    assert len(active)==28,active
    for e in range(28):
        st[:rows].copy_(torch.where(eid==e,p.to(torch.int32),padding).to(out.device))
        m=torch.zeros_like(mask);m[active[e]]=1
        r,y=run(False,st,m),run(True,st,m)
        assert torch.equal(r,y),f'expert {e}'
    result={'routes':int(live.sum()),'sorted_rows':rows,'replays':3,
        'one_two_route_bit_exact':True,'all_28_experts_bit_exact':True,'all_output_written':True}
    Path('g2var/check80_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result,flush=True)
    return original(*a,**kw)
impl.dispatch=check
args=bench_moe.build_parser().parse_args(['run','--output-dir','g2var/check80_bench',
    '--batch-sizes','4096','--ep','32','--rank','0','--warmup','1','--samples','1'])
bench_moe.run(args)
assert done,'BM80 target path not reached'
