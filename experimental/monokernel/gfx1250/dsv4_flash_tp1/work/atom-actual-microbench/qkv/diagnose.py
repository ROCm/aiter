from native import *
sys.path.insert(0,'/work/mhc-optimized')
from qkv_region import QKVRegion
from tail import quant
from split_kernels import qa_reduce,qa_norm
n=NativeQKV();f=n.f;r=QKVRegion(f['W_bytes'].cuda().view(torch.float8_e4m3fn),f['WS'].cuda(),f['NW'].cuda(),f['QB_bytes'].cuda().view(torch.float8_e4m3fn),f['QBS'].cuda())
def err(a,b):return ((a.float()-b.float()).square().mean()/b.float().square().mean().clamp_min(1e-30)).sqrt().item()
rows=[]
for index in [0,2,3,35]:
 if index==0:x=f['live'].cuda()
 elif index<10:x=f['inputs'][index-1].cuda()
 else:torch.manual_seed(12000+index-10);x=torch.randn(4096,dtype=torch.bfloat16,device='cpu').cuda()
 q,k=n(x);r(x,n.cos[1024].flatten().float().contiguous(),n.sin[1024].flatten().float().contiguous());torch.cuda.synchronize()
 aq,asc=linear.per_1x128_e8m0_quant(x.reshape(1,4096),quant_dtype=torch.float8_e4m3fn,transpose_scale=True)
 bq=torch.empty(4096,device='cuda',dtype=torch.float8_e4m3fn);bsc=torch.empty(32,device='cuda',dtype=torch.uint8)
 quant[(32,)](x,bq,bsc,num_warps=4,enable_fp_fusion=False)
 raw=torch.empty(1536,device='cuda');qa_reduce[(2,)](r.p,raw,num_warps=4,enable_fp_fusion=False)
 # Isolate norm+quant with exact identical native QA output as input.
 y=torch.empty_like(raw);ql=torch.empty_like(r.lp);qs=torch.empty_like(r.ls)
 qa_norm[(2,)](n.qaout.float(),r.NW,y,num_warps=4,enable_fp_fusion=False);quant[(8,)](y,ql,qs,num_warps=4,enable_fp_fusion=False)
 torch.cuda.synchronize()
 rows.append({'input':index,'input_fp8_equal':torch.equal(aq.view(torch.uint8).flatten(),bq.view(torch.uint8)),'input_scales_equal':torch.equal(asc.view(torch.uint8).flatten(),bsc),'qa_raw_rms':err(raw[:1024],n.qaout[0,:1024]),'kv_raw_rms':err(raw[1024:],n.qaout[0,1024:]),'latent_fp8_mismatch_count':(r.lp.view(torch.uint8)!=n.qr.view(torch.uint8).flatten()).sum().item(),'latent_scales_equal':torch.equal(r.ls,n.qrs.view(torch.uint8).flatten()),'norm_on_same_qa_fp8_mismatch_count':(ql.view(torch.uint8)!=n.qr.view(torch.uint8).flatten()).sum().item(),'norm_on_same_qa_scales_equal':torch.equal(qs,n.qrs.view(torch.uint8).flatten()),'qb_rms':err(r.qp,n.qbout.flatten()),'q':err(r.q,q),'kv':err(r.kv,k)})
json.dump(rows,open('diagnosis.json','w'),indent=2);print(rows)
