from native_output import *
from aiter.ops.inverse_rope_group_quant import inverse_rope_group_quant

def err(a,b):return (((a.float()-b.float())**2).mean()/(b.float()**2).mean()).sqrt().item()
def dequant(w,s):return w.float()*torch.pow(2.,s.view(torch.uint8).float()-127).repeat_interleave(128,0).repeat_interleave(128,1)
n=NativeOutput();f=n.fixture;torch.manual_seed(612)
x=torch.randn(1,64,512,device='cuda',dtype=torch.bfloat16);res=torch.randn(1,4,4096,device='cuda',dtype=torch.bfloat16);post=torch.rand(1,4,device='cuda');comb=torch.rand(1,4,4,device='cuda')
for _ in range(3):y=n(x);z=n.post(y,res,post,comb)
torch.cuda.synchronize()
cos,sin=n.rotary_emb.cos_sin_2d();xq,xs=inverse_rope_group_quant(x,n.positions,cos,sin,8,128,'row')
xd=xq.float()*torch.pow(2.,xs.view(torch.uint8).float()-127).repeat_interleave(128,-1)
wad=dequant(f['OA'].cuda(),f['OAS'].cuda()).view(8,1024,4096)
aref=torch.einsum('bgk,gnk->bgn',xd,wad).bfloat16().flatten(1);a=n.oa(x)
aq,asc=linear.per_1x128_e8m0_quant(a,torch.float8_e4m3fn,True);ad=aq.float()*torch.pow(2.,asc.view(torch.uint8).float()-127).repeat_interleave(128,-1)
wbd=dequant(f['OB'].cuda(),f['OBS'].cuda());bref=(ad@wbd.T).bfloat16()
zref=(post[:,:,None]*y[:,None,:].float()+torch.einsum('bij,bid->bjd',comb,res.float())).bfloat16()
checks={'OA_reference_error':err(a,aref),'OB_reference_error':err(y,bref),'post_reference_error':err(z,zref),'post_maxabs':(z.float()-zref.float()).abs().max().item(),'finite':bool(torch.isfinite(z).all()),'metadata':n.metadata()}
json.dump(checks,open('checks.json','w'),indent=2);print(checks,flush=True)
for name,fn in [('native_output',lambda:n(x)),('native_post',lambda:n.post(y,res,post,comb)),('native_output_post',lambda:n.full(x,res,post,comb))]:
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
  fn();torch.cuda.synchronize()
 p.export_chrome_trace(name+'-trace.json');e=json.load(open(name+'-trace.json'))['traceEvents'];ks=[{'name':v['name'],'dur_us':v['dur']} for v in e if v.get('cat')=='kernel'];json.dump(ks,open(name+'-kernels.json','w'),indent=2);print(name,ks,flush=True)
