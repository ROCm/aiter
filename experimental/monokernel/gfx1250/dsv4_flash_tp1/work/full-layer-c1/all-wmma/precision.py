exec(compile((__import__('pathlib').Path(__file__).parent/'setup.py').read_text(),'setup.py','exec'))
from split_control import split_stage
from region_kernel import region_kernel
from full_kernel import all_wmma_layer
import statistics,re,importlib.util
kw=dict(GRID=args.grid,num_warps=4,enable_fp_fusion=False)
extra=dict(XP=torch.empty(4096,device='cuda',dtype=torch.float8_e4m3fn),XS=torch.empty(32,device='cuda',dtype=torch.uint8),LP=torch.empty(1024,device='cuda',dtype=torch.float8_e4m3fn),LS=torch.empty(8,device='cuda',dtype=torch.uint8),QPART=torch.empty(8192,device='cuda',dtype=torch.float32),KPART=torch.empty(4096,device='cuda',dtype=torch.float32),MHC_TEMP=torch.empty(128*24,device='cuda'))
regions=[(0,4),(5,10),(11,12),(13,17),(18,22),(23,30),(31,32)]
watch=[*sizes,'Out','Cache']
def split(a=0,b=32):
 for i in range(a,b+1):split_stage[(args.grid,)](*values,STAGE=i,**kw)
def region(i):return region_kernel[(args.grid,)](*values,REGION=i,**kw,**extra)
def full():return all_wmma_layer[(args.grid,)](*values,**kw,**extra)
def snapshot():return {k:d[k].clone() for k in watch}
def restore(s):
 for k,v in s.items():d[k].copy_(v)
def compare(s):
 result={}
 for k,v in s.items():
  diff=d[k].float()-v.float()
  result[k]=dict(relative_rms=(diff.square().mean()/v.float().square().mean().clamp_min(1e-30)).sqrt().item(),max_abs=diff.abs().max().item(),exact=torch.equal(d[k],v),finite=torch.isfinite(d[k]).all().item())
 return result
def audit(c,name):
 isa=c.asm['amdgcn'];(root/(name+'.amdgcn')).write_text(isa)
 return dict(n_regs=c.n_regs,shared=c.metadata.shared,sgpr_spills=re.findall(r'\.sgpr_spill_count:\s*(\d+)',isa),vgpr_spills=re.findall(r'\.vgpr_spill_count:\s*(\d+)',isa),scratch=re.findall(r'ScratchSize:\s*(\d+)',isa),wmma=sorted(set(re.findall(r'v_wmma\w+',isa))))

from reference import check
for tok in [42,7]:
 with safe_open(str(p/index['embed.weight']),framework='pt',device='cpu') as sf:emb=sf.get_slice('embed.weight')[tok:tok+1].cuda()
 d['R'].copy_(emb.repeat(4,1));d['Token'].fill_(tok)
 full();torch.cuda.synchronize();assert d['Health'].item()==0
 outdir=root/f'mhc-third-token-{tok}';outdir.mkdir(exist_ok=True)
 try:check(d,outdir)
 except AssertionError:print('REFERENCE GATE FAILED',tok,flush=True)
 print('CASE RESULT',tok,json.loads((outdir/'reference-comparison.json').read_text())['Out'],flush=True)
