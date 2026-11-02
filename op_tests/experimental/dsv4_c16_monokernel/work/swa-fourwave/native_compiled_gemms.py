"""Adapt native compiled GEMMs: no replacement tile, layout, or pipeline."""
from pathlib import Path
import hashlib,json,re
from export_native_fly import build
ROOT=Path('/work/swa-fourwave')
def apply(st):
 configs={
 4:('gemm_a8w8_mx128_t16x32x1024_mw1_nw2_nb4_sk1_cm1_cn1',(16,1536,4096,16,4096,1536),(1,48,1)),
 6:('gemm_a8w8_mx128_t16x128x512_mw1_nw4_nb2_sk1_cm1_cn1',(16,32768,1024,16,1024,32768),(1,256,1)),
 11:('batched_gemm_a8w8_mx128_t32x32x1024_mw2_nw2_nb4_sk1_cm1_cn1_mbn',(16,1024,4096,256,32768,8192),(1,32,8)),
 13:('gemm_a8w8_mx128_t16x64x512_mw1_nw4_nb4_sk1_cm1_cn1',(16,4096,8192,16,8192,4096),(1,64,1)),
 17:('gemm_a8w8_mx128_t16x32x1024_mw1_nw2_nb4_sk1_cm1_cn1',(16,4096,4096,16,4096,4096),(1,128,1)),
 20:('gemm_a8w8_mx128_t16x32x1024_mw1_nw2_nb2_sk1_cm1_cn1',(16,4096,2048,16,2048,4096),(1,128,1))}
 provenance=[]
 for i,(name,scalars,grid) in configs.items():
  old=st[i];source=ROOT/'source/native_parity'/f'gemm_{i}.ll'
  text=source.read_text();assert f'@{name}(' in text
  md=re.search(r'!reqd_work_group_size !(\d+)',text)[1]
  dims=[int(x) for x in re.findall(r'i32 (\d+)',re.search(r'^!'+md+r' = (.*)$',text,re.M)[1])]
  assert dims==[old['threads'],1,1],(i,dims)
  local=ROOT/'generated'/('native_parity_'+str(i)+'.native.ll');local.write_bytes(source.read_bytes())
  args=list(old.get('values',old.get('args')))+list(scalars)
  new=build(local,args,ROOT/'generated','native_parity_'+str(i),grid,old['threads'],physical_warps=4)
  assert new['total']==old['total']
  new.update(name=old['name']+'_native_compiled',lds=max(new['lds'],old['lds']))
  st[i]=new
  provenance.append({'stage':i,'native_symbol':name,'native_source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'grid':grid,'threads':old['threads'],'lds':new['lds'],'adapter':'ABI, logical CTA IDs, dynamic LDS, subgroup barrier scope only'})
 (ROOT/'NATIVE_COMPILED_GEMMS.json').write_text(json.dumps(provenance,indent=2))
