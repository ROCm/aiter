import json,pathlib,re,sys
root=pathlib.Path('/work/triton-cache');rows=[]
for p in root.glob('*/_mhc_pre_reduce_apply_gfx1250_kernel.amdgcn'):
 j=p.with_suffix('.json')
 if not j.exists():continue
 d=json.load(j.open());s=p.read_text();name=d.get('name','')
 if 'C_7168_' not in name:continue
 def field(x):
  m=re.search(r'\.'+x+r'\s+(\d+)',s);return int(m.group(1)) if m else None
 rows.append(dict(name=name,shared=d.get('shared'),vgpr=field('amdhsa_next_free_vgpr'),
                  sgpr=field('amdhsa_next_free_sgpr'),scratch_bytes=field('amdhsa_private_segment_fixed_size'),path=str(p)))
pathlib.Path('/work/results/resources_'+sys.argv[1]+'.json').write_text(json.dumps(rows,indent=2))
