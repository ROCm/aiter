from pathlib import Path
root=Path('/home/vigokhal/dsv4-monokernel');out=root/'rollout-c1248'
for kind,fn in [('swa','whole_tiles.py'),('compressed','whole_compressed.py')]:
 s=(root/'latency-overlap'/fn).read_text()
 s=s.replace("pos=torch.arange(1023,1023+m,device='cuda',dtype=torch.int32)","pos=torch.full((m,),1023,device='cuda',dtype=torch.int32)")
 s=s.replace(' for use in [0,1]:',' variants=[(0,0),(192,2),(192,0),(128,0),(128,2)]\n for use in range(len(variants)):')
 s=s.replace("kw['EDGE_MASK']=int(os.getenv('EDGE_MASK','0'))", "kw['EDGE_MASK']=0")
 s=s.replace("kw['TILE_MASK']=int(os.getenv('TILE_MASK','1'))", "kw['TILE_MASK']=variants[use][0]")
 s=s.replace("kw['OVERLAP']=int(os.getenv('EARLY_MASK','0')) if use else 0", "kw['OVERLAP']=variants[use][1]")
 s=s.replace(' for mode in [1]:',' for mode in [1,2,3,4]:')
 s=s.replace('dict(mode=use,vgpr=', 'dict(mode=use,tile_mask=variants[use][0],overlap=variants[use][1],vgpr=')
 s=s.replace("'/work/latency-overlap/'+os.getenv('RUN_ID','whole-tile-'+os.getenv('TILE_MASK','1'))+'.json'", "'/work/rollout-c1248/variants-'+str(m)+'-'+str(kw['RATIO'])+'.json'")
 (out/('variants_'+kind+'.py')).write_text(s)
