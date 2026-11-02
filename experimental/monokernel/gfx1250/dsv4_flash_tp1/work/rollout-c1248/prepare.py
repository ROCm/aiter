from pathlib import Path
import re
root=Path('/home/vigokhal/dsv4-monokernel');out=root/'rollout-c1248'
src=(root/'integrated-c8/model_layer_batch.py').read_text()
header,body=src.split(' p=gl.program_id(0)',1)
header=header.replace('def model_layer(', 'def region(').replace('TILE_MASK:gl.constexpr=15):','TILE_MASK:gl.constexpr=15,GROUP:gl.constexpr=0):')
body=' p=gl.program_id(0)'+body
# Isolated entry points copy the actual serving region bodies; never prefix-time subtraction.
start_comp=body.index(' if RATIO!=0:\n  N:')
start_qkv=body.index(' if Edges is None:\n')
start_attn=body.index(' if Edges is not None and RATIO!=0 and EDGE_MASK&4 and')
start_out=body.index(' if OVERLAP & 1:\n  prefetch_bytes(MHC[0]')
start_mid=body.index(' if OPT_MHC:\n  post_pre')
start_ffn=body.index(' RunF=')
pieces=[body[:start_comp],body[start_comp:start_qkv],body[start_qkv:start_attn],body[start_attn:start_out],body[start_out:start_mid],body[start_mid:start_ffn],body[start_ffn:]]
pieces[2]=' if p<M:\n  i=gl.arange(0,32,layout=gl.BlockedLayout([1],[32],[4],[0]));pos=gl.load(Positions+p)\n  gl.store(A[21]+p*32+i,gl.load(CosTable+pos*32+i));gl.store(A[22]+p*32+i,gl.load(SinTable+pos*32+i))\n sync(A[29],A[30],256,3)\n'+pieces[2][pieces[2].index(' qkv_batch('):]
new=header+' p=gl.program_id(0)\n Epoch=1\n'
for i,piece in enumerate(pieces):
 piece=re.sub(r'^ if STOP[^\n]*return\n','',piece,flags=re.M)
 new+=f' if GROUP=={i}:\n'+''.join(' '+l+'\n' for l in piece.splitlines())
(out/'regions.py').write_text(new)
# Build one fixture harness per attention type from validated snapshot, retaining all full-layer checks.
for kind,fn in [('swa','whole_tiles.py'),('compressed','whole_compressed.py')]:
 s=(root/'latency-overlap'/fn).read_text()
 s=s.replace("from model_layer_batch import model_layer as batch", "from model_layer_batch import model_layer as batch\nfrom regions import region\nfrom bench_regions import benchmark_regions")
 s=s.replace("pos=torch.arange(1023,1023+m,device='cuda',dtype=torch.int32)","pos=torch.full((m,),1023,device='cuda',dtype=torch.int32)")
 # Explicitly disable all experimental handshakes and enable only validated changes in B.
 s=s.replace("kw['EDGE_MASK']=int(os.getenv('EDGE_MASK','0'))", "kw['EDGE_MASK']=0")
 s=s.replace("kw['TILE_MASK']=int(os.getenv('TILE_MASK','1'))", "kw['TILE_MASK']=192 if use else 0")
 s=s.replace("kw['OVERLAP']=int(os.getenv('EARLY_MASK','0')) if use else 0", "kw['OVERLAP']=2 if use else 0")
 s=s.replace(" results.append(dict(m=m,ratio=kw['RATIO'],checks=checks,resources=metrics))", " groups=benchmark_regions(region,batch,args,kw,s,m,saved)\n results.append(dict(m=m,ratio=kw['RATIO'],checks=checks,resources=metrics,groups=groups))")
 s=s.replace("'/work/latency-overlap/'+os.getenv('RUN_ID','whole-tile-'+os.getenv('TILE_MASK','1'))+'.json'", "'/work/rollout-c1248/micro-'+str(m)+'-'+str(kw['RATIO'])+'.json'")
 (out/('micro_'+kind+'.py')).write_text(s)
