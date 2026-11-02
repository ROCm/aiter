from pathlib import Path
r=Path('/home/vigokhal/dsv4-monokernel');o=r/'integrated-c8'
# Explicit optimized module names keep legacy reference imports unchanged.
for src,dst in [('moe_multitoken_prototype','optimized_moe'),('moe_task_stage','optimized_moe_stage'),('routing_batch','optimized_routing'),('compressor_batch','optimized_compressor')]:
 s=(r/'c8-tuning'/f'{src}.py').read_text().replace('from moe_task_stage import','from optimized_moe_stage import')
 (o/f'{dst}.py').write_text(s)
 if (r/'batch-full'/f'{src}.py').exists():(o/f'{src}.py').write_text((r/'batch-full'/f'{src}.py').read_text())
 else:(o/f'{src}.py').unlink(missing_ok=True)
s=(o/'ffn_batch.py').read_text().replace('from moe_multitoken_prototype import fused as moe_body','from optimized_moe import fused as moe_body').replace('from routing_batch import route_body','from optimized_routing import route_body');(o/'ffn_batch.py').write_text(s)
s=(o/'optimized_compressor.py').read_text().replace('SSP:gl.constexpr):\n p=gl.program_id(0);row=p//4;part=p%4','SSP:gl.constexpr,M:gl.constexpr=8,PLAN_STRIDE:gl.constexpr=4):\n p=gl.program_id(0);row=p//4;part=p%4').replace('if p<32:\n  pos=gl.load(Plan+row*4+2)','if p<M*4:\n  pos=gl.load(Plan+row*PLAN_STRIDE+2)')
(o/'optimized_compressor.py').write_text(s)
s=(o/'mhc_native_inline.py').read_text().replace('PIDM=None,PIDK=None,','PIDM=None,PIDK=None,COEF_STRIDE:gl.constexpr=0,')
s=s.replace('gl.expand_dims(rows, 0) * HC + gl.expand_dims(h_idx, 1)','gl.expand_dims(rows, 0) * (COEF_STRIDE if COEF_STRIDE else HC) + gl.expand_dims(h_idx, 1)').replace('gl.expand_dims(rows, 0) * (HC * HC) + gl.expand_dims(h_idx, 1)','gl.expand_dims(rows, 0) * (COEF_STRIDE if COEF_STRIDE else HC * HC) + gl.expand_dims(h_idx, 1)')
for expr,var in [('post_c','x_t'),('comb1_c','r1'),('comb2_c','r2'),('comb3_c','r3')]:s=s.replace(f'a = gl.expand_dims({expr}, 2) * {var}.to(gl.float32) + a',f'a = gl.fma(gl.expand_dims({expr}, 2), {var}.to(gl.float32), a)')
s=s.replace('sq = a * a + sq','sq = gl.fma(a, a, sq)')
(o/'mhc_native_inline.py').write_text(s)
s=(r/'batch-full/model_layer_batch.py').read_text().replace('from qkv_full_batch import qkv_batch','from qkv_candidate import qkv_batch').replace('from attention_batch import attention_body','from attention_paged import attention_paged').replace('from compressor_batch import compressor_body','from optimized_compressor import compressor_body,pool_partial\nfrom mhc_integrated import post_pre')
s=s.replace('BatchIds=None):','BatchIds=None,CacheRows=0,Work=None,CompCounter=None,MHC=(),ML:gl.constexpr=(),MV:gl.constexpr=(),OPT_MHC:gl.constexpr=True,OPT_COMP:gl.constexpr=True):')
# replace compact staging with cos/sin preparation only
start=s.index('  n=gl.load(RealPtr+p+1)');end=s.index(' sync(A[29],A[30],256,3)\n qkv_batch',start)
s=s[:start]+s[end:]
s=s.replace('attention_body(A[28],A[23],A[31],A[32],A[33],A[34],A[35],A[36],A[37],A[38],A[39],A[29],A[30],M,T,64)','attention_paged(A[28],A[23],RealCache,RealIndices,RealPtr,A[34],A[35],A[36],A[37],A[38],A[39],A[29],A[30],Dest,CacheRows,M,8)')
# subgroup includes all projection CTAs (N//16 >=32), plans written by first M
s=s.replace('  sync(A[29],A[30],256,3)\n  if p<M:\n   CR=', '''  if OPT_COMP:
   if p<N//16:sync(CompCounter,A[30],N//16,3)
  else:sync(A[29],A[30],256,3)
  if RATIO==128 and OPT_COMP:
   pool_partial(MC[8],MC[9],MainPlan,MC[6],Work,MM[4],MM[5],MM[6],MM[7],M,8)
   if p<N//16:sync(CompCounter,A[30],N//16,3)
  if p<M:
   CR=''')
s=s.replace('compressor_body(MY+p*N,MY+p*N+N//2,*CR,*MM,OWNER=p)','compressor_body(MY+p*N,MY+p*N+N//2,*CR,*MM,OWNER=p,Partial=Work+p*6144 if RATIO==128 and OPT_COMP else None)')
start=s.index(' post_batch(A[1],A[51]');end=s.index(' sync(F[-2],F[-1],256,3)\n shuffle',start)
s=s[:start]+''' if OPT_MHC:
  post_pre(A[51],A[1],A[7],F[0],F[6],F[9],F[3],F[4],F[5],MHC,F[-1],M,ML,MV)
  sync(F[-2],F[-1],256,3)
 else:
  post_batch(A[1],A[51],A[7],A[2],M,False,FP32MID)
  sync(F[-2],F[-1],256,3)
 if STOP==5:return
 ffn_body(*F,W1S,S1S,W2S,S2S,M,Chosen,Weights,DO_PRE=not OPT_MHC,POST_SHUFFLED=False,Bias=Bias,HAS_BIAS=HAS_BIAS,FP32IN=FP32MID,FP32OUT=FP32OUT)
'''+s[end:]
s=s.replace('(dest>=0)&valid','(dest>=0)&(dest<CacheRows)&valid')
(o/'model_layer_batch.py').write_text(s)
s=(r/'batch-full/runtime_batch.py').read_text().replace("'/work/batch-full'","'/work/integrated-c8','/work/batch-full'")
s=s.replace('   w=self.weights','   w=self.weights\n   import aiter\n   w.mhc_packed=aiter.mhc_shuffle_fn(w.f[2].contiguous().float()).view(torch.bfloat16)')
s=s.replace('  obj=model_layer[(256,)](*args,**kwargs)', '''  from aiter.ops.triton.fusions.mhc import _gluon_post_pre_layouts,_gluon_reduce_apply_layouts
  ml=_gluon_post_pre_layouts(4,16,64,32,2,True,4);mv=_gluon_reduce_apply_layouts(4,1,4,64)
  kwargs.update(CacheRows=a.unified_kv.numel()//512,Work=s.pool,CompCounter=s.comp_counter,MHC=(w.mhc_packed,*s.mhc),ML=ml,MV=mv,OPT_MHC=os.getenv('ATOM_DSV4_OPT_MHC','1')=='1',OPT_COMP=os.getenv('ATOM_DSV4_OPT_COMP','1')=='1')
  key=(m,r,kwargs['OPT_MHC'],kwargs['OPT_COMP'])
  if key not in getattr(self,'compiled',set()):
   obj=model_layer.warmup(*args,grid=(256,),**kwargs);obj._init_handles()
   if obj.n_spills or obj.metadata.shared>327680:raise RuntimeError(f'Unsafe persistent resource budget: {obj.n_regs=} {obj.n_spills=} {obj.metadata.shared=}')
   if not hasattr(self,'compiled'):self.compiled=set()
   self.compiled.add(key)
  obj=model_layer[(256,)](*args,**kwargs)''')
s=s.replace('MULTIROW FULL MONOKERNEL','OPTIMIZED FULL MONOKERNEL')
(o/'runtime_batch.py').write_text(s)
s=(r/'batch-full/scratch_batch.py').read_text().replace(' def extra(self):', '''  self.pool=empty((m,4,3,512),torch.float32)
  self.comp_counter=torch.zeros(1,device='cuda',dtype=torch.int32)
  self.mhc=(empty((64,m,32),torch.float32),empty((64,m),torch.float32),empty((m,4,1),torch.float32),empty((m,4,4),torch.float32),torch.zeros(1,device='cuda',dtype=torch.int32))
 def extra(self):''')
(o/'scratch_batch.py').write_text(s)
