from pathlib import Path
r=Path('/home/vigokhal/dsv4-monokernel/integrated-c8')
def edit(name,fn):
 p=r/name;p.write_text(fn(p.read_text()))
imports='from edge_sync import edge,publish,consume,prefetch_bytes\n'
# Shared state slots:0pre projection;1pre output;2comp projection;3pool;4comp completion;
# 5QA projection;6QA finishing;7postpre projection;8postpre output;
# 9unused;10shared quant;11routed quant;12router;13selection;14metadata;15shared up;16activation
edit('scratch_batch.py',lambda s:s.replace('  self.comp_counter=',"  self.edges=torch.zeros((20,257),device='cuda',dtype=torch.int32)\n  self.comp_counter="))
for name in ['mhc_pre_aiter.py','mhc_integrated.py','qa_candidate.py','qkv_candidate.py','optimized_routing.py','ffn_batch.py','output_batch.py','model_layer_batch.py']:
 edit(name,lambda s:imports+s)
edit('mhc_pre_aiter.py',lambda s:s.replace('V:gl.constexpr,FP32=None):','V:gl.constexpr,FP32=None,Edges=None):').replace('  sync(CT,H,64,3)','  if Edges is None:sync(CT,H,64,3)\n if Edges is not None:edge(Edges,H,64,p<64,p<M)'))
edit('mhc_integrated.py',lambda s:s.replace('V:gl.constexpr):','V:gl.constexpr,Edges=None):').replace(' if p<64:sync(CT,H,64,3)',' if Edges is not None:edge(Edges,H,64,p<64,p<M)\n elif p<64:sync(CT,H,64,3)'))
edit('qa_candidate.py',lambda s:s.replace('ATOMIC:gl.constexpr=False):\n pid=', 'ATOMIC:gl.constexpr=False,Edges=None):\n pid=').replace(' sync(Counter,Health,256,SYNC_STYLE)\n if not PRE', ' if Edges is not None:edge(Edges,Health,(1536//BN)*SPLITS,pid<(1536//BN)*SPLITS,pid<2*M)\n else:sync(Counter,Health,256,SYNC_STYLE)\n if not PRE'))
edit('qkv_candidate.py',lambda s:s.replace('ATOMIC:gl.constexpr=False):','ATOMIC:gl.constexpr=False,Edges=None,OVERLAP:gl.constexpr=False):').replace('ATOMIC=ATOMIC)\n if SEM:', 'ATOMIC=ATOMIC,Edges=Edges)\n if Edges is not None:\n  edge(Edges+257,Health,M,pid<M,True)\n elif SEM:'))
# Hoist the 8 scalar QB scales and force the returned register value to stay live.
edit('qkv_candidate.py',lambda s:s.replace(' SW:gl.constexpr=', " if OVERLAP:\n  s0=gl.load(QBS+pid*8);s1=gl.load(QBS+pid*8+1);s2=gl.load(QBS+pid*8+2);s3=gl.load(QBS+pid*8+3)\n  s4=gl.load(QBS+pid*8+4);s5=gl.load(QBS+pid*8+5);s6=gl.load(QBS+pid*8+6);s7=gl.load(QBS+pid*8+7)\n SW:gl.constexpr=",1).replace('bsc=gl.load(QBS+pid*8+j)+gl.full((64,4),0,gl.uint8,BSL)', 'bsc=((s0,s1,s2,s3,s4,s5,s6,s7)[j] if OVERLAP else gl.load(QBS+pid*8+j))+gl.full((64,4),0,gl.uint8,BSL)'))
edit('optimized_routing.py',lambda s:s.replace('M:gl.constexpr):','M:gl.constexpr,Edges=None):').replace(' sync(Counter,Health,256,3)',' if Edges is not None:edge(Edges,Health,M,p<M,p<M)\n else:sync(Counter,Health,256,3)'))
# Replace FFN schedule while retaining full old implementation for A/B.
p=r/'ffn_batch.py';s=p.read_text().replace('FP32IN=None,FP32OUT=None):','FP32IN=None,FP32OUT=None,Edges=None,OVERLAP:gl.constexpr=False):')
start=s.index(' quant_body(X,XQ,XS,4096,128,8,M)',s.index('def ffn_body'))
end=s.index(' sync(Counter,Health,256,3)\n if gl.program_id(0)<M*32:',start)
old=s[start:end]
new=''' if Edges is not None:
  p=gl.program_id(0)
  if OVERLAP:
   prefetch_bytes(Gate,256*4096*2)
   prefetch_bytes(SW1,4096*4096)
   prefetch_bytes(SS1,4096*4096//128//128)
   if HAS_BIAS:prefetch_bytes(Bias,256*2)
  quant_body(X,XQ,XS,4096,128,8,M)
  publish(Edges+10*257,p<gl.minimum(256,M*32))
  quant_body(X,MX,MS,4096,32,1,M)
  publish(Edges+11*257,p<gl.minimum(256,M*128))
  router_body(X,Gate,Logits,M)
  edge(Edges+12*257,Health,16,p<16,p<M)
  route_body(Logits,Hist,Offs,Pad,Map,Gather,Scatter,Gamma,Bias,HAS_BIAS,Chosen,Weights,Counter,Health,M,Edges+13*257)
  publish(Edges+14*257,p<M)
  consume(Edges+10*257,Health,min(256,M*32),p<64)
  shared_body(XQ,XS,SW1,SS1,GU,4096,64,M)
  publish(Edges+15*257,p<64)
  if OVERLAP:
   prefetch_bytes(SW2,4096*2048)
   prefetch_bytes(SS2,4096*2048//128//128)
  consume(Edges+15*257,Health,64,p<M*16)
  activation_quant(GU,ACT,AQ,AS,M)
  edge(Edges+16*257,Health,M*16,p<M*16,p<128)
  shared_body(AQ,AS,SW2,SS2,SO,2048,32,M)
  consume(Edges+11*257,Health,min(256,M*128),True)
  consume(Edges+14*257,Health,M,True)
  moe_body(MX,MS,W1,S1,W2,S2,I,IS,Y,Gather,Hist,Offs,Pad,Map,Gamma,Count,Health,Z,Scatter,W1S,S1S,W2S,S2S,10.,M,256,3)
 else:
'''+''.join(' '+line+'\n' for line in old.splitlines())
s=s[:start]+new+s[end:];p.write_text(s)
# Layer-level scheduling.
p=r/'model_layer_batch.py';s=p.read_text().replace('OPT_PRE:gl.constexpr=False):','OPT_PRE:gl.constexpr=False,Edges=None,OVERLAP:gl.constexpr=False):')
rope=''' if p<M:
  i=gl.arange(0,32,layout=gl.BlockedLayout([1],[32],[4],[0]));pos=gl.load(Positions+p)
  gl.store(A[21]+p*32+i,gl.load(CosTable+pos*32+i));gl.store(A[22]+p*32+i,gl.load(SinTable+pos*32+i))
'''
s=s.replace(rope+' sync(A[29],A[30],256,3)\n qkv_batch', ' if Edges is None:\n'+''.join(' '+l+'\n' for l in rope.splitlines())+'  sync(A[29],A[30],256,3)\n qkv_batch')
s=s.replace(' unshuffle(A[0],A[1],M)',' if Edges is not None:\n'+''.join(' '+l+'\n' for l in rope.splitlines())+' if OVERLAP and OPT_PRE:prefetch_bytes(PRE_W,24*16384*4)\n unshuffle(A[0],A[1],M)')
s=s.replace('M,ML,MV,FP32IN)','M,ML,MV,FP32IN,Edges)')
s=s.replace(' sync(A[29],A[30],256,3)\n if STOP==1:', ' if Edges is not None and OPT_PRE:\n  publish(Edges+257,p<M)\n  if OVERLAP and RATIO!=0:\n   prefetch_bytes(MW,(2048 if RATIO==4 else 1024)*4096*2)\n   if RATIO==4:prefetch_bytes(IW,512*4096*2)\n  consume(Edges+257,A[30],M,p<192)\n else:sync(A[29],A[30],256,3)\n if STOP==1:')
s=s.replace('  if OPT_COMP and RATIO!=128:', '  if Edges is not None:\n   edge(Edges+2*257,A[30],N//16,p<N//16,p<(2*M if RATIO==4 else 4*M))\n  elif OPT_COMP and RATIO!=128:')
s=s.replace('   sync(A[29],A[30],256,3)\n  if DBG is not None and p==0:gl.store(DBG+1', '   if Edges is not None:edge(Edges+3*257,A[30],M*4,p<M*4,p<M)\n   else:sync(A[29],A[30],256,3)\n  if DBG is not None and p==0:gl.store(DBG+1')
s=s.replace('  sync(A[29],A[30],256,3)\n if DBG is not None and p==0:gl.store(DBG+2', '  if Edges is not None:publish(Edges+4*257,p<(M*4 if RATIO==4 else M))\n  else:sync(A[29],A[30],256,3)\n if DBG is not None and p==0:gl.store(DBG+2')
s=s.replace('PREFETCH_STYLE=1)','PREFETCH_STYLE=1,Edges=None if Edges is None else Edges+5*257,OVERLAP=OVERLAP)')
s=s.replace(' attention_paged(A[28]', ' if Edges is not None and RATIO!=0:consume(Edges+4*257,A[30],M*4 if RATIO==4 else M,True)\n if OVERLAP:\n  prefetch_bytes(A[40],8192*4096)\n  prefetch_bytes(A[41],8192*4096//128//128)\n attention_paged(A[28]')
s=s.replace('STOP=STOP-10 if STOP>10 else 0)','STOP=STOP-10 if STOP>10 else 0,OVERLAP=OVERLAP)')
s=s.replace(' output_body(A[38]', ' if OVERLAP:\n  prefetch_bytes(MHC[0],24*16384*4)\n  prefetch_bytes(F[5],4096*2)\n output_body(A[38]')
s=s.replace('M,ML,MV)\n  sync(F[-2],F[-1],256,3)', 'M,ML,MV,None if Edges is None else Edges+7*257)\n  if Edges is not None:edge(Edges+8*257,F[-1],M,p<M,True)\n  else:sync(F[-2],F[-1],256,3)')
s=s.replace('FP32OUT=FP32OUT)','FP32OUT=FP32OUT,Edges=Edges,OVERLAP=OVERLAP)')
p.write_text(s)
# Output weights already in L2 before preceding compute finishes.
edit('output_batch.py',lambda s:s.replace('STOP:gl.constexpr=0):','STOP:gl.constexpr=0,OVERLAP:gl.constexpr=False):').replace(' inverse_quant(O,Cos,Sin,XQ,XS,M)',' if OVERLAP:\n  prefetch_bytes(OB,4096*8192)\n  prefetch_bytes(OBS,4096*8192//128//128)\n inverse_quant(O,Cos,Sin,XQ,XS,M)'))
# Wire runtime. Keep rollout switches off until full checks pass.
edit('runtime_batch.py',lambda s:s.replace("key=(m,r,kwargs['OPT_MHC'],kwargs['OPT_COMP'],kwargs['OPT_PRE'])", "kwargs.update(Edges=s.edges if os.getenv('ATOM_DSV4_EDGE_SYNC','0')=='1' else None,OVERLAP=os.getenv('ATOM_DSV4_OVERLAP','0')=='1')\n  key=(m,r,kwargs['OPT_MHC'],kwargs['OPT_COMP'],kwargs['OPT_PRE'],kwargs['Edges'] is not None,kwargs['OVERLAP'])"))
