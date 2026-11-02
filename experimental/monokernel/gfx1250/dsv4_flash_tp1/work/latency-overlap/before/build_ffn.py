from pathlib import Path
s=Path('modelwide/ffn_precise.py').read_text();s=s[:s.index('@g.jit\ndef ffn_body')]
s=s.replace('from mhc_precise import candidate_body as mhc_body','from mhc_batch import mhc_batch as mhc_body').replace('from post_precise import post_body','from post_batch import post_batch as post_body')
s=s.replace('STRIDE:gl.constexpr=1):','STRIDE:gl.constexpr=1,M:gl.constexpr=1):')
s=s.replace(' p=gl.program_id(0)\n if p<N//GROUP:', ' p=gl.program_id(0)\n for task in range(p,M*(N//GROUP),256):\n  row=task//(N//GROUP);pcol=task%(N//GROUP)')
s=s.replace('make_tensor_descriptor(X,[1,N]','make_tensor_descriptor(X+row*N,[1,N]').replace('[0,p*GROUP]','[0,pcol*GROUP]').replace('Q+p*GROUP+i','Q+row*N+pcol*GROUP+i').replace('S+p*STRIDE','S+(row*(N//GROUP)+pcol if STRIDE==1 else pcol*STRIDE+row)')
s=s.replace('def router_body(X,W,Y):','def router_body(X,W,Y,M:gl.constexpr):').replace('router_gemm(X,W,Y,None,1,256','router_gemm(X,W,Y,None,M,256')
s=s.replace('BN:gl.constexpr):','BN:gl.constexpr,M:gl.constexpr):').replace('shared_gemm(X,W,Y,XS,WS,1,4096','shared_gemm(X,W,Y,XS,WS,M,4096')
s=s.replace('from routing import route_body','from routing_batch import route_body')
s=s.replace('def activation_quant(GU,A,AQ,AS):','def activation_quant(GU,A,AQ,AS,M:gl.constexpr):').replace(' if p<16:\n  SL:', ' if p<M*16:\n  row=p//16;p=p%16\n  SL:')
s=s.replace('make_tensor_descriptor(GU,[1,4096]','make_tensor_descriptor(GU+row*4096,[1,4096]').replace('A+p*128+i','A+row*2048+p*128+i').replace('AQ+p*128+i','AQ+row*2048+p*128+i').replace('AS+p*8','AS+p*8+row')
Path('batch-full/ffn_batch.py').write_text(s)
