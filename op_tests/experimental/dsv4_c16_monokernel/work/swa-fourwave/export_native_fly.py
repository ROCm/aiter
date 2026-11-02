"""Export captured FlyDSL native LLVM without altering arithmetic.

Dynamic Tensor metadata arguments are specialized to the captured shape and
outer strides; caller pointers and explicit strides remain runtime ABI arguments.
"""
import re,json,importlib.util,hashlib
from pathlib import Path
from bridge.export import export,split_args

def build(ll,args,directory,symbol,grid,threads,physical_warps=None):
 physical_warps=physical_warps or max(9,threads//32)
 p=Path(directory);s=Path(ll).read_text();match=re.search(r'define amdgpu_kernel void @([^ (]+)\((.*)\).*?\{\n',s);orig=split_args(match[2]);kept=[]
 prep=[];arg_index=-1
 for arg in orig:
  if arg.startswith('<{'):
   name=arg.rsplit(' ',1)[1];ty=arg.rsplit(' ',1)[0];t=args[arg_index]
   # Native FlyDSL Tensor metadata contains dynamic dimensions then outer
   # strides; contiguous innermost stride is compile-time one.
   groups=re.findall(r'<\{ ([^{}<>]+) \}>',ty)
   vals=[]
   for gi,g in enumerate(groups):
    tys=g.split(', ');vv=list(t.shape) if gi==0 else list(t.stride())[:-1]
    assert len(tys)==len(vv),(ty,t.shape,t.stride())
    vals.append('<{ '+g+' }> <{ '+', '.join(f'{tt} {v}' for tt,v in zip(tys,vv))+' }>')
   prep.append('  '+name+' = freeze '+ty+' <{ '+', '.join(vals)+' }>')
  else:kept.append(arg);arg_index+=1
 original_body=s[match.end():]
 s=s[:match.start(2)]+', '.join(kept)+s[match.end(2):match.end()]+'\n'.join(prep)+'\n'+original_body

 offset=0
 for line in list(s.splitlines()):
  m=re.match(r'(@__shared_alloc_\d+) = .*addrspace\(3\) global \[(\d+) x i8\]',line)
  if m:
   align=re.search(r'align (\d+)',line);alignment=int(align[1]) if align else 1;offset=((offset+alignment-1)//alignment)*alignment
   s=s.replace(line+'\n','').replace(m[1],f'getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 {offset})');offset+=int(m[2])
 if offset:s+='\n@__dynamic_shared_0 = external addrspace(3) global [0 x i8], align 16\n'
 # Canonicalize equivalent full-CTA signal/wait pairs for existing bridge scoping.
 # Surrounding workgroup release/acquire fences remain unchanged.
 s,pairs=re.subn(r'(?:tail )?call void @llvm\.amdgcn\.s\.barrier\.signal\(i32 -1\)[^\n]*\n\s*(?:tail )?call void @llvm\.amdgcn\.s\.barrier\.wait\(i16 -1\)[^\n]*', 'call void @llvm.amdgcn.s.barrier()',s)
 if pairs and 'declare void @llvm.amdgcn.s.barrier()' not in s:s+='\ndeclare void @llvm.amdgcn.s.barrier()\n'
 pre=p/(symbol+'.abi.ll');pre.write_text(s)
 manifest=export(pre,p/(symbol+'.bc'),symbol,threads//32,physical_warps=physical_warps)
 manifest.update(full_cta_barrier_pairs_canonicalized=pairs,native_flydsl_source=str(ll),native_flydsl_sha256=hashlib.sha256(Path(ll).read_bytes()).hexdigest(),changes=['Tensor shape/outer-stride metadata specialized to captured tensors','static LDS allocations relocated with original alignment into shared stage arena','logical program IDs explicit; native active waves preserved'],lds=offset)
 (p/(symbol+'.json')).write_text(json.dumps(manifest,indent=2))
 vals=[a.data_ptr() if hasattr(a,'data_ptr') else a for a in args];assert len(vals)==len(manifest['types'])-4,(len(vals),manifest['types'])
 names=[f'a{i}' for i in range(len(vals))];types=manifest['types'][:-4];ann={'uint64':'Uint64','uint32':'Uint32','float32':'Float32'};sig=','.join(f'{n}:fx.{ann[t]}' for n,t in zip(names,types));call=','.join(names)
 code=f'''import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.extern_link import BoundExternFunction
from native_gemm import _named_qa_barrier_pointer
B=BoundExternFunction({symbol!r},{manifest['types']!r},'void',bitcode_path={str(p/(symbol+'.bc'))!r})
@flyc.jit
def body({sig},logical:fx.Int32):
 B({call},fx.Uint32(logical%{grid[0]}),fx.Uint32((logical//{grid[0]})%{grid[1]}),fx.Uint32(logical//{grid[0]*grid[1]}),fx.Uint64(0))
@flyc.kernel(known_block_size=[{physical_warps*32},1,1])
def kernel({sig}):
 a=fx.SharedAllocator(static=False);a.allocate({max(offset,16)})
 _named_qa_barrier_pointer()
 if fx.Int32(fx.thread_idx.x)<{threads}:
  for logical in range(fx.Int32(fx.block_idx.x),{grid[0]*grid[1]*grid[2]},256):body({call},logical)
@flyc.jit
def launch({sig},stream:fx.Stream):kernel({call}).launch(grid=(256,1,1),block=({physical_warps*32},1,1),stream=stream)
'''
 out=p/(symbol+'_adapter.py');out.write_text(code);sp=importlib.util.spec_from_file_location(symbol+'_adapter',out);mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
 return dict(body=mod.body,module=mod,values=vals,types=types,threads=threads,total=grid[0]*grid[1]*grid[2],lds=offset)
