"""Export a native Triton LLVM kernel as an inline device stage.

Only ABI, logical program IDs, LDS base and CTA barrier scope are adapted.
The input must be the compiled specialization actually used by the native call.
"""
import re, json, hashlib, subprocess
from pathlib import Path

def split_args(s):
    depth=0; out=[]; start=0
    for i,c in enumerate(s):
        depth += (c in '(<[')-(c in ')>]')
        if c==',' and depth==0: out.append(s[start:i].strip()); start=i+1
    if s[start:].strip(): out.append(s[start:].strip())
    return out

def export(llir, output, symbol, num_warps, physical_warps=4, preserve_fp_fusion=False):
    source=Path(llir).read_text(); text=source
    # Named SSA values allow an ABI wrapper without renumbering original values.
    text=re.sub(r'%(\d+)',r'%v\1',text)
    text=re.sub(r'^(\d+):',r'v\1:',text,flags=re.M)
    m=re.search(r'define amdgpu_kernel void @([^ (]+)\((.*)\).*?\{\n',text)
    if not m: raise ValueError('Expected one native AMDGPU kernel')
    end=text.index('\n}',m.end())
    args=split_args(m[2]); abi=[]; prep=[]; types=[]
    for i,a in enumerate(args):
        name=re.search(r'%[\w.]+$',a)[0]
        if a.startswith('ptr'):
            ty=re.match(r'ptr(?: addrspace\(\d+\))?',a)[0]
            abi.append(f'i64 %arg{i}'); prep.append(f'  {name} = inttoptr i64 %arg{i} to {ty}'); types.append('uint64')
        else:
            ty=a.split()[0]; abi.append(f'{ty} {name}'); types.append({'i32':'uint32','i64':'uint64','float':'float32','double':'float64'}[ty])
    abi += ['i32 %logical_x','i32 %logical_y','i32 %logical_z','i64 %lds_addr']
    types += ['uint32']*3+['uint64']
    prep.append('  %stage_lds = inttoptr i64 %lds_addr to ptr addrspace(3)')
    body=text[m.end():end]
    for axis in 'xyz':
        body=re.sub(r'(?:tail )?call i32 @llvm.amdgcn.workgroup.id.'+axis+r'\(\)',f'add i32 %logical_{axis}, 0',body)
    # All sequential stages share the parent's dynamic LDS arena at offset zero.
    # Keep constant TDM descriptors intact; their LDS addresses must stay constant.
    body=body.replace('@global_smem','@__dynamic_shared_0')
    # Native Triton enable_fp_fusion is a backend option, absent from plain
    # LLIR arithmetic flags. Preserve that observed policy only when requested.
    # No reassociation/fast-math permissions are added.
    if preserve_fp_fusion:
        body=re.sub(r'\b(fmul|fadd|fsub) (?!contract\b|fast\b)',r'\1 contract ',body)
    declarations='' 
    if num_warps != physical_warps:
        if num_warps==1:
            body=body.replace('@llvm.amdgcn.s.barrier()', '@llvm.amdgcn.wave.barrier()')
            if 'declare void @llvm.amdgcn.wave.barrier()' not in text:
                declarations+='\ndeclare void @llvm.amdgcn.wave.barrier()\n'
        else:
            body=body.replace('tail call void @llvm.amdgcn.s.barrier()', 'call void @native_bridge_subgroup_barrier()')
            body=body.replace('call void @llvm.amdgcn.s.barrier()', 'call void @native_bridge_subgroup_barrier()')
            declarations+=f'''
@native_qa_named_barrier = external addrspace(3) global target("amdgcn.named.barrier", 0)
declare void @llvm.amdgcn.s.barrier.join(ptr addrspace(3))
declare void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3), i32)
declare void @llvm.amdgcn.s.barrier.wait(i16 immarg)
define internal void @native_bridge_subgroup_barrier() alwaysinline {{
  call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) @native_qa_named_barrier)
  call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) @native_qa_named_barrier, i32 {num_warps})
  call void @llvm.amdgcn.s.barrier.wait(i16 1)
  ret void
}}
'''
    entry_id=max([int(n) for a in args for n in re.findall(r'%v(\d+)',a)],default=-1)+1
    entry_label=f'v{entry_id}'
    explicit=re.match(r'\s*([\w.]+):[^\n]*\n',body)
    if explicit:
        entry_label=explicit[1];body=body[explicit.end():]
    header=f'define void @{symbol}('+', '.join(abi)+f') alwaysinline {{\n{entry_label}:\n'
    # Native worker-specialized kernels may already declare these intrinsics.
    declarations='\n'.join(line for line in declarations.splitlines() if not (line.startswith('declare ') and re.search(r'@([^ (]+)',line)[0]+'(' in text))
    text=text[:m.start()]+header+'\n'.join(prep)+'\n'+body+text[end:]+'\n'+declarations+'\n'
    text=text.replace('@global_smem','@__dynamic_shared_0')
    # Original debug subprogram names the kernel ABI; discard debug metadata.
    text=re.sub(r',? !dbg !\d+','',text)
    text=re.sub(r'^!llvm.module.flags = .*\n','',text,flags=re.M)
    # Preserve MMRA fences, range and alias metadata: these affect native memory semantics.
    # The ROCm LLVM assembler predates some Triton debug fields. Remove the
    # debug metadata graph only; retain memory-model and scheduling metadata.
    defs=dict(re.findall(r'^!(\d+) = (.*)$',text,re.M))
    bad={k for k,v in defs.items() if '!DI' in v}
    while True:
        newer={k for k,v in defs.items() if any(r in bad for r in re.findall(r'!(\d+)',v))}
        if newer <= bad: break
        bad |= newer
    text='\n'.join(line for line in text.splitlines() if not (line.startswith('!') and any(r in bad for r in re.findall(r'!(\d+)',line))))+'\n'
    text=re.sub(r',? ![\w.]+ !(\d+)',lambda m:'' if m[1] in bad else m[0],text)
    text=re.sub(r'^target triple = .*$', 'target triple = "amdgcn-amd-amdhsa"',text,flags=re.M)
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    output.with_suffix('.ll').write_text(text)
    subprocess.run(['/opt/rocm/llvm/bin/llvm-as',str(output.with_suffix('.ll')),'-o',str(output)],check=True)
    manifest=dict(preserve_fp_fusion=preserve_fp_fusion,symbol=symbol,types=types,native_symbol=m[1],native_args=args,num_warps=num_warps,physical_warps=physical_warps,source=str(llir),source_sha256=hashlib.sha256(source.encode()).hexdigest(),bitcode=str(output))
    output.with_suffix('.json').write_text(json.dumps(manifest,indent=2))
    return manifest

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('llir');p.add_argument('output');p.add_argument('symbol');p.add_argument('--warps',type=int,required=True);p.add_argument('--physical-warps',type=int,default=4)
    a=p.parse_args();print(json.dumps(export(a.llir,a.output,a.symbol,a.warps,a.physical_warps),indent=2))
