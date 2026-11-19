# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 MONO code-generation policy: DEV reads, DEV+HT writes, full EXEC.

This is an explicit, hashed compiler stage. It changes memory control operands
and the standalone epilogue only; arithmetic, masks and task mapping are retained.
Native modules are never selected. The final machine code remains the authority.
"""
import contextlib,functools,hashlib,json,os,re,shutil,struct,subprocess,tempfile
from pathlib import Path
from .elf import Elf,body_instructions
LLVM=Path('/opt/rocm/llvm/bin')
POLICY_ID=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def rewrite_instruction(text):
    op=text.split()[0] if text.split() else ''
    if op.startswith(('global_load','buffer_load','flat_load','tensor_load','global_prefetch')):
        clean=re.sub(r'\s+(scope|th):\S+','',text)
        return clean+' scope:SCOPE_DEV'
    if op.startswith(('global_store','buffer_store','flat_store','tensor_store')):
        clean=re.sub(r'\s+(scope|th):\S+','',text)
        return clean+' scope:SCOPE_DEV th:TH_STORE_HT'
    if text=='s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)':
        # s_endpgm releases registers for standalone execution. A callable body
        # must retain VGPRs; replace the early release with the required EXEC.
        return 's_mov_b32 exec_lo, -1'
    return text

def rewrite_assembly(src):
    rows=[];changed=[];exits=0
    for line in src.splitlines():
        text=line.strip()
        if text.startswith(('global_','buffer_','flat_','tensor_','s_sendmsg')):
            result=rewrite_instruction(text)
            if result!=text:
                changed.append([text,result])
                line='\t'+result
                if text=='s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)':
                    line='\ts_wait_xcnt 0\n'+line
        if text=='s_endpgm':
            exits+=1
            # Small shape specializations can omit MSG_DEALLOC_VGPRS.
            # Restore their EXEC explicitly as well.
            if not rows or not rows[-1].rstrip().endswith('s_mov_b32 exec_lo, -1'):
                rows.append('\ts_wait_xcnt 0\n\ts_mov_b32 exec_lo, -1')
        rows.append(line)
    if not exits:raise ValueError('MONO: no recognized exit to legalize')
    return '\n'.join(rows)+'\n'

def compile_hook(*args):
    if not args:return ('mono-standard-'+POLICY_ID,POLICY_ID)
    backend,stages,options,language,_=args
    if not any(k=='mono-kernel-contract' and v==POLICY_ID for k,v in options.llvm_fn_attrs):return
    old=stages['amdgcn']
    def assembly(src,metadata):
        result=old(src,metadata)
        if not metadata['name'].startswith('_rope_window'):raise ValueError('MONO stage selected unexpected kernel')
        return rewrite_assembly(result)
    stages['amdgcn']=assembly

def compiled_a05(fn):
    @functools.wraps(fn)
    def invoke(*args,**kwargs):
        from triton import knobs
        previous=knobs.runtime.add_stages_inspection_hook
        if previous not in (None,compile_hook):raise RuntimeError('Another Triton pipeline hook is active')
        knobs.runtime.add_stages_inspection_hook=compile_hook
        try:return fn(*args,**kwargs)
        finally:knobs.runtime.add_stages_inspection_hook=previous
    return invoke

def assemble_instructions(lines):
    cmd=[str(LLVM/'llvm-mc'),'-triple=amdgcn-amd-amdhsa','-mcpu=gfx1250','-show-encoding']
    r=subprocess.run(cmd,input='\n'.join(lines)+'\n',text=True,capture_output=True,check=True)
    enc=[]
    for l in r.stdout.splitlines():
        if 'encoding: [' in l:
            values=l.split('encoding: [')[1].split(']')[0].split(',')
            enc.append(bytes(int(v,16) for v in values))
    if len(enc)!=len(lines):raise ValueError('assembler instruction count mismatch')
    return enc

def legalize_hsaco(src,dst,evidence):
    elf=Elf(src);blob=bytearray(elf.data);changes=[];preserved=[]
    symbols=[n for n in elf.symbols if 'inverse_rope_group_quant_persistent_kernel' in n and not n.endswith('.kd')]
    if len(symbols)!=2:raise ValueError(f'Expected two A07 persistent specializations, got {symbols}')
    for symbol in symbols:
        body=elf.symbol_bytes(symbol);section,value,size=elf.symbols[symbol]
        sec=elf.sections[section];base=sec[4]+value-sec[3]
        dis=subprocess.check_output([str(LLVM/'llvm-objdump'),'-d','--mcpu=gfx1250','--disassemble-symbols='+symbol,str(src)],text=True)
        ins=body_instructions(dis,value,size)
        if b''.join(x[2] for x in ins)!=body:raise ValueError('incomplete symbol disassembly')
        edits=[(address,text,raw,rewrite_instruction(text)) for address,text,raw in ins if rewrite_instruction(text)!=text]
        replacements=assemble_instructions([x[3] for x in edits])
        exits=[i for i,x in enumerate(ins) if x[1]=='s_endpgm']
        if not exits or any(i==0 or ins[i-1][1]!='s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)' for i in exits):
            raise ValueError('Every A07 exit must have a recognized epilogue')
        if sum(x[1]=='s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)' for x in edits)!=len(exits):
            raise ValueError('Unpaired A07 epilogue')
        for (address,text,raw,rewritten),new in zip(edits,replacements):
            if len(new)!=len(raw):raise ValueError('MONO instruction width changed')
            offset=base+address-value
            assert blob[offset:offset+len(raw)]==raw
            blob[offset:offset+len(raw)]=new
            changes.append(dict(symbol=symbol,address=address,original=text,replacement=rewritten,before=raw.hex(),after=new.hex()))
        preserved.append(dict(symbol=symbol,bytes=size,arithmetic_and_control_flow_unchanged=True))
    Path(dst).write_bytes(blob)
    evidence.mkdir(parents=True,exist_ok=True)
    (evidence/'policy.json').write_text(json.dumps(dict(policy_id=POLICY_ID,input_sha256=hashlib.sha256(elf.data).hexdigest(),output_sha256=hashlib.sha256(blob).hexdigest(),changes=changes,symbols=preserved),indent=2)+'\n')
    return bytes(blob)

def legalize_library(library,evidence):
    evidence.mkdir(parents=True,exist_ok=True)
    src=evidence/'compiler-library.so';shutil.copy2(library,src)
    fat=evidence/'original.fatbin'
    subprocess.run([str(LLVM/'llvm-objcopy'),'--dump-section',f'.hip_fatbin={fat}',str(src),str(evidence/'host-copy.so')],check=True)
    bundler=str(LLVM/'clang-offload-bundler')
    targets=subprocess.check_output([bundler,'--list','--type=o',f'--input={fat}'],text=True).splitlines()
    selected=[x for x in targets if x.startswith('hip') and x.endswith('gfx1250')]
    if len(selected)!=1:raise ValueError('Expected one gfx1250 code object')
    hsaco=evidence/'compiler.hsaco'
    subprocess.run([bundler,'--unbundle','--type=o','--targets='+selected[0],f'--input={fat}',f'--output={hsaco}'],check=True)
    transformed=legalize_hsaco(hsaco,evidence/'mono.hsaco',evidence)
    old=hsaco.read_bytes();raw=fat.read_bytes()
    if raw.count(old)!=1:raise ValueError('Code object not uniquely embedded in uncompressed fatbin')
    (evidence/'mono.fatbin').write_bytes(raw.replace(old,transformed))
    temp=evidence/'mono-library.so'
    subprocess.run([str(LLVM/'llvm-objcopy'),'--update-section',f'.hip_fatbin={evidence / "mono.fatbin"}',str(src),str(temp)],check=True)
    shutil.copy2(temp,library)
