"""Fail-closed audit of the actual MONO op instructions, returns and resources."""
import argparse,hashlib,json,re,struct,subprocess
from pathlib import Path
from aiter.ops.mono_standard.elf import Elf,body_instructions
from aiter.ops.mono_standard.elf import REGISTER
p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stages',nargs='+',default=['a05','a07']);a=p.parse_args()
rows=[]
for stage in a.stages:
 audit=json.loads(Path(str(a.suite)+'-audit-'+stage,'audit.json').read_text());assert audit['passed']
 for obj in audit['objects']:
  path=Path(obj['binary']);elf=Elf(path)
  assert hashlib.sha256(path.read_bytes()).hexdigest()==obj['sha256']
  for k in obj['kernels']:
   symbol=k['.name'];_,start,size=elf.symbols[symbol]
   dis=subprocess.check_output(['/opt/rocm/llvm/bin/llvm-objdump','-d','--mcpu=gfx1250','--disassemble-symbols='+symbol,str(path)],text=True)
   ins=body_instructions(dis,start,size);assert b''.join(i[2] for i in ins)==elf.symbol_bytes(symbol)
   loads=[];stores=[];scalar_loads=[];exits=[];branches=[];addresses={i[0] for i in ins}
   for i,(addr,text,raw) in enumerate(ins):
    op=text.split()[0]
    assert '<unknown>' not in text and not op.startswith('.')
    for single,first,last in REGISTER.findall(text):
     lo,hi=(int(single),int(single)) if single else (int(first),int(last))
     assert not (lo<=105 and hi>=96), text
    assert 'MSG_DEALLOC_VGPRS' not in text
    assert not re.match(r's_(getpc|set_pc|setpc|swappc|call|rfe)',op)
    if op.startswith(('global_','buffer_','flat_','tensor_')):
     assert any(x in op for x in ('load','store','prefetch')),text
     assert 'scope:SCOPE_DEV' in text,text
     if 'store' in op:
      assert 'th:TH_STORE_HT' in text,text
      stores.append(text)
     else:
      assert 'th:' not in text or 'th:TH_LOAD_RT' in text,text
      loads.append(text)
    if op.startswith(('s_load','s_buffer_load')):scalar_loads.append(dict(offset=addr-start,instruction=text))
    if op=='s_endpgm':
     assert i>0 and ins[i-1][1]=='s_mov_b32 exec_lo, -1'
     exits.append(addr)
    if re.match(r's_(?:cbranch_\w+|branch)$',op):
     target=addr+4+struct.unpack('<h',raw[:2])[0]*4
     assert target in addresses
     # Cannot bypass EXEC restoration by branching directly to endpgm.
     assert next(x[1] for x in ins if x[0]==target)!='s_endpgm'
     branches.append([addr,target])
   assert loads and stores and exits
   assert k['.wavefront_size']==32 and k['.private_segment_fixed_size']==0
   rows.append(dict(stage=stage.upper(),symbol=symbol,binary=str(path),sha256=obj['sha256'],loads=len(loads),stores=len(stores),scalar_loads=scalar_loads,exits=len(exits),sgpr_count=k['.sgpr_count'],vgpr_count=k['.vgpr_count'],lds_static_bytes=k['.group_segment_fixed_size'],scratch_bytes=k['.private_segment_fixed_size'],all_loads_DEV=True,all_stores_DEV_HT=True,all_exits_full_EXEC=True,no_s96_s105_references=True,internal_branches=len(branches),passed=True))
a.output.write_text(json.dumps(dict(passed=True,objects=rows,scope='Selected runtime op bodies. Scalar loads are separately enumerated: the C16 review maps them to immutable positions/batch_ids/cu_seqlens_q/slots supplied before launch, not inter-op activations. See scalar-input-review.json. The DEV assertion covers VMEM/TDM activation instructions; it does not assert DEV operands on SMEM. Device calls, live SGPR preservation, all-TG join and payload visibility require run_contract hardware results.'),indent=2)+'\n')
print(json.dumps(rows,indent=2))
