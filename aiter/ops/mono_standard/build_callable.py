# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Export exact MONO op bytes with an explicit SGPR return ABI and join harness.

Callable ABI: original kernel argument/preload and workitem registers on entry;
s100:s101 return PC; s96:s99/s102:s105 preserved. In particular the MONO s104/s105
contract is unchanged. The caller owns flags and waits. Only s_endpgm is replaced
by s_setpc; all computation/memory/EXEC-restoration bytes are copied verbatim.
"""
import argparse,copy,hashlib,json,re,struct,subprocess
from pathlib import Path
import yaml
from .elf import Elf,body_instructions
from .elf import REGISTER
LLVM=Path('/opt/rocm/llvm/bin')

def build(binary,symbol,output,corrupt=False):
    binary,output=Path(binary),Path(output);output.mkdir(parents=True,exist_ok=False)
    elf=Elf(binary);body=elf.symbol_bytes(symbol);desc=bytearray(elf.symbol_bytes(symbol+'.kd'))
    dis=subprocess.check_output([str(LLVM/'llvm-objdump'),'-d','--mcpu=gfx1250','--disassemble-symbols='+symbol,str(binary)],text=True)
    ins=body_instructions(dis,elf.symbols[symbol][1],len(body));assert b''.join(i[2] for i in ins)==body
    exits=[i for i,x in enumerate(ins) if x[1]=='s_endpgm']
    assert exits and all(i>0 and ins[i-1][1]=='s_mov_b32 exec_lo, -1' for i in exits)
    start=elf.symbols[symbol][1]
    addresses={x[0] for x in ins}
    for address,text,raw in ins:
        if re.match(r's_(?:cbranch_\w+|branch)\s',text):
            assert address+4+struct.unpack('<h',raw[:2])[0]*4 in addresses
    for addr,text,raw in ins:
        for single,first,last in REGISTER.findall(text):
            lo,hi=(int(single),int(single)) if single else (int(first),int(last))
            if lo<=105 and hi>=96:raise ValueError('callee uses caller-reserved SGPRs')
        if re.search(r'\bs_(getpc|set_pc|setpc|swappc|call|rfe)',text):raise ValueError('nonrelocatable callee control flow')
    notes=subprocess.check_output([str(LLVM/'llvm-readelf'),'--notes',str(binary)],text=True)
    meta=yaml.safe_load(notes[notes.index('amdhsa.kernels:'):]);k=copy.deepcopy(next(k for k in meta['amdhsa.kernels'] if k['.name']==symbol))
    assert len(desc)==64 and k['.kernarg_segment_size']==120 and k['.wavefront_size']==32
    struct.pack_into('<I',desc,8,160)
    k.update({'.name':'mono_harness','.symbol':'mono_harness.kd','.kernarg_segment_size':160,'.sgpr_count':106})
    k['.args'] += [{'.offset':off,'.size':size,'.value_kind':'global_buffer' if size==8 else 'by_value',**({'.address_space':'global'} if size==8 else {})} for off,size in [(120,8),(128,8),(136,4),(140,4),(144,8),(152,4),(156,4)]]
    meta['amdhsa.kernels']=[k]
    # All original branches keep their offsets. Each exit has the same width.
    from .codegen import assemble_instructions
    patched=bytearray(body)
    return_bytes=assemble_instructions(['s_setpc_b64 s[100:101]'])[0]
    for i in exits:
        offset=ins[i][0]-start
        replacement=assemble_instructions([f's_branch {(len(body)-offset-4)//4}'])[0] if corrupt else return_bytes
        assert len(replacement)==len(ins[i][2])==4
        patched[offset:offset+4]=replacement
    (output/'body.bin').write_bytes(patched)
    call_src=f'''.text
.globl mono_op
.protected mono_op
.type mono_op,@function
.p2align 8
mono_op:
.incbin "{output/'body.bin'}"
'''
    if corrupt:call_src+='s_xor_b32 s104, s104, 1\ns_xor_b32 s105, s105, 2\ns_mov_b32 exec_lo, 0\n'
    if corrupt:call_src+='s_setpc_b64 s[100:101]\n'
    call_src+='.size mono_op, .-mono_op\n'
    (output/'mono_op.s').write_text(call_src)
    subprocess.run([str(LLVM/'clang'),'-x','assembler','-target','amdgcn-amd-amdhsa','-mcpu=gfx1250','-c',str(output/'mono_op.s'),'-o',str(output/'mono_op.o')],check=True)
    asm='''.text
.globl mono_harness
.protected mono_harness
.type mono_harness,@function
.p2align 8
mono_harness:
    s_mov_b64 s[96:97], s[0:1]
    v_readfirstlane_b32 s98, v0
    s_lshr_b32 s98, s98, 5
    s_and_b32 s99, ttmp7, 0xffff
    s_lshl_b32 s99, s99, 4
    s_add_co_u32 s99, s99, ttmp9
    s_load_b64 s[104:105], s[96:97], 136
    s_load_b32 s102, s[96:97], 156
    s_wait_kmcnt 0
    s_cmp_eq_u32 s99, 255
    s_cbranch_scc0 .Lcall
    s_cmp_eq_u32 s102, 0
    s_cbranch_scc1 .Lcall
.Ldelay:
    s_sleep 1
    s_sub_co_u32 s102, s102, 1
    s_cmp_lg_u32 s102, 0
    s_cbranch_scc1 .Ldelay
.Lcall:
    s_call_b64 s[100:101], mono_op
    s_mov_b32 s16, s104
    s_mov_b32 s17, s105
    s_mov_b32 s18, exec_lo
    s_wait_xcnt 0
    s_mov_b32 exec_lo, -1
    s_load_b64 s[2:3], s[96:97], 120
    s_load_b64 s[4:5], s[96:97], 128
    s_load_b64 s[6:7], s[96:97], 144
    s_load_b32 s8, s[96:97], 152
    s_load_b32 s104, s[96:97], 136
    s_wait_kmcnt 0
    s_add_co_u32 s19, s104, 8
    s_mov_b32 s21, 0
.Ljoin:
    s_wait_tensorcnt 0
    s_wait_storecnt 0
    s_wait_xcnt 0
    s_barrier_signal -1
    s_barrier_wait -1
    s_cmp_eq_u32 s98, 0
    s_cbranch_scc0 .Lpoll_setup
    s_mov_b32 exec_lo, 1
    s_lshl_b32 s10, s99, 2
    v_mov_b32 v0, s10
    v_mov_b32 v1, s104
    s_wait_alu 0
    global_store_b32 v0, v1, s[2:3] scope:SCOPE_DEV th:TH_STORE_HT
.Lpoll_setup:
    s_wait_xcnt 0
    s_mov_b32 exec_lo, -1
    v_mbcnt_lo_u32_b32 v1, -1, 0
    s_wait_alu 0
    v_lshlrev_b32 v0, 3, v1
    s_lshl_b32 s10, s98, 8
    s_wait_alu 0
    v_add_nc_u32 v0, s10, v0
    s_wait_alu 0
    s_mov_b32 s20, 100000
.Lpoll:
    global_load_b64 v[4:5], v0, s[2:3] scope:SCOPE_DEV
    s_wait_loadcnt 0
    v_min_u32 v6, v4, v5
    s_wait_alu 0
    v_cmpx_lt_u32 v6, s104
    s_wait_alu 0
    s_cbranch_execz .Ljoined
    s_sub_co_u32 s20, s20, 1
    s_cmp_eq_u32 s20, 0
    s_cbranch_scc1 .Ltimeout
    s_sleep 1
    s_branch .Lpoll
.Ltimeout:
    s_mov_b32 s21, 1
.Ljoined:
    s_barrier_signal -1
    s_barrier_wait -1
    s_mov_b32 exec_lo, -1
    s_cmp_eq_u32 s104, s19
    s_cbranch_scc1 .Lregions
    s_add_co_u32 s104, s104, 1
    s_branch .Ljoin
.Lregions:
    s_cmp_eq_u32 s8, 0
    s_cbranch_scc1 .Lmonitor
.Lregion:
    s_load_b256 s[24:31], s[6:7], 0
    s_wait_kmcnt 0
    s_lshl_b32 s10, s99, 7
    s_lshl_b32 s11, s98, 5
    s_add_co_u32 s10, s10, s11
    v_mbcnt_lo_u32_b32 v8, -1, 0
    s_wait_alu 0
    v_add_nc_u32 v8, s10, v8
    s_wait_alu 0
    v_lshlrev_b32 v9, 2, v8
    s_wait_alu 0
.Lcopy:
    v_cmpx_lt_u32 v9, s28
    s_wait_alu 0
    s_cbranch_execz .Lnext_region
    global_load_b32 v11, v9, s[24:25] scope:SCOPE_DEV
    s_wait_loadcnt 0
    global_store_b32 v9, v11, s[26:27] scope:SCOPE_DEV th:TH_STORE_HT
    s_wait_storecnt 0
    v_add_nc_u32 v9, 0x20000, v9
    s_wait_alu 0
    s_branch .Lcopy
.Lnext_region:
    s_mov_b32 exec_lo, -1
    s_add_nc_u64 s[6:7], s[6:7], 32
    s_sub_co_u32 s8, s8, 1
    s_cmp_lg_u32 s8, 0
    s_cbranch_scc1 .Lregion
.Lmonitor:
    s_lshl_b32 s10, s99, 2
    s_add_co_u32 s10, s10, s98
    s_lshl_b32 s10, s10, 5
    s_mov_b32 s11, 0
    s_add_nc_u64 s[4:5], s[4:5], s[10:11]
    s_mov_b32 exec_lo, 1
    v_mov_b32 v0, 0
    v_mov_b32 v4, s16
    v_mov_b32 v5, s17
    v_mov_b32 v6, s18
    v_mov_b32 v7, 0xcafe
    v_mov_b32 v8, s104
    v_mov_b32 v9, s21
    v_mov_b32 v10, 1
    v_mov_b32 v11, 0
    s_wait_alu 0
    global_store_b128 v0, v[4:7], s[4:5] scope:SCOPE_DEV th:TH_STORE_HT
    global_store_b128 v0, v[8:11], s[4:5] offset:16 scope:SCOPE_DEV th:TH_STORE_HT
    s_wait_storecnt 0
    s_endpgm
.size mono_harness, .-mono_harness
'''+call_src
    asm+='\n.section .rodata,"a",@progbits\n.p2align 6\n.globl mono_harness.kd\n.type mono_harness.kd,@object\nmono_harness.kd:\n'
    bs=lambda b:'.byte '+','.join(map(str,b))+'\n'
    asm+=bs(desc[:16])+'.quad mono_harness-mono_harness.kd\n'+bs(desc[24:])+'.size mono_harness.kd, 64\n.amdgpu_metadata\n---\n'+yaml.safe_dump(meta,sort_keys=False)+'...\n.end_amdgpu_metadata\n'
    (output/'harness.s').write_text(asm)
    subprocess.run([str(LLVM/'clang'),'-x','assembler','-target','amdgcn-amd-amdhsa','-mcpu=gfx1250','-c',str(output/'harness.s'),'-o',str(output/'harness.o')],check=True)
    subprocess.run([str(LLVM/'ld.lld'),'-shared',str(output/'harness.o'),'-o',str(output/'harness.hsaco')],check=True)
    final=Elf(output/'harness.hsaco');assert final.symbol_bytes('mono_op')[:len(body)]==patched
    (output/'harness.disasm').write_text(subprocess.check_output([str(LLVM/'llvm-objdump'),'-d','--mcpu=gfx1250',str(output/'harness.hsaco')],text=True))
    r=dict(binary=str(binary),original_sha256=hashlib.sha256(elf.data).hexdigest(),symbol=symbol,callable_object=str(output/'mono_op.o'),harness=str(output/'harness.hsaco'),preserved_compute_bytes=len(body)-4*len(exits),return_sites=len(exits),compute_bytes_equal=True,full_exec_instruction_before_return=True,return_PC='s100:s101',caller_reserved_sgprs='s96:s105',joins=9,flag_scope='DEV/HT store, DEV/RT load',bounded_poll_iterations=100000,corrupt_control=corrupt,scope='One exact operator callable followed by nine joins and DEV copying of all observable outputs; not a full A01-A10 model pipeline')
    (output/'build.json').write_text(json.dumps(r,indent=2)+'\n');return r
