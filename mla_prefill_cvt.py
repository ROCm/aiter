import sys
import re

def add_kargs(k_out, name, k_type, offset, rw_type):
    if k_type == 0:
        arg_line = '    - {.name: ' + name + ', .size: ' + '4' + ', .offset: ' + str(offset) + \
                   ', .value_kind: by_value, .value_type: i32}'
        k_out.write(arg_line + '\n')
        offset += 4
        for i in range(0, 3):
            arg_line = '    - {.name: ' + 'pad' + ', .size: ' + '4' + ', .offset: ' + str(offset) + \
                       ', .value_kind: by_value, .value_type: i32}'
            k_out.write(arg_line + '\n')
            offset += 4
    elif k_type == 1:
        rw_str = 'read_only'
        if rw_type == 1:
            rw_str = 'read_write'

        arg_line = '    - {.name: ' + name + ', .size: ' + '8' + ', .offset: ' + str(offset) + \
                   ', .value_kind: global_buffer, .address_space: global, .actual_access: ' + rw_str + '}'
        k_out.write(arg_line + '\n')
        offset += 8
        arg_line = '    - {.name: ' + 'pad' + ', .size: ' + '8' + ', .offset: ' + str(offset) + \
                   ', .value_kind: by_value, .value_type: i32}'
        k_out.write(arg_line + '\n')
        offset += 8
    return offset

if len(sys.argv) < 4:
    print("Usage: ",sys.argv[0],"origin_sp3 raw_sp3 target_s fn_name");
    sys.exit()

orig_filename = sys.argv[1]
i_filename = sys.argv[2]
o_filename = sys.argv[3]

# get fn name from sys.argv, if not get default
fn_name = sys.argv[4] if len(sys.argv) >= 5 else "mla_prefill_kernel_func"
    
# print(filename)
orig_sp3_in = open(orig_filename, 'r')
sp3_in = open(i_filename, 'r')
hsa_out = open(o_filename, 'w')

hsa_out.write('.text' + '\n')
hsa_out.write('.global ' + fn_name + '\n')
hsa_out.write('.p2align 8' + '\n')
hsa_out.write('.type ' + fn_name + ',@function' + '\n')

hsa_out.write("//Config:"+ orig_filename.replace(".sp3","").replace("_",";")+"\n")
foundauthor = 0
for line in orig_sp3_in:
    match = re.match('.*Author.*', line)
    if match != None:
        hsa_out.write(line)
        foundauthor=1
orig_sp3_in.close()
if foundauthor==0:
    print("No author found in sp3 file, please check and rerun!\n")
    sys.exit()

hsa_out.write( '\n'+ fn_name + ':' + '\n')
for line in sp3_in:
    # print(line)
    is_find = line.find("shader main")
    if is_find != -1:
        continue

    is_find = line.find("asic")
    if is_find != -1:
        continue

    match = re.match('.*buffer_load_dword.*lds.*', line)
    if match != None:
        line = line.replace('v0,', '')
        line = line.replace('v[0:3],', '')

    match = re.match('.*global_atomic_pk_add_bf16.*', line)
    if match != None:
        line = line.replace('v0,', '')
        mgrp = re.match(r'.*v\[(\d+)\:\d+\].*', line)
        #print(mgrp.group(1))
        #line = line.replace('v[\d+:\d+]', 'v'+mgrp.group(1))
        line = re.sub(r'v\[\d+\:\d+\]', 'v'+mgrp.group(1), line)

    match = re.match('.*global_load_dword.*', line)
    if match != None:
        mgrp = re.match(r'.*v\[(\d+)\:\d+\].*', line)
        #print(mgrp.group(1))
        #line = line.replace('v[\d+:\d+]', 'v'+mgrp.group(1))
        line = re.sub(r'v\[\d+\:\d+\]', 'v'+mgrp.group(1), line)

    match = re.match('end', line)
    if match != None:
        continue

    hsa_out.write(line)
    # hsa_out.write(line[:-1]+' '+hex(NA)+' bank:'+str(bank)+' row:'+str(row)+' pc:'+str(PC)+'\n')
    # print(hex(NA))

hsa_out.write( '\n' + '.rodata' + '\n')
hsa_out.write('.p2align 6' + '\n')
hsa_out.write('.amdhsa_kernel ' + fn_name + '\n')
hsa_out.write('    .amdhsa_group_segment_fixed_size 163840' + '\n')
hsa_out.write('    .amdhsa_user_sgpr_kernarg_segment_ptr 1' + '\n')
hsa_out.write('    .amdhsa_system_sgpr_workgroup_id_x 1' + '\n')
hsa_out.write('    .amdhsa_system_sgpr_workgroup_id_y 1' + '\n')
hsa_out.write('    .amdhsa_system_sgpr_workgroup_id_z 1' + '\n')
hsa_out.write('    .amdhsa_system_vgpr_workitem_id 0' + '\n')
hsa_out.write('    .amdhsa_next_free_vgpr 256 ' + '\n')
hsa_out.write('    .amdhsa_next_free_sgpr 96' + '\n')
hsa_out.write('    .amdhsa_accum_offset 128' + '\n')
hsa_out.write('    .amdhsa_ieee_mode 0' + '\n')
hsa_out.write('    .amdhsa_dx10_clamp 0' + '\n')
hsa_out.write('.end_amdhsa_kernel' + '\n')

hsa_out.write( '\n'+'.amdgpu_metadata' + '\n')
hsa_out.write('---' + '\n')
hsa_out.write('amdhsa.version: [ 1, 0 ]' + '\n')
hsa_out.write('amdhsa.kernels:' + '\n')
hsa_out.write('  - .name: ' + fn_name + '\n')
hsa_out.write('    .symbol: ' + fn_name + '.kd' + '\n')
hsa_out.write('    .sgpr_count: 96' + '\n')
hsa_out.write('    .vgpr_count: 256' + '\n')
hsa_out.write('    .kernarg_segment_align: 4' + '\n')
hsa_out.write('    .kernarg_segment_size: 304' + '\n')
hsa_out.write('    .group_segment_fixed_size: 163840' + '\n')
hsa_out.write('    .private_segment_fixed_size: 0' + '\n')
hsa_out.write('    .wavefront_size: 64' + '\n')
hsa_out.write('    .reqd_workgroup_size : [512, 1, 1]' + '\n')
hsa_out.write('    .max_flat_workgroup_size: 512' + '\n')
hsa_out.write('    .args:' + '\n')


arg_offset = 0
arg_offset = add_kargs(hsa_out, "ptr_Q", 1, arg_offset, 0)
arg_offset = add_kargs(hsa_out, "ptr_K", 1, arg_offset, 0)
arg_offset = add_kargs(hsa_out, "ptr_V", 1, arg_offset, 0)
arg_offset = add_kargs(hsa_out, "ptr_O", 1, arg_offset, 1)
arg_offset = add_kargs(hsa_out, "ptr_PartialO", 1, arg_offset, 1)
arg_offset = add_kargs(hsa_out, "ptr_PartialLSE", 1, arg_offset, 1)
arg_offset = add_kargs(hsa_out, "ptr_WorkIndptr", 1, arg_offset, 0)
arg_offset = add_kargs(hsa_out, "ptr_WorkInfo", 1, arg_offset, 0)
arg_offset = add_kargs(hsa_out, "ptr_QOIndptr", 1, arg_offset, 0)
arg_offset = add_kargs(hsa_out, "ptr_KVIndptr", 1, arg_offset, 0)
arg_offset = add_kargs(hsa_out, "ptr_KVPageIndices", 1, arg_offset, 0)
arg_offset = add_kargs(hsa_out, "ptr_QScale", 1, arg_offset, 0)
arg_offset = add_kargs(hsa_out, "ptr_KScale", 1, arg_offset, 0)
arg_offset = add_kargs(hsa_out, "ptr_VScale", 1, arg_offset, 0)

arg_offset = add_kargs(hsa_out, "scalar", 0, arg_offset, 0)
arg_offset = add_kargs(hsa_out, "num_q_tokens", 0, arg_offset, 0)
arg_offset = add_kargs(hsa_out, "num_head_q", 0, arg_offset, 0) 
arg_offset = add_kargs(hsa_out, "num_page", 0, arg_offset, 0)
arg_offset = add_kargs(hsa_out, "num_used_page", 0, arg_offset, 0)


hsa_out.write('...' + '\n')
hsa_out.write('.end_amdgpu_metadata' + '\n')

sp3_in.close()
hsa_out.close()
