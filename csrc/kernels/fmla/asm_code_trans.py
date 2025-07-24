
file_r = open("./fmla_gfx9_a16w16_qh16_m16x4_n16x1_coex0_mask1_split_r.inc", mode="r")
file_w = open("./fmla_gfx9_a16w16_qh16_m16x4_n16x1_coex0_mask1_split_w.inc", mode="w")

# acc_trans_table_stride_2 = {
#     f"acc[{i}:{i+1}]" : f"[%a{i}, %a{i+1}]" for i in range(0, 72, 2)
#     }
#
# acc_trans_table_stride_4 = {
#     f"acc[{i}:{i+3}]" : f"[%a{i}, %a{i+1}, %a{i+2}, %a{i+3}]" for i in range(0, 72, 4)
#     }
#
# v_trans_table_stride_1 = {
#     f"v{i+40}" : f"%v{i}" for i in range(0, 128)
#     }
# v_trans_table_stride_2 = {
#     f"v[{i+40}:{i+41}]" : f"[%v{i}, %v{i+1}]" for i in range(0, 128, 2)
#     }
# v_trans_table_stride_4 = {
#     f"v[{i+40}:{i+43}]" : f"[%v{i}, %v{i+1}, %v{i+2}, %v{i+3}]" for i in range(0, 128, 4)
#     }

o_regs_trans_table_stride_1 = {
    f"v{i+40}" : f"%[o_regs_{i}]" for i in range(0, 128)
    }
o_regs_trans_table_stride_2 = {
    f"v[{i+40}:{i+41}]" : f"[%[o_regs_{i}], %[o_regs_{i+1}]]" for i in range(0, 128, 2)
    }
o_regs_trans_table_stride_4 = {
    f"v[{i+40}:{i+43}]" : f"[%[o_regs_{i}], %[o_regs_{i+1}], %[o_regs_{i+2}], %[o_regs_{i+3}]]" for i in range(0, 128, 4)
    }

o_res_trans_table_stride_1_blank = {
    f"s{i} " : f"%[o_res_{i-8}] " for i in range(8, 12)
    }
o_res_trans_table_stride_1_comma = {
    f"s{i}," : f"%[o_res_{i-8}]," for i in range(8, 12)
    }
o_res_trans_table_stride_2 = {
    f"s[{i}:{i + 1}]" : f"[%[o_res_{i - 8}], %[o_res_{i - 8+1}]]" for i in range(8, 12, 2)
    }
o_res_trans_table_stride_4 = {
    f"s[{i}:{i + 3}]" : f"[%[o_res_{i - 8}], %[o_res_{i - 8 +1}], %[o_res_{i-8+2}], %[o_res_{i-8+3}]]" for i in range(8, 9)
    }

lse_res_trans_table_stride_1 = {
    f"s{i}" : f"%[lse_res_{i - 12}]" for i in range(12, 16)
    }
lse_res_trans_table_stride_2 = {
    f"s[{i}:{i + 1}]" : f"[%[lse_res_{i - 12}], %[lse_res_{i - 12+1}]]" for i in range(12, 16, 2)
    }
lse_res_trans_table_stride_4 = {
    f"s[{i}:{i + 3}]" : f"[%[lse_res_{i - 12}], %[lse_res_{i - 12+1}], %[lse_res_{i - 12+2}], %[lse_res_{i - 12+3}]]" for i in range(12, 13)
    }

q_res_trans_table_stride_1 = {
    f"s{i}" : f"%[q_res_{i - 16}]" for i in range(16, 20)
    }
q_res_trans_table_stride_2 = {
    f"s[{i}:{i + 1}]" : f"[%[q_res_{i - 16}], %[q_res_{i - 16+1}]]" for i in range(16, 20, 2)
    }
q_res_trans_table_stride_4 = {
    f"s[{i}:{i + 3}]" : f"[%[q_res_{i - 16}], %[q_res_{i - 16+1}], %[q_res_{i - 16+2}], %[q_res_{i - 16+3}]]" for i in range(16, 17)
    }

kv_res_trans_table_stride_1 = {
    f"s{i}" : f"%[kv_res_{i - 20}]" for i in range(20, 24)
    }
kv_res_trans_table_stride_2 = {
    f"s[{i}:{i + 1}]" : f"[%[kv_res_{i - 20}], %[kv_res_{i - 20+1}]]" for i in range(20, 24, 2)
    }
kv_res_trans_table_stride_4 = {
    f"s[{i}:{i + 3}]" : f"[%[kv_res_{i - 20}], %[kv_res_{i - 20+1}], %[kv_res_{i - 20+2}], %[kv_res_{i - 20+3}]]" for i in range(20, 21)
    }

kv_indices_res_trans_table_stride_1 = {
    f"s{i}" : f"%[kv_indices_res_{i - 24}]" for i in range(24, 28)
    }
kv_indices_res_trans_table_stride_2 = {
    f"s[{i}:{i + 1}]" : f"[%[kv_indices_res_{i - 24}], %[kv_indices_res_{i - 24+1}]]" for i in range(24, 28, 2)
    }
kv_indices_res_trans_table_stride_4 = {
    f"s[{i}:{i + 3}]" : f"[%[kv_indices_res_{i - 24}], %[kv_indices_res_{i - 24+1}], %[kv_indices_res_{i - 24+2}], %[kv_indices_res_{i - 24+3}]]" for i in range(24, 25)
    }


kv_indptr_res_trans_table_stride_1= {
    f"s{i}" : f"%[kv_indptr_res_{i - 28}]" for i in range(28, 30)
    }
kv_indptr_res_trans_table_stride_2 = {
    f"s[{i}:{i + 1}]" : f"[%[kv_indptr_res_{i - 28}], %[kv_indptr_res_{i - 28+1}]]" for i in range(28, 29)
    }



qo_res_trans_table_stride_1 = {
    f"s{i}" : f"%[qo_res_{i - 32}]" for i in range(32, 34)
    }
qo_res_trans_table_stride_2 = {
    f"s[{i}:{i + 1}]" : f"[%[qo_res_{i - 32}], %[qo_res_{i - 32+1}]]" for i in range(32, 34, 2)
    }
# qo_res_trans_table_stride_4 = {
#     f"s[{i}:{i + 3}]" : f"[%[qo_res_{i}], %[qo_res_{i+1}], %[qo_res_{i+2}], %[qo_res_{i+3}]]" for i in range(32, 33)
#     }

kv_splits_res_trans_table_stride_1 = {
    f"s{i}" : f"%[kv_splits_res_{i - 88}]" for i in range(88, 92)
    }
kv_splits_res_trans_table_stride_2 = {
    f"s[{i}:{i + 1}]" : f"[%[kv_splits_res_{i - 88}], %[kv_splits_res_{i - 88+1}]]" for i in range(88, 90, 2)
    }
# kv_splits_res_trans_table_stride_4 = {
#     f"s[{i}:{i + 3}]" : f"[%[kv_splits_res_{i}, %[kv_splits_res_{i+1}, %[kv_splits_res_{i+2}, %[kv_splits_res_{i+3}]" for i in range(88, 89)
#     }

lines = file_r.readlines()

table_list = [
    # v_trans_table_stride_1,
    # v_trans_table_stride_2,
    # v_trans_table_stride_4,
    o_regs_trans_table_stride_1, 
    o_regs_trans_table_stride_2, 
    o_regs_trans_table_stride_4, 
    o_res_trans_table_stride_1_blank,
    o_res_trans_table_stride_1_comma,
    o_res_trans_table_stride_2, 
    o_res_trans_table_stride_4, 
    lse_res_trans_table_stride_1,
    lse_res_trans_table_stride_2,
    lse_res_trans_table_stride_4,
    q_res_trans_table_stride_1,
    q_res_trans_table_stride_2,
    q_res_trans_table_stride_4,
    kv_res_trans_table_stride_1,
    kv_res_trans_table_stride_2,
    kv_res_trans_table_stride_4,
    kv_indices_res_trans_table_stride_1,
    kv_indices_res_trans_table_stride_2,
    kv_indices_res_trans_table_stride_4,
    kv_indptr_res_trans_table_stride_1,
    kv_indptr_res_trans_table_stride_2,
    qo_res_trans_table_stride_1,
    qo_res_trans_table_stride_2,
    kv_splits_res_trans_table_stride_1,
    kv_splits_res_trans_table_stride_2,
]


for line in lines:
    for table in table_list:
        for key in table:
            if key in line:
                # import pdb;pdb.set_trace()
                line = line.replace(key, table[key])
    file_w.write(line)


