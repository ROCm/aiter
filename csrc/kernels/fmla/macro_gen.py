
file_w = open("./macro_utils_b.hpp", mode="w")

file_w.write("#define LOOPEND()\n")

for loop in range(8):
    for i in range(4):
        for j in range(4):
            if j != 3:
                line = f"#define LOOP_STRIDE4_{16 * loop + i + 4 * j}(func)  func({j}, {16 * loop + i + 4 * j})  LOOP_STRIDE4_{16 * loop + i + 4 * j + 4}(func) \n"
            else:
                line = f"#define LOOP_STRIDE4_{16 * loop + i + 4 * j}(func)  func(3, {16 * loop + i + 4 * j})  LOOPEND() \n"

            file_w.write(line)
        file_w.write("\n")


for r in range(2):
    for i in range(2):
        for j in range(4):
            if j != 3:
                line = f"#define B16_LDS_2_VGPR_LOOP_STRIDE1_{r}_{i}_{j}(func, lds_ld_base)  func({r * 8 + i * 2 + j + j // 2 * 2}, lds_ld_base + {r * 576 + i * 288 + j + (j // 2) * 14})  B16_LDS_2_VGPR_LOOP_STRIDE1_{r}_{i}_{j + 1}(func, lds_ld_base) \n"
            elif i == 0:
                line = f"#define B16_LDS_2_VGPR_LOOP_STRIDE1_{r}_{i}_{j}(func, lds_ld_base)  func({r * 8 + i * 2 + j + j // 2 * 2}, lds_ld_base + {r * 576 + i * 288 + j + (j // 2) * 14})  B16_LDS_2_VGPR_LOOP_STRIDE1_{r}_{i + 1}_{0}(func, lds_ld_base) \n"
            elif r == 0:
                line = f"#define B16_LDS_2_VGPR_LOOP_STRIDE1_{r}_{i}_{j}(func, lds_ld_base)  func({r * 8 + i * 2 + j + j // 2 * 2}, lds_ld_base + {r * 576 + i * 288 + j + (j // 2) * 14})  B16_LDS_2_VGPR_LOOP_STRIDE1_{r + 1}_{0}_{0}(func, lds_ld_base) \n"
            else:
                line = f"#define B16_LDS_2_VGPR_LOOP_STRIDE1_{r}_{i}_{j}(func, lds_ld_base)  func({r * 8 + i * 2 + j + j // 2 * 2}, lds_ld_base + {r * 576 + i * 288 + j + (j // 2) * 14})  LOOPEND() \n"
            file_w.write(line)


for r in range(2):
    for i in range(2):
        for j in range(4):
            if j != 3:
                line = f"#define B16_VGPR_2_DRAM_LOOP_STRIDE1_{r}_{i}_{j}(func, dram_st_base)  func({r * 4 + i * 8 + j}, dram_st_base + {r * 2048 + i * 32 + j})  B16_VGPR_2_DRAM_LOOP_STRIDE1_{r}_{i}_{j + 1}(func, dram_st_base) \n"
            elif i == 0:
                line = f"#define B16_VGPR_2_DRAM_LOOP_STRIDE1_{r}_{i}_{j}(func, dram_st_base)  func({r * 4 + i * 8 + j}, dram_st_base + {r * 2048 + i * 32 + j})  B16_VGPR_2_DRAM_LOOP_STRIDE1_{r}_{i + 1}_{0}(func, dram_st_base) \n"
            elif r == 0:
                line = f"#define B16_VGPR_2_DRAM_LOOP_STRIDE1_{r}_{i}_{j}(func, dram_st_base)  func({r * 4 + i * 8 + j}, dram_st_base + {r * 2048 + i * 32 + j})  B16_VGPR_2_DRAM_LOOP_STRIDE1_{r + 1}_{0}_{0}(func, dram_st_base) \n"
            else:
                line = f"#define B16_VGPR_2_DRAM_LOOP_STRIDE1_{r}_{i}_{j}(func, dram_st_base)  func({r * 4 + i * 8 + j}, dram_st_base + {r * 2048 + i * 32 + j})  LOOPEND() \n"
            file_w.write(line)
        file_w.write("\n")


for r in range(2):
    for i in range(4):
        for j in range(4):
            if j != 3:
                line = f"#define F32_LDS_2_VGPR_LOOP_STRIDE1_{r}_{i}_{j}(func, lds_ld_base)  func({r * 16 + i * 4 + j}, lds_ld_base + {r * 1056 + i * 16 + j})  F32_LDS_2_VGPR_LOOP_STRIDE1_{r}_{i}_{j + 1}(func, lds_ld_base) \n"
            elif i != 3:
                line = f"#define F32_LDS_2_VGPR_LOOP_STRIDE1_{r}_{i}_{j}(func, lds_ld_base)  func({r * 16 + i * 4 + j}, lds_ld_base + {r * 1056 + i * 16 + j})  F32_LDS_2_VGPR_LOOP_STRIDE1_{r}_{i + 1}_{0}(func, lds_ld_base) \n"
            elif r == 0:
                line = f"#define F32_LDS_2_VGPR_LOOP_STRIDE1_{r}_{i}_{j}(func, lds_ld_base)  func({r * 16 + i * 4 + j}, lds_ld_base + {r * 1056 + i * 16 + j})  F32_LDS_2_VGPR_LOOP_STRIDE1_{r + 1}_{0}_{0}(func, lds_ld_base) \n"
            else:
                line = f"#define F32_LDS_2_VGPR_LOOP_STRIDE1_{r}_{i}_{j}(func, lds_ld_base)  func({r * 16 + i * 4 + j}, lds_ld_base + {r * 1056 + i * 16 + j})  LOOPEND() \n"
            file_w.write(line)
        file_w.write("\n")


# for r in range(4):
#     for i in range(2):
#         for j in range(4):
#             if j != 3:
#                 line = f"#define F32_VGPR_2_DRAM_LOOP_STRIDE1_{r}_{i}_{j}(func, dram_st_base)  func({r * 4 + i * 16 + j}, dram_st_base + {r * 2048 + i * 64 + j})  F32_VGPR_2_DRAM_LOOP_STRIDE1_{r}_{i}_{j + 1}(func, dram_st_base) \n"
#             elif i == 0:
#                 line = f"#define F32_VGPR_2_DRAM_LOOP_STRIDE1_{r}_{i}_{j}(func, dram_st_base)  func({r * 4 + i * 16 + j}, dram_st_base + {r * 2048 + i * 64 + j})  F32_VGPR_2_DRAM_LOOP_STRIDE1_{r}_{i + 1}_{0}(func, dram_st_base) \n"
#             elif r != 3:
#                 line = f"#define F32_VGPR_2_DRAM_LOOP_STRIDE1_{r}_{i}_{j}(func, dram_st_base)  func({r * 4 + i * 16 + j}, dram_st_base + {r * 2048 + i * 64 + j})  F32_VGPR_2_DRAM_LOOP_STRIDE1_{r + 1}_{0}_{0}(func, dram_st_base) \n"
#             else:
#                 line = f"#define F32_VGPR_2_DRAM_LOOP_STRIDE1_{r}_{i}_{j}(func, dram_st_base)  func({r * 4 + i * 16 + j}, dram_st_base + {r * 2048 + i * 64 + j})  LOOPEND() \n"
#             file_w.write(line)
#         file_w.write("\n")

for i in range(2):
    for r in range(4):
        for j in range(4):
            if j != 3:
                line = f"#define F32_VGPR_2_DRAM_LOOP_STRIDE1_{r}_{i}_{j}(func, dram_st_base)  func({r * 4 + i * 16} + {j}, dram_st_base + {r * 2048 + i * 64} + {j})  F32_VGPR_2_DRAM_LOOP_STRIDE1_{r}_{i}_{j + 1}(func, dram_st_base) \n"
            elif i == 0:
                line = f"#define F32_VGPR_2_DRAM_LOOP_STRIDE1_{r}_{i}_{j}(func, dram_st_base)  func({r * 4 + i * 16} + {j}, dram_st_base + {r * 2048 + i * 64} + {j})  F32_VGPR_2_DRAM_LOOP_STRIDE1_{r}_{i + 1}_{0}(func, dram_st_base) \n"
            elif r != 3:
                line = f"#define F32_VGPR_2_DRAM_LOOP_STRIDE1_{r}_{i}_{j}(func, dram_st_base)  func({r * 4 + i * 16} + {j}, dram_st_base + {r * 2048 + i * 64} + {j})  F32_VGPR_2_DRAM_LOOP_STRIDE1_{r + 1}_{0}_{0}(func, dram_st_base) \n"
            else:
                line = f"#define F32_VGPR_2_DRAM_LOOP_STRIDE1_{r}_{i}_{j}(func, dram_st_base)  func({r * 4 + i * 16} + {j}, dram_st_base + {r * 2048 + i * 64} + {j})  LOOPEND() \n"
            file_w.write(line)
        file_w.write("\n")
