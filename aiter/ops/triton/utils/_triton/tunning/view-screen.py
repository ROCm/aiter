import os
import sys
from aiter.ops.triton.utils._triton.tunning._utils import (
    config_parms_key,
    read_screen_file,
)

import aiter.ops.triton.utils._triton.arch_info as arch_info

DEVICE_ARCH = arch_info.get_arch()

TP = 8
# DS-R1 TP8
list_of_shapes = [
    (2112, 7168),  # fused_qkv_a_proj
    (3072, 1536),  # b_proj
    (7168, 2048),  # o_proj
    (256, 7168),  # moe gate
    (4608, 7168),  # dense layer1
    (7168, 2304),  # dense layer2
    (4096, 512),  # prefill kv_proj
]
# # LL3-405B TP-n
# list_of_shapes = [
#     (106496//TP, 16384),
#     (16384, 53248//TP),
#     (18432//TP, 16384),
#     (16384, 16384//TP),
# ]
# # LL3-70B TP-n
# list_of_shapes = [
#     (57344//TP, 8192),
#     (8192, 28672//TP),
#     (10240//TP, 8192),
#     (8192, 8192//TP),
# ]
# # LL3-8B TP-n
# list_of_shapes = [
#     (28672//TP, 4096),
#     (4096, 14336//TP),
#     (6144//TP, 4096),
#     (4096, 4096//TP),
# ]
# # LL4-Maverick TP8
# list_of_shapes = [
#     (4096, 5120),
#     (5120, 2048),
#     (128, 5120),
#     (2048, 5120),
#     (5120, 1024),
#     (896, 5120),
#     (5120, 640),
# ]
# # Qwen3 MoE TP8
# list_of_shapes = [
#     (128, 4096),
#     (1152, 4096),
#     (4096, 1024),
# ]
# # GPT-OSS-120B TP-n
# list_of_shapes = [
#     (10240//TP, 8192),
#     (8192, 8192//TP),
#     (57344//TP, 8192),
#     (8192, 28672//TP),
# ]

ut_filename = sys.argv[1]
filename_prefix = f"{DEVICE_ARCH}-{sys.argv[2]}"  # example "GEMM-A8W8_BLOCKSCALE"
m_config_map = {v: [f"M_LEQ_{v}"] for v in [8, 16, 32, 64, 128, 256]}
m_config_map[16384] = ["any"]

mlist = list(m_config_map.keys())
last_config_name = []
for a_config_name in m_config_map.values():
    last_config_name += a_config_name
last_config_name = last_config_name[-1]
print(f"M\tN\tK\tTriton (us)\tconfig")
for n, k in list_of_shapes:
    get_at_least_one_config = False
    fout = open(f"{filename_prefix}-N={n}-K={k}.json", "w")
    fout.write("{\n")

    last_config_list = None
    for m in mlist:
        case_data = []
        file_tag = f"{ut_filename}-{m}-{n}-{k}"
        read_screen_file(f"screen-{file_tag}.log", case_data)
        case_data = sorted(case_data, key=lambda x: x[0])

        if len(case_data) > 0:
            get_at_least_one_config = True
            triton_runtime = f"{case_data[0][0]:8.3f}"
            config_str = f"(config = {case_data[0][1]})"
        else:
            triton_runtime = "     N/A"
            config_str = "Warning: your config files is not complete!"

        print(f"{m}\t{n}\t{k}\t{triton_runtime}\t{config_str}")

        if len(case_data) == 0:
            if last_config_list is None:
                continue
            config_list = last_config_list
        else:
            config_list = case_data[0][1].split()
            last_config_list = config_list

        for config_name in m_config_map[m]:

            fout.write("""  "%s": {\n""" % (config_name))
            for i_parms_key, parms_key in enumerate(config_parms_key):
                parm = config_list[i_parms_key]

                if parms_key == "cache_modifier":
                    fout.write(
                        """    "%s": %s"""
                        % (
                            parms_key,
                            """".cg\"""" if parm == "0" else "null",
                        )
                    )
                else:
                    fout.write("""    "%s": %s""" % (parms_key, parm))

                if i_parms_key != len(config_parms_key) - 1:
                    fout.write(""",\n""")
                else:
                    fout.write("""\n  }""")

        if config_name == last_config_name:
            fout.write("\n")
        else:
            fout.write(",\n")

    fout.write("}\n")
    fout.close()
    if get_at_least_one_config == False:
        os.popen(f"rm {filename_prefix}-N={n}-K={k}.json").read()
        print(f"No file is created")
    else:
        print(f"{filename_prefix}-N={n}-K={k}.json is created")
