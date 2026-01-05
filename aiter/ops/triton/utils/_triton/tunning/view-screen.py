import sys

def read_file(filename, case_data):
    err_lines_limit = 100000
    time_rec_map = {}
    if os.path.isfile(filename):
        with open(filename, "r") as f:
            for newline in f:
                try:
                    err_lines = 0
                    while not newline.startswith("screencase"):
                        newline = f.readline()
                        err_lines += 1
                        if err_lines >= err_lines_limit:
                            break
                    screencaseline = newline[:]
                    err_lines = 0
                    while not newline.strip().endswith("(us)"):
                        if newline.startswith("screencase"):
                            screencaseline = newline[:]
                        newline = f.readline()
                        err_lines += 1
                        if err_lines >= err_lines_limit:
                            break
                    r = float(newline.strip().split()[0])
                    # if int(screencaseline[len("screencase")+1:].strip().split()[-1]) == 1: continue # remove this comment to consider only split-k case
                    # if int(screencaseline[len("screencase")+1:].strip().split()[-1]) != 1: continue # remove this comment to consider only split-k case
                    # if int(screencaseline[len("screencase")+1:].strip().split()[2]) != 128: continue # remove this comment to consider only BK=128
                    case_data.append([r, screencaseline[len("screencase")+1:].strip()])
                except:
                    break
               
import os

m_config_map = {v: [f"M_LEQ_{v}"] for v in [8, 16, 32, 64, 128, 256]}
m_config_map[16384] = ["any"]

mlist = list(m_config_map.keys())
last_config_name = []
for a_config_name in m_config_map.values():
    last_config_name += a_config_name
last_config_name = last_config_name[-1]
filename_prefix = "gfx950-GEMM-A8W8_PRESHUFFLED"
print(f"M\tN\tK\tTriton (us)\tconfig")
for n, k in [
    (2112, 7168),
    (3072, 1536),
]:
    fout = open(f"{filename_prefix}-N={n}-K={k}.json", "w")
    fout.write("{\n")

    last_config_list = None
    for m in mlist:
        case_data = []
        read_file(f"screen-{m}-{n}-{k}.txt", case_data)
        case_data = sorted(case_data, key=lambda x: x[0])

        if len(case_data) > 0:
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

            fout.write("""  "%s": {
    "BLOCK_SIZE_M": %s,
    "BLOCK_SIZE_N": %s,
    "BLOCK_SIZE_K": %s,
    "GROUP_SIZE_M": %s,
    "num_warps": %s,
    "num_stages": %s,
    "waves_per_eu": %s,
    "matrix_instr_nonkdim": %s,
    "cache_modifier": %s,
    "NUM_KSPLIT": %s
  }"""%(
      config_name,
      config_list[0],
      config_list[1],
      config_list[2],
      config_list[3],
      config_list[4],
      config_list[5],
      config_list[6],
      config_list[7],
      """".cg\"""" if config_list[8] == "0" else 'null',
      config_list[9],
  ))
            
            if config_name == last_config_name:
                fout.write("\n")
            else:
                fout.write(",\n")

    fout.write("}\n")
    fout.close()