from itertools import product
import os
import sys
import triton

M = int(sys.argv[1])
N = int(sys.argv[2])
K = int(sys.argv[3])
G = int(sys.argv[4])
ut_filename = sys.argv[5]

assert M == triton.next_power_of_2(M), "M has to be power of 2"
assert os.path.isfile(ut_filename), f"{ut_filename} not found"

NUM_KSPLITs = [1]
possible_split = [3, 4, 7, 8, 14, 16, 28]

############################################################
# default settings
Ms = [4, 8]
possible_ms = [16, 32, 64, 128, 256]
Ms += [v for v in possible_ms if v <= M]

Ns = [16]
possible_ns = [32, 64, 128, 256]
Ns += [v for v in possible_ns if v <= N]

Ks = [128]
possible_ks = [256, 512, 1024]
Ks += [v for v in possible_ks if v <= K]
############################################################

############################################################
# # for AFP4WFP4_GEMM_preshuffe please use this
# if M >= 256:
#     Ms = [32, 64, 128, 256]
# elif M >= 128:
#     Ms = [32, 64, 128]
# elif M >= 64:
#     Ms = [32, 64]
# elif M >= 32:
#     Ms = [32]
# else:
#     Ms = [4, 8, 16]
# Ns = [32, 64, 128]
# Ks = [256, 512, 1024]
############################################################

############################################################
# # for a8w8_GEMM_blockscale/a8w8_GEMM_blockscale_preshuffe/a16w8_GEMM_blockscale/a16w8_GEMM_blockscale_preshuffe, Ks can only be 128
# Ks = [128]
############################################################

for a_possible_split in possible_split:
    if K % a_possible_split == 0:
        NUM_KSPLITs.append(a_possible_split)

parms = {
    "BLOCK_SIZE_M": Ms,
    "BLOCK_SIZE_N": Ns,
    "BLOCK_SIZE_K": Ks,
    "GROUP_SIZE_M": [1, 4, 8],
    "num_warps": [2, 4, 8],
    "num_stages": [1, 2],
    "waves_per_eu": [1, 2, 4, 6, 8],
    "matrix_instr_nonkdim": [16],
    "cache_modifier": [0, 1],
    "NUM_KSPLIT": NUM_KSPLITs,
}
comb = list(product(*parms.values()))
comb_p = []
for a_comb in comb:
    (
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
        num_warps,
        num_stages,
        waves_per_eu,
        matrix_instr_nonkdim,
        cache_modifier,
        NUM_KSPLIT,
    ) = a_comb
    # skip cases
    if NUM_KSPLIT > 1 and GROUP_SIZE_M > 1:
        continue
    if BLOCK_SIZE_K > K // NUM_KSPLIT:
        continue
    if BLOCK_SIZE_K == K // NUM_KSPLIT and num_stages != 1:  # k_itr == 1 case
        continue
    if BLOCK_SIZE_K < K // NUM_KSPLIT and num_stages == 1:  # k_itr > 1 case
        continue
    comb_p.append(a_comb)
comb = comb_p
file_tag = f"{ut_filename}-{M}-{N}-{K}"
filename = f"screen-{file_tag}.txt"
s = " ".join([str(v) for v in parms.keys()])
os.popen(f"echo 'Number of combinations = {len(comb)}' > {filename}")
os.popen(f"echo '{s}' >> {filename}")
i_comb_start = 0
comb_max_batch = 100
os.popen(f"date >> {filename}").read()
while i_comb_start < len(comb):
    i_comb_end = i_comb_start + 1
    while (
        i_comb_end < len(comb)
        and i_comb_end - i_comb_start < comb_max_batch
        and comb[i_comb_start][0:3] == comb[i_comb_end][0:3]
    ):
        i_comb_end += 1
    os.popen(
        f"echo 'running case {i_comb_start} ~ {i_comb_end - 1}' >> {filename}"
    ).read()
    s = ""
    for a_comb in comb[i_comb_start:i_comb_end]:
        s += " ".join([str(v) for v in a_comb])
        s += " "
    s = s.strip()

    cmd = f"""HIP_VISIBLE_DEVICES={G} rocprofv3 --kernel-trace -f csv -o res-{file_tag} -- python3 {ut_filename} {M} {N} {K} {s}"""
    cmd_rprof = f"""python3 rprof.py res-{file_tag}_kernel_trace.csv -k gemm"""

    os.popen(f"rm res-{file_tag}_kernel_trace.csv").read()
    os.popen(cmd).read()
    if os.path.isfile(f"res-{file_tag}_kernel_trace.csv") == True:
        prof_output = os.popen(cmd_rprof).read().split("\n")
        if prof_output[-1].strip() == "":
            prof_output.pop()
        prof_output_i = 0

        for a_comb in comb[i_comb_start:i_comb_end]:
            s = " ".join([str(v) for v in a_comb])
            os.popen(f"echo 'screencase {s}' >> {filename}").read()
            assert prof_output[prof_output_i] == "Kernel detected:"
            prof_output_i += 1
            while (
                prof_output_i < len(prof_output)
                and prof_output[prof_output_i] != "Kernel detected:"
            ):
                os.popen(f"echo '{prof_output[prof_output_i]}' >> {filename}").read()
                prof_output_i += 1

    i_comb_start = i_comb_end
    os.popen(f"date >> {filename}").read()
