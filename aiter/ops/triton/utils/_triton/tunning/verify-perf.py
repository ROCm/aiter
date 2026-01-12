from itertools import product
import os
import sys
import triton
import argparse
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int, help="M dim")
    parser.add_argument("N", type=int, help="N dim")
    parser.add_argument("K", type=int, help="K dim")
    parser.add_argument("F", type=str, help="Unit test filename")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    M = args.M
    N = args.N
    K = args.K
    ut_filename = args.F

    file_tag = f"{ut_filename}-{M}-{N}-{K}"
    cmd = f"""rocprofv3 --kernel-trace -f csv -o verf_{file_tag} -- python3 {ut_filename} {M} {N} {K}"""
    cmd = cmd.split(" ")

    rocprof_filename = f"verf_{file_tag}_kernel_trace.csv"

    if os.path.isfile(rocprof_filename) == True:
        process = subprocess.Popen(
            ["rm", rocprof_filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        process.communicate()

    env = os.environ.copy()
    process = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    stdout_data, stderr_data = process.communicate()

    if process.returncode == 0:
        if os.path.isfile(rocprof_filename) == True:
            cmd_rprof = f"""python3 rprof.py {rocprof_filename} -k gemm"""
            cmd_rprof = cmd_rprof.split(" ")
            process = subprocess.Popen(
                cmd_rprof, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            stdout_data, stderr_data = process.communicate()
            if process.returncode == 0:
                prof_output = stdout_data.split("\n")
                if prof_output[-1].strip() == "":
                    prof_output.pop()

                prof_output_i = 0
                assert prof_output[prof_output_i] == "Kernel detected:"
                prof_output_i += 1
                while (
                    prof_output_i < len(prof_output)
                    and prof_output[prof_output_i] != "Kernel detected:"
                ):
                    print(prof_output[prof_output_i], flush=True)
                    prof_output_i += 1
            else:
                for l in stderr_data:
                    print(f"\t{l}")
        else:
            print(f"[Error]: {rocprof_filename} not found")
    else:
        stderr_data = stderr_data.split("\n")
        print(f"[Error]: when running rocprof, error message:")
        for l in stderr_data:
            print(f"\t{l}")


if __name__ == "__main__":
    sys.exit(main())
