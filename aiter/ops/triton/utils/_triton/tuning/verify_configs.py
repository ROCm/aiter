import argparse
import os
import subprocess
import sys

from harness import kernel_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int, help="M dim")
    parser.add_argument("N", type=int, help="N dim")
    parser.add_argument("K", type=int, help="K dim")
    parser.add_argument(
        "F", type=kernel_name, help="Kernel name (see harness.py --help)"
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    M = args.M
    N = args.N
    K = args.K
    kernel = args.F
    harness_filename = f"harness_{kernel}.py"

    file_tag = f"{harness_filename}-{M}-{N}-{K}"
    cmd = [
        "rocprofv3",
        "--kernel-trace",
        "-f",
        "csv",
        "-o",
        f"verf_{file_tag}",
        "--",
        sys.executable,
        os.path.join(os.path.dirname(__file__), "harness.py"),
        kernel,
        str(M),
        str(N),
        str(K),
    ]

    rocprof_filename = f"verf_{file_tag}_kernel_trace.csv"

    if os.path.isfile(rocprof_filename):
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
        if os.path.isfile(rocprof_filename):
            cmd_parse = f"python3 parse_kernel_trace.py {rocprof_filename} -k gemm"
            cmd_parse = cmd_parse.split(" ")
            process = subprocess.Popen(
                cmd_parse, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
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
                for stderr_str in stderr_data:
                    print(f"\t{stderr_str}")
        else:
            print(f"[Error]: {rocprof_filename} not found")
    else:
        stderr_data = stderr_data.split("\n")
        print("[Error]: when running rocprof, error message:")
        for stderr_str in stderr_data:
            print(f"\t{stderr_str}")


if __name__ == "__main__":
    sys.exit(main())
