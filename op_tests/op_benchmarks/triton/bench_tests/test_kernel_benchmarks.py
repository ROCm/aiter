import importlib
import os
import warnings
import argparse
from op_tests.op_benchmarks.triton.utils.argparse import get_parser, add_argparse_ff

"""
Loads the appropriate kernel using importlib and runs the benchmark script.
To get the mock argparse arguments for the kernel, import get_parser and add_argparse_ff from op_tests.op_benchmarks.triton.utils.argparse.
"""


def get_benchmark_output(
    bench_filename: str, mock_args: argparse.Namespace, defaults: argparse.Namespace
):
    kernel_benchmark_dir = os.path.join(__file__, "../../")
    kernel_benchmark_dir = os.path.abspath(kernel_benchmark_dir)
    kernel_benchmark_path = os.path.join(kernel_benchmark_dir, bench_filename)
    print(f"Loading kernel from {kernel_benchmark_path}")
    spec = importlib.util.spec_from_file_location(
        name=bench_filename, location=kernel_benchmark_path
    )
    kernel_benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel_benchmark)

    return kernel_benchmark.run_benchmark(mock_args, defaults)


def test_bench_gemm_a8w8_model():
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = add_argparse_ff(get_parser("A8W8 GEMM"))
    defaults = parser.parse_args([])  # get default arguments
    mock_args = parser.parse_args("--model llama3-8B -fc1".split())
    get_benchmark_output("bench_gemm_a8w8.py", mock_args, defaults)

    output_file = "GEMM A8W8 Benchmark.csv"
    assert os.path.exists(output_file)

    with open(output_file, "r") as f:
        content = f.read()
        assert "4096" in content and "14336" in content

    os.remove(output_file)
    os.remove("GEMM A8W8 Benchmark.png")


def test_bench_gemm_a8w8_shape():
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = add_argparse_ff(get_parser("A8W8 GEMM"))
    defaults = parser.parse_args([])  # get default arguments
    mock_args = parser.parse_args("--shape 4096 1024 1024".split())
    get_benchmark_output("bench_gemm_a8w8.py", mock_args, defaults)

    output_file = "GEMM A8W8 Benchmark.csv"
    assert os.path.exists(output_file)

    with open(output_file, "r") as f:
        content = f.read()
        assert "4096" in content and "1024" in content

    os.remove(output_file)
    os.remove("GEMM A8W8 Benchmark.png")


def test_bench_gemm_a8w8_tp():
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = add_argparse_ff(get_parser("A8W8 GEMM"))
    defaults = parser.parse_args([])  # get default arguments
    mock_args = parser.parse_args("--model llama3-8B -tp 8".split())
    get_benchmark_output("bench_gemm_a8w8.py", mock_args, defaults)

    output_file = "GEMM A8W8 Benchmark.csv"
    assert os.path.exists(output_file)

    with open(output_file, "r") as f:
        content = f.read()
        assert "4096" in content and "14336" in content

    os.remove(output_file)
    os.remove("GEMM A8W8 Benchmark.png")
