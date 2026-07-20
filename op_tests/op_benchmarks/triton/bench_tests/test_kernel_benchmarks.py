import importlib
import json
import os
import re
import warnings
import argparse
from op_tests.op_benchmarks.triton.utils.argparse import get_parser, add_argparse_ff

MODEL_SHAPES_JSON = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "model_benchmarking_tool",
        "model_shapes.json",
    )
)

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


def test_kimi_k2_model_shapes_entry():
    """Structural regression guard for the Kimi K2.6 (Kimi-K2 Thinking) entry
    in model_shapes.json. Runs without a GPU.

    Kimi K2 is a 1T-param DeepSeek-V3-style MoE with 384 experts, top_k=8,
    64 attention heads, and MLA (q_lora=1536, kv_lora=512, qk_nope=128,
    qk_rope=64, v_head=128). Source: huggingface.co/moonshotai/Kimi-K2-Thinking
    config.json. Removing or misshaping this entry will silently drop Kimi K2.6
    from the per-model kernel sweep and let regressions slip through.
    """
    from op_tests.op_benchmarks.triton.model_benchmarking_tool.bench_models import (
        KERNEL_DICT,
    )

    with open(MODEL_SHAPES_JSON, "r") as f:
        data = json.load(f)

    assert "Kimi-K2 Thinking" in data, "Kimi-K2 Thinking entry missing"
    kimi = data["Kimi-K2 Thinking"]

    # Every declared kernel must be dispatchable by bench_models.py.
    unknown = [k for k in kimi if k not in KERNEL_DICT]
    assert not unknown, f"Kimi-K2 declares kernels not in KERNEL_DICT: {unknown}"

    # MLA decode hot path (hq=64 distinguishes Kimi K2 from DeepSeek-R1's 128).
    assert "mla" in kimi
    mla = kimi["mla"][0]
    assert (mla["hq"], mla["hkv"], mla["dqk"], mla["dv"]) == (64, 1, 576, 512)

    # MHA prefill hot path.
    assert "mha" in kimi
    mha = kimi["mha"][0]
    assert (mha["hq"], mha["hkv"], mha["dqk"], mha["dv"]) == (64, 64, 192, 128)

    # Routed MoE GEMM must reflect Kimi K2's E=384, TopK=8, hidden=7168,
    # 2*moe_intermediate_size=4096.
    moe_kernels = [k for k in kimi if k.startswith("moe_op_gemm_")]
    assert moe_kernels, "Kimi-K2 must exercise at least one MoE GEMM kernel"
    for k in moe_kernels:
        shape = kimi[k][0]
        assert shape["E"] == 384, (k, shape)
        assert shape["TopK"] == 8, (k, shape)
        assert shape["Dim1"] == 7168, (k, shape)
        assert shape["Dim2"] == 4096, (k, shape)

    # MLA projection GEMMs (q_b, kv_b, o_proj) — these differ from DSR1
    # because Kimi K2 halves the head count.
    dense_gemm_kernels = [k for k in kimi if k.startswith("gemm_") and "moe" not in k]
    assert dense_gemm_kernels, "Kimi-K2 must exercise at least one dense GEMM"
    for k in dense_gemm_kernels:
        nk = {(s["N"], s["K"]) for s in kimi[k]}
        if k in ("gemm_a8w8_blockscale", "gemm_afp4wfp4"):
            assert (12288, 1536) in nk, f"{k} missing q_b_proj shape"
            assert (16384, 512) in nk, f"{k} missing kv_b_proj shape"
            assert (7168, 8192) in nk, f"{k} missing o_proj shape"

    # --model 'kimi' regex (case-insensitive, from bench_models.parse_args)
    # must match exactly this entry.
    pat = re.compile("kimi", re.IGNORECASE)
    assert [m for m in data if pat.search(m)] == ["Kimi-K2 Thinking"]


def test_bench_models_kimi_k2_rmsnorm_runs():
    """End-to-end smoke: bench_models.run_benchmarks() must execute at least
    one Kimi-K2 kernel and produce a numeric result. Pinned to rmsnorm because
    it has no FP4/MX arch gating and is GPU-cheap, while still validating the
    full path: JSON load -> model filter -> handler dispatch -> kernel run.
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    from op_tests.op_benchmarks.triton.model_benchmarking_tool.bench_models import (
        filter_models_and_kernels,
        read_json,
        run_benchmarks,
    )

    data = read_json("model_shapes.json")
    data = filter_models_and_kernels(
        data,
        available_models=sorted(data.keys()),
        model_pattern="kimi",
        kernel_pattern="rmsnorm",
    )
    assert data is not None and "Kimi-K2 Thinking" in data

    results = run_benchmarks(
        data,
        batch_sizes=[1],
        seq_lens=[1024],
        TP=1,
        gemm_layout="TN",
        mha_layout="thd",
        metric="throughput",
    )
    assert results, "No results produced for Kimi-K2 rmsnorm benchmark"
    for row in results:
        assert row["Model"] == "Kimi-K2 Thinking"
        assert row["Kernel"] == "rmsnorm"
        value = row.get("throughput")
        assert value not in (None, "N/A"), f"Benchmark failed for row: {row}"
        assert float(value) > 0, f"Non-positive throughput for row: {row}"
