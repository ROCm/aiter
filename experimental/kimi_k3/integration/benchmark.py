"""Use the same InferenceX serving client and 1k/1k workload as the saved runs."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-script",
        type=Path,
        required=True,
        help="InferenceX utils/bench_serving/benchmark_serving.py",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--output", type=Path, default=Path("results/serving"))
    args = parser.parse_args()
    count = max(10, 2 * args.concurrency)
    args.output.mkdir(parents=True, exist_ok=True)
    filename = f"c{args.concurrency}-1k1k.json"
    command = [
        sys.executable,
        str(args.benchmark_script),
        "--model",
        str(args.model),
        "--served-model-name",
        "kimi-k3",
        "--backend",
        "vllm",
        "--base-url",
        "http://127.0.0.1:8243",
        "--dataset-name",
        "random",
        "--random-input-len",
        "1024",
        "--random-output-len",
        "1024",
        "--random-range-ratio",
        "1.0",
        "--num-prompts",
        str(count),
        "--num-warmups",
        "2",
        "--max-concurrency",
        str(args.concurrency),
        "--request-rate",
        "inf",
        "--ignore-eos",
        "--seed",
        "42",
        "--save-result",
        "--save-detailed",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--result-dir",
        str(args.output),
        "--result-filename",
        filename,
        "--trust-remote-code",
    ]
    (args.output / f"c{args.concurrency}-command.json").write_text(
        json.dumps(command, indent=2) + "\n"
    )
    subprocess.run(command, check=True)
    result = json.loads((args.output / filename).read_text())
    assert result["completed"] == count
    assert result["total_input_tokens"] == result["total_output_tokens"] == count * 1024
    print(
        {
            k: result[k]
            for k in (
                "mean_tpot_ms",
                "mean_ttft_ms",
                "total_token_throughput",
                "output_throughput",
            )
        }
    )


if __name__ == "__main__":
    main()
