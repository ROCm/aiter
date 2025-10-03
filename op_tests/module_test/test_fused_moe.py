#!/usr/bin/env python3
"""
MoE 2-Stage Test Runner
Manages test parameters and executes test_moe_2stage.py with user-defined argument combinations
"""

import subprocess
import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TestConfig:
    """
    Test configuration data class

    Parameters:
    -----------
    model_name : str
        Name of the model (e.g., "DeepSeek-R1", "Qwen3-235B-A22B")

    tp_size : int
        Tensor Parallel size (e.g., 8 for TP8, 4 for TP4)

    dim : str
        Model dimensions in format "hidden_dim,moe_intermediate_dim"
        Examples: "6144,4096", "8192,4096", "4096,2048"

    expert : int
        Number of experts in the MoE layer

    topk : int
        Number of top experts to use for each token
    """

    model_name: str
    tp_size: int
    dim: str
    expert: int
    topk: int

    def __str__(self):
        # Parse dim string to extract hidden_dim and moe_intermediate_size
        hidden_dim, moe_intermediate_size = self.dim.split(",")
        return f"model={self.model_name}.tp{self.tp_size}.hidden_dim={hidden_dim}.moe_intermediate_size={moe_intermediate_size}.expert={self.expert}.topk={self.topk}"


TEST_CONFIGS = [
    #          model_name,        tp_size, dim(hidden_dim,inter_dim), expert, topk
    # DeepSeek-R1 -- TP8
    TestConfig("DeepSeek-R1", 8, "7168,256", 256, 8),
    # DeepSeek-R1 -- TP4
    TestConfig("DeepSeek-R1", 4, "7168,512", 256, 8),
    # Qwen3-235B-A22B -- TP8
    TestConfig("Qwen3-235B-A22B", 8, "4096,192", 128, 8),
    # Qwen3-235B-A22B -- TP4
    TestConfig("Qwen3-235B-A22B", 4, "4096,384", 128, 8),
    # Qwen3-30B-A3B -- TP8
    TestConfig("Qwen3-30B-A3B", 8, "2048,96", 128, 8),
    # Qwen3-30B-A3B -- TP4
    TestConfig("Qwen3-30B-A3B", 4, "2048,192", 128, 8),
]


def filter_test_configs(
    model_name: str = None, tp_size: int = None
) -> List[TestConfig]:
    """
    Filter test configurations based on model name and/or tp_size

    Parameters:
    -----------
    model_name : str, optional
        Filter by specific model name (e.g., "DeepSeek-R1", "Qwen3-235B-A22B")
    tp_size : int, optional
        Filter by specific tensor parallel size (e.g., 8, 4)

    Returns:
    --------
    List[TestConfig]
        Filtered list of test configurations
    """
    filtered_configs = []

    for config in TEST_CONFIGS:
        # Apply model name filter if specified
        if model_name and config.model_name != model_name:
            continue

        # Apply tp_size filter if specified
        if tp_size and config.tp_size != tp_size:
            continue

        filtered_configs.append(config)

    return filtered_configs


def list_available_configs():
    """List all available test configurations"""
    logger.info("Available test configurations:")
    logger.info("=" * 60)

    # Group by model
    models = {}
    for config in TEST_CONFIGS:
        if config.model_name not in models:
            models[config.model_name] = []
        models[config.model_name].append(config)

    for model_name, configs in models.items():
        logger.info(f"\n{model_name}:")
        for config in configs:
            hidden_dim, moe_intermediate_size = config.dim.split(",")
            logger.info(
                f"  TP{config.tp_size:2d} | hidden_dim={hidden_dim:4s} | moe_intermediate_size={moe_intermediate_size:4s} | expert={config.expert:3d} | topk={config.topk}"
            )


def get_available_models():
    """Get list of available model names"""
    return sorted({config.model_name for config in TEST_CONFIGS})


def get_available_tp_sizes():
    """Get list of available tensor parallel sizes"""
    return sorted({config.tp_size for config in TEST_CONFIGS})


class FusedmoeTestRunner:
    def __init__(self):
        """Initialize the test runner"""
        self.test_script = "../test_moe_2stage.py"

        # Check if we're in the right directory
        if not os.path.exists(self.test_script):
            logger.error(f"Test script {self.test_script} not found")
            logger.error("Please run this script from the module_test directory")
            sys.exit(1)

        # Initialize test results
        self.results = []
        self.log_dir = Path("test_logs")
        self.log_dir.mkdir(exist_ok=True)

        # Clear previous results
        self.results_file = Path("test_results.txt")
        if self.results_file.exists():
            self.results_file.unlink()

    def run_single_test(
        self, config: TestConfig, timeout: int = 1000
    ) -> Dict[str, Any]:
        """Run a single test with given configuration"""
        test_name = str(config)
        logger.info(f"Running test: {test_name}")

        # Build command
        cmd = [
            "python",
            "-u",
            self.test_script,
            "-dim",
            config.dim,
            "-e",
            str(config.expert),
            "-k",
            str(config.topk),
        ]

        # Create log file path - use clean name without str() wrapper
        hidden_dim, moe_intermediate_size = config.dim.split(",")
        clean_name = f"model={config.model_name}.tp{config.tp_size}.hidden_dim={hidden_dim}.moe_intermediate_size={moe_intermediate_size}.expert={config.expert}.topk={config.topk}"
        log_file = self.log_dir / f"{clean_name}.log"

        # Print and log the command being executed
        command_str = " ".join(cmd)
        logger.info(f"Executing command: {command_str}")

        # Write command to log file
        with open(log_file, "w") as f:
            f.write(f"Command: {command_str}\n")
            f.write(f"Test: {test_name}\n")
            f.write("=" * 50 + "\n\n")

        start_time = time.time()
        result = {
            "test_name": test_name,
            "config": config,
            "start_time": start_time,
            "status": "UNKNOWN",
            "exit_code": None,
            "duration": None,
            "error": None,
            "log_file": str(log_file),
        }

        try:
            # Run test with timeout
            with open(log_file, "a") as f:
                process = subprocess.run(
                    cmd, timeout=timeout, stdout=f, stderr=subprocess.STDOUT, text=True
                )

            result["exit_code"] = process.returncode
            result["duration"] = time.time() - start_time

            if process.returncode == 0:
                result["status"] = "PASS"
                logger.info(f"Test passed: {test_name}")
            else:
                result["status"] = "FAIL"
                result["error"] = f"Exit code: {process.returncode}"
                logger.error(
                    f"Test failed: {test_name} (exit code: {process.returncode})"
                )

        except subprocess.TimeoutExpired:
            result["status"] = "TIMEOUT"
            result["duration"] = time.time() - start_time
            logger.warning(f"Test timed out: {test_name}")
        except Exception as e:
            result["status"] = "ERROR"
            result["error"] = str(e)
            result["duration"] = time.time() - start_time
            logger.error(f"Test error: {test_name} - {e}")

        return result

    def run_tests(
        self, test_configs: List[TestConfig], delay: float = 1.0
    ) -> List[Dict[str, Any]]:
        """Run multiple tests"""
        results = []

        for i, config in enumerate(test_configs):
            logger.info(f"Progress: {i+1}/{len(test_configs)}")
            result = self.run_single_test(config)
            results.append(result)

            # Save result to file
            with open(self.results_file, "a") as f:
                f.write(f"{result['test_name']}: {result['status']}\n")

            # Delay between tests (except for the last one)
            if i < len(test_configs) - 1:
                time.sleep(delay)

        return results

    def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all defined tests"""
        logger.info("Starting all MoE 2-stage tests...")
        logger.info(f"Total tests: {len(TEST_CONFIGS)}")
        return self.run_tests(TEST_CONFIGS)

    def print_summary(self, results: List[Dict[str, Any]]):
        """Print test summary"""
        total = len(results)
        passed = sum(1 for r in results if r["status"] == "PASS")
        failed = sum(1 for r in results if r["status"] == "FAIL")
        timeout = sum(1 for r in results if r["status"] == "TIMEOUT")
        error = sum(1 for r in results if r["status"] == "ERROR")

        logger.info("\n" + "=" * 50)
        logger.info("TEST SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Timeout: {timeout}")
        logger.info(f"Error: {error}")
        logger.info("=" * 50)

        if failed == 0 and error == 0:
            logger.info("All tests completed successfully!")
        else:
            logger.warning("Some tests failed or encountered errors")

        # Print failed tests
        if failed > 0 or error > 0:
            logger.info("\nFailed/Error tests:")
            for result in results:
                if result["status"] in ["FAIL", "ERROR"]:
                    logger.error(f"  {result['test_name']}: {result['status']}")
                    if result["error"]:
                        logger.error(f"    Error: {result['error']}")
                    logger.info(f"    Log file: {result['log_file']}")


def main():
    """Main function - runs tests based on command line arguments or all tests if no arguments provided"""
    # Note: tp_size can only be specified when model is also specified
    # This prevents ambiguity since the same tp_size might exist across different models
    parser = argparse.ArgumentParser(
        description="MoE 2-Stage Test Runner - Run tests for specific models and tensor parallel sizes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests (default behavior)
  python test_fused_moe.py

  # Run tests for a specific model
  python test_fused_moe.py --model "DeepSeek-R1"

  # Run tests for a specific model and tensor parallel size
  python test_fused_moe.py --model "Qwen3-235B-A22B" --tp_size 4

  # List all available test configurations
  python test_fused_moe.py --list-configs

Available models: {models}
Available tp_sizes: {tp_sizes}
        """.format(
            models=", ".join(get_available_models()),
            tp_sizes=", ".join(map(str, get_available_tp_sizes())),
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Filter tests by model name (e.g., 'DeepSeek-R1', 'Qwen3-235B-A22B')",
    )

    parser.add_argument(
        "--tp_size",
        type=int,
        choices=[4, 8],
        help="Filter tests by tensor parallel size (4 or 8). Must be used with --model.",
    )

    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List all available test configurations",
    )

    args = parser.parse_args()

    if args.list_configs:
        list_available_configs()
        sys.exit(0)

    # Validate that tp_size is only specified when model is also specified
    if args.tp_size and not args.model:
        logger.error("--tp_size can only be specified when --model is also specified")
        logger.info(
            "This prevents ambiguity since the same tp_size might exist across different models"
        )
        logger.info(
            "Please specify both --model and --tp_size, or use --model alone to run all tp_sizes for that model"
        )
        sys.exit(1)

    # Validate model name if provided
    if args.model:
        available_models = get_available_models()
        if args.model not in available_models:
            logger.error(f"Invalid model name: {args.model}")
            logger.info("Available models:")
            for model in available_models:
                logger.info(f"  - {model}")
            sys.exit(1)

    # Filter test configurations based on arguments
    test_configs = filter_test_configs(args.model, args.tp_size)

    if not test_configs:
        logger.error("No test configurations match the specified criteria")
        logger.info("Available configurations:")
        for config in TEST_CONFIGS:
            logger.info(f"  - {config.model_name} (TP{config.tp_size})")
        sys.exit(1)

    # Log what tests will be run
    if args.model or args.tp_size:
        logger.info("Running filtered tests:")
        if args.model:
            logger.info(f"  Model: {args.model}")
        if args.tp_size:
            logger.info(f"  TP Size: {args.tp_size}")
        else:
            logger.info(f"  TP Size: All available sizes for {args.model}")
        logger.info(f"  Total tests to run: {len(test_configs)}")

        # Show which specific configurations will be run
        logger.info("  Selected configurations:")
        for config in test_configs:
            logger.info(f"    - {config.model_name} (TP{config.tp_size})")
    else:
        logger.info("No filters specified, running all tests")
        logger.info(f"Total tests to run: {len(test_configs)}")

    # Initialize and run tests
    runner = FusedmoeTestRunner()
    results = runner.run_tests(test_configs)
    runner.print_summary(results)

    # Exit with appropriate code
    failed_count = sum(1 for r in results if r["status"] in ["FAIL", "ERROR"])
    sys.exit(1 if failed_count > 0 else 0)


if __name__ == "__main__":
    main()
