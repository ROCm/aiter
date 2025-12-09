import argparse
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
import triton
from datetime import datetime
from typing import List, Dict, Union, Tuple, Optional, Any
import torch
import pytz
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.live import Live
from rich.layout import Layout
from rich.progress import Progress, BarColumn, TextColumn


from aiter.ops.triton.gemm_a8w8_per_token_scale import gemm_a8w8_per_token_scale  # type: ignore
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH  # type: ignore
from aiter.ops.triton.utils.types import get_fp8_dtypes

mp.set_start_method("spawn", force=True)

# Get FP8 data types
e5m2_type, e4m3_type = get_fp8_dtypes()


class TimeoutError(Exception):
    """Custom exception for timeout errors."""

    pass


# Global variables to track bad configurations and current state
BAD_CONFIGS = {
    "timeouts": [],
    "out_of_resources": [],
    "assert_errors": [],
    "other_errors": [],
}

# Global variables to track current state for SIGINT handling
CURRENT_CONFIG: Dict[str, Any] = {
    "M": None,
    "N": None,
    "K": None,
    "config": None,
    "config_index": None,
    "total_configs": None,
    "batch_size": None,
    "weight_shape_index": None,
    "total_weight_shapes": None,
    "gpu_id": None,
}

INTERRUPTED = False


class GMT8Formatter(logging.Formatter):
    """Custom formatter that uses GMT+8 timezone."""

    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self.gmt8 = pytz.timezone("Asia/Shanghai")  # GMT+8

    def formatTime(self, record, datefmt=None):
        # Convert timestamp to GMT+8
        dt = datetime.fromtimestamp(record.created, tz=self.gmt8)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.strftime("%Y-%m-%d %H:%M:%S")

    def format(self, record):
        # Add timezone info to the formatted message
        original = super().format(record)
        return original.replace("[GMT+8]", "")  # Remove any existing timezone tag


def get_timestamped_filename(base_name: str, extension: str = ".log") -> str:
    """Generate a filename with timestamp in GMT+8 timezone."""
    gmt8 = pytz.timezone("Asia/Shanghai")
    timestamp = datetime.now(gmt8).strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}{extension}"


def sigint_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) gracefully by logging the current configuration."""
    global INTERRUPTED
    global CURRENT_CONFIG
    INTERRUPTED = True

    print("\n" + "=" * 80)
    print("🛑 TUNING INTERRUPTED BY USER (Ctrl+C)")
    print("=" * 80)

    if CURRENT_CONFIG["M"] is not None:
        print("📍 Last configuration being processed:")
        print(f"   🎯 GPU: {CURRENT_CONFIG['gpu_id']}")
        print(
            f"   📊 Matrix: M={CURRENT_CONFIG['M']} N={CURRENT_CONFIG['N']} K={CURRENT_CONFIG['K']}"
        )
        print(f"   📦 Batch Size: {CURRENT_CONFIG['batch_size']}")
        print(
            f"   🔄 Progress: Config {CURRENT_CONFIG['config_index'] + 1}/{CURRENT_CONFIG['total_configs']}"
        )
        print(
            f"   🏗️  Weight Shape: {CURRENT_CONFIG['weight_shape_index'] + 1}/{CURRENT_CONFIG['total_weight_shapes']}"
        )

        if CURRENT_CONFIG["config"]:
            config = CURRENT_CONFIG["config"]
            print("   ⚙️  Parameters:")
            print(f"      BLOCK_SIZE_M: {config.get('BLOCK_SIZE_M', 'N/A')}")
            print(f"      BLOCK_SIZE_N: {config.get('BLOCK_SIZE_N', 'N/A')}")
            print(f"      BLOCK_SIZE_K: {config.get('BLOCK_SIZE_K', 'N/A')}")
            print(f"      num_warps: {config.get('num_warps', 'N/A')}")
            print(f"      num_stages: {config.get('num_stages', 'N/A')}")
            print(f"      NUM_KSPLIT: {config.get('NUM_KSPLIT', 'N/A')}")
            print(f"      waves_per_eu: {config.get('waves_per_eu', 'N/A')}")
            print(f"      kpack: {config.get('kpack', 'N/A')}")
            print(f"      cache_modifier: {config.get('cache_modifier', 'N/A')}")
            print(f"      GROUP_SIZE_M: {config.get('GROUP_SIZE_M', 'N/A')}")

            # Show config in same format as console output for consistency
            config_num = CURRENT_CONFIG["config_index"] + 1
            console_format = f"   💻 Config {config_num} (INTERRUPTED): | BM:{config.get('BLOCK_SIZE_M')} BN:{config.get('BLOCK_SIZE_N')} BK:{config.get('BLOCK_SIZE_K')} W:{config.get('num_warps')} S:{config.get('num_stages')} KS:{config.get('NUM_KSPLIT')} kpack:{config.get('kpack')} cache:{config.get('cache_modifier')}"
            print(console_format)

        # Log the interruption to the file if logger is available
        try:
            logger = logging.getLogger("gemm_a8w8_per_token_scale_tuning")
            if logger.handlers:
                # Use GMT+8 timestamp for consistency
                gmt8 = pytz.timezone("Asia/Shanghai")

                # Create detailed log entry
                detailed_log_entry = {
                    "timestamp": datetime.now(gmt8).isoformat(),
                    "event_type": "user_interrupt",
                    "gpu_id": CURRENT_CONFIG.get("gpu_id", "N/A"),
                    "batch_size": CURRENT_CONFIG["batch_size"],
                    "matrix_dims": f"M={CURRENT_CONFIG['M']} N={CURRENT_CONFIG['N']} K={CURRENT_CONFIG['K']}",
                    "config": CURRENT_CONFIG["config"],
                    "progress": f"Config {CURRENT_CONFIG['config_index'] + 1}/{CURRENT_CONFIG['total_configs']}",
                    "weight_shape_progress": f"Shape {CURRENT_CONFIG['weight_shape_index'] + 1}/{CURRENT_CONFIG['total_weight_shapes']}",
                }

                # Log detailed interruption info
                logger.info(f"=== USER INTERRUPT ===")
                logger.info(
                    f"Interrupted while testing: Config {CURRENT_CONFIG['config_index'] + 1}/{CURRENT_CONFIG['total_configs']}"
                )
                logger.info(f"GPU: {CURRENT_CONFIG.get('gpu_id', 'N/A')}")
                logger.info(
                    f"Matrix: M={CURRENT_CONFIG['M']} N={CURRENT_CONFIG['N']} K={CURRENT_CONFIG['K']}"
                )
                logger.info(
                    f"Weight Shape Progress: {CURRENT_CONFIG['weight_shape_index'] + 1}/{CURRENT_CONFIG['total_weight_shapes']}"
                )

                # Log config details in same format as console output for consistency
                if CURRENT_CONFIG["config"]:
                    config = CURRENT_CONFIG["config"]
                    config_num = CURRENT_CONFIG["config_index"] + 1
                    config_str = f"Config {config_num} (INTERRUPTED): | BM:{config.get('BLOCK_SIZE_M')} BN:{config.get('BLOCK_SIZE_N')} BK:{config.get('BLOCK_SIZE_K')} W:{config.get('num_warps')} S:{config.get('num_stages')} KS:{config.get('NUM_KSPLIT')} kpack:{config.get('kpack')} cache:{config.get('cache_modifier')} GROUP_SIZE_M:{config.get('GROUP_SIZE_M')}"
                    logger.info(f"CONFIG_DETAILS: {config_str}")

                logger.info(f"DETAILED_ENTRY: {detailed_log_entry}")
                logger.info(f"=== END USER INTERRUPT ===")

                # Force flush to write immediately
                for handler in logger.handlers:
                    if hasattr(handler, "stream"):
                        handler.stream.flush()

                print("   📝 Interruption logged to tuning log file")
        except Exception as e:
            print(f"   ⚠️  Could not log interruption: {e}")

    print("\n💡 You can use this information to:")
    print("   • Skip this problematic configuration in future runs")
    print("   • Analyze why this specific config might be causing issues")
    print("   • Adjust the search space to avoid similar parameter combinations")
    print("=" * 80)

    # Exit gracefully
    import sys

    sys.exit(1)


def setup_logger(log_file_path: str, mode: str = "a") -> logging.Logger:
    """
    Setup logger for recording bad configurations during tuning.

    Args:
        log_file_path: Path to the log file
        mode: File write mode - 'a' to append to existing logs, 'w' to overwrite

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("gemm_a8w8_per_token_scale_tuning")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # Create file handler with live writing (immediate flush)
    # Default to append mode to preserve logs across resume sessions
    file_handler = logging.FileHandler(log_file_path, mode=mode)
    file_handler.setLevel(logging.INFO)

    # Create custom formatter that flushes immediately
    file_handler.flush = lambda: file_handler.stream.flush()  # type: ignore

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    # Create GMT+8 formatter
    formatter = GMT8Formatter(
        "%(asctime)s [GMT+8] - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def log_bad_config(
    logger: logging.Logger,
    error_type: str,
    M: int,
    N: int,
    K: int,
    config: Dict[str, Union[str, int]],
    error_msg: str = "",
):
    """
    Log a bad configuration that failed during tuning.

    Args:
        logger: Logger instance
        error_type: Type of error ('timeout', 'out_of_resources', 'assert_error', 'other_error')
        M, N, K: Matrix dimensions
        config: Configuration that failed
        error_msg: Additional error message
    """
    # Use GMT+8 timestamp for consistency
    gmt8 = pytz.timezone("Asia/Shanghai")
    log_entry = {
        "timestamp": datetime.now(gmt8).isoformat(),
        "error_type": error_type,
        "batch_size": M,
        "matrix_dims": f"M={M} N={N} K={K}",
        "config": config,
        "error_msg": str(error_msg),
    }

    # Log to file
    logger.info(f"BAD_CONFIG_{error_type.upper()}: {log_entry}")

    # Force flush to write immediately
    for handler in logger.handlers:
        if hasattr(handler, "stream"):
            handler.stream.flush()

    # Store in global list for summary
    if error_type == "timeout":
        BAD_CONFIGS["timeouts"].append(log_entry)
    elif error_type == "out_of_resources":
        BAD_CONFIGS["out_of_resources"].append(log_entry)
    elif error_type == "assert_error":
        BAD_CONFIGS["assert_errors"].append(log_entry)
    else:
        BAD_CONFIGS["other_errors"].append(log_entry)


def log_bad_config_summary(logger: logging.Logger, total_configs_tested: int):
    """
    Log a summary of all bad configurations encountered during tuning.

    Args:
        logger: Logger instance
        total_configs_tested: Total number of configurations tested
    """
    total_bad = (
        len(BAD_CONFIGS["timeouts"])
        + len(BAD_CONFIGS["out_of_resources"])
        + len(BAD_CONFIGS["assert_errors"])
        + len(BAD_CONFIGS["other_errors"])
    )
    success_rate = (
        ((total_configs_tested - total_bad) / total_configs_tested * 100)
        if total_configs_tested > 0
        else 0
    )

    logger.info("=" * 80)
    logger.info("BAD CONFIGURATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total configurations tested: {total_configs_tested}")
    logger.info(f"Successful configurations: {total_configs_tested - total_bad}")
    logger.info(f"Failed configurations: {total_bad}")
    logger.info(f"Success rate: {success_rate:.2f}%")
    logger.info("")

    logger.info(f"Timeouts: {len(BAD_CONFIGS['timeouts'])}")
    logger.info(f"Out of Resources: {len(BAD_CONFIGS['out_of_resources'])}")
    logger.info(f"Assert Errors: {len(BAD_CONFIGS['assert_errors'])}")
    logger.info(f"Other Errors: {len(BAD_CONFIGS['other_errors'])}")
    logger.info("")

    if BAD_CONFIGS["timeouts"]:
        logger.info("TIMEOUT CONFIGS (most problematic):")
        for entry in BAD_CONFIGS["timeouts"]:
            config = entry["config"]
            logger.info(
                f"  - Batch {entry['batch_size']} | {entry['matrix_dims']} | BM:{config.get('BLOCK_SIZE_M', 'N/A')}, BN:{config.get('BLOCK_SIZE_N', 'N/A')}, BK:{config.get('BLOCK_SIZE_K', 'N/A')}, W:{config.get('num_warps', 'N/A')}, S:{config.get('num_stages', 'N/A')}, KS:{config.get('NUM_KSPLIT', 'N/A')}"
            )

    if BAD_CONFIGS["out_of_resources"]:
        logger.info("OUT OF RESOURCE CONFIGS:")
        for entry in BAD_CONFIGS["out_of_resources"]:
            config = entry["config"]
            logger.info(
                f"  - Batch {entry['batch_size']} | {entry['matrix_dims']} | BM:{config.get('BLOCK_SIZE_M', 'N/A')}, BN:{config.get('BLOCK_SIZE_N', 'N/A')}, BK:{config.get('BLOCK_SIZE_K', 'N/A')}, W:{config.get('num_warps', 'N/A')}, S:{config.get('num_stages', 'N/A')}, KS:{config.get('NUM_KSPLIT', 'N/A')}"
            )

    logger.info("=" * 80)

    # Print summary to console as well
    print("\n📊 Bad Configuration Summary:")
    print(f"   Total tested: {total_configs_tested}")
    print(f"   ✅ Successful: {total_configs_tested - total_bad}")
    print(
        f"   ❌ Failed: {total_bad} ({len(BAD_CONFIGS['timeouts'])} timeouts, {len(BAD_CONFIGS['out_of_resources'])} OOM, {len(BAD_CONFIGS['assert_errors'])} assert, {len(BAD_CONFIGS['other_errors'])} other)"
    )
    print(f"   📈 Success rate: {success_rate:.1f}%")


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Kernel execution timed out")


def run_with_timeout(func, timeout_seconds=3, *args, **kwargs):
    """
    Run a function with a timeout limit.

    Args:
        func: Function to execute
        timeout_seconds: Timeout in seconds (default: 3)
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Result of the function call

    Raises:
        TimeoutError: If function execution exceeds timeout
    """
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        result = func(*args, **kwargs)
        return result
    finally:
        # Cancel the alarm and restore the old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def generate_gemm_a8w8_per_token_scale_inputs(M, N, K, dtype, output=True, bias=False):
    """
    Generate inputs for gemm_a8w8_per_token_scale kernel.

    Args:
        M, N, K: Matrix dimensions
        dtype: Output data type
        output: Whether to generate output tensor
        bias: Whether to generate bias tensor

    Returns:
        Tuple of (x, w, x_scale, w_scale, bias, y)
    """
    # Generate input matrix x (M, K) - FP8 E4M3 (matching test file pattern)
    x = (torch.rand((M, K), dtype=torch.float16, device="cuda") / 10).to(e4m3_type)

    # Generate weight matrix w (N, K) - FP8 E4M3 (matching test file pattern)
    w = (torch.rand((N, K), dtype=torch.float16, device="cuda") / 10).to(e4m3_type)

    # Generate per-token and per-channel scale tensors - 2D [M,1] and [N,1]
    x_scale = torch.rand([M, 1], dtype=torch.float32, device="cuda")
    w_scale = torch.rand([N, 1], dtype=torch.float32, device="cuda")

    # Generate bias tensor if needed
    bias_tensor = None
    if bias:
        bias_tensor = torch.empty((N), dtype=dtype, device="cuda")

    # Generate output tensor if needed
    y = None
    if output:
        y = torch.empty((M, N), dtype=dtype, device="cuda")

    return x, w, x_scale, w_scale, bias_tensor, y


def get_configs_compute_bound() -> List[Dict[str, int | str]]:
    """
    Generate configuration space for tuning the gemm_a8w8_per_token_scale kernel.
    Focus on parameters that affect performance for this specific kernel.
    Comprehensive search space matching atomic kernel patterns.
    """
    configs = []

    # Explore optimized parameter space (removed large block sizes that cause slowdowns)
    # for num_stages in [1, 2, 3, 4]:
    #         for block_m in [32, 64, 128]:  # Removed 256 (causes slowdowns)
    #             for block_n in [32, 64, 128]:  # Removed 256 (causes slowdowns)
    #                 for block_k in [64, 128, 256]:
    #                     for group_size in [1, 8, 16]:
    #                         for num_warps in [2, 4, 8]:
    #                             for num_ksplit in [
    #                                 1,
    #                                 2,
    #                                 4,
    #                             ]:  # Key parameter for K-splitting
    #                                 for waves_per_eu in [2, 4, 8]:
    #                                     for kpack in [2]:
    #                                         for cache_modifier in ["", ".cg"]:
    #                                             configs.append(
    #                                                 {
    #                                                     "BLOCK_SIZE_M": block_m,
    #                                                     "BLOCK_SIZE_N": block_n,
    #                                                     "BLOCK_SIZE_K": block_k,
    #                                                     "GROUP_SIZE_M": group_size,
    #                                                     "num_warps": num_warps,
    #                                                     "num_stages": num_stages,
    #                                                     "NUM_KSPLIT": num_ksplit,
    #                                                     "waves_per_eu": waves_per_eu,
    #                                                     "kpack": kpack,
    #                                                     "matrix_instr_nonkdim": 16,  # Fixed value from atomic kernel
    #                                                     "cache_modifier": cache_modifier,
    #                                                 }
    #                                             )
    #     return configs
    for num_stages in [
        1,
    ]:
        for block_m in [
            32,
        ]:  # Removed 256 (causes slowdowns)
            for block_n in [
                32,
            ]:  # Removed 256 (causes slowdowns)
                for block_k in [
                    64,
                ]:
                    for group_size in [
                        1,
                    ]:
                        for num_warps in [
                            2,
                        ]:
                            for num_ksplit in [
                                1,
                            ]:  # Key parameter for K-splitting
                                for waves_per_eu in [
                                    2,
                                ]:
                                    for kpack in [2]:
                                        for cache_modifier in ["", ".cg"]:
                                            configs.append(
                                                {
                                                    "BLOCK_SIZE_M": block_m,
                                                    "BLOCK_SIZE_N": block_n,
                                                    "BLOCK_SIZE_K": block_k,
                                                    "GROUP_SIZE_M": group_size,
                                                    "num_warps": num_warps,
                                                    "num_stages": num_stages,
                                                    "NUM_KSPLIT": num_ksplit,
                                                    "waves_per_eu": waves_per_eu,
                                                    "kpack": kpack,
                                                    "matrix_instr_nonkdim": 16,  # Fixed value from atomic kernel
                                                    "cache_modifier": cache_modifier,
                                                }
                                            )
    return configs


def get_weight_shapes(tp_size: int) -> List[Tuple[int, int]]:
    """Get weight shapes to test during tuning."""
    total = [
        # (1024, 1024),
        # (4096, 1024),
        # (1024, 2048),
        # (6144, 1024),
        (1024, 3072),
    ]

    weight_shapes: List[Tuple[int, int]] = []
    for t in total:
        weight_shapes.append(t)

    return weight_shapes


def run_torch_reference(x, w, x_scale, w_scale, bias, dtype=torch.bfloat16):
    """
    Run reference implementation using PyTorch.
    This is used for correctness verification.
    """
    # Apply scaling as in test file: convert to scale dtype, multiply, then compute
    x = x.to(x_scale.dtype) * x_scale
    w = w.to(w_scale.dtype) * w_scale

    # Compute the matrix multiplication - note weights need to be transposed for torch linear
    out = torch.nn.functional.linear(x.to(torch.float32), w.to(torch.float32))

    return out.to(dtype)


def benchmark_config(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    dtype: torch.dtype,
    config: Dict[str, Union[str, int]],
    y: Optional[torch.Tensor] = None,
    num_iters=10,
) -> float:
    """
    Benchmark the performance of a GEMM operation with a specific configuration.

    This function measures the execution time of the gemm_a8w8_per_token_scale kernel by running
    it multiple times with synchronization points to ensure accurate timing. It performs
    warmup runs before the actual benchmarking to account for JIT compilation overhead.

    Args:
        x (torch.Tensor): Input tensor of shape (M, K) representing the first matrix operand.
        w (torch.Tensor): Weight tensor of shape (N, K) representing the second matrix operand.
        x_scale (torch.Tensor): Per-token scale tensor for x with shape (M, 1).
        w_scale (torch.Tensor): Per-output-channel scale tensor for w with shape (N, 1).
        dtype (torch.dtype): Data type for the computation (e.g., torch.bfloat16).
        config (Dict[str, Union[str, int]]): Configuration dictionary containing kernel
            parameters such as block sizes, number of warps, etc.
        y (Optional[torch.Tensor], optional): Output tensor to store the result. If None,
            a new tensor will be allocated. Defaults to None.
        num_iters (int, optional): Number of benchmark iterations to run. Defaults to 10.

    Returns:
        float: Average execution time in microseconds (us) per iteration.

    Note:
        The function performs 5 warmup iterations before benchmarking to account for
        JIT compilation and GPU warmup effects. The timing is measured using CUDA events
        for accurate GPU kernel timing.
    """
    # Get reference output for correctness verification
    torch_out = run_torch_reference(x, w, x_scale, w_scale, bias, dtype)

    # Add SPLITK_BLOCK_SIZE computation as done in the kernel function
    _, K = x.shape
    _, K = w.shape
    num_ksplit = int(config["NUM_KSPLIT"])
    block_k = int(config["BLOCK_SIZE_K"])
    splitk_block_size = triton.cdiv(K, num_ksplit)

    config["SPLITK_BLOCK_SIZE"] = splitk_block_size
    if block_k > splitk_block_size:
        block_k = triton.next_power_of_2(splitk_block_size)
        if block_k > splitk_block_size:
            block_k = block_k // 4
    block_k = max(block_k, 16)
    config["BLOCK_SIZE_K"] = block_k

    # Run kernel
    def run():
        return gemm_a8w8_per_token_scale(x, w, x_scale, w_scale, dtype, y, config)

    torch.cuda.synchronize()

    # JIT compilation & warmup with timeout for entire warmup phase
    def run_warmup():
        for i in range(5):
            run()

    try:
        run_with_timeout(run_warmup, timeout_seconds=3)
    except TimeoutError:
        # If warmup times out, this config is likely bad, skip it
        raise TimeoutError("Warmup phase timed out after 3 seconds")
    torch.cuda.synchronize()

    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)

    latencies: list[float] = []
    for i in range(num_iters):
        torch.cuda.synchronize()
        start_event.record()
        try:
            triton_out_raw = run_with_timeout(run, timeout_seconds=3)
        except TimeoutError:
            # If benchmark iteration times out, skip this config
            raise TimeoutError(f"Benchmark iteration {i + 1} timed out after 3 seconds")

        # Convert to the same dtype as the reference for comparison
        # Handle the case where triton_out_raw might be None
        if triton_out_raw is not None:
            triton_out = triton_out_raw.to(torch_out.dtype)
        else:
            triton_out = torch_out  # Fallback to reference output
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
        torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    return avg


# Global variable to store console output
console_output = []


def create_live_display(
    M: int,
    N: int,
    K: int,
    current_config: Dict[str, Union[str, int]],
    best_config: Dict[str, Union[str, int]],
    best_time: float,
    config_index: int,
    total_configs: int,
    console_messages: Optional[List[str]] = None,
) -> Layout:
    """Create a live display layout with current and best configuration tables."""

    layout = Layout()

    # Use global console_output if none provided
    if console_messages is None:
        global console_output
        console_messages = console_output

    # Create progress bar
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
    )
    task = progress.add_task(
        f"🔧 Tuning M={M} N={N} K={K} | Batch Size={M}",
        total=total_configs,
        completed=config_index,
    )

    # Create status information
    status_text = ""
    if best_time != float("inf"):
        status_text = f"🏆 Best Performance: {best_time:.1f}μs"
    else:
        status_text = "🔍 Searching for best configuration..."

    # Create console area
    console_text = (
        "\n".join(console_messages[-10:])
        if console_messages
        else "Waiting for results..."
    )
    console_table = Table(show_header=False, box=None, padding=0)
    console_table.add_column("Output", style="white")
    console_table.add_row(console_text)

    # Create current config table
    config_table = Table(
        title="Current Configuration",
        show_header=True,
        header_style="bold magenta",
    )
    config_table.add_column("Parameter", style="cyan", width=15)
    config_table.add_column("Value", style="green", width=10)

    # Add matrix dimensions and batch size first
    config_table.add_row("[bold yellow]Matrix M[/bold yellow]", str(M), style="yellow")
    config_table.add_row("[bold yellow]Matrix N[/bold yellow]", str(N), style="yellow")
    config_table.add_row("[bold yellow]Matrix K[/bold yellow]", str(K), style="yellow")
    config_table.add_row(
        "[bold yellow]Batch Size[/bold yellow]", str(M), style="yellow"
    )
    config_table.add_row("", "")  # Separator
    config_table.add_row("BLOCK_SIZE_M", str(current_config.get("BLOCK_SIZE_M", "N/A")))
    config_table.add_row("BLOCK_SIZE_N", str(current_config.get("BLOCK_SIZE_N", "N/A")))
    config_table.add_row("BLOCK_SIZE_K", str(current_config.get("BLOCK_SIZE_K", "N/A")))
    config_table.add_row("num_warps", str(current_config.get("num_warps", "N/A")))
    config_table.add_row("num_stages", str(current_config.get("num_stages", "N/A")))
    config_table.add_row("NUM_KSPLIT", str(current_config.get("NUM_KSPLIT", "N/A")))
    config_table.add_row("waves_per_eu", str(current_config.get("waves_per_eu", "N/A")))
    config_table.add_row("kpack", str(current_config.get("kpack", "N/A")))
    config_table.add_row(
        "cache_modifier", str(current_config.get("cache_modifier", "N/A"))
    )
    config_table.add_row("GROUP_SIZE_M", str(current_config.get("GROUP_SIZE_M", "N/A")))

    # Create best config table if we have a best configuration
    best_config_table = None
    if best_time != float("inf"):
        best_config_table = Table(
            title="🏆 Best Configuration So Far",
            show_header=True,
            header_style="bold green",
        )
        best_config_table.add_column("Parameter", style="cyan", width=15)
        best_config_table.add_column("Value", style="green", width=10)

        # Add performance and matrix dimensions
        best_config_table.add_row(
            "[bold green]Performance[/bold green]", f"{best_time:.1f}μs", style="green"
        )
        best_config_table.add_row("", "")  # Separator
        best_config_table.add_row(
            "[bold yellow]BLOCK_SIZE_M[/bold yellow]",
            str(best_config.get("BLOCK_SIZE_M", "N/A")),
            style="yellow",
        )
        best_config_table.add_row(
            "[bold yellow]BLOCK_SIZE_N[/bold yellow]",
            str(best_config.get("BLOCK_SIZE_N", "N/A")),
            style="yellow",
        )
        best_config_table.add_row(
            "[bold yellow]BLOCK_SIZE_K[/bold yellow]",
            str(best_config.get("BLOCK_SIZE_K", "N/A")),
            style="yellow",
        )
        best_config_table.add_row("num_warps", str(best_config.get("num_warps", "N/A")))
        best_config_table.add_row(
            "num_stages", str(best_config.get("num_stages", "N/A"))
        )
        best_config_table.add_row(
            "NUM_KSPLIT", str(best_config.get("NUM_KSPLIT", "N/A"))
        )
        best_config_table.add_row(
            "waves_per_eu", str(best_config.get("waves_per_eu", "N/A"))
        )
        best_config_table.add_row("kpack", str(best_config.get("kpack", "N/A")))
        best_config_table.add_row(
            "cache_modifier", str(best_config.get("cache_modifier", "N/A"))
        )
        best_config_table.add_row(
            "GROUP_SIZE_M", str(best_config.get("GROUP_SIZE_M", "N/A"))
        )

    # Create combined layout
    if best_config_table:
        # Display tables side by side
        tables = Columns([config_table, best_config_table], equal=True, expand=True)
        layout.split_column(
            Layout(Panel(progress, title="Progress", border_style="blue"), size=5),
            Layout(Panel(status_text, title="Status", border_style="green"), size=3),
            Layout(tables),
            Layout(
                Panel(console_table, title="Console Output", border_style="cyan"),
                size=10,
            ),
        )
    else:
        # Display only current config
        layout.split_column(
            Layout(Panel(progress, title="Progress", border_style="blue"), size=5),
            Layout(Panel(status_text, title="Status", border_style="green"), size=3),
            Layout(config_table),
            Layout(
                Panel(console_table, title="Console Output", border_style="cyan"),
                size=10,
            ),
        )

    return layout


def tune(
    M: int,
    N: int,
    K: int,
    search_space: List[Dict[str, int | str]],
    input_type: str,
    logger: logging.Logger,
):
    """Tune the kernel for specific matrix dimensions."""
    # Register SIGINT handler if not already registered
    if not signal.getsignal(signal.SIGINT) == sigint_handler:
        signal.signal(signal.SIGINT, sigint_handler)

    if input_type == "bfloat16":
        # Use the same input generation as test file
        x, w, x_scale, w_scale, bias, y = generate_gemm_a8w8_per_token_scale_inputs(
            M, N, K, torch.bfloat16, bias=True
        )
    else:
        raise RuntimeError(
            "Currently, only support tune a8w8 per-token scale kernel with bfloat16 output."
        )

    best_config: Dict[str, Union[int, str]] = {}
    best_time = float("inf")
    slow_config_threshold = (
        1000  # microseconds - configs slower than this get highlighted
    )

    # Clear console output for fresh start
    global console_output
    console_output = []

    # Initialize Rich console for better formatting
    console = Console()

    # Create initial live display
    initial_layout = create_live_display(
        M,
        N,
        K,
        search_space[0] if search_space else {},
        best_config,
        best_time,
        0,
        len(search_space),
        console_output,
    )

    # Create progress display with Rich and Live
    with Live(initial_layout, refresh_per_second=4, console=console) as live:
        for i, config in enumerate(search_space):
            # Check if we were interrupted
            if INTERRUPTED:
                break

            # Update global state for SIGINT handling
            CURRENT_CONFIG.update(
                {
                    "M": M,
                    "N": N,
                    "K": K,
                    "config": config,
                    "config_index": i,
                    "total_configs": len(search_space),
                    "batch_size": M,
                }
            )

            # Update live display with current configuration
            layout = create_live_display(
                M,
                N,
                K,
                config,
                best_config,
                best_time,
                i + 1,
                len(search_space),
                console_output,
            )
            live.update(layout)

            try:
                kernel_time = benchmark_config(
                    x=x,
                    w=w,
                    x_scale=x_scale,
                    w_scale=w_scale,
                    bias=bias,
                    dtype=torch.bfloat16,
                    y=None,
                    config=config,
                    num_iters=10,
                )

                # Add kernel time to console output
                if kernel_time > slow_config_threshold:
                    console_msg = f"[yellow]⚠️  Config {i + 1} (SLOW): {kernel_time:.1f}μs | BM:{config.get('BLOCK_SIZE_M')} BN:{config.get('BLOCK_SIZE_N')} BK:{config.get('BLOCK_SIZE_K')} W:{config.get('num_warps')} S:{config.get('num_stages')} KS:{config.get('NUM_KSPLIT')}[/yellow]"
                    live.console.print(console_msg)
                    console_output.append(console_msg)
                else:
                    console_msg = f"[green]✅ Config {i + 1}: {kernel_time:.1f}μs | BM:{config.get('BLOCK_SIZE_M')} BN:{config.get('BLOCK_SIZE_N')} BK:{config.get('BLOCK_SIZE_K')} W:{config.get('num_warps')} S:{config.get('num_stages')} KS:{config.get('NUM_KSPLIT')}[/green]"
                    console_output.append(console_msg)

                # Update best time and config
                if kernel_time < best_time:
                    best_time = kernel_time
                    best_config = config
                    best_msg = (
                        f"[bold green]🏆 NEW BEST: {kernel_time:.1f}μs![/bold green]"
                    )
                    console_output.append(best_msg)

                # Update live display with current configuration and console output
                layout = create_live_display(
                    M,
                    N,
                    K,
                    config,
                    best_config,
                    best_time,
                    i + 1,
                    len(search_space),
                    console_output,
                )
                live.update(layout)

            except triton.runtime.autotuner.OutOfResources as e:
                # Log and skip out of resources configurations
                log_bad_config(logger, "out_of_resources", M, N, K, config, str(e))
                error_msg = f"[red]❌ Config {i + 1} (OOM): Out of resources | BM:{config.get('BLOCK_SIZE_M')} BN:{config.get('BLOCK_SIZE_N')} BK:{config.get('BLOCK_SIZE_K')} W:{config.get('num_warps')} S:{config.get('num_stages')} KS:{config.get('NUM_KSPLIT')}[/red]"
                console_output.append(error_msg)
                live.console.print(error_msg)
                continue
            except AssertionError as e:
                # Log and skip assert error configurations
                log_bad_config(logger, "assert_error", M, N, K, config, str(e))
                error_msg = f"[red]❌ Config {i + 1} (ASSERT): Assert error | BM:{config.get('BLOCK_SIZE_M')} BN:{config.get('BLOCK_SIZE_N')} BK:{config.get('BLOCK_SIZE_K')} W:{config.get('num_warps')} S:{config.get('num_stages')} KS:{config.get('NUM_KSPLIT')} | {e}[/red]"
                console_output.append(error_msg)
                live.console.print(error_msg)
                continue
            except TimeoutError as e:
                # Log and skip timeout configurations
                log_bad_config(logger, "timeout", M, N, K, config, str(e))
                error_msg = f"[orange1]⏱️ Config {i + 1} (TIMEOUT): {e} | BM:{config.get('BLOCK_SIZE_M')} BN:{config.get('BLOCK_SIZE_N')} BK:{config.get('BLOCK_SIZE_K')} W:{config.get('num_warps')} S:{config.get('num_stages')} KS:{config.get('NUM_KSPLIT')}[/orange1]"
                console_output.append(error_msg)
                live.console.print(error_msg)
                continue
            except Exception as e:
                # Log and skip other error configurations
                log_bad_config(logger, "other_error", M, N, K, config, str(e))
                error_msg = f"[red]💥 Config {i + 1} (ERROR): {e} | BM:{config.get('BLOCK_SIZE_M')} BN:{config.get('BLOCK_SIZE_N')} BK:{config.get('BLOCK_SIZE_K')} W:{config.get('num_warps')} S:{config.get('num_stages')} KS:{config.get('NUM_KSPLIT')}[/red]"
                console_output.append(error_msg)
                live.console.print(error_msg)
                continue

    # Show final completion message with Rich
    print("\n" + "=" * 70)

    # Create best config table with matrix dimensions
    best_table = Table(
        title="🏆 Best Configuration Found", show_header=True, header_style="bold green"
    )
    best_table.add_column("Parameter", style="cyan", width=15)
    best_table.add_column("Value", style="green", width=10)

    # Add matrix dimensions and batch size first
    best_table.add_row(
        "[bold yellow]Matrix M (Batch)[/bold yellow]", str(M), style="yellow"
    )
    best_table.add_row("[bold yellow]Matrix N[/bold yellow]", str(N), style="yellow")
    best_table.add_row("[bold yellow]Matrix K[/bold yellow]", str(K), style="yellow")
    best_table.add_row("", "")  # Separator
    best_table.add_row(
        "[bold green]Performance[/bold green]", f"{best_time:.1f}μs", style="green"
    )
    best_table.add_row("", "")  # Separator
    best_table.add_row(
        "[bold yellow]BLOCK_SIZE_M[/bold yellow]",
        str(best_config.get("BLOCK_SIZE_M", "N/A")),
        style="yellow",
    )
    best_table.add_row(
        "[bold yellow]BLOCK_SIZE_N[/bold yellow]",
        str(best_config.get("BLOCK_SIZE_N", "N/A")),
        style="yellow",
    )
    best_table.add_row(
        "[bold yellow]BLOCK_SIZE_K[/bold yellow]",
        str(best_config.get("BLOCK_SIZE_K", "N/A")),
        style="yellow",
    )
    best_table.add_row("num_warps", str(best_config.get("num_warps", "N/A")))
    best_table.add_row("num_stages", str(best_config.get("num_stages", "N/A")))
    best_table.add_row("NUM_KSPLIT", str(best_config.get("NUM_KSPLIT", "N/A")))
    best_table.add_row("waves_per_eu", str(best_config.get("waves_per_eu", "N/A")))
    best_table.add_row("kpack", str(best_config.get("kpack", "N/A")))
    best_table.add_row("cache_modifier", str(best_config.get("cache_modifier", "N/A")))
    best_table.add_row("GROUP_SIZE_M", str(best_config.get("GROUP_SIZE_M", "N/A")))
    best_table.add_row(
        "matrix_instr_nonkdim", str(best_config.get("matrix_instr_nonkdim", "N/A"))
    )

    completion_panel = Panel(
        best_table,
        title=f"[bold green]✅ Completed Tuning for M={M} N={N} K={K} (Batch Size={M})[/bold green]",
        border_style="green",
    )
    console.print(completion_panel)
    print("=" * 70)

    assert best_config is not None
    return best_config


def save_configs(
    N,
    K,
    configs,
    save_path,
    is_incremental=False,
    completed_batch_sizes=None,
) -> None:
    """Save the best configurations to a JSON file."""
    os.makedirs(save_path, exist_ok=True)
    device_name = "R9700"  # TODO: Hardcoded, make it dynamic

    if is_incremental:
        # Save incremental progress with batch size info in filename
        batch_sizes_str = (
            "_".join(map(str, completed_batch_sizes))
            if completed_batch_sizes
            else "partial"
        )
        json_file_name = f"{device_name}-GEMM-A8W8_PER_TOKEN_SCALE-N={N}-K={K}_batch_{batch_sizes_str}.json"
        progress_file = os.path.join(
            save_path,
            f"{device_name}-GEMM-A8W8_PER_TOKEN_SCALE-N={N}-K={K}_progress.json",
        )

        # Save progress info
        progress_info = {
            "completed_batch_sizes": completed_batch_sizes or [],
            "configs": configs,
            "last_updated": datetime.now().isoformat(),
        }

        with open(progress_file, "w") as f:
            json.dump(progress_info, f, indent=4)
            f.write("\n")
    else:
        json_file_name = f"{device_name}-GEMM-A8W8_PER_TOKEN_SCALE-N={N}-K={K}.json"

    config_file_path = os.path.join(save_path, json_file_name)

    # Add incremental flag to filename
    action = "Updating incremental" if is_incremental else "Writing"
    print(f"{action} config to {config_file_path}...")

    with open(config_file_path, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")


def load_progress(N, K, save_path):
    """Load previously saved progress for a given N,K configuration."""
    device_name = "R9700"  # TODO: Hardcoded, make it dynamic
    progress_file = os.path.join(
        save_path, f"{device_name}-GEMM-A8W8_PER_TOKEN_SCALE-N={N}-K={K}_progress.json"
    )

    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                progress_info = json.load(f)
            return progress_info.get("completed_batch_sizes", []), progress_info.get(
                "configs", {}
            )
        except Exception as e:
            print(f"Warning: Could not load progress file {progress_file}: {e}")
            return [], {}
    return [], {}


def tune_on_gpu(
    gpu_id: int,
    batch_sizes: List[int],
    weight_shapes: List[Tuple[int, int]],
    input_type: str,
    resume: bool = True,
    log_filename: Optional[str] = None,
) -> None:
    """Run tuning on a specific GPU."""
    # Register SIGINT handler and set GPU ID in global state
    signal.signal(signal.SIGINT, sigint_handler)
    CURRENT_CONFIG["gpu_id"] = gpu_id

    torch.cuda.set_device(gpu_id)
    print(f"🚀 Starting tuning on GPU {gpu_id} with batch sizes {batch_sizes}")

    save_path = AITER_TRITON_CONFIGS_PATH + "/gemm/"

    # Setup logger for this GPU with custom or timestamped filename
    if log_filename:
        # Use custom filename, ensure it has .log extension
        if not log_filename.endswith(".log"):
            log_filename += ".log"
        # If no path separator, assume it's just a filename
        if "/" not in log_filename and "\\" not in log_filename:
            log_filename = os.path.join(save_path, log_filename)
        else:
            log_filename = log_filename  # Use full path as provided
    else:
        # Fall back to timestamped filename
        log_filename = os.path.join(
            save_path,
            get_timestamped_filename(
                f"tune_a8w8_per_token_scale_bad_configs_gpu{gpu_id}"
            ),
        )

    # Choose appropriate logging mode: append for resume, overwrite for fresh start
    log_mode = "a" if resume else "w"
    logger = setup_logger(log_filename, mode=log_mode)

    # Log the start time in GMT+8
    gmt8 = pytz.timezone("Asia/Shanghai")
    start_time_gmt8 = datetime.now(gmt8).strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"=== TUNING SESSION STARTED AT {start_time_gmt8} [GMT+8] ===")
    logger.info(f"GPU: {gpu_id}")
    logger.info(f"Batch sizes: {batch_sizes}")

    search_space = get_configs_compute_bound()
    total_configs = len(search_space)
    total_tests = total_configs * len(batch_sizes) * len(weight_shapes)

    print(f"   📊 Search space: {total_configs:,} configurations")
    print(f"   🎯 Total tests to run: {total_tests:,}")
    print(
        f"   ⚡  Estimated tests per weight shape: {total_configs * len(batch_sizes):,}"
    )
    log_action = (
        "Appending to existing"
        if resume and os.path.exists(log_filename)
        else "Writing to new"
    )
    print(f"   📝 Bad configurations will be logged to: {log_filename}")
    print(f"   📝 Logging mode: {log_action}")

    start = time.time()

    # Collect all configs to determine the best overall config
    all_configs: List[Dict[str, Dict[str, int | str]]] = []

    for i, shape in enumerate(weight_shapes):
        # Check if we were interrupted
        if INTERRUPTED:
            break

        # Update weight shape tracking
        CURRENT_CONFIG.update(
            {"weight_shape_index": i, "total_weight_shapes": len(weight_shapes)}
        )

        N, K = shape[0], shape[1]
        print(
            f"\n🚀 [GPU {gpu_id}] Shape {i + 1}/{len(weight_shapes)}: Starting tuning for N:{N}, K:{K}"
        )
        print(
            f"   📊 Testing {len(search_space):,} configurations across {len(batch_sizes)} batch sizes"
        )

        # Check for existing progress and resume from there (if resume is enabled)
        if resume:
            completed_batch_sizes, existing_configs = load_progress(N, K, save_path)
        else:
            completed_batch_sizes, existing_configs = [], {}

        # Filter batch_sizes to only those not yet completed
        remaining_batch_sizes = [
            bs for bs in batch_sizes if bs not in completed_batch_sizes
        ]

        if completed_batch_sizes and resume:
            print(f"\n   📂 [GPU {gpu_id}] Found progress for N={N}, K={K}")
            print(f"   ✅ Already completed batch sizes: {completed_batch_sizes}")
            print(f"   🔄 Remaining batch sizes to tune: {remaining_batch_sizes}")
        elif not resume:
            print(
                f"\n   🔄 [GPU {gpu_id}] Starting fresh (resume disabled) for N={N}, K={K}"
            )
        elif not remaining_batch_sizes:
            print(
                f"\n   ✅ [GPU {gpu_id}] All batch sizes already completed for N={N}, K={K}"
            )
            # Add existing configs to all_configs and continue to next shape
            if existing_configs:
                all_configs.append(existing_configs)
                save_configs(N, K, existing_configs, save_path)
            continue

        # Initialize benchmark_results with existing results if any
        benchmark_results :List[Dict[str, str |int]]= []
        if existing_configs:
            # Reconstruct benchmark_results from existing configs
            # We need to map the configs back to their corresponding batch sizes
            for i, batch_size in enumerate(batch_sizes):
                if batch_size in completed_batch_sizes:
                    # Find the config for this batch size
                    config_to_add = None

                    # Try to find matching config based on batch size category
                    if batch_size < 32 and "small" in existing_configs:
                        config_to_add = existing_configs["small"]
                    elif batch_size <= 128:
                        BLK_M = triton.next_power_of_2(batch_size)
                        if BLK_M == 32 and "medium_M32" in existing_configs:
                            config_to_add = existing_configs["medium_M32"]
                        elif BLK_M == 64 and "medium_M64" in existing_configs:
                            config_to_add = existing_configs["medium_M64"]
                        elif BLK_M == 128 and "medium_M128" in existing_configs:
                            config_to_add = existing_configs["medium_M128"]
                    elif batch_size <= 256 and "large" in existing_configs:
                        config_to_add = existing_configs["large"]
                    elif batch_size > 256 and "xlarge" in existing_configs:
                        config_to_add = existing_configs["xlarge"]

                    if config_to_add:
                        benchmark_results.append(config_to_add)
                    else:
                        # If we couldn't find a matching config, we'll need to retune this batch size
                        remaining_batch_sizes.append(batch_size)

        for batch_size in remaining_batch_sizes:
            # Check if we were interrupted
            if INTERRUPTED:
                break

            print(
                f"\n   🔍 [GPU {gpu_id}] Testing batch size M={batch_size} for N={N}, K={K}"
            )
            result = tune(
                batch_size,
                N,
                K,
                search_space,
                input_type,
                logger,
            )

            # Check if tune() was interrupted
            if INTERRUPTED:
                break

            benchmark_results.append(result)

            # Save incremental progress immediately after each batch size
            updated_completed_batch_sizes = completed_batch_sizes + [batch_size]

            # Create configs for different M size categories as expected by the kernel
            incremental_configs: Dict[str, Dict[str, int | str]] = {}
            for i, (M, config) in enumerate(
                zip(batch_sizes[: len(benchmark_results)], benchmark_results)
            ):
                if i == len(batch_sizes[: len(benchmark_results)]) - 1:
                    incremental_configs["any"] = config
                elif M < 32:
                    incremental_configs["small"] = config
                elif M <= 128:
                    BLK_M = triton.next_power_of_2(M)
                    if BLK_M == 32:
                        incremental_configs["medium_M32"] = config
                    elif BLK_M == 64:
                        incremental_configs["medium_M64"] = config
                    elif BLK_M == 128:
                        incremental_configs["medium_M128"] = config
                elif M <= 256:
                    incremental_configs["large"] = config
                else:
                    incremental_configs["xlarge"] = config

            # Save the incremental progress
            save_configs(
                N,
                K,
                incremental_configs,
                save_path,
                is_incremental=True,
                completed_batch_sizes=updated_completed_batch_sizes,
            )

            print(f"   💾 [GPU {gpu_id}] Saved progress for batch size {batch_size}")

            # Update completed_batch_sizes for next iteration
            completed_batch_sizes = updated_completed_batch_sizes

        # Create final configs for different M size categories as expected by the kernel
        best_configs: Dict[str, Dict[str, int | str]] = {}
        for i, (M, config) in enumerate(zip(batch_sizes, benchmark_results)):
            if i == len(batch_sizes) - 1:
                best_configs["any"] = config
            elif M < 32:
                best_configs["small"] = config
            elif M <= 128:
                BLK_M = triton.next_power_of_2(M)
                if BLK_M == 32:
                    best_configs["medium_M32"] = config
                elif BLK_M == 64:
                    best_configs["medium_M64"] = config
                elif BLK_M == 128:
                    best_configs["medium_M128"] = config
            elif M <= 256:
                best_configs["large"] = config
            else:
                best_configs["xlarge"] = config

        # Store configs for later analysis
        all_configs.append(best_configs)

        # Save the final complete config (non-incremental)
        save_configs(N, K, best_configs, save_path)

        # Clean up progress file since we completed successfully
        device_name = "R9700"  # TODO: Hardcoded, make it dynamic
        progress_file = os.path.join(
            save_path,
            f"{device_name}-GEMM-A8W8_PER_TOKEN_SCALE-N={N}-K={K}_progress.json",
        )
        if os.path.exists(progress_file):
            os.remove(progress_file)
            print(f"   🧹 [GPU {gpu_id}] Cleaned up progress file for N={N}, K={K}")

    # Create a default config file (without N,K parameters) by selecting the most common config
    default_config = create_default_config(all_configs)
    save_default_config(default_config, save_path)

    end = time.time()

    # Log session end time in GMT+8
    gmt8 = pytz.timezone("Asia/Shanghai")
    end_time_gmt8 = datetime.now(gmt8).strftime("%Y-%m-%d %H:%M:%S")
    duration = end - start
    logger.info(f"=== TUNING SESSION COMPLETED AT {end_time_gmt8} [GMT+8] ===")
    logger.info(f"Total duration: {duration:.2f} seconds")

    # Log summary of bad configurations
    log_bad_config_summary(logger, total_tests)

    print(f"Tuning on GPU {gpu_id} took {duration:.2f} seconds")


def create_default_config(
    all_configs: List[Dict[str, Dict[str, Union[int, str]]]],
) -> Dict[str, Dict[str, Union[int, str]]]:
    """Create a default config by selecting the most common config across all shapes."""
    from collections import Counter

    # Collect all configs for each category
    category_configs = {
        "small": [],
        "medium_M32": [],
        "medium_M64": [],
        "medium_M128": [],
        "large": [],
        "xlarge": [],
        "any": [],
    }

    for config in all_configs:
        for category, params in config.items():
            if category in category_configs:
                # Convert config to a hashable tuple for counting
                config_tuple = tuple(sorted(params.items()))
                category_configs[category].append(config_tuple)

    # Find the most common config for each category
    default_config: Dict[str, Dict[str, Union[int, str]]] = {}
    for category, configs in category_configs.items():
        if configs:
            most_common = Counter(configs).most_common(1)[0][0]
            default_config[category] = dict(most_common)

    return default_config


def save_default_config(
    config: Dict[str, Dict[str, Union[int, str]]], save_path: str
) -> None:
    """Save the default config file (without N,K parameters)."""
    os.makedirs(save_path, exist_ok=True)
    device_name = "R9700"  # TODO: Hardcoded, make it dynamic
    json_file_name = f"{device_name}-GEMM-A8W8_PER_TOKEN_SCALE.json"

    config_file_path = os.path.join(save_path, json_file_name)
    print(f"Writing default config to {config_file_path}...")

    with open(config_file_path, "w") as f:
        json.dump(config, f, indent=4)
        f.write("\n")


def distribute_batch_sizes(batch_sizes: List[int], num_gpus: int) -> List[List[int]]:
    """Distribute batch sizes across available GPUs."""
    batches_per_gpu: List[List[int]] = []
    for i in range(num_gpus):
        start_idx = i * len(batch_sizes) // num_gpus
        end_idx = (i + 1) * len(batch_sizes) // num_gpus
        batches_per_gpu.append(batch_sizes[start_idx:end_idx])
    return batches_per_gpu


def main(args):
    print(args)
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPU available for tuning")
    print(f"Found {num_gpus} GPUs for parallel tuning")

    torch.cuda.init()

    if args.batch_size is None:
        batch_sizes = [
            16,  # For small config
            32,  # For medium_M32 config
            64,  # For medium_M64 config
            128,  # For medium_M128 config
            256,  # For large config
            512,  # For large config
            2048,  # For xlarge config
            4096,  # For xlarge config
        ]
    else:
        batch_sizes = [args.batch_size]
        num_gpus = 1  # If only one batch size, use only one GPU

    weight_shapes = get_weight_shapes(args.tp_size)

    batches_per_gpu = distribute_batch_sizes(batch_sizes, num_gpus)

    # Prepare arguments for each GPU process
    process_args = []
    for gpu_id in range(num_gpus):
        process_args.append(
            (
                gpu_id,
                batches_per_gpu[gpu_id],
                weight_shapes,  # Each GPU processes all weight shapes
                args.input_type,
                args.resume,
                args.log_filename,
            )
        )

    # Set up signal handler for main process to gracefully terminate workers
    def main_sigint_handler(signum, frame):  # type: ignore
        print("\n" + "=" * 80)
        print("🛑 MAIN PROCESS INTERRUPTED BY USER (Ctrl+C)")
        print("📡 Sending termination signal to worker processes...")
        print("⏳ Giving workers 3 seconds to log their current state...")
        print("=" * 80)
        # Set a flag for workers to check and give them time to cleanup
        global INTERRUPTED
        INTERRUPTED = True
        import time

        time.sleep(3)  # Give workers time to handle the signal and log
        sys.exit(1)

    # Register main process signal handler
    if not signal.getsignal(signal.SIGINT) == main_sigint_handler:
        signal.signal(signal.SIGINT, main_sigint_handler)

    ctx = mp.get_context("spawn")
    try:
        with ctx.Pool(num_gpus) as pool:
            pool.starmap(tune_on_gpu, process_args)
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received in main process")
        print("📡 Worker processes terminated")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error in main process: {e}")
        sys.exit(1)

    print("Multi-GPU tuning completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--tp-size", "-tp", type=int, default=1)
    parser.add_argument(
        "--input-type", type=str, choices=["bfloat16"], default="bfloat16"
    )
    parser.add_argument(
        "--out-dtype",
        type=str,
        choices=["float32", "float16", "bfloat16", "half"],
        default="bfloat16",
    )
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument(
        "--log-filename",
        type=str,
        default=None,
        help="Custom log filename (without .log extension). If not provided, timestamped filename will be used.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume functionality and start fresh tuning",
    )
    args = parser.parse_args()

    # Convert no_resume flag to resume boolean
    args.resume = not args.no_resume

    main(args)
