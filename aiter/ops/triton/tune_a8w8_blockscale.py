import argparse
import json
import logging
import multiprocessing as mp
import os
import signal
import time
import triton
from datetime import datetime
from typing import List, Dict, Union, Tuple, Optional, Any
import torch
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.panel import Panel
from tqdm import tqdm


from aiter.ops.triton.gemm_a8w8_blockscale import gemm_a8w8_blockscale  # type: ignore
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


def sigint_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) gracefully by logging the current configuration."""
    global INTERRUPTED
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

        # Log the interruption to the file if logger is available
        try:
            logger = logging.getLogger("gemm_a8w8_blockscale_tuning")
            if logger.handlers:
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "user_interrupt",
                    "batch_size": CURRENT_CONFIG["batch_size"],
                    "matrix_dims": f"M={CURRENT_CONFIG['M']} N={CURRENT_CONFIG['N']} K={CURRENT_CONFIG['K']}",
                    "config": CURRENT_CONFIG["config"],
                    "progress": f"Config {CURRENT_CONFIG['config_index'] + 1}/{CURRENT_CONFIG['total_configs']}",
                    "weight_shape_progress": f"Shape {CURRENT_CONFIG['weight_shape_index'] + 1}/{CURRENT_CONFIG['total_weight_shapes']}",
                }
                logger.info(f"USER_INTERRUPT: {log_entry}")

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


def setup_logger(log_file_path: str) -> logging.Logger:
    """
    Setup logger for recording bad configurations during tuning.

    Args:
        log_file_path: Path to the log file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("gemm_a8w8_blockscale_tuning")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # Create file handler with live writing (immediate flush)
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setLevel(logging.INFO)

    # Create custom formatter that flushes immediately
    file_handler.flush = lambda: file_handler.stream.flush()  # type: ignore

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
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
    log_entry = {
        "timestamp": datetime.now().isoformat(),
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


def generate_gemm_a8w8_blockscale_inputs(
    M: int,
    N: int,
    K: int,
    block_shape_n: int,
    block_shape_k: int,
    dtype=torch.bfloat16,
    layout: str = "TN",
    output=False,
):
    """
    The GEMM kernel expects:
    - x: (M, K) -> row-major format
    - w: (N, K) -> column-major format
    """
    scale_n = (N + block_shape_n - 1) // block_shape_n
    scale_k = (K + block_shape_k - 1) // block_shape_k

    if layout[0] == "T":
        x = (torch.rand((M, K), dtype=torch.float16, device="cuda") / 10).to(e4m3_type)
    else:
        x = (
            (torch.rand((K, M), dtype=torch.float16, device="cuda") / 10)
            .to(e4m3_type)
            .T
        )

    if layout[1] == "N":
        weight = (torch.rand((N, K), dtype=torch.float16, device="cuda") / 10).to(
            e4m3_type
        )
    else:
        weight = (
            (torch.rand((K, N), dtype=torch.float16, device="cuda") / 10)
            .to(e4m3_type)
            .T
        )

    x_scale = torch.rand([M, scale_k], dtype=torch.float32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=torch.float32, device="cuda")

    y = None
    if output:
        y = torch.empty((M, N), dtype=dtype, device="cuda").cuda()

    return x, weight, x_scale, w_scale, y


def get_configs_compute_bound() -> List[Dict[str, int | str]]:
    """
    Generate configuration space for tuning the gemm_a8w8_blockscale kernel.
    Based on the sample config file, we'll tune around those values.
    Note: GROUP_K must equal BLOCK_SIZE_K as required by the kernel.
    """
    configs = []
    # Start with the known working configuration from MI300X-GEMM-A8W8_BLOCKSCALE.json
    base_config = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 2,
        "matrix_instr_nonkdim": 16,
        "NUM_KSPLIT": 1,
        "kpack": 2,
        "cache_modifier": ".cg",
    }

    # Add the base config first (known to work)
    configs.append(base_config.copy())

    # Generate variations around the base config, but be conservative
    for block_m in [
        32,
        64,
        128,
    ]:
        for block_n in [
            32,
            64,
            128,
        ]:
            for block_k in [64, 128]:  # Keep as power of 2
                for num_warps in [4, 8]:
                    for num_stages in [2, 3, 4, 5]:
                        for waves_per_eu in [2, 4, 8]:
                            for cache_modifier in [
                                ".cg",
                                "",
                            ]:  # Start with cache modifier
                                config = {
                                    "BLOCK_SIZE_M": block_m,
                                    "BLOCK_SIZE_N": block_n,
                                    "BLOCK_SIZE_K": block_k,
                                    "GROUP_K": block_k,
                                    "GROUP_SIZE_M": 1,  # Keep fixed for now
                                    "num_warps": num_warps,
                                    "num_stages": num_stages,
                                    "waves_per_eu": waves_per_eu,  # Keep fixed for now
                                    "matrix_instr_nonkdim": 16,
                                    "NUM_KSPLIT": 1,
                                    "kpack": 2,  # Keep fixed for now
                                    "cache_modifier": cache_modifier,
                                }
                                configs.append(config)

    print(f"Generated {len(configs)} configurations")
    return configs


def get_weight_shapes(tp_size: int = 1) -> List[Tuple[int, int]]:
    """Get weight shapes to test during tuning."""
    total = [
        (1024, 1024),
        (4096, 1024),
        (1024, 2048),
        (6144, 1024),
        (1024, 3072),
    ]

    weight_shapes: List[Tuple[int, int]] = []
    for t in total:
        weight_shapes.append(t)

    return weight_shapes


def run_torch(
    x, weight, x_scale, w_scale, block_shape: Tuple[int, int], dtype=torch.bfloat16
):
    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = weight.shape[0]

    # Expand scales to match the block sizes
    x_scale_expanded = x_scale.repeat_interleave(block_shape_k, dim=1)
    x_dequant = x.to(x_scale_expanded.dtype) * x_scale_expanded[:m, :k]

    # Expand weight scales: first repeat along N dimension, then along K dimension
    w_scale_expanded = w_scale.repeat_interleave(block_shape_n, dim=0)
    w_scale_expanded = w_scale_expanded.repeat_interleave(block_shape_k, dim=1)
    w_scale_expanded = w_scale_expanded[:n, :k]
    weight_dequant = weight.to(w_scale_expanded.dtype) * w_scale_expanded

    out = torch.nn.functional.linear(
        x_dequant.to(torch.float32), weight_dequant.to(torch.float32)
    )

    return out.to(dtype)


def benchmark_config(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: torch.dtype,
    config: Dict[str, Union[str, int]],
    y: Optional[torch.Tensor] = None,
    num_iters=10,
) -> float:
    """
    Benchmark the performance of a GEMM operation with a specific configuration.

    This function measures the execution time of the gemm_a8w8_blockscale kernel by running
    it multiple times with synchronization points to ensure accurate timing. It performs
    warmup runs before the actual benchmarking to account for JIT compilation overhead.

    Args:
        x (torch.Tensor): Input tensor of shape (M, K) representing the first matrix operand.
        w (torch.Tensor): Weight tensor of shape (N, K) representing the second matrix operand.
        x_scale (torch.Tensor): Scale tensor for x with shape (M, scale_k).
        w_scale (torch.Tensor): Scale tensor for w with shape (scale_n, scale_k).
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

    torch_out = run_torch(
        x,
        w,
        x_scale,
        w_scale,
        (128, 128),  # follow test using (128,128)
        dtype,
    )

    # Run kernel
    def run():
        # Pass the modified config to the kernel
        return gemm_a8w8_blockscale(
            x, w, x_scale, w_scale, dtype, y, config, skip_reduce=False
        )

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

    if input_type != "bfloat16":
        raise RuntimeError(
            "Currently, only support tune a8w8 blockscale kernel with bfloat16 output."
        )

    best_config: Dict[str, Union[int, str]] = {}
    best_time = float("inf")
    slow_config_threshold = (
        1000  # microseconds - configs slower than this get highlighted
    )

    # Initialize Rich console for better formatting
    console = Console()

    # Create progress display with Rich
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,  # Keep progress bar visible
    ) as progress:
        task = progress.add_task(
            f"🔧 Tuning M={M} N={N} K={K}",
            total=len(search_space),
        )

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

            # Update progress
            progress.update(
                task,
                advance=1,
                description=f"🔧 Testing config {i + 1}/{len(search_space)}",
            )

            # Show current config (only every 10 configs to avoid flicker)
            if i % 10 == 0 or i == len(search_space) - 1:
                # Create fresh config table with matrix dimensions and batch size
                config_table = Table(
                    title="Current Configuration",
                    show_header=True,
                    header_style="bold magenta",
                )
                config_table.add_column("Parameter", style="cyan", width=15)
                config_table.add_column("Value", style="green", width=10)

                # Add matrix dimensions and batch size first
                config_table.add_row(
                    "[bold yellow]Matrix M[/bold yellow]", str(M), style="yellow"
                )
                config_table.add_row(
                    "[bold yellow]Matrix N[/bold yellow]", str(N), style="yellow"
                )
                config_table.add_row(
                    "[bold yellow]Matrix K[/bold yellow]", str(K), style="yellow"
                )
                config_table.add_row(
                    "[bold yellow]Batch Size[/bold yellow]", str(M), style="yellow"
                )
                config_table.add_row("", "")  # Separator
                config_table.add_row(
                    "BLOCK_SIZE_M", str(config.get("BLOCK_SIZE_M", "N/A"))
                )
                config_table.add_row(
                    "BLOCK_SIZE_N", str(config.get("BLOCK_SIZE_N", "N/A"))
                )
                config_table.add_row(
                    "BLOCK_SIZE_K", str(config.get("BLOCK_SIZE_K", "N/A"))
                )
                config_table.add_row("num_warps", str(config.get("num_warps", "N/A")))
                config_table.add_row("num_stages", str(config.get("num_stages", "N/A")))
                config_table.add_row("NUM_KSPLIT", str(config.get("NUM_KSPLIT", "N/A")))
                config_table.add_row(
                    "waves_per_eu", str(config.get("waves_per_eu", "N/A"))
                )

                # Create summary header with all tuning parameters
                header_text = f"[bold blue]🔧 Tuning M={M} N={N} K={K} | Batch Size={M} | Config {i + 1}/{len(search_space)}[/bold blue]"

                # Show config info (don't clear screen to avoid issues in multiprocessing)
                console.print(f"\n{header_text}")
                console.print(config_table)

                if best_time != float("inf"):
                    console.print(
                        f"[yellow]🏆 Best time so far: {best_time:.1f}μs[/yellow]"
                    )
                console.print("─" * 70)

            try:
                # Use the same input generation as test file
                x, w, x_scale, w_scale, _ = generate_gemm_a8w8_blockscale_inputs(
                    M,
                    N,
                    K,
                    int(config["BLOCK_SIZE_N"]),
                    int(config["BLOCK_SIZE_K"]),
                    torch.bfloat16,
                )
                kernel_time = benchmark_config(
                    x=x,
                    w=w,
                    x_scale=x_scale,
                    w_scale=w_scale,
                    dtype=torch.bfloat16,
                    y=None,
                    config=config,
                    num_iters=10,
                )

                # Warn about slow configs
                if kernel_time > slow_config_threshold:
                    console.print(
                        f"\n[bold yellow]⚠️  SLOW CONFIG DETECTED: {kernel_time:.1f}μs[/bold yellow]"
                    )
                    console.print(
                        f"[cyan]📊 Matrix: M={M} N={N} K={K} | Config:[/cyan] BM:{config.get('BLOCK_SIZE_M', 'N/A')}, BN:{config.get('BLOCK_SIZE_N', 'N/A')}, BK:{config.get('BLOCK_SIZE_K', 'N/A')}, W:{config.get('num_warps', 'N/A')}, S:{config.get('num_stages', 'N/A')}, KS:{config.get('NUM_KSPLIT', 'N/A')}"
                    )

                # Update best time and config
                if kernel_time < best_time:
                    best_time = kernel_time
                    best_config = config

            except triton.runtime.autotuner.OutOfResources as e:
                # Log and skip out of resources configurations
                log_bad_config(logger, "out_of_resources", M, N, K, config, str(e))
                console.print(
                    f"\n[bold red]⚠️  Out of resources for M={M} N={N} K={K} - logged[/bold red]"
                )
                continue
            except AssertionError as e:
                # Log and skip assert error configurations
                log_bad_config(logger, "assert_error", M, N, K, config, str(e))
                console.print(
                    f"\n[bold red]❌ Assert error for M={M} N={N} K={K} - logged[/bold red]"
                )
                console.print(f"[red]💬 Error:[/red] {e}")
                continue
            except TimeoutError as e:
                # Log and skip timeout configurations
                log_bad_config(logger, "timeout", M, N, K, config, str(e))
                console.print(
                    f"\n[bold orange1]⏱️  TIMEOUT for M={M} N={N} K={K} - logged[/bold orange1]"
                )
                console.print(f"[orange1]💬 Timeout:[/orange1] {e}")
                continue
            except Exception as e:
                # Log and skip other error configurations
                log_bad_config(logger, "other_error", M, N, K, config, str(e))
                console.print(
                    f"\n[bold red]💥 Unexpected error for M={M} N={N} K={K} - logged[/bold red]"
                )
                console.print(f"[red]💬 Error:[/red] {e}")
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
    best_table.add_row("Performance", f"{best_time:.1f}μs")
    best_table.add_row("BLOCK_SIZE_M", str(best_config.get("BLOCK_SIZE_M", "N/A")))
    best_table.add_row("BLOCK_SIZE_N", str(best_config.get("BLOCK_SIZE_N", "N/A")))
    best_table.add_row("BLOCK_SIZE_K", str(best_config.get("BLOCK_SIZE_K", "N/A")))
    best_table.add_row("num_warps", str(best_config.get("num_warps", "N/A")))
    best_table.add_row("num_stages", str(best_config.get("num_stages", "N/A")))
    best_table.add_row("NUM_KSPLIT", str(best_config.get("NUM_KSPLIT", "N/A")))
    best_table.add_row("waves_per_eu", str(best_config.get("waves_per_eu", "N/A")))

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
) -> None:
    """Save the best configurations to a JSON file."""
    os.makedirs(save_path, exist_ok=True)
    device_name = "R9700"  # TODO: Hardcoded, make it dynamic
    json_file_name = f"{device_name}-GEMM-A8W8_BLOCKSCALE-N={N}-K={K}.json"

    config_file_path = os.path.join(save_path, json_file_name)
    print(f"Writing best config to {config_file_path}...")

    with open(config_file_path, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")


def tune_on_gpu(
    gpu_id: int,
    batch_sizes: List[int],
    weight_shapes: List[Tuple[int, int]],
    input_type: str,
) -> None:
    """Run tuning on a specific GPU."""
    # Register SIGINT handler and set GPU ID in global state
    signal.signal(signal.SIGINT, sigint_handler)
    CURRENT_CONFIG["gpu_id"] = gpu_id

    torch.cuda.set_device(gpu_id)
    print(f"🚀 Starting tuning on GPU {gpu_id} with batch sizes {batch_sizes}")

    save_path = AITER_TRITON_CONFIGS_PATH + "/gemm/"

    # Setup logger for this GPU with proper prefix
    log_file_path = os.path.join(
        save_path, f"tune_a8w8_blockscale_bad_configs_gpu{gpu_id}.log"
    )
    logger = setup_logger(log_file_path)
    logger.info(f"Starting tuning on GPU {gpu_id} with batch sizes {batch_sizes}")

    search_space = get_configs_compute_bound()
    total_configs = len(search_space)
    total_tests = total_configs * len(batch_sizes) * len(weight_shapes)

    print(f"   📊 Search space: {total_configs:,} configurations")
    print(f"   🎯 Total tests to run: {total_tests:,}")
    print(
        f"   ⚡  Estimated tests per weight shape: {total_configs * len(batch_sizes):,}"
    )
    print(f"   📝 Bad configurations will be logged to: {log_file_path}")

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

        benchmark_results = []
        for batch_size in batch_sizes:
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
        best_configs: Dict[str, Dict[str, int | str]] = {}
        # Create configs for different M size categories as expected by the kernel
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
        save_configs(N, K, best_configs, save_path)

    # Create a default config file (without N,K parameters) by selecting the most common config
    default_config = create_default_config(all_configs)
    save_default_config(default_config, save_path)

    end = time.time()

    # Log summary of bad configurations
    log_bad_config_summary(logger, total_tests)

    print(f"Tuning on GPU {gpu_id} took {end - start:.2f} seconds")


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
    json_file_name = f"{device_name}-GEMM-A8W8_BLOCKSCALE.json"

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
            )
        )

    ctx = mp.get_context("spawn")
    with ctx.Pool(num_gpus) as pool:
        pool.starmap(tune_on_gpu, process_args)

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
    args = parser.parse_args()

    main(args)
