import pytest
from collections import defaultdict
import os
import numpy as np
from prettytable import PrettyTable
import torch
import torch.profiler as tpf
from typing import Callable, Generator, TypeVar
from functools import partial

T = TypeVar("T")

AITER_LOG_MORE = int(os.environ.get('AITER_LOG_MORE', 0))
NUM_ITERATIONS = int(os.environ.get('NUM_ITERATIONS', 100))
NUM_WARMUP_ITERATION = os.environ.get('NUM_WARMUP_ITERATION', 10)
USE_CUDA_GRAPH = os.environ.get('USE_CUDA_GRAPH', 0)

def execute_callback(
    num_iterations: int,
    func: Callable[..., T],
    *args,
    **kwargs
) -> T:
    """
    Executes a callback for a given number of iterations.

    Parameters
    ----------
        num_iterations : int.
            The number of iterations to use for profiling the callback.
        func : Callable[..., T].
            A callback function with arbitrary arguments to be executed.
        *args
            Variable length argument list for the callback function.
        **kwargs
            Keyword arguments for the callback function.

    Returns
    -------
        T: The last value returned by the callback function.

    """
    for _ in range(num_iterations):
        result = func(*args, **kwargs)
    return result

def time_callback_with_cuda_event(
    num_iterations: int,
    func: Callable[..., T],
    *args,
    **kwargs
) -> list[float]:
    """
    Measure the average execution time of a given function using CUDA
    events in milliseconds.
    
    Parameters
    ----------
        num_iterations : int.
            The number of iterations to use for profiling the callback.
        func : Callable[..., T].
            A callback function with arbitrary arguments to be executed.
        *args
            Variable length argument list for the callback function.
        **kwargs
            Keyword arguments for the callback function.
    Returns
    -------
        list[float]: The executions times in milliseconds.
    """
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    latency_list = []
    for _ in range(num_iterations):
        start_event.record()
        func(*args, **kwargs)
        end_event.record()
        end_event.synchronize()
        latency = start_event.elapsed_time(end_event)
        latency_list.append(latency)
    return latency_list

def extract_cuda_time_trace(torch_profiler: torch.profiler) -> list[float]:
    """
    Extract the CUDA times from a PyTorch profiler trace.

    This function extracts self device time for all CUDA events
    in the profiler trace, excluding the first event.

    Parameters
    ----------
        torch_profiler : torch.profiler
            A PyTorch profiler object containing trace events.

    Returns
    -------
        float: list[float]: The executions times in milliseconds.

    """
    get_cuda_total_time = lambda event: getattr(event, "self_device_time_total", 0.0)
    is_cuda = lambda event: getattr(event, "device_type", None) == torch.profiler.DeviceType.CUDA
    
    if len(torch_profiler.events()) <=1:
        return 0.0
    
    return[
            get_cuda_total_time(event) / 1000
            for event in torch_profiler.events()[1:]
            if is_cuda(event) 
        ]

def profile(
    num_iterations: int,
    num_warmup_iterations:int,
    func: Callable[..., T],
    *args,
    **kwargs
) -> tuple[list[float], T]:
    """
    Profile the execution of a function using PyTorch Profiler.

    This function performs warmup iterations, then profiles the actual execution
    of the function for a specified number of iterations.
    
    Parameters
    ----------
        num_iterations : int.
            The number of iterations to use for profiling the callback.
        num_warmup_iterations : int.
            The number of iterations to use for warmup before profiling the callback.
        func : Callable[..., T].
            A callback function with arbitrary arguments to be executed.
        *args
            Variable length argument list for the callback function.
        **kwargs
            Keyword arguments for the callback function.

    Returns
    -------
        tuple[list[float], T]: A tuple containing:
            - list[float]: The executions times in milliseconds.
            - T: The result of the last function execution.
    """

    #  warmup
    result = execute_callback(num_warmup_iterations, func, *args, **kwargs)
    
    # Profile using cuda.Event
    # Note: The use of AITER_LOG_MORE variable 
    # is temporary (until we completly shift to using pytest)
    # and should be replaced with a more descriptive
    # flag.
    if AITER_LOG_MORE:
        latency_list = time_callback_with_cuda_event(
            num_iterations, func, *args, **kwargs)
        return latency_list, result
    
    # Profile using torch.profiler 
    with tpf.profile(
        activities=[
            tpf.ProfilerActivity.CPU,
            tpf.ProfilerActivity.CUDA
        ],
        profile_memory=True,
        with_stack=True,
        with_modules=True,
    ) as prof:
        execute_callback(num_iterations, func, *args, **kwargs)

    latency_list = extract_cuda_time_trace(prof)
    return latency_list, result

def profile_cuda_graph(
    num_iterations: int,
    num_warmup_iterations:int,
    func: Callable[..., T],
    *args,
    **kwargs
) -> tuple[list[float], T]:
    """
    Profile the execution of a function using CUDA Graph and PyTorch Profiler.

    This function creates a CUDA Graph for the given function, then profiles its
    execution using the standard profile function.

    Parameters
    ----------
        num_iterations : int.
            The number of iterations to use for profiling the callback.
        num_warmup_iterations : int.
            The number of iterations to use for warmup before profiling the callback.
        func : Callable[..., T].
            A callback function with arbitrary arguments to be executed.
        *args
            Variable length argument list for the callback function.
        **kwargs
            Keyword arguments for the callback function.

    Returns
    -------
        tuple[list[float], T]: A tuple containing:
            - list[float]: The executions times in milliseconds.
            - T: The result of the last function execution.
    """
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        execute_callback(1, func, *args, **kwargs)
    return profile(num_iterations, num_warmup_iterations, func, *args, **kwargs)


@pytest.fixture(scope="session")
def benchmark() -> Generator[Callable[..., tuple[float, T]], None, None]:
    """
    A pytest fixture that provides a benchmarking function for performance testing.

    This fixture creates a generator that yields a function which can be used to benchmark 
    other functions, optionally using CUDA graphs. It collects statistics on execution times 
    and presents the results in a formatted table at the end of the test session.

    Yields
    ------
        Callable[..., Tuple[float, T]]: A benchmarking function that can be called within tests.

    Usage
    -----
        def test_example(benchmark):
            avg_time, result = benchmark(
                func=my_function_to_test,
                arg1=value1,
                arg2=value2
            )
    """
    test_result_data: dict[str, list[dict[str, float]]] = defaultdict(list)

    def _benchmark(
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        
        test_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
        profile_func = profile_cuda_graph if USE_CUDA_GRAPH else profile

        execution_time_list, result = profile_func(
            NUM_ITERATIONS,
            NUM_WARMUP_ITERATION,
            func,
            *args,
            **kwargs
        )

        stats = {
            "test_name": test_name,
            "avg_time": np.mean(execution_time_list),
            "std_time": np.std(execution_time_list),
            "min_time": np.min(execution_time_list),
            "max_time": np.max(execution_time_list),
            "iterations": NUM_ITERATIONS,
            "warmup": NUM_WARMUP_ITERATION,
            "cuda_graph": USE_CUDA_GRAPH
        }
        test_result_data[test_name].append(stats)
        
        return result

    yield _benchmark

    table = PrettyTable()
    table.field_names = ["Test Name", "Avg Time (ms)", "Std Dev (ms)", "Min Time (ms)", "Max Time (ms)", "Iterations", "Warmup", "CUDA Graph"]
    
    for test_results in test_result_data.values():
        for result in test_results:
            table.add_row([
                result["test_name"],
                f"{result['avg_time']:.6f}",
                f"{result['std_time']:.6f}",
                f"{result['min_time']:.6f}",
                f"{result['max_time']:.6f}",
                result["iterations"],
                result["warmup"],
                "Yes" if result["cuda_graph"] else "No"
            ])

    table.align = "r"  # Right-align all columns
    table.align["Test"] = "l"  # Left-align the Test Name column
    table.float_format = ".6"  # Set float precision to 6 decimal places
    table.title = "Benchmark Results"
    table.border = True
    table.header = True

    print("\n", table)

