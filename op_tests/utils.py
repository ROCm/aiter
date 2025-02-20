from functools import partial
import numpy as np
import os
import torch
import torch.profiler as tpf
from typing import Callable, Generic, TypeVar

T = TypeVar("T")

MAX_RAND_INT = 20
MIN_RAND_INT = -20

def rand_tensor(
    size: torch.Size,
    dtype: torch.dtype
) -> torch.tensor:
    """
    Generate a random PyTorch tensor with specified shape and data type.

    - For integer types: Uses torch.randint to generate random integers within a fixed range.
    - For float types: Uses torch.rand to generate random floats between 0 and 1.

    Parameters
    ----------
    size: torch.Size
        The size of the generated tensor.
    dtype: torch.dtype
        The data type of the generated tensor.

    Returns
    -------
    torch.Tensor : A random tensor of the specified shape and data type.

    Raises
    ------
    ValueError
        If an unsupported data type is provided.
    """
    if dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        # For integer types, use randint
        return torch.randint(MIN_RAND_INT, MAX_RAND_INT, size=size, dtype=dtype)
    elif dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
        # For float types, use rand
        return torch.rand(size=size, dtype=dtype)
    elif dtype == torch.float8_e4m3fnuz:
        # Special case for float8_e4m3fnuz
        return torch.rand(size=size, dtype=torch.float16).to(torch.float8_e4m3fnuz)

    raise ValueError(f"Unsupported dtype: {dtype}")

def check_all_close(
    a: torch.tensor,
    b: torch.tensor,
    rtol: float,
    atol: float,
) -> None:
    """
    Check if all elements in two tensors are close within specified tolerances.
    
    Parameters
    ----------
    a: torch.tensor
        First input tensor.
    b: torch.tensor
        Second input tensor.
    rtol: float
        Relative tolerence.
    atol: float
        Absolute tolerence.

    Raises
    -------
    AssertionError
        If any elements in 'a' and 'b' are not close within the specified tolerances.
        The error message includes details about the maximum and average delta,
        and the percentage of elements that are not close.
    """
    is_close = torch.isclose(a, b, rtol=rtol, atol=atol)
    is_not_close = ~is_close
    num_not_close = is_not_close.sum()
    delta = (a-b)[is_not_close]
    percent = num_not_close/a.numel()
    message = "" if num_not_close == 0 else f"""
check_all_close failed!
max delta:{delta.max()}
average delta:{delta.mean()}
delta details: {percent:.1%} ({num_not_close} of {a.numel()}) elements
        """
    assert is_close.all(), message

def extract_avg_cuda_time_trace(torch_profiler: torch.profiler) -> float:
    """
    Extract the average CUDA time from a PyTorch profiler trace.

    This function calculates the mean of the self device time for all CUDA events
    in the profiler trace, excluding the first event.

    Parameters
    ----------
        torch_profiler : torch.profiler
            A PyTorch profiler object containing trace events.

    Returns
    -------
        float: The average CUDA time across all CUDA events, or 0.0 if there are
               insufficient events or no CUDA events.
    """
    get_cuda_total_time = lambda event: getattr(event, "self_device_time_total", 0.0)
    is_cuda = lambda event: getattr(event, "device_type", None) == torch.profiler.DeviceType.CUDA
    
    if len(torch_profiler.events()) <=1:
        return 0.0
    
    return np.mean(
        [
            get_cuda_total_time(event)
            for event in torch_profiler.events()[1:]
            if is_cuda(event) 
        ]
    )

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
) -> float:
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
        float: The average execution time in milliseconds over all iterations.
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
    return np.mean(latency_list)

def profile(
    num_iterations: int,
    num_warmup_iterations:int,
    func: Callable[..., T],
    *args,
    **kwargs
) -> tuple[float, T]:
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
        tuple[float, T]: A tuple containing:
            - float: The average execution time in milliseconds.
            - T: The result of the last function execution.
    """

    #  warmup
    result = execute_callback(num_warmup_iterations, func, *args, **kwargs)
    
    # Profile using cuda.Event
    # Note: The use of AITER_LOG_MORE variable 
    # is temporary (until we completly shift to using pytest)
    # and should be replaced with a more descriptive
    # flag.
    if int(os.environ.get('AITER_LOG_MORE', 0)):
        average_latency = time_callback_with_cuda_event(
            num_iterations, func, *args, **kwargs)
        return average_latency, result
    
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

    average_latency = extract_avg_cuda_time_trace(prof)
    return average_latency, result

def profile_cuda_graph(
    num_iterations: int,
    num_warmup_iterations:int,
    func: Callable[..., T],
    *args,
    **kwargs
) -> tuple[float, T]:
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
        tuple[float, T]: A tuple containing:
            - float: The average execution time in milliseconds.
            - T: The result of the last function execution.
    """
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        execute_callback(1, func, *args, **kwargs)
    return profile(num_iterations, num_warmup_iterations, func, *args, **kwargs)


class BenchmarkHook(Generic[T]):
    """
    A generic class for custom benchmarking in pytest using pytest-benchmark.

    This class allows for fine-grained control over benchmarking parameters and
    can be used with the pytest-benchmark fixture to measure function performance.

    Type Parameters:
    ----------------
    T : The return type of the function being benchmarked.

    Attributes
    ----------
    num_iterations : int
        The number of times the function will be executed during benchmarking.
    num_warmup_iterations : int
        The number of times the function will be executed before actual benchmarking begins.
        These iterations are not included in the final measurements.
    func : Callable[..., T]
        The function to be benchmarked.
    args : tuple
        Positional arguments to be passed to the benchmarked function.
    kwargs : dict
        Keyword arguments to be passed to the benchmarked function.

    Methods
    -------
    __call__() -> tuple[float, T]
        Executes the benchmarking process and returns a tuple containing
        the execution time and the result of the benchmarked function.

    Usage
    -----
    def test_example(benchmark):
        def function_to_benchmark(x: int) -> int:
            return x * 2

        hook = BenchmarkHook(
            num_iterations=1000,
            num_warmup_iterations=100,
            func=function_to_benchmark,
            5
        )
        
        execution_time, result = benchmark(hook)
        
        assert result == 10
        assert execution_time > 0
    """

    def __init__(self,
        num_iterations:int,
        num_warmup_iterations:int,
        use_cuda_graph: bool,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> None:
        self.num_iterations = num_iterations
        self.num_warmup_iterations = num_warmup_iterations
        self.use_cuda_graph = use_cuda_graph
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self) -> tuple[float, T]:
        if self.use_cuda_graph:
            return profile_cuda_graph(
                self.num_iterations,
                self.num_warmup_iterations,
                self.func,
                *self.args,
                **self.kwargs
            )
        return profile(
            self.num_iterations,
            self.num_warmup_iterations,
            self.func,
            *self.args,
            **self.kwargs
        )
    
# Define some BenchmarkHook partials for conveniance
DefaultBenchmarkHook = partial(BenchmarkHook, 100, 10, False)
DefaultCudaGraphBenchmarkHook = partial(BenchmarkHook, 100, 10, True)