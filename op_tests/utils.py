from functools import partial
import pandas as pd
import torch
import torch.profiler as tpf
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")

MAX_RAND_INT = 20
MIN_RAND_INT = -20

def rand_tensor(
    shape: tuple[int, int],
    dtype: torch.dtype
) -> torch.tensor:
    """
    Generate a random PyTorch tensor with specified shape and data type.

    - For integer types: Uses torch.randint to generate random integers within a fixed range.
    - For float types: Uses torch.rand to generate random floats between 0 and 1.

    Parameters:
    -----------
    shape : tuple[int, int]
        The shape of the output tensor. Must be a tuple of two integers.
    dtype : torch.dtype
        The desired data type of the output tensor.

    Returns:
    --------
    torch.Tensor
        A random tensor of the specified shape and data type.

    Raises:
    -------
    ValueError
        If an unsupported data type is provided.
    """
    if dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        # For integer types, use randint
        return torch.randint(MIN_RAND_INT, MAX_RAND_INT, shape, dtype=dtype)
    elif dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
        # For float types, use rand
        return torch.rand(shape, dtype=dtype)
    elif dtype == torch.float8_e4m3fnuz:
        # Special case for float8_e4m3fnuz
        return torch.rand(shape, dtype=torch.float16).to(torch.float8_e4m3fnuz)

    raise ValueError(f"Unsupported dtype: {dtype}")

def check_all_close(
    a: torch.tensor,
    b: torch.tensor,
    rtol: float,
    atol: float,
) -> None:
    """
    Check if all elements in two tensors are close within specified tolerances.

    Parameters:
    -----------
    a : torch.Tensor
        First input tensor.
    b : torch.Tensor
        Second input tensor to compare with 'a'.
    rtol : float
        Relative tolerance.
    atol : float
        Absolute tolerance.

    Raises:
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

def get_trace_perf(prof, num_iters):
    # TODO: clean up
    assert (num_iters > 1)
    num_iters -= 1
    df = []
    cols = ['name', 'self_cpu_time_total', 'self_device_time_total',
            'device_type', 'device_index',]
    for el in prof.events():
        df.append([getattr(el, x, None) for x in cols])
    df = pd.DataFrame(df, columns=cols)
    df['cnt'] = 1
    rets = []
    for name, d in df.groupby('name', sort=False):
        r = d.iloc[1:][['cnt',
                        'self_cpu_time_total',
                        'self_device_time_total']].sum()
        if not r.empty:
            device_type = str(d['device_type'].iat[0]).split('.')[-1]
            r['name'] = name
            r['device_type'] = device_type
            r['device_index'] = str(d['device_index'].iat[0])
            if device_type == 'CUDA':
                r['device_time_total'] = r['self_device_time_total']
                r['host_time_total'] = 0
            else:
                r['host_time_total'] = r['self_device_time_total']
                r['device_time_total'] = 0

        rets.append(r)
    df = pd.DataFrame(rets)

    cols = ['name', 'cnt', 'host_time_total', 'device_time_total',
            'device_type', 'device_index',]
    cols = [el for el in cols if el in df.columns]
    df = df[(df.host_time_total > 0) | (df.device_time_total > 0)]

    timerList = ['host_time_total', 'device_time_total', ]
    df = df[cols].sort_values(timerList, ignore_index=True)
    avg_name = '[avg us/iter]'
    for el in timerList:
        df.at[avg_name, el] = df[el].sum()/num_iters
    return df.at[avg_name, 'device_time_total']

def execute_callback(num_iterations: int, func: Callable[..., T], *args, **kwargs) -> T:
    """"
    Execute a function multiple times and return the result of the last execution.

    Returns:
        T: The result of the last function execution.
    """
    for _ in range(num_iterations):
        result = func(*args, **kwargs)
    return result

def profile(
    num_iterations: int,
    num_warmup_iterations:int,
    func: Callable[..., T],
    *args,
    **kwargs
) -> T:
    """
    Profile the execution of a function using PyTorch Profiler.

    This function performs warmup iterations, then profiles the actual execution
    of the function for a specified number of iterations.

    Returns:
        tuple[float, T]: A tuple containing:
            - float: The average execution time.
            - T: The result of the last function execution.
    """
    #  warmup
    execute_callback(num_warmup_iterations, func, *args, **kwargs)
    with tpf.profile(
        activities=[
            tpf.ProfilerActivity.CPU,
            tpf.ProfilerActivity.CUDA
        ],
        profile_memory=True,
        with_stack=True,
        with_modules=True,
    ) as prof:
        result = execute_callback(num_iterations, func, *args, **kwargs)

    return get_trace_perf(prof, num_iterations), result

def profile_cuda_graph(
    num_iterations: int,
    num_warmup_iterations:int,
    func: Callable[..., T],
    *args,
    **kwargs
) -> T:
    """
    Profile the execution of a function using CUDA Graph and PyTorch Profiler.

    This function creates a CUDA Graph for the given function, then profiles its
    execution using the standard profile function.

    Returns:
        tuple[float, T]: A tuple containing:
            - float: The average execution time or other performance metric.
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

    Attributes:
    -----------
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

    Methods:
    --------
    __call__() -> tuple[float, T]
        Executes the benchmarking process and returns a tuple containing
        the execution time and the result of the benchmarked function.

    Usage:
    ------
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

DefaultBenchmarkHook = partial(BenchmarkHook, 100, 10, False)
DefaultCudaGraphBenchmarkHook = partial(BenchmarkHook, 100, 10, True)