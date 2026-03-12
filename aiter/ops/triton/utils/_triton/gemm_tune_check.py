import importlib
import inspect
from typing import Optional
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def gemm_tune_check(
    func: callable,
    N: int,
    K: int,
    M: Optional[int] = None,
    shuffle: Optional[bool] = None,
):
    """
    This function returns if a AITER Triton GEMM is tunned for a specific shape

    example 1: FP4 GEMM preshuffled weight scales for shape (16, 1280, 8192)
        from aiter.ops.triton.utils._triton.gemm_tune_check import gemm_tune_check
        from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffle
        is_tunned = gemm_tune_check(gemm_afp4wfp4_preshuffle, N=1280, K=8192//2, M=16, shuffle=True)
        print(is_tunned) # return True or False

    example 2: FP8 GEMM blockscale for shape (16, 1280, 8192)
        from aiter.ops.triton.utils._triton.gemm_tune_check import gemm_tune_check
        from aiter.ops.triton.gemm_a8w8_blockscale import gemm_a8w8_blockscale
        is_tunned = gemm_tune_check(gemm_a8w8_blockscale, N=1024, K=8192, M=16)
        print(is_tunned) # return True or False
    """

    module_pth = func.__module__.split(".")
    module_pth = module_pth[:-1] + ["_triton_kernels"] + module_pth[-1:]
    module_pth = ".".join(module_pth)
    module = importlib.import_module(module_pth)
    _LOGGER.info(f"Function {func} found at {module}")

    if hasattr(module, "_get_config"):

        get_config_func = getattr(module, "_get_config")
        sig = inspect.signature(get_config_func)

        if (
            sig.parameters.get("M") is not None
            and sig.parameters.get("N") is not None
            and sig.parameters.get("K") is not None
        ):
            if sig.parameters.get("shuffle") is None:
                _, is_tunned = get_config_func(M, N, K)
            else:
                if shuffle is not None:
                    _, is_tunned = get_config_func(M, N, K, shuffle=shuffle)
                else:
                    raise Exception(f"Please specify shuffle (True/False) for {module}")

            return is_tunned

        raise NotImplementedError(
            f"{module} _get_config not yet supported for inspection"
        )

    raise Exception(f"{module} does not have _get_config function")
