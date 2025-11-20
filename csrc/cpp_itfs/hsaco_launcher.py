#!/usr/bin/env python3
"""
Launch Triton kernels from HSACO files using hip-python

This module provides a pure Python implementation using hip-python
to load and launch Triton-compiled kernels from HSACO binary files.
"""

import os
import sys
from typing import List, Tuple

try:
    from hip import hip
except ImportError:
    print("Error: hip-python not found. Please install: pip install hip-python")
    sys.exit(1)

try:
    import torch
except ImportError:
    torch = None
    print("Warning: PyTorch not found. Tensor support will be limited.")


def hip_check(result):
    """Helper function to check HIP API call results"""
    if isinstance(result, tuple):
        err = result[0]
    else:
        err = result

    if err != hip.hipError_t.hipSuccess:
        error_string = hip.hipGetErrorString(err)
        if isinstance(error_string, tuple):
            error_string = error_string[1]
        raise RuntimeError(f"HIP Error: {error_string} (code: {err})")

    return result


def read_hsaco(path: str) -> bytes:
    """
    Read HSACO file and return as bytes

    Args:
        path: Path to the HSACO file

    Returns:
        HSACO binary data as bytes
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"HSACO file not found: {path}")

    with open(path, "rb") as f:
        data = f.read()

    print(f"Loaded HSACO file: {len(data)} bytes from {path}")
    return data


class HsacoLauncher:
    """
    Launcher for HIP/Triton kernels using hip-python
    """

    def __init__(self):
        """Initialize the launcher and HIP context"""
        # Get device count
        err, device_count = hip.hipGetDeviceCount()
        hip_check(err)
        print(f"Found {device_count} HIP device(s)")

        # Set device 0
        err = hip.hipSetDevice(0)
        hip_check(err)

        # Get device properties
        props = hip.hipDeviceProp_t()
        err = hip.hipGetDeviceProperties(props, 0)
        hip_check(err)
        print(f"Using device: {props.gcnArchName}")

        self.module = None
        self.kernel_func = None

    def load_module(self, hsaco_data: bytes):
        """
        Load HIP module from HSACO binary data

        Args:
            hsaco_data: HSACO binary data
        """
        err, self.module = hip.hipModuleLoadData(hsaco_data)
        hip_check(err)
        print("Module loaded successfully")

    def get_function(self, kernel_name: str):
        """
        Get kernel function from loaded module

        Args:
            kernel_name: Name of the kernel function
        """
        if self.module is None:
            raise RuntimeError("Module not loaded. Call load_module() first.")

        err, self.kernel_func = hip.hipModuleGetFunction(
            self.module, kernel_name.encode("utf-8")
        )
        if err != hip.hipError_t.hipSuccess:
            error_string = hip.hipGetErrorString(err)
            if isinstance(error_string, tuple):
                error_string = error_string[1]
            raise RuntimeError(
                f"Failed to get kernel function '{kernel_name}': {error_string}"
            )

        print(f"Kernel function retrieved: {kernel_name}")

    def launch_kernel(
        self,
        kernel_args: List,
        grid: Tuple[int, int, int] = (1, 1, 1),
        block: Tuple[int, int, int] = (256, 1, 1),
        shared_mem_bytes: int = 0,
        stream=None,
    ):
        """
        Launch the kernel

        Args:
            kernel_args: List of kernel arguments (pointers and scalars)
            grid: Grid dimensions (gridX, gridY, gridZ)
            block: Block dimensions (blockX, blockY, blockZ)
            shared_mem_bytes: Shared memory size in bytes
            stream: HIP stream (None for default stream)
        """
        if self.kernel_func is None:
            raise RuntimeError(
                "Kernel function not retrieved. Call get_function() first."
            )

        # Convert stream to hipStream_t if None
        if stream is None:
            stream = 0  # Use default stream

        # Prepare kernel arguments
        # hip-python expects a list of ctypes pointers or values
        import ctypes

        args = []  # Keep references to prevent garbage collection

        for arg in kernel_args:
            if isinstance(arg, int):
                # Handle integer arguments
                val = ctypes.c_int32(arg)
                args.append(val)

            elif isinstance(arg, float):
                # Handle float arguments
                val = ctypes.c_float(arg)
                args.append(val)

            elif torch is not None and isinstance(arg, torch.Tensor):
                # Handle PyTorch tensors
                ptr = arg.data_ptr()
                ptr_obj = ctypes.c_void_p(ptr)
                args.append(ptr_obj)

            elif isinstance(arg, (ctypes.c_void_p, ctypes.c_int32, ctypes.c_float)):
                # Already a ctypes value
                args.append(arg)

            else:
                raise TypeError(f"Unsupported argument type: {type(arg)}")

        print(
            f"Launching kernel with grid=({grid[0]},{grid[1]},{grid[2]}) "
            f"block=({block[0]},{block[1]},{block[2]}) "
            f"shared_mem={shared_mem_bytes} bytes"
        )
        # Launch the kernel
        err = hip.hipModuleLaunchKernel(
            self.kernel_func,
            *grid,
            *block,
            shared_mem_bytes,
            stream,
            None,
            extra=tuple(args),
        )
        hip_check(err)
        print("Kernel launched successfully")

    def unload_module(self):
        """Unload the HIP module"""
        if self.module is not None:
            err = hip.hipModuleUnload(self.module)
            hip_check(err)
            self.module = None
            self.kernel_func = None

    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.unload_module()
        except Exception:
            pass


def launch_triton_kernel(
    hsaco_data: bytes,
    kernel_name: str,
    kernel_args: List,
    grid: Tuple[int, int, int] = (1, 1, 1),
    block: Tuple[int, int, int] = (256, 1, 1),
    shared_mem_bytes: int = 0,
    stream=None,
) -> int:
    """
    Launch a Triton kernel from HSACO binary data

    Args:
        hsaco_data: HSACO binary data
        kernel_name: Name of the kernel function to launch
        kernel_args: List of kernel arguments (tensors, pointers, scalars)
        grid: Grid dimensions (gridX, gridY, gridZ)
        block: Block dimensions (blockX, blockY, blockZ)
        shared_mem_bytes: Shared memory size in bytes
        stream: HIP stream (None for default stream)

    Returns:
        0 on success, 1 on failure
    """
    launcher = HsacoLauncher()

    try:
        # Load module
        launcher.load_module(hsaco_data)

        # Get kernel function
        launcher.get_function(kernel_name)

        # Launch kernel
        launcher.launch_kernel(
            kernel_args=kernel_args,
            grid=grid,
            block=block,
            shared_mem_bytes=shared_mem_bytes,
            stream=stream,
        )

        # Cleanup
        launcher.unload_module()

        return 0

    except Exception as e:
        print(f"Error launching kernel: {e}")
        import traceback

        traceback.print_exc()
        return 1


def launch_triton_kernel_from_file(
    hsaco_path: str,
    kernel_name: str,
    kernel_args: List,
    grid: Tuple[int, int, int] = (1, 1, 1),
    block: Tuple[int, int, int] = (256, 1, 1),
    shared_mem_bytes: int = 0,
    stream=None,
) -> int:
    """
    Launch a Triton kernel from HSACO file

    Args:
        hsaco_path: Path to HSACO file
        kernel_name: Name of the kernel function to launch
        kernel_args: List of kernel arguments (tensors, pointers, scalars)
        grid: Grid dimensions (gridX, gridY, gridZ)
        block: Block dimensions (blockX, blockY, blockZ)
        shared_mem_bytes: Shared memory size in bytes
        stream: HIP stream (None for default stream)

    Returns:
        0 on success, 1 on failure
    """
    # Load HSACO file
    hsaco_data = read_hsaco(hsaco_path)

    # Launch kernel
    return launch_triton_kernel(
        hsaco_data=hsaco_data,
        kernel_name=kernel_name,
        kernel_args=kernel_args,
        grid=grid,
        block=block,
        shared_mem_bytes=shared_mem_bytes,
        stream=stream,
    )
