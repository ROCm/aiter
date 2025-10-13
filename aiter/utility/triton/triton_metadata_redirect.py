# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Triton Metadata Redirect Module

This module provides decorators and utilities for customizing Triton kernel
metadata file paths during compilation. It allows redirecting .json and
.hsaco files to custom directories.

Example usage0 for jit:
    from aiter.utility.triton_metadata_redirect import with_custom_metadata_path

    @with_custom_metadata_path("/custom/path")
    @triton.jit
    def my_kernel(...):
        ...

Example usage1 for aot:
    from aiter.utility.triton_metadata_redirect import AOTMetadataContext

    with AOTMetadataContext("kernel_name", "/custom/path"):
        kernel = compile_kernel(...)
"""

import functools
import os
from typing import Callable, Dict, Optional
import threading
import triton.compiler.compiler as triton_compiler


# Use thread-local storage to avoid multi-threading race conditions
_thread_local = threading.local()


def _get_thread_registry():
    """Get the registry for the current thread"""
    if not hasattr(_thread_local, "replacement_registry"):
        _thread_local.replacement_registry = {}
    return _thread_local.replacement_registry


# Lock to ensure patching happens only once
_patch_lock = threading.Lock()
# Flag indicating whether patching has been performed
_patched = False


def _ensure_patched():
    """Ensure CompiledKernel.__init__ method is patched only once"""
    global _patched
    with _patch_lock:
        if not _patched:
            # Save the original __init__ method
            _original_compiled_kernel_init = triton_compiler.CompiledKernel.__init__

            def _replacement_init(self, src, metadata_group, hash):
                # Find kernel name from metadata group
                kernel_name = None
                for key in metadata_group:
                    if key.endswith(".json"):
                        kernel_name = key[:-5]  # Remove '.json' suffix
                        break

                # Replace metadata paths using thread-local registry
                if kernel_name:
                    registry = _get_thread_registry()
                    if kernel_name in registry:
                        dir = registry[kernel_name]
                        metadata_group[kernel_name + ".json"] = os.path.join(
                            dir, f"{kernel_name}.json"
                        )
                        metadata_group[kernel_name + ".hsaco"] = os.path.join(
                            dir, f"{kernel_name}.hsaco"
                        )

                # Call the original initialization method
                _original_compiled_kernel_init(self, src, metadata_group, hash)

            # Replace the original __init__ method with our patched version
            triton_compiler.CompiledKernel.__init__ = _replacement_init
            _patched = True


# Ensure patching is done when module is loaded
_ensure_patched()


class AOTMetadataContext:
    """
    Context manager for AOT compilation with custom metadata paths

    Uses thread-local storage to avoid multi-threading race conditions.

    Example usage:
        with AOTMetadataContext("kernel_name", "/custom/path"):
            kernel = compile_kernel(...)
    """

    def __init__(self, kernel_name: str, dir: str):
        self.kernel_name = kernel_name
        self.dir = dir
        self._previously_registered = False
        self._previous_dir: Optional[str] = None

    def __enter__(self):
        registry = _get_thread_registry()

        # Save previous registration if it exists
        if self.kernel_name in registry:
            self._previously_registered = True
            self._previous_dir = registry[self.kernel_name]

        # Register the new path
        registry[self.kernel_name] = self.dir

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        registry = _get_thread_registry()

        # Restore previous registration or remove current one
        if self._previously_registered:
            registry[self.kernel_name] = self._previous_dir
        else:
            # Remove our registration if it wasn't previously registered
            registry.pop(self.kernel_name, None)

        # Don't suppress exceptions
        return False


def with_custom_metadata_path(dir):
    """
    Decorator to register a kernel for metadata path replacement

    This decorator uses thread-local storage to ensure consistency
    with the context manager approach.

    Args:
        dir: Directory path where kernel metadata files should be stored
    """

    def decorator(func: Callable) -> Callable:
        # Save the original function
        original_func = func

        # Create a wrapper function
        # Currently does nothing but maintains signature
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return original_func(*args, **kwargs)

        # Register the kernel in the thread-local registry
        registry = _get_thread_registry()
        registry[original_func.__name__] = dir

        # Add decorator markers
        wrapper._with_custom_metadata_path_applied = True
        wrapper._metadata_directory = dir

        # Return the original function to avoid interfering with @triton.jit
        return original_func

    return decorator
