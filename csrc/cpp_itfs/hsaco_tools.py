import ctypes
from ctypes.util import find_library
import functools
import torch
import os
import subprocess

_is_hip_library_api_supported_ = False


@functools.cache
def get_amdhip():
    global _is_hip_library_api_supported_

    try:
        lib = ctypes.CDLL(find_library("amdhip64"))
    except Exception as e:
        print(e)
        torch_amdhip64 = os.path.join(torch.__path__[0], "lib", "libamdhip64.so")
        print(f"Try {torch_amdhip64} instead...")
        lib = ctypes.CDLL(torch_amdhip64)
    lib.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    lib.hipModuleLoad.restype = ctypes.c_int32
    lib.hipModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.hipModuleGetFunction.restype = ctypes.c_int32
    lib.hipModuleLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,  # unsigned int sharedMemBytes
        ctypes.c_void_p,  # hipStream_t stream
        ctypes.c_void_p,  # void **kernelParams
        ctypes.c_void_p,  # void **extra
    ]
    lib.hipModuleLaunchKernel.restype = ctypes.c_int32
    lib.hipGetErrorString.argtypes = [ctypes.c_int32]
    lib.hipGetErrorString.restype = ctypes.c_char_p

    try:
        lib.hipLibraryLoadFromFile.restype = ctypes.c_int32
        lib.hipLibraryLoadFromFile.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_void_p,  # hipJitOption *jitOptions
            ctypes.c_void_p,  # void **jitOptionsValues
            ctypes.c_uint32,  # unsigned int numJitOptions,
            ctypes.c_void_p,  # hipLibraryOption *libraryOptions
            ctypes.c_void_p,  # void **libraryOptionValues
            ctypes.c_uint32,  # unsigned int numLibraryOptions
        ]

        lib.hipLibraryGetKernelCount.restype = ctypes.c_int32
        lib.hipLibraryGetKernelCount.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),  # unsigned int *count,
            ctypes.c_void_p,  # hipLibrary_t library
        ]

        lib.hipLibraryEnumerateKernels.restype = ctypes.c_int32
        lib.hipLibraryEnumerateKernels.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # hipKernel_t *kernels
            ctypes.c_uint32,  # unsigned int numKernels,
            ctypes.c_void_p,  # hipLibrary_t library
        ]

        lib.hipKernelGetName.restype = ctypes.c_int32
        lib.hipKernelGetName.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),  # const char **name
            ctypes.c_void_p,  # hipKernel_t kernel
        ]
        _is_hip_library_api_supported_ = True
    except Exception:
        _is_hip_library_api_supported_ = False

    return lib


def hip_check_error(err, *args):
    if err != 0:
        raise Exception(
            "HIP error:"
            + get_amdhip().hipGetErrorString(err).decode("utf-8")
            + repr(args)
        )


@functools.cache
def get_lib(lib_fpath):
    hip = get_amdhip()
    p_lib = ctypes.c_void_p()
    hip_check_error(
        (
            hip.hipLibraryLoadFromFile(
                ctypes.byref(p_lib),
                lib_fpath.encode("utf-8"),
                None,
                None,
                0,
                None,
                None,
                0,
            )
            if _is_hip_library_api_supported_
            else hip.hipModuleLoad(ctypes.byref(p_lib), lib_fpath.encode("utf-8"))
        ),
        lib_fpath,
    )
    return p_lib


@functools.cache
def get_all_kernel_names(co_path):
    # we need both demangle & symbol name for loading & argtype parsing
    dynamic_syms_raw = subprocess.check_output(
        ["/opt/rocm/llvm/bin/llvm-objdump", "--dynamic-syms", co_path]
    ).decode("utf-8")
    kernel_names = []
    for line_raw in dynamic_syms_raw.splitlines():
        ls = line_raw.split()
        if len(ls) < 7:
            continue
        if ls[3] != ".text":
            continue
        symbol_name = line_raw.split()[6]
        kernel_names.append(symbol_name)
    return kernel_names


@functools.cache
def get_kernel(kernel_path_prefix, constexpr_args: tuple = ()):
    """
    constexpr_args is compile-time args which are part of co-file name
    """
    hip = get_amdhip()

    co_suffix = ""
    for k, v in constexpr_args:
        co_suffix += f"-{k}={v}"
    co_suffix += ".co"

    if ":" in kernel_path_prefix:
        # file contain many kernels, filename is not started with kernel name
        kernel_path_base, kernel_name = kernel_path_prefix.split(":", 1)
        lib_fpath = kernel_path_base + co_suffix
    else:
        # file contain only one kernel, filename starts with kernel name
        _, kernel_name = os.path.split(kernel_path_prefix)
        lib_fpath = kernel_path_prefix + co_suffix

    lib_fpath = os.path.abspath(os.path.normpath(lib_fpath))
    print("lib_fpath: ", lib_fpath)

    p_lib = get_lib(lib_fpath)

    if _is_hip_library_api_supported_:
        kernel_cnt = ctypes.c_uint32()
        hip_check_error(hip.hipLibraryGetKernelCount(ctypes.byref(kernel_cnt), p_lib))

        assert kernel_cnt.value > 0
        kernels = (ctypes.c_void_p * kernel_cnt.value)()

        hip_check_error(hip.hipLibraryEnumerateKernels(kernels, kernel_cnt, p_lib))

        p_func = None
        for k in kernels:
            p_name = ctypes.c_char_p()
            hip_check_error(hip.hipKernelGetName(ctypes.byref(p_name), k))
            assert p_name.value is not None
            cur_kernel_name = p_name.value.decode("utf-8")
            if kernel_name in cur_kernel_name:
                p_func = k
                break
    else:
        p_func = None
        for cur_kernel_name in get_all_kernel_names(lib_fpath):
            if kernel_name in cur_kernel_name:
                p_func = ctypes.c_void_p()
                hip_check_error(
                    hip.hipModuleGetFunction(
                        ctypes.byref(p_func), p_lib, cur_kernel_name.encode("utf-8")
                    )
                )
                break

    assert p_func is not None, f"kernel {kernel_name} is not found in {lib_fpath}"

    def CallableKernel(
        gridDims: list[int],
        blockDims: list[int],
        *args,
        sharedMemBytes=0,
    ):
        fields = []
        for i, arg in enumerate(args):
            # Device pointers must be c_void_p — bare int from Tensor.data_ptr() is 64-bit and
            # must not be packed as c_int (truncation → GPU page fault / abort).
            if arg is None or isinstance(arg, torch.Tensor):
                fields.append((f"arg_{i}", ctypes.c_void_p))
            elif isinstance(arg, int):
                fields.append((f"arg_{i}", ctypes.c_int))
            elif isinstance(arg, float):
                fields.append((f"arg_{i}", ctypes.c_float))
            else:
                raise Exception(f"Unsupported arg type: {arg}")

        class Args(ctypes.Structure):
            _fields_ = fields

        kernel_args = Args()
        for i, a in enumerate(args):
            if isinstance(a, torch.Tensor):
                ptr = a.data_ptr()
                setattr(kernel_args, f"arg_{i}", ctypes.c_void_p(ptr))
            elif a is None:
                setattr(kernel_args, f"arg_{i}", ctypes.c_void_p())
            else:
                setattr(kernel_args, f"arg_{i}", a)
        ExtraType = ctypes.c_void_p * 5
        kernel_args_size = ctypes.c_uint64(ctypes.sizeof(kernel_args))
        kernel_config = ExtraType(
            1, ctypes.addressof(kernel_args), 2, ctypes.addressof(kernel_args_size), 3
        )
        stream = ctypes.cast(torch.cuda.current_stream(), ctypes.c_void_p)
        while len(gridDims) < 3:
            gridDims.append(1)
        while len(blockDims) < 3:
            blockDims.append(1)
        hip_check_error(
            hip.hipModuleLaunchKernel(
                p_func,
                *gridDims,
                *blockDims,
                sharedMemBytes,
                stream,
                0,
                ctypes.byref(kernel_config),
            )
        )

    return CallableKernel