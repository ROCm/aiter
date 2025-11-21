# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.


import shutil
import os
import subprocess
from jinja2 import Template
import ctypes
from packaging.version import parse, Version
from collections import OrderedDict
from functools import lru_cache, partial
import binascii
import hashlib
import logging
import time
import inspect
import json


logger = logging.getLogger("aiter")
this_dir = os.path.dirname(os.path.abspath(__file__))
AITER_CORE_DIR = os.path.abspath(f"{this_dir}/../../")
if os.path.exists(os.path.join(AITER_CORE_DIR, "aiter_meta")):
    AITER_CORE_DIR = os.path.join(AITER_CORE_DIR, "aiter_meta")
DEFAULT_GPU_ARCH = (
    subprocess.run(
        "/opt/rocm/llvm/bin/amdgpu-arch", shell=True, capture_output=True, text=True
    )
    .stdout.strip()
    .split("\n")[0]
)
GPU_ARCH = os.environ.get("GPU_ARCHS", DEFAULT_GPU_ARCH)
AITER_REBUILD = int(os.environ.get("AITER_REBUILD", 0))

HOME_PATH = os.environ.get("HOME")
AITER_MAX_CACHE_SIZE = os.environ.get("AITER_MAX_CACHE_SIZE", None)
AITER_ROOT_DIR = os.environ.get("AITER_ROOT_DIR", f"{HOME_PATH}/.aiter")
BUILD_DIR = os.path.abspath(os.path.join(AITER_ROOT_DIR, "build"))
AITER_LOG_MORE = int(os.getenv("AITER_LOG_MORE", 0))
LOG_LEVEL = {
    0: logging.ERROR,
    1: logging.WARNING,
    2: logging.INFO,
    3: logging.DEBUG,
}
logger.setLevel(LOG_LEVEL[AITER_LOG_MORE])
AITER_DEBUG = int(os.getenv("AITER_DEBUG", 0))

if AITER_REBUILD >= 1:
    subprocess.run(f"rm -rf {BUILD_DIR}/*", shell=True)

if not os.path.exists(BUILD_DIR):
    os.makedirs(BUILD_DIR, exist_ok=True)

CK_DIR = os.environ.get("CK_DIR", f"{AITER_CORE_DIR}/3rdparty/composable_kernel")

makefile_template = Template(
    """
CXX=hipcc
TARGET=lib.so

SRCS = {{sources | join(" ")}}
OBJS = $(SRCS:.cpp=.o)

build: $(OBJS)
	$(CXX) -shared $(OBJS) -o $(TARGET)

%.o: %.cpp
	$(CXX) -fPIC {{cxxflags | join(" ")}} {{includes | join(" ")}} -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)
"""
)


def mp_lock(
    lock_path: str,
    main_func: callable,
    final_func: callable = None,
    wait_func: callable = None,
):
    """
    Using FileBaton for multiprocessing.
    """
    from aiter.jit.utils.file_baton import FileBaton

    baton = FileBaton(lock_path)
    if baton.try_acquire():
        try:
            ret = main_func()
        finally:
            if final_func is not None:
                final_func()
            baton.release()
    else:
        baton.wait()
        if wait_func is not None:
            ret = wait_func()
        ret = None
    return ret


def get_hip_version():
    version = subprocess.run(
        "/opt/rocm/bin/hipconfig --version", shell=True, capture_output=True, text=True
    )
    return parse(version.stdout.split()[-1].rstrip("-").replace("-", "+"))


@lru_cache()
def hip_flag_checker(flag_hip: str) -> bool:
    ret = os.system(f"hipcc {flag_hip} -x hip -c /dev/null -o /dev/null")
    if ret == 0:
        return True
    else:
        logger.warning(f"{flag_hip} is not supported by hipcc.")
        return False


def validate_and_update_archs():
    archs = GPU_ARCH.split(";")
    archs = [arch.strip().split(":")[0] for arch in archs]
    # List of allowed architectures
    allowed_archs = [
        "native",
        "gfx90a",
        "gfx940",
        "gfx941",
        "gfx942",
        "gfx950",
    ]

    # Validate if each element in archs is in allowed_archs
    assert all(
        arch in allowed_archs for arch in archs
    ), f"One of GPU archs of {archs} is invalid or not supported"
    for i in range(len(archs)):
        if archs[i] == "native":
            archs[i] = DEFAULT_GPU_ARCH

    return archs


def compile_lib(src_file, folder, includes=None, sources=None, cxxflags=None):
    sub_build_dir = os.path.join(BUILD_DIR, folder)
    include_dir = f"{sub_build_dir}/include"
    if not os.path.exists(include_dir):
        os.makedirs(include_dir, exist_ok=True)
    lock_path = f"{sub_build_dir}/lock"
    start_ts = time.perf_counter()

    def main_func(includes=None, sources=None, cxxflags=None):
        logger.info(f"start build {sub_build_dir}")
        if includes is None:
            includes = []
        if sources is None:
            sources = []
        if cxxflags is None:
            cxxflags = []

        for include in includes + [f"{CK_DIR}/include"]:
            if os.path.isdir(include):
                shutil.copytree(include, include_dir, dirs_exist_ok=True)
            else:
                shutil.copy(include, include_dir)
        for source in sources:
            if os.path.isdir(source):
                shutil.copytree(source, sub_build_dir, dirs_exist_ok=True)
            else:
                shutil.copy(source, sub_build_dir)
        with open(f"{sub_build_dir}/{folder}.cpp", "w") as f:
            f.write(src_file)

        sources += [f"{folder}.cpp"]
        cxxflags += [
            "-DUSE_ROCM",
            "-DENABLE_FP8",
            "-O3" if not AITER_DEBUG else "-O0",
            "-std=c++20",
            "-DLEGACY_HIPBLAS_DIRECT",
            "-DUSE_PROF_API=1",
            "-D__HIP_PLATFORM_HCC__=1",
            "-D__HIP_PLATFORM_AMD__=1",
            "-U__HIP_NO_HALF_CONVERSIONS__",
            "-U__HIP_NO_HALF_OPERATORS__",
            "-mllvm --amdgpu-kernarg-preload-count=16",
            "-Wno-unused-result",
            "-Wno-switch-bool",
            "-Wno-vla-cxx-extension",
            "-Wno-undefined-func-template",
            "-fgpu-flush-denormals-to-zero",
        ]

        if AITER_DEBUG:
            cxxflags += [
                "-g",
                "-ggdb",
                "-fverbose-asm",
                "--save-temps",
                "-Wno-gnu-line-marker",
            ]

        # Imitate https://github.com/ROCm/composable_kernel/blob/c8b6b64240e840a7decf76dfaa13c37da5294c4a/CMakeLists.txt#L190-L214
        hip_version = get_hip_version()
        if hip_version > Version("5.5.00000"):
            cxxflags += ["-mllvm --lsr-drop-solution=1"]
        if hip_version > Version("5.7.23302"):
            cxxflags += ["-fno-offload-uniform-block"]
        if hip_version > Version("6.1.40090"):
            cxxflags += ["-mllvm -enable-post-misched=0"]
        if hip_version > Version("6.2.41132"):
            cxxflags += [
                "-mllvm -amdgpu-early-inline-all=true",
                "-mllvm -amdgpu-function-calls=false",
            ]
        if hip_version > Version("6.2.41133"):
            cxxflags += ["-mllvm -amdgpu-coerce-illegal-types=1"]
        archs = validate_and_update_archs()
        cxxflags += [f"--offload-arch={arch}" for arch in archs]
        cxxflags = [flag for flag in set(cxxflags) if hip_flag_checker(flag)]
        makefile_file = makefile_template.render(
            includes=[f"-I{include_dir}"], sources=sources, cxxflags=cxxflags
        )
        with open(f"{sub_build_dir}/Makefile", "w") as f:
            f.write(makefile_file)
        subprocess.run(
            f"cd {sub_build_dir} && make build -j{len(sources)}",
            shell=True,
            capture_output=AITER_LOG_MORE < 2,
            check=True,
        )

    def final_func():
        logger.info(
            f"finish build {sub_build_dir}, cost {time.perf_counter()-start_ts:.8f}s"
        )

    main_func = partial(
        main_func, includes=includes, sources=sources, cxxflags=cxxflags
    )

    mp_lock(lock_path=lock_path, main_func=main_func, final_func=final_func)


@lru_cache(maxsize=AITER_MAX_CACHE_SIZE)
def run_lib(func_name, folder=None):
    if folder is None:
        folder = func_name
    lib = ctypes.CDLL(f"{BUILD_DIR}/{folder}/lib.so", os.RTLD_LAZY)
    return getattr(lib, func_name)


def hash_signature(signature: str):
    return hashlib.md5(signature.encode("utf-8")).hexdigest()


@lru_cache(maxsize=None)
def get_default_func_name(md_name, args: tuple):
    signature = "_".join([str(arg).lower() for arg in args])
    return f"{md_name}_{hash_signature(signature)}"


def not_built(folder):
    return not os.path.exists(f"{BUILD_DIR}/{folder}/lib.so")


def compile_template_op(
    src_template,
    md_name,
    includes=None,
    sources=None,
    cxxflags=None,
    func_name=None,
    folder=None,
    **kwargs,
):
    kwargs = OrderedDict(kwargs)
    if func_name is None:
        func_name = get_default_func_name(md_name, tuple(kwargs.values()))
    if folder is None:
        folder = func_name

    if not_built(folder):
        if includes is None:
            includes = []
        if sources is None:
            sources = []
        if cxxflags is None:
            cxxflags = []
        logger.info(f"compile_template_op {func_name = } with {locals()}...")
        src_file = src_template.render(func_name=func_name, **kwargs)
        compile_lib(src_file, folder, includes, sources, cxxflags)
    return run_lib(func_name, folder)


def transfer_hsaco(hsaco_path):
    with open(hsaco_path, "rb") as f:
        hsaco = f.read()
    hsaco_hex = binascii.hexlify(hsaco).decode("utf-8")
    return len(hsaco_hex), ", ".join(
        [f"0x{x}{y}" for x, y in zip(hsaco_hex[::2], hsaco_hex[1::2])]
    )


def str_to_bool(s):
    return True if s.lower() == "true" else False


def compile_hsaco_from_triton(kernel, *args, grid=(1, 1, 1), **kwargs):
    import triton
    import triton.language as tl

    if not isinstance(kernel, triton.JITFunction):
        raise ValueError(f"Kernel {kernel} is not a triton.JITFunction")
    sig = inspect.signature(kernel.fn)

    constant_indices = []
    for idx, param in enumerate(sig.parameters.values()):
        if param.annotation == tl.constexpr:
            constant_indices.append(idx)
    ccinfo = kernel.warmup(*args, grid=grid, **kwargs)
    constants = {}
    keys = list(sig.parameters.keys())
    for idx, arg in enumerate(args):
        if idx in constant_indices:
            constants[keys[idx]] = arg
    extra_metadata = {}
    extra_metadata["waves_per_eu"] = kwargs.get("waves_per_eu", 1)
    extra_metadata["num_stages"] = kwargs.get("num_stages", 1)
    extra_metadata["num_warps"] = kwargs.get("num_warps", 1)
    extra_metadata["num_ctas"] = kwargs.get("num_ctas", 1)
    return compile_hsaco(
        kernel.fn.__name__,
        ccinfo.asm["hsaco"],
        ccinfo.metadata.shared,
        ccinfo.metadata.target.arch,
        constants,
        extra_metadata,
    )


def compile_hsaco(
    kernel_name,
    hsaco,
    shared=0,
    gcnArchName=GPU_ARCH,
    constants=None,
    extra_metadata=None,
):
    build_dir = f"{BUILD_DIR}/{gcnArchName}"
    constants = OrderedDict(constants or {})
    func_name = get_default_func_name(kernel_name, tuple(constants.values()))
    lock_path = f"{build_dir}/{func_name}.lock"
    if not os.path.exists(build_dir):
        os.makedirs(build_dir, exist_ok=True)

    def main_func(constants):
        metadata = {}
        metadata["shared"] = shared
        metadata["name"] = kernel_name
        metadata["gcnArchName"] = gcnArchName
        metadata.update(extra_metadata or {})
        for key, value in constants.items():
            metadata[key] = str(value)
        with open(f"{build_dir}/{func_name}.hsaco", "wb") as f:
            f.write(hsaco)
        with open(f"{build_dir}/{func_name}.json", "w") as f:
            json.dump(metadata, f)

    def final_func():
        logger.info(f"finish build {func_name}")

    main_func = partial(main_func, constants=constants)
    mp_lock(lock_path=lock_path, main_func=main_func, final_func=final_func)


def check_hsaco(func_name, constants=None):
    constants = OrderedDict(constants or {})
    hsaco_name = get_default_func_name(func_name, tuple(constants.values()))
    return os.path.exists(f"{BUILD_DIR}/{GPU_ARCH}/{hsaco_name}.hsaco")


@lru_cache(maxsize=None)
def get_hsaco_launcher(hsaco_name, kernel_name):
    from csrc.cpp_itfs.hsaco_launcher import HsacoLauncher, read_hsaco

    hsaco = read_hsaco(f"{BUILD_DIR}/{GPU_ARCH}/{hsaco_name}.hsaco")
    hsaco_launcher = HsacoLauncher()
    hsaco_launcher.load_module(hsaco)
    hsaco_launcher.get_function(kernel_name)
    return hsaco_launcher


def run_hsaco(
    func_name, *args, grid=(1, 1, 1), block=(256, 1, 1), stream=None, constants=None
):
    constants = OrderedDict(constants or {})
    hsaco_name = get_default_func_name(func_name, tuple(constants.values()))
    metadata_path = f"{BUILD_DIR}/{GPU_ARCH}/{hsaco_name}.json"
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    kernel_name = metadata["name"]
    hsaco_launcher = get_hsaco_launcher(hsaco_name, kernel_name)
    hsaco_launcher.launch_kernel(
        args, grid=grid, block=block, shared_mem_bytes=metadata["shared"], stream=stream
    )
