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
from aiter.jit.utils.file_baton import FileBaton
import logging
import time


logger = logging.getLogger("aiter")
this_dir = os.path.dirname(os.path.abspath(__file__))
AITER_CORE_DIR = os.path.abspath(f"{this_dir}/../../")
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
AITER_DEBUG = int(os.getenv("AITER_DEBUG", 0))

if AITER_REBUILD >= 1:
    subprocess.run(f"rm -rf {BUILD_DIR}/*", shell=True)
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


def validate_and_update_archs():
    archs = GPU_ARCH.split(";")
    archs = [arch.strip() for arch in archs]
    # List of allowed architectures
    allowed_archs = [
        "native",
        "gfx90a",
        "gfx940",
        "gfx941",
        "gfx942",
        "gfx950",
        "gfx1100",
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
            "-O3",
            "-std=c++17",
            "-DLEGACY_HIPBLAS_DIRECT",
            "-DUSE_PROF_API=1",
            "-D__HIP_PLATFORM_HCC__=1",
            "-D__HIP_PLATFORM_AMD__=1",
            "-U__HIP_NO_HALF_CONVERSIONS__",
            "-U__HIP_NO_HALF_OPERATORS__",
            "-mllvm",
            "--amdgpu-kernarg-preload-count=16",
            "-Wno-unused-result",
            "-Wno-switch-bool",
            "-Wno-vla-cxx-extension",
            "-Wno-undefined-func-template",
            "-fgpu-flush-denormals-to-zero",
        ]

        if AITER_DEBUG:
            cxxflags += ["-g", "-fverbose-asm", "--save-temps", "-Wno-gnu-line-marker"]

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
        makefile_file = makefile_template.render(
            includes=[f"-I{include_dir}"], sources=sources, cxxflags=cxxflags
        )
        with open(f"{sub_build_dir}/Makefile", "w") as f:
            f.write(makefile_file)
        subprocess.run(
            f"cd {sub_build_dir} && make build -j{len(sources)}", shell=True, check=True
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
