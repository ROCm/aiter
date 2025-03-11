import shutil
import os
import subprocess
from jinja2 import Template
import ctypes
import importlib
from packaging.version import parse, Version
import psutil
from collections import OrderedDict

AITER_ROOT_DIR = os.environ.get("AITER_ROOT_DIR", "~/.aiter")
BUILD_DIR=os.path.abspath(os.path.join(AITER_ROOT_DIR, "build"))
os.makedirs(BUILD_DIR, exist_ok=True)

libs = {}


def get_max_jobs():
    max_num_jobs_cores = int(max(1, os.cpu_count()*0.8))
    # calculate the maximum allowed NUM_JOBS based on free memory
    free_memory_gb = psutil.virtual_memory().available / \
        (1024 ** 3)  # free memory in GB
    # each JOB peak memory cost is ~8-9GB when threads = 4
    max_num_jobs_memory = int(free_memory_gb / 9)

    # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
    max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
    return max_jobs


def get_hip_version():
    version = subprocess.run(f"/opt/rocm/bin/hipconfig --version", shell=True, capture_output=True, text=True)
    return parse(version.stdout.split()[-1].rstrip('-').replace('-', '+'))


def validate_and_update_archs():
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    # List of allowed architectures
    allowed_archs = ["native", "gfx90a",
                     "gfx940", "gfx941", "gfx942", "gfx1100"]

    # Validate if each element in archs is in allowed_archs
    assert all(
        arch in allowed_archs for arch in archs
    ), f"One of GPU archs of {archs} is invalid or not supported"
    return archs


def init_build_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        subprocess.run(f"rm -rf {dir}/*", shell=True)

makefile_template = Template("""
build:
	hipcc -DUSE_ROCM -DENABLE_FP8 -fPIC -shared {{cxxflags | join(" ")}} {{includes | join(" ")}} {{sources | join(" ")}} -o lib.so

clean:
	rm -rf lib.so test
""")

def compile_lib(src_file, folder, includes=[], sources=[], cxxflags=[]):
    init_build_dir(os.path.join(BUILD_DIR, folder))
    os.makedirs(f"{BUILD_DIR}/include", exist_ok=True)
    # includes += [f"{AITER_ROOT_DIR}/csrc/include"]
    for include in includes:
        # if not os.path.exists(include):
        if os.path.isdir(include):
            shutil.copytree(include, f"{BUILD_DIR}/include", dirs_exist_ok=True)
        else:
            shutil.copy(include, f"{BUILD_DIR}/include")
    # includes = [f"-I{BUILD_DIR}"]
    for source in sources:
        if os.path.isdir(source):
            shutil.copytree(source, os.path.join(BUILD_DIR, folder), dirs_exist_ok=True)
        else:
            shutil.copy(source, os.path.join(BUILD_DIR, folder))
    with open(f"{BUILD_DIR}/{folder}/call_lib.cpp", "w") as f:
        f.write(src_file)
    sources += ["call_lib.cpp"]
    cxxflags += [
            "-O3", "-std=c++17",
            "-DLEGACY_HIPBLAS_DIRECT",
            "-DUSE_PROF_API=1",
            "-D__HIP_PLATFORM_HCC__=1",
            "-D__HIP_PLATFORM_AMD__=1",
            "-U__HIP_NO_HALF_CONVERSIONS__",
            "-U__HIP_NO_HALF_OPERATORS__",
            "-mllvm", "-enable-post-misched=0",
            "-mllvm", "-amdgpu-early-inline-all=true",
            "-mllvm", "-amdgpu-function-calls=false",
            "-mllvm", "--amdgpu-kernarg-preload-count=16",
            "-mllvm", "-amdgpu-coerce-illegal-types=1",
            # "-v", "--save-temps",
            "-Wno-unused-result",
            "-Wno-switch-bool",
            "-Wno-vla-cxx-extension",
            "-Wno-undefined-func-template",
            "-fgpu-flush-denormals-to-zero",
    ]

    # Imitate https://github.com/ROCm/composable_kernel/blob/c8b6b64240e840a7decf76dfaa13c37da5294c4a/CMakeLists.txt#L190-L214
    hip_version = get_hip_version()
    if hip_version > Version('5.7.23302'):
        cxxflags += ["-fno-offload-uniform-block"]
    if hip_version > Version('6.1.40090'):
        cxxflags += ["-mllvm", "-enable-post-misched=0"]
    if hip_version > Version('6.2.41132'):
        cxxflags += ["-mllvm", "-amdgpu-early-inline-all=true",
                        "-mllvm", "-amdgpu-function-calls=false"]
    if hip_version > Version('6.2.41133') and hip_version < Version('6.3.00000'):
        cxxflags += ["-mllvm", "-amdgpu-coerce-illegal-types=1"]
    archs = validate_and_update_archs()
    cxxflags+=[f"--offload-arch={arch}" for arch in archs]
    makefile_file = makefile_template.render(includes=[f"-I{BUILD_DIR}/include"], sources=sources, cxxflags=cxxflags)
    with open(f"{BUILD_DIR}/{folder}/Makefile", "w") as f:
        f.write(makefile_file)
    subprocess.run(f"cd {BUILD_DIR}/{folder} && make build -j{get_max_jobs()}", shell=True, check=True)

def run_lib(folder, *args):
    if folder in libs:
        lib = libs[folder]
    else:
        lib = ctypes.CDLL(f"{BUILD_DIR}/{folder}/lib.so", os.RTLD_LAZY)
        libs[folder] = lib
    lib.call(*args)

def compile_template_op(src_template, md_name, includes=[], sources=[], cxxflags=[], **kwargs):
    kwargs = OrderedDict(kwargs)
    folder = f"{md_name}_{'_'.join([str(v) for v in kwargs.values()])}"
    src_file = src_template.render(**kwargs)
    if not os.path.exists(f"{BUILD_DIR}/{folder}/lib.so") or os.environ.get("AITER_FORCE_COMPILE", "0") == "1":
        compile_lib(src_file, folder, includes, sources, cxxflags)
    def wrapper(*args):
        return run_lib(folder, *args)
    return wrapper
