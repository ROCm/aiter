import shutil
import os
import subprocess
from jinja2 import Template
import ctypes
from packaging.version import parse, Version
from collections import OrderedDict
from functools import lru_cache


HOME_PATH = os.environ.get('HOME')
AITER_MAX_CACHE_SIZE = os.environ.get('AITER_MAX_CACHE_SIZE', None)
AITER_ROOT_DIR = os.environ.get("AITER_ROOT_DIR", f"{HOME_PATH}/.aiter")
BUILD_DIR=os.path.abspath(os.path.join(AITER_ROOT_DIR, "build"))
os.makedirs(BUILD_DIR, exist_ok=True)


makefile_template = Template("""
build:
	hipcc -DUSE_ROCM -DENABLE_FP8 -fPIC -shared {{cxxflags | join(" ")}} {{includes | join(" ")}} {{sources | join(" ")}} -o lib.so

clean:
	rm -rf lib.so test
""")


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


def compile_lib(src_file, folder, includes=None, sources=None, cxxflags=None):
    if includes is None:
        includes = []
    if sources is None:
        sources = []
    if cxxflags is None:
        cxxflags = []
    sub_build_dir = os.path.join(BUILD_DIR, folder)
    init_build_dir(sub_build_dir)
    include_dir = f"{sub_build_dir}/include"
    os.makedirs(include_dir, exist_ok=True)
    for include in includes:
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
            "-O3", "-std=c++17",
            "-DLEGACY_HIPBLAS_DIRECT",
            "-DUSE_PROF_API=1",
            "-D__HIP_PLATFORM_HCC__=1",
            "-D__HIP_PLATFORM_AMD__=1",
            "-U__HIP_NO_HALF_CONVERSIONS__",
            "-U__HIP_NO_HALF_OPERATORS__",
            "-mllvm", "--amdgpu-kernarg-preload-count=16",
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
    makefile_file = makefile_template.render(includes=[f"-I{include_dir}"], sources=sources, cxxflags=cxxflags)
    with open(f"{sub_build_dir}/Makefile", "w") as f:
        f.write(makefile_file)
    subprocess.run(f"cd {sub_build_dir} && make build -j8", shell=True, check=True)


@lru_cache(maxsize=AITER_MAX_CACHE_SIZE)
def run_lib(folder):
    lib = ctypes.CDLL(f"{BUILD_DIR}/{folder}/lib.so", os.RTLD_LAZY)
    return lib.call


def compile_template_op(src_template, md_name, includes=None, sources=None, cxxflags=None, folder=None, **kwargs):
    if includes is None:
        includes = []
    if sources is None:
        sources = []
    if cxxflags is None:
        cxxflags = []
    kwargs = OrderedDict(kwargs)
    if folder is None:
        folder = f"{md_name}_{'_'.join([str(v) for v in kwargs.values()])}"
    src_file = src_template.render(**kwargs)
    if not os.path.exists(f"{BUILD_DIR}/{folder}/lib.so") or os.environ.get("AITER_FORCE_COMPILE", "0") == "1":
        compile_lib(src_file, folder, includes, sources, cxxflags)
    return run_lib(folder)
