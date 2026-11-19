# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Build isolated A05/A07 modules and apply the mandatory MONO policy."""
import copy
import functools
import fcntl
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import types

ROOT = Path(__file__).resolve().parent
from . import codegen


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_source(name):
    path = ROOT / 'sources' / name
    expected = json.loads((ROOT / 'sources.json').read_text())['sources/' + name]['sha256']
    if sha(path) != expected:
        raise RuntimeError(f'candidate source checksum mismatch: {path}')
    return path


def evidence():
    from aiter.jit.core import get_user_jit_dir
    path = Path(os.environ.get('MONO_EVIDENCE_DIR', str(Path(get_user_jit_dir()) / 'mono-evidence' / codegen.POLICY_ID)))
    path.mkdir(parents=True, exist_ok=True)
    return path


@functools.lru_cache(None)
def load_a05():
    source = checked_source('rope_window_mono.py')
    spec = importlib.util.spec_from_file_location('aiter.ops.mono_standard.sources.rope_window_mono', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(None)
def load_a07():
    from aiter.jit.core import get_user_jit_dir
    with (Path(get_user_jit_dir()) / 'mono-a07.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _load_a07_locked()


def _load_a07_locked():
    import torch
    from aiter.jit.core import build_module, compile_ops, get_args_of_build, get_user_jit_dir
    native = importlib.import_module('aiter.ops.inverse_rope_group_quant')
    source = checked_source('inverse_rope_group_quant_mono.cu')
    args = get_args_of_build('module_inverse_rope_group_quant')
    sources = list(args['srcs'])
    replace = [i for i, p in enumerate(sources) if str(p).endswith('/kernels/inverse_rope_group_quant.cu')]
    if len(replace) != 1:
        raise RuntimeError(f'expected exactly one A07 kernel source, got {sources}')
    native_source = Path(sources[replace[0]])
    binding = {
        'candidate_source_sha256': sha(source),
        'candidate_source_name': source.name,
        'mono_codegen_policy_sha256': codegen.POLICY_ID,
        'native_source_sha256': sha(native_source),
        'native_wrapper_sha256': sha(native.__file__),
        'torch': torch.__version__, 'hip': torch.version.hip,
        'build_configuration': copy.deepcopy(args),
    }
    name = 'module_inverse_rope_mono_' + hashlib.sha256(json.dumps(binding, sort_keys=True, default=str).encode()).hexdigest()[:20]
    sources[replace[0]] = str(source)
    args.update(srcs=sources, md_name=name)
    # Always invoke the builder; an ordinary decorated AITER import may reuse
    # a stale .so. The name also binds this candidate to the native interface.
    build_module(**{k: v for k, v in args.items() if k in inspect.signature(build_module).parameters})
    library = Path(get_user_jit_dir()) / (name + '.so')
    if not library.is_file():
        raise RuntimeError(f'A07 build produced no library: {library}')

    def lowlevel(o: torch.Tensor, x_fp8: torch.Tensor, x_scale: torch.Tensor,
                 positions: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor,
                 num_groups: int, quant_group_size: int = 128, scale_layout: int = 0) -> None: ...

    policy_evidence = evidence() / 'a07-policy'
    # The build name contains the policy hash. Build once, then rewrite before
    # the library can be loaded by the runtime. Repeated imports copy its proof.
    proof = library.with_suffix('.mono-policy.json')
    if not proof.exists():
        codegen.legalize_library(library, policy_evidence)
        proof.write_text(json.dumps({'policy_id':codegen.POLICY_ID,'library_sha256':sha(library),'evidence':str(policy_evidence)}))
    else:
        saved=json.loads(proof.read_text())
        if saved['policy_id']!=codegen.POLICY_ID:
            raise RuntimeError('MONO library policy identity mismatch')
        if saved['library_sha256']!=sha(library):
            original=Path(saved['evidence'])/'compiler-library.so'
            if not original.is_file() or sha(original)!=sha(library):
                raise RuntimeError('MONO library is neither proven output nor recognized compiler rebuild')
            codegen.legalize_library(library,policy_evidence)
            if saved['library_sha256']!=sha(library):
                raise RuntimeError('MONO repeated legalization is not deterministic')
        elif Path(saved['evidence']).resolve()!=policy_evidence.resolve():
            shutil.copytree(saved['evidence'], policy_evidence, dirs_exist_ok=True)

    # AITER registers by Python name, independent of the JIT module name.
    lowlevel.__name__ = name + '_entry'
    lowlevel = compile_ops(name, fc_name='inverse_rope_group_quant', develop=True)(lowlevel)

    original = native.inverse_rope_group_quant
    namespace = dict(original.__globals__)
    namespace['_inverse_rope_group_quant_kernel'] = lowlevel
    wrapper = types.FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    wrapper.__kwdefaults__ = original.__kwdefaults__
    wrapper.__annotations__ = dict(original.__annotations__)
    target = evidence()
    shutil.copy2(library, target / library.name)
    shutil.copy2(source, target / source.name)
    shutil.copy2(native_source, target / 'native_inverse_rope_group_quant.cu')
    (target / 'a07-build.json').write_text(json.dumps({
        **binding, 'module': name, 'library': str(library),
        'library_sha256': sha(library), 'explicit_build': True, 'effective_build_configuration': args,
        'native_module_unmodified': native._inverse_rope_group_quant_kernel is original.__globals__['_inverse_rope_group_quant_kernel'],
    }, indent=2, default=str) + '\n')
    return wrapper
