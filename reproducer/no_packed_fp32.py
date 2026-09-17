"""Disable packed FP32 VALU ops (v_pk_fma/mul/add/mov_b32) for named Triton/Gluon kernels on AMD.

    import no_packed_fp32
    no_packed_fp32.enable("_gluon_fp8_mqa_logits_kernel")   # substring match on the kernel name

Uses Triton's own per-function API, `add_fn_target_feature`, the same one the AMD backend already
calls for "+xnack", so it appends to the feature string instead of overwriting it. Scoped by kernel
name, so unrelated kernels in the same process are untouched. No-ops loudly if the API moves.
Requires a Triton cache dir not shared with an unpatched run: the cache key does not see this.
"""
import triton.backends.amd.compiler as _c

_TARGETS, _installed = [], False

def enable(*name_substrings):
    global _installed
    _TARGETS.extend(name_substrings)
    if _installed:
        return
    orig = _c.llvm.optimize_module
    def patched(mod, *a, **k):
        for fn in mod.get_functions():
            if fn.is_declaration():
                continue
            if any(t in fn.name for t in _TARGETS):
                fn.add_fn_target_feature("-packed-fp32-ops")
        return orig(mod, *a, **k)
    if not hasattr(_c.llvm, "optimize_module"):
        raise RuntimeError("triton.backends.amd.compiler.llvm.optimize_module missing; API moved")
    _c.llvm.optimize_module = patched
    _installed = True
