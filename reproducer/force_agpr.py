"""Undo Triton 3.8.0's unconditional `amdgpu-agpr-alloc="0"` for named kernels.

    import force_agpr
    force_agpr.enable("_gluon_fp8_mqa_logits_kernel")            # default "0,256"
    force_agpr.enable("_gluon_fp8_mqa_logits_kernel", value="0,128")

3.8.0's AMD backend sets the attribute on every kernel it compiles:

    # Workaround, remove once the LLVM fix lands
    # With this set, waves_per_eu >= 2 uses no AGPRs; waves_per_eu = 1 stills gets 256.
    kernel_fn.add_fn_attr("amdgpu-agpr-alloc", "0")

so any kernel compiled at `waves_per_eu >= 2` gets no AGPRs at all.

Same hook as `no_packed_fp32.py` -- wrap `llvm.optimize_module`, which runs
after the backend has finished setting attributes -- and the same per-function
API the backend itself uses, so this is scoped to the kernels you name rather
than to the process. 3.7.0 never sets the attribute, so `enable()` is harmless
there and `removed()` reports 0.
"""
import triton.backends.amd.compiler as _c

ATTR = "amdgpu-agpr-alloc"

_TARGETS, _installed, _value = [], False, "0,256"
_touched = []


def enable(*name_substrings, value="0,256"):
    """Replace the attribute on matching kernels. Substring match on the name."""
    global _installed, _value
    _TARGETS.extend(name_substrings)
    _value = value
    if _installed:
        return
    if not hasattr(_c.llvm, "optimize_module"):
        raise RuntimeError("triton.backends.amd.compiler.llvm.optimize_module "
                           "missing; API moved")
    orig = _c.llvm.optimize_module

    def patched(mod, *a, **k):
        for fn in mod.get_functions():
            if fn.is_declaration():
                continue
            if any(t in fn.name for t in _TARGETS):
                # remove first: add_fn_attr on an existing name would otherwise
                # leave the original in place, which is what the backend's own
                # llvm_fn_attrs loop does too
                fn.remove_fn_attr(ATTR)
                if _value is not None:
                    fn.add_fn_attr(ATTR, _value)
                _touched.append(fn.name)
        return orig(mod, *a, **k)

    _c.llvm.optimize_module = patched
    _installed = True


def touched():
    """Names this actually rewrote -- 0 on 3.7.0, which never sets the attribute."""
    return list(_touched)
