# SPDX-License-Identifier: MIT
"""Shared helpers for the Gluon kernels.

Python 3.14 (PEP 649) compatibility.

Triton's ``@aggregate`` decorator builds ``hash_attrs`` from
``inspect.getmembers(cls)``.  On Python 3.14, PEP 649 gives every annotated
class a compiler-generated ``__annotate__`` function, which lands in that list;
the JIT's dependency walker (``triton/runtime/jit.py: record_reference``) then
rejects it because it is a callable that is neither a type nor a ``constexpr``::

    RuntimeError: Unsupported function referenced:
        <function MQAAsyncKVLoader.__annotate__ at 0x...>

Fixed upstream by triton-lang/triton@47f7e923 ("Exclude __annotate__ from
aggregate_value-s hash_attrs", PR #9529, merged 2026-02-27).  That commit is on
``main`` but is NOT in any 3.7.x release -- the release/3.7 branch was cut
before it and never got the backport -- so builds on Triton 3.7.x with Python
3.14 still need this shim.  Delete this module once Triton carries the fix.
"""

def _strip_annotate(cls):
    """Neutralize ``__annotate__`` so Triton's aggregate hash walker skips it.

    Apply *below* ``@aggregate`` so it runs first (decorators apply bottom-up):

        @aggregate
        @_strip_annotate
        class MyLoader:
            field: gl.constexpr
    """
    _ = cls.__annotations__
    cls.__annotate__ = None
    return cls
