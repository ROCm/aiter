# AITER MK1 persistent decoder

`aiter.MK1` is an optional binary-only AITER provider that lets ATOM use the
MK1 persistent megakernel for GPT-OSS-120B decoding on AMD Instinct MI355X
(`gfx950`). Private kernel source is not included in this package.

ATOM enables the provider with only:

```text
--persistent-decoder
--persistent-decoder-checkpoint /path/to/compiled-checkpoint
```

The checkpoint argument must name an existing local snapshot directory. ATOM
continues to own request scheduling, ordinary model loading, and KV-cache
allocation. AITER validates and loads the megakernel-specific checkpoint only
when persistent decoding is enabled, owns those tensors for the provider's
lifetime, and binds the megakernel directly to ATOM's existing KV cache. It
does not allocate or copy into a second serving cache.

ATOM imports the integration directly from the MK1 package:

```python
from aiter.MK1 import (
    AtomCacheBinding,
    MK1Config,
    PersistentDecoder,
    QuantumRequest,
)
```

`module_persistent_decoder.so` provides the Python/native boundary. The
`MK1_persistent_decoder_gpt_oss_gfx950_atom_bf16.so` and
`MK1_persistent_decoder_gpt_oss_gfx950_atom_fp16.so` libraries contain the
packaged provider code; their suffixes identify the KV-cache scalar explicitly.
The extension reports its compiled ABI to AITER when loaded, and its host
runtime resolves the provider's exact versioned ELF factory symbol before
execution. Checkpoint discovery is local-only; AITER does not download model or
checkpoint files.
