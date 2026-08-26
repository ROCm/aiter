# AITER MK1 persistent decoder

`aiter.MK1` is an optional binary-only persistent-decoder provider for
GPT-OSS-120B on AMD Instinct MI355X (`gfx950`). Private kernel source is not
included in this package.

The serving runtime owns request scheduling, ordinary model loading, KV-cache
allocation, and fallback execution. AITER validates and loads the compiled
persistent checkpoint only when the provider is enabled, retains those tensors
for the provider lifetime, and binds the native decoder to the runtime-owned KV
cache without allocating a second serving cache.

The public integration surface is intentionally small:

```python
from aiter.MK1 import KVCacheBinding, MK1Config, PersistentDecoder, QuantumRequest
```

`KVCacheBinding` describes separate shuffled K/V planes using byte views,
physical block counts, block strides, and pool identifiers. `QuantumRequest`
contains one frozen decode command and its physical block maps.

`module_persistent_decoder.so` provides the Python/native boundary. The
`MK1_persistent_decoder_gpt_oss_gfx950_bf16.so` and
`MK1_persistent_decoder_gpt_oss_gfx950_fp16.so` libraries contain the packaged
provider implementations; their suffixes identify the KV-cache scalar.

The checkpoint path must name an existing local snapshot directory. The
provider performs no network download and accepts the `gpt_oss_gfx950_v1`
checkpoint backend.
