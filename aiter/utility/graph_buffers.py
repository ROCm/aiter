import functools

import torch


def per_stream_buffers_shared_by_graphs(alloc):
    """Cache scratch buffers so they stay correct across CUDA graph capture.

    Capture records operations instead of running them, so a ``torch.zeros``
    issued there only adds a memset *node* to the graph being captured; the
    memory itself still holds whatever the graph pool last left in it. A cache
    keyed on the capture stream misses exactly once, so one graph records that
    memset and every other graph shares the same never-zeroed buffer. Replaying
    any of the others first starts the kernel's atomic protocol on residue.

    Buffers are therefore allocated outside capture -- by an eager launch or by
    ``preallocate`` -- and all graphs share that one really-zeroed set. Eager
    launches keep a per-stream buffer: concurrent launches on different streams
    must not share an atomic counter, or their arrival counts mix and the
    reduction never fires. Graphs cannot be separated that way (a graph replays
    on whatever stream launches it, not the capture stream) and are assumed not
    to replay concurrently.
    """
    per_stream = functools.lru_cache(maxsize=128)(
        lambda device, stream_id, *key: alloc(device, *key)
    )
    shared = {}

    def preallocate(device, *key):
        buf = shared.get((device, *key))
        if buf is None:
            buf = shared[(device, *key)] = alloc(device, *key)
        return buf

    @functools.wraps(alloc)
    def get(device, *key):
        if torch.cuda.is_current_stream_capturing():
            buf = shared.get((device, *key))
            if buf is None:
                raise RuntimeError(
                    f"{alloc.__name__}: buffers must be allocated before CUDA "
                    "graph capture. Run one eager launch on this shape first, "
                    "or call .preallocate() outside capture."
                )
            return buf
        preallocate(device, *key)
        stream_id = torch.cuda.current_stream(device).cuda_stream
        return per_stream(device, stream_id, *key)

    def cache_clear():
        per_stream.cache_clear()
        shared.clear()

    get.preallocate = preallocate
    get.cache_clear = cache_clear
    return get
