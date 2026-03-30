"""Shared pipeline utilities for gfx1250 GEMM kernels. """


def make_tail_plan(num_buffers, pre_loaded, extra):
    """Compute a compile-time tail execution plan for the N-stage pipeline.

    Returns a list of (load_stage, compute_stage, outstanding) tuples, one per
    tail step.  outstanding=-1 means "last step, use compute_tile (no barrier)".

    Args:
        num_buffers: total number of pipeline stages.
        pre_loaded:  stages already loaded and ready to compute (= num_buffers - 1).
        extra:       additional tiles that must be loaded in the tail.
    """
    steps = pre_loaded + extra
    plan = []
    for i in range(steps):
        compute_stage = (
            i if i < pre_loaded
            else (i - pre_loaded + num_buffers - 1) % num_buffers
        )
        load_stage = (
            (i + num_buffers - 1) % num_buffers if i < extra
            else None
        )
        is_last = (i == steps - 1)
        if is_last:
            outstanding = -1
        else:
            j = i + 1
            next_compute = (
                j if j < pre_loaded
                else (j - pre_loaded + num_buffers - 1) % num_buffers
            )
            outstanding = (
                2 * (num_buffers - 2) if (load_stage is not None and load_stage != next_compute)
                else 0
            )
        plan.append((load_stage, compute_stage, outstanding))
    return plan


__all__ = ["make_tail_plan"]
