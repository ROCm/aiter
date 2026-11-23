# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


@triton.jit
def remap_xcd_chunked(
    pid, GRID_MN, NUM_XCDS: tl.constexpr, CHUNK_SIZE: tl.constexpr = 2
):
    # Compute current XCD and local PID
    xcd = pid % NUM_XCDS
    # distribute the modulo pids in round robin
    if pid > (GRID_MN // (NUM_XCDS * CHUNK_SIZE)) * (NUM_XCDS * CHUNK_SIZE):
        return pid
    local_pid = pid // NUM_XCDS
    # Calculate chunk index and position within chunk
    chunk_idx = local_pid // CHUNK_SIZE
    pos_in_chunk = local_pid % CHUNK_SIZE
    # Calculate new PID
    new_pid = chunk_idx * NUM_XCDS * CHUNK_SIZE + xcd * CHUNK_SIZE + pos_in_chunk
    return new_pid


@triton.jit
def remap_xcd_aid(
    pid,
    GRID_MN,
    NUM_AIDS: tl.constexpr = 2,
    XCDS_PER_AID: tl.constexpr = 4,
):
    """Queue-aligned XCD remap for parts whose L2 is shared by the XCDs of an AID, such as gfx1250.

    Hardware dispatches round-robin over the W = NUM_AIDS * XCDS_PER_AID XCDs. This gives each
    XCD one contiguous run of GRID_MN // W tiles, in XCD order. The XCDs of an AID are
    consecutive, so an AID holds one contiguous range of tiles and its XCDs share their
    operands in its L2. Neighbouring tiles stay on one XCD.

    The round robin does not start at XCD 0. With n = 4 * pipe + queue of the hardware queue
    that the launch goes through, dispatch id i runs on XCD (i - n) mod W. A stream's queue is
    chosen by the runtime, a graph replay uses a stream of its own, and a process with more
    than four streams moves launches between queues, so n cannot be assumed. The wave reads
    its own pipe and queue from HW_REG_WAVE_HW_ID2.

    Taking pid % W as the XCD, as remap_xcd does, rotates the runs by n. With many dispatch ids
    per XCD an AID still holds one contiguous range and nothing is lost. With few, such as the
    two ids per XCD of a 16-CTA cluster, the rotation splits operand rows and columns across
    the AIDs: measured 1.3 to 1.5% on 4096x7168x7168 with 4x4 clusters.

    Interleaving the XCDs of an AID inside its range, instead of one run per XCD, puts
    neighbouring tiles on different XCDs and measured 0.9 to 1.8% slower on a two-row grid.

    The map is a bijection for every GRID_MN and every n: the permuted range is exactly
    [0, (GRID_MN // W) * W), and the ragged tail above it maps to itself.
    """
    W: tl.constexpr = NUM_AIDS * XCDS_PER_AID
    blk = GRID_MN // W  # pids for one XCD
    if pid >= blk * W:
        return pid
    # Bits 3:0 of HW_REG_WAVE_HW_ID2 (register 24) are the pipe and bits 7:4 the queue.
    hw = tl.inline_asm_elementwise(
        "s_getreg_b32 $0, hwreg(24, 0, 8)",
        "=r",
        [],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    n = 4 * (hw & 0xF) + ((hw >> 4) & 0xF)
    # The true XCD of this dispatch id. Inside one round of W ids this is a rotation of the
    # residues, so the map stays a bijection for every n.
    xcd = (pid % W + W - n % W) % W
    return xcd * blk + pid // W


@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    # Legacy callers discard the result and rely on the default. Callers using
    # the remapped PID must pass the device's XCD count.
    ## pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    if tall_xcds == 0:
        tall_xcds = tl.cast(NUM_XCDS, tall_xcds.type)
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Calculate new pid based on the new grouping
    # Note that we need to consider the following two cases:
    # 1. the current pid is on a tall xcd
    # 2. the current pid is on a short xcd
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )

    return pid


@triton.jit
def pid_grid(pid: int, num_pid_m: int, num_pid_n: int, GROUP_SIZE_M: tl.constexpr = 1):
    """
    Maps 1D pid to 2D grid coords (pid_m, pid_n).

    Args:
        - pid: 1D pid
        - num_pid_m: grid m size
        - num_pid_n: grid n size
        - GROUP_SIZE_M: tl.constexpr: default is 1
    """
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        tl.assume(group_size_m >= 0)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    return pid_m, pid_n


@triton.jit
def pid_grid_3d(pid: int, num_pid_m: int, num_pid_n: int, num_pid_k):
    """
    Maps 1D pid to 3D grid coords (pid_m, pid_n, pid_k).
    Args:
        - pid: 1D pid
        - num_pid_m: grid m size
        - num_pid_n: grid n size
        - num_pid_k: grid k size

    Returns:
        - pid_m, pid_n, pid_k: 3D grid coordinates
    """
    pid_m = pid % num_pid_m
    pid_n = (pid // num_pid_m) % num_pid_n
    pid_k = pid // (num_pid_m * num_pid_n) % num_pid_k

    return pid_m, pid_n, pid_k


@triton.jit
def remap_workgroup_spatial(
    wid,
    NUM_Q_HEADS,
    NUM_BLOCKS,
    BATCH,
    NUM_QUERIES_PER_KV: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    """
    XCD-aware workgroup mapping for multi-head attention on AMD CDNA3/3.5 GPUs.

    Hardware assigns workgroup `wid` to XCD `wid % NUM_XCDS` (round-robin).
    This function remaps (wid) -> (off_q_head, start_m, off_z) so that workgroups
    sharing KV data land on the same XCD, keeping that data hot in its 4 MB L2 slice.

    MHA path  (NUM_QUERIES_PER_KV == 1):
        Groups ceil(NUM_Q_HEADS / NUM_XCDS) consecutive Q heads onto each XCD so each
        head's full KV tensor is processed by one XCD only.  Within each XCD the
        ordering is head-first: all sequence blocks for head i complete before head i+1.

    GQA path  (NUM_QUERIES_PER_KV > 1):
        Assigns each KV head exclusively to one XCD so all Q heads that share a KV head
        run on the same XCD.  Within each XCD the ordering is block-first: all Q heads
        in the group process sequence block b together before advancing to block b+1.
        Block-first avoids the causal load-imbalance that head-first ordering creates.

        Three regimes depending on NUM_KV_HEADS vs NUM_XCDS:
          HK == NUM_XCDS : one KV head per XCD (aligned).
          HK >  NUM_XCDS : ceil(HK / NXCD) KV heads per XCD, each processed fully
                           before the next, so only one KV head's data is live at a time.
          HK <  NUM_XCDS : NXCD / HK XCDs share each KV head, splitting the per-group
                           work evenly across the cluster of co-owners.

    Args:
        wid               : flat workgroup ID (tl.program_id(0))
        NUM_Q_HEADS       : total number of Q heads
        NUM_BLOCKS        : number of sequence blocks along the Q dimension
        BATCH             : batch size
        NUM_QUERIES_PER_KV: Q heads per KV head (1 for MHA, >1 for GQA)
        NUM_XCDS          : number of XCDs on the device

    Returns:
        off_q_head : Q head index for this workgroup
        start_m    : sequence block index for this workgroup
        off_z      : batch index for this workgroup
    """
    xcd = wid % NUM_XCDS
    pos = wid // NUM_XCDS

    # ── MHA path (spatial_mha) ────────────────────────────────────────────────
    if NUM_QUERIES_PER_KV == 1:
        wgs_per_head = NUM_BLOCKS * BATCH
        local_head_idx = pos // wgs_per_head
        remainder = pos % wgs_per_head
        off_z = remainder // NUM_BLOCKS
        start_m = remainder % NUM_BLOCKS
        off_q_head = local_head_idx * NUM_XCDS + xcd
        # Bounds check for non-divisible NUM_Q_HEADS / NUM_XCDS.
        if off_q_head >= NUM_Q_HEADS:
            off_q_head = wid % NUM_Q_HEADS
            start_m = (wid // NUM_Q_HEADS) % NUM_BLOCKS
            off_z = (wid // (NUM_BLOCKS * NUM_Q_HEADS)) % BATCH
        return off_q_head, start_m, off_z

    # ── GQA path (spatial_gqa) ────────────────────────────────────────────────
    NUM_KV_HEADS: tl.constexpr = NUM_Q_HEADS // NUM_QUERIES_PER_KV

    if NUM_KV_HEADS >= NUM_XCDS:
        NUM_KV_REPLICAS: tl.constexpr = (NUM_KV_HEADS + NUM_XCDS - 1) // NUM_XCDS
        wgs_per_block = NUM_QUERIES_PER_KV * BATCH
        if NUM_KV_REPLICAS == 1:
            # Aligned: one KV head per XCD.
            kv_head = xcd % NUM_KV_HEADS
            start_m = (pos // wgs_per_block) % NUM_BLOCKS
            remainder = pos % wgs_per_block
            off_z = remainder // NUM_QUERIES_PER_KV
            local_q_in_group = remainder % NUM_QUERIES_PER_KV
            off_q_head = kv_head * NUM_QUERIES_PER_KV + local_q_in_group
        else:
            # HK > NXCD: each XCD owns ceil(HK/NXCD) KV heads processed one at a time.
            wgs_per_kv_group = NUM_QUERIES_PER_KV * NUM_BLOCKS * BATCH
            kv_slot = pos // wgs_per_kv_group
            kv_head = xcd + kv_slot * NUM_XCDS
            local = pos % wgs_per_kv_group
            start_m = (local // (NUM_QUERIES_PER_KV * BATCH)) % NUM_BLOCKS
            remainder = local % (NUM_QUERIES_PER_KV * BATCH)
            off_z = remainder // NUM_QUERIES_PER_KV
            local_q_in_group = remainder % NUM_QUERIES_PER_KV
            off_q_head = kv_head * NUM_QUERIES_PER_KV + local_q_in_group
    else:
        # HK < NXCD: NXCD/HK XCDs share each KV head, splitting the per-group work.
        NXCD_PER_KV: tl.constexpr = NUM_XCDS // NUM_KV_HEADS
        xcd_in_cluster = xcd // NUM_KV_HEADS
        kv_head = xcd % NUM_KV_HEADS
        wgs_per_kv_group = NUM_QUERIES_PER_KV * NUM_BLOCKS * BATCH
        wgs_per_xcd_in_cluster = wgs_per_kv_group // NXCD_PER_KV
        local = (pos % wgs_per_xcd_in_cluster) + xcd_in_cluster * wgs_per_xcd_in_cluster
        start_m = (local // (NUM_QUERIES_PER_KV * BATCH)) % NUM_BLOCKS
        remainder = local % (NUM_QUERIES_PER_KV * BATCH)
        off_z = remainder // NUM_QUERIES_PER_KV
        local_q_in_group = remainder % NUM_QUERIES_PER_KV
        off_q_head = kv_head * NUM_QUERIES_PER_KV + local_q_in_group

    # Bounds check for non-divisible edge cases.
    if kv_head >= NUM_KV_HEADS or off_q_head >= NUM_Q_HEADS:
        off_q_head = wid % NUM_Q_HEADS
        start_m = (wid // NUM_Q_HEADS) % NUM_BLOCKS
        off_z = (wid // (NUM_BLOCKS * NUM_Q_HEADS)) % BATCH

    return off_q_head, start_m, off_z
