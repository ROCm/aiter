# SPDX-License-Identifier: Apache-2.0
"""把 tile Stage-1 的产物接到 MegaMoEv2 stage2(kernel1) 的 ABI 上。

kernel1 吃 15 个指针。按交接 §21 探底的结果，适配面很小:

  直接映射(6)  arg_aq/arg_ascale <- h1_output_q/h1_output_scale
               arg_eids          <- tile_expert
               arg_stids         <- tile_row_source   (含 source | topk_slot<<24 打包)
               arg_sweights      <- tile_row_weight
               arg_bq/arg_bscale <- 外部权重, 直接传
  等价替换(1)  arg_cumsum        <- num_valid (都是"总 sorted 行数")
  退化(1)      arg_trb           <- 恒等数组 trb[i] = i*BM
  关掉(4)      arg_max_expert_tiles / arg_expert_tile_end / arg_count_matrix /
               arg_pair_config: 只在 skew_cu>0 或 runtime_pair_skip 下被读
  运行时构造(1) arg_p2p_comb_inp <- peer 的 plane_slot_inbox 基地址表

`arg_trb` 是唯一始终被读的那个: kernel 里

    sort_block_idx    = (m_block_idx*BM) // SBM
    row_in_sort_block = m_block_idx*BM - sort_block_idx*SBM
    srcmap_row_base   = trb_buf[sort_block_idx] + row_in_sort_block

SBM == BM == 32 时 sort_block_idx == m_block_idx 且 row_in_sort_block == 0,
所以 srcmap_row_base = trb[m_block_idx]; 而 tile 侧 arg_stids 本来就按 m_row + tx
直接索引, 因此 trb[i] = i*BM 让两边对齐。
"""
from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels.megamoe_tile import comm_ops
import mori.cco.device.flydsl as cco


def identity_trb(row_capacity: int, BM: int, device) -> torch.Tensor:
    """SBM==BM 时让 kernel1 的 sort-block 间接层退化为恒等。"""
    if BM <= 0 or row_capacity <= 0:
        raise ValueError("row_capacity 和 BM 必须为正")
    blocks = (int(row_capacity) + int(BM) - 1) // int(BM)
    # 多给一格: kernel 的 total_m_blocks 按 cumsum 向上取整, 末块可能触到 blocks。
    return (torch.arange(blocks + 1, dtype=torch.int32, device=device) * int(BM))


def plane_slot_offset(arena, parity: int) -> int:
    """plane_slot_inbox 在 window 里的字节偏移(含 parity)。"""
    s2 = arena.stage2
    region = s2.region("plane_slot_inbox")
    return (int(arena.stage2_offset) + region.offset
            + int(parity) * (region.nbytes // s2.parity_depth))


def arrival_delta(arena, parity):
    """peer 的 arrival 字基址 - peer 的 plane_slot_inbox 基址。

    每个 rank 的 window 布局相同, 所以这个差在所有 peer 上是同一个常量 ——
    kernel1 复用 LDS 里已有的 peer_base 加上它就能寻址, 不需要第二张 peer 表。
    `stage2_offset` 在相减时抵消, 只剩两个 region 的偏移之差。

    单位: arrival 区是每个 (token, top-k slot) 一个 i64 字, 和 payload 用的是
    同一个 slot 下标, 所以 kernel1 那边直接复用已经算好的 `slot`。
    """
    s2 = arena.stage2

    def off(name):
        region = s2.region(name)
        return region.offset + int(parity) * (region.nbytes // s2.parity_depth)

    return off("plane_slot_arrived") - off("plane_slot_inbox")


def peer_table_launcher(gpus_per_node: int, npes: int):
    """填 arg_p2p_comb_inp: 每个 peer 上 plane_slot_inbox 的基地址。

    CCO 的 `lsa_ptr` 只在 device 侧可用(host 拿不到 peer 的虚拟地址), 所以这张
    表由一个一次性的小 kernel 写出来, 用法和 bench 里的 epoch_launcher 相同。

    ep16_plane_slots 下 dest_pe = owner_rank & (gpus_per_node-1), 只有前
    gpus_per_node 项会被真正寻址。其余填 peer 0 的基地址: kernel 的 valid 位
    已经把它们判掉, 这里只是不让越界索引变成野指针。
    """
    local_peers = int(gpus_per_node)
    total = int(npes)

    @flyc.kernel(known_block_size=[64, 1, 1])
    def fill_peer_table(window: fx.Int64, table: fx.Int64, offset: fx.Int64):
        if fx.Int32(fx.thread_idx.x) == fx.Int32(0):
            for peer in range(local_peers):
                address = cco.Window(window).lsa_ptr(fx.Int32(peer), offset)
                comm_ops.store_i64_global_relaxed(
                    table + fx.Int64(peer * 8), address)
            fallback = cco.Window(window).lsa_ptr(fx.Int32(0), offset)
            for peer in range(local_peers, total):
                comm_ops.store_i64_global_relaxed(
                    table + fx.Int64(peer * 8), fallback)

    @flyc.jit
    def launch(window: fx.Int64, table: fx.Int64, offset: fx.Int64,
               stream: fx.Stream):
        fill_peer_table(window, table, offset).launch(
            grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


def build_kernel1_args(*, window, arena, parity, tensors, trb, p2p_table,
                       zero_i32, expert_major=False):
    """返回 run_mega_moe_stage2 的前 15 个指针参数。"""
    s1 = arena.stage1
    base = int(window.local_ptr)

    def s1_ptr(name):
        region = s1.region(name)
        return fx.Int64(base + region.offset
                        + int(parity) * (region.nbytes // 2))

    zero = fx.Int64(int(zero_i32.data_ptr()))
    return (
        s1_ptr("h1_output_q"),                       # arg_aq
        s1_ptr("h1_output_scale"),                   # arg_ascale
        fx.Int64(int(tensors["bq"].data_ptr())),     # arg_bq
        fx.Int64(int(tensors["bs"].data_ptr())),     # arg_bscale
        s1_ptr("tile_expert_sorted"                  # arg_eids
               if expert_major else "tile_expert"),
        s1_ptr("num_valid"),                         # arg_cumsum
        zero,                                        # arg_max_expert_tiles
        s1_ptr("tile_row_source_sorted"              # arg_stids
               if expert_major else "tile_row_source"),
        s1_ptr("tile_row_weight_sorted"              # arg_sweights
               if expert_major else "tile_row_weight"),
        fx.Int64(int(trb.data_ptr())),               # arg_trb
        zero,                                        # arg_expert_tile_end
        zero,                                        # arg_count_matrix
        zero,                                        # arg_pair_config
        zero,                                        # arg_parity
        fx.Int64(int(p2p_table.data_ptr())),         # arg_p2p_comb_inp
    )
