# SPDX-License-Identifier: MIT

"""Fused packed MXFP4 dequantization and BF16 WMMA for RDNA3.5."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import gpu, llvm, scf
from flydsl.expr.typing import Vector as Vec

MX_BLOCK_K = 32


def _copy_valid_columns(atom, source, destination, coordinates, extent, width, first=0):
    """Guard the contiguous axis with one predicate per copy atom."""
    values = fx.size(coordinates.shape[0]).to_py_value()
    rest_rows = fx.size(coordinates.shape[1]).to_py_value()
    rest_columns = fx.size(coordinates.shape[2]).to_py_value()
    atoms = values // width
    # The leading value mode has one boolean per vector, not per BF16.
    predicate = fx.make_rmem_tensor(
        fx.make_layout(
            ((1, atoms), rest_rows, rest_columns), ((0, 1), atoms, atoms * rest_rows)
        ),
        fx.Boolean,
    )
    for z in fx.range_constexpr(rest_columns):
        for y in fx.range_constexpr(rest_rows):
            for x in fx.range_constexpr(atoms):
                coordinate = coordinates[(x * width, y, z)]
                column = fx.get_scalar(coordinate[1])
                valid = column + width <= extent
                if first:
                    valid = (column >= first) & valid
                predicate[(x, y, z)] = valid
    fx.copy(atom, source, destination, pred=predicate)


def _tail_segments(extent):
    """Decompose the suffix after the last eight-value chunk at trace time."""
    column = extent // 8 * 8
    segments = []
    for width in (4, 2, 1):
        if extent & width:
            segments.append((column, width))
            column += width
    return tuple(segments)


def _copy_tail_segments(
    segments, cache_modifier, make_thread_copy, tile, coordinates, fragment
):
    # Each narrow plan retains the wide plan's lane ownership.
    for index in fx.range_constexpr(len(segments)):
        first, width = segments[index]
        atom = fx.make_copy_atom(
            fx.rocdl.BufferCopy(16 * width, cache_modifier), fx.BFloat16
        )
        thread_copy = make_thread_copy(atom)
        source = thread_copy.partition_S(tile)
        destination = thread_copy.retile(fragment)
        thread_coordinates = thread_copy.partition_S(coordinates)
        _copy_valid_columns(
            atom, source, destination, thread_coordinates, first + width, width, first
        )


def _create_staging_views(
    a_group_global_view,
    GROUP_TILE_M,
    GROUP_TILE_N,
    STAGE_K,
    LEFT_IN_REGISTERS,
    WAVE_GROUPS_M,
    WAVE_GROUPS_N,
    group_row,
    group_col,
    RIGHT_IN_REGISTERS=False,
):
    WAVE_GROUPS = WAVE_GROUPS_M * WAVE_GROUPS_N

    @fx.struct
    class BSharedStorage:
        b: fx.Array[fx.BFloat16, WAVE_GROUPS * STAGE_K * GROUP_TILE_N, 16]

    @fx.struct
    class ASharedStorage:
        a: fx.Array[fx.BFloat16, WAVE_GROUPS * GROUP_TILE_M * (STAGE_K + 8), 16]

    @fx.struct
    class BothSharedStorage:
        a: fx.Array[fx.BFloat16, WAVE_GROUPS * GROUP_TILE_M * (STAGE_K + 8), 16]
        b: fx.Array[fx.BFloat16, WAVE_GROUPS * STAGE_K * GROUP_TILE_N, 16]

    if fx.const_expr(not LEFT_IN_REGISTERS and not RIGHT_IN_REGISTERS):
        lds = fx.SharedAllocator().allocate(BothSharedStorage).peek()
    elif fx.const_expr(not LEFT_IN_REGISTERS):
        lds = fx.SharedAllocator().allocate(ASharedStorage).peek()
    elif fx.const_expr(not RIGHT_IN_REGISTERS):
        lds = fx.SharedAllocator().allocate(BSharedStorage).peek()

    a_group_staging_view = a_group_global_view
    if fx.const_expr(not LEFT_IN_REGISTERS):
        a_lds_all = lds.a.view(
            fx.make_layout(
                (WAVE_GROUPS_M, WAVE_GROUPS_N, GROUP_TILE_M, STAGE_K),
                (
                    WAVE_GROUPS_N * GROUP_TILE_M * (STAGE_K + 8),
                    GROUP_TILE_M * (STAGE_K + 8),
                    STAGE_K + 8,
                    1,
                ),
            )
        )
        a_group_staging_view = fx.slice(a_lds_all, (group_row, group_col, None, None))
    b_group_staging_view = None
    if fx.const_expr(not RIGHT_IN_REGISTERS):
        b_lds_all = lds.b.view(
            fx.make_layout(
                (WAVE_GROUPS_M, WAVE_GROUPS_N, STAGE_K, GROUP_TILE_N),
                (
                    WAVE_GROUPS_N * STAGE_K * GROUP_TILE_N,
                    STAGE_K * GROUP_TILE_N,
                    GROUP_TILE_N,
                    1,
                ),
            )
        )
        # Slice both group axes to select this group's disjoint B tile.
        b_group_staging_view = fx.slice(b_lds_all, (group_row, group_col, None, None))
    return a_group_staging_view, b_group_staging_view


def _load_buffer_word(resource, byte_offset, dtype, cache_modifier):
    # Buffer copy atoms require typed alignment. The raw byte offset also
    # handles the unaligned row starts of packed odd-width matrices.
    result_type = (
        ir.VectorType.get([2], fx.Uint32.ir_type)
        if dtype == fx.Uint64
        else dtype.ir_type
    )
    word = fx.rocdl.raw_ptr_buffer_load(
        result_type, resource, fx.Int32(byte_offset), fx.Int32(0), aux=cache_modifier
    )
    if dtype == fx.Uint64:
        return Vec(word, dtype=fx.Uint32).bitcast(fx.Uint64)[0]
    return dtype(word)


def _load_tail_bytes(resource, byte_offset, count, dtype, cache_modifier):
    # Assemble 4-, 2-, and 1-byte pieces at their original word offsets.
    word = dtype(0)
    byte = 0
    for width, piece_type in ((4, fx.Uint32), (2, fx.Uint16), (1, fx.Uint8)):
        if count & width:
            piece = _load_buffer_word(
                resource, byte_offset + byte, piece_type, cache_modifier
            )
            word = word | (piece.to(dtype) << (8 * byte))
            byte += width
    return word


def _load_packed_word(resource, row, column, k, n, cache_modifier, check_boundary=True):
    first = row * n + column
    byte_offset = first // 2
    if n % 8 == 0:
        return _load_buffer_word(resource, byte_offset, fx.Uint32, cache_modifier)

    if n % 2:
        # Odd row pitches can start on the high nibble. The fifth byte
        # supplies the last nibble of a full chunk in that case.
        shift = (first % 2 * 4).to(fx.Uint32)

    def load_wide():
        low = _load_buffer_word(resource, byte_offset, fx.Uint32, cache_modifier)
        if n % 2:
            high = _load_buffer_word(
                resource, byte_offset + 4, fx.Uint8, cache_modifier
            )
            word = low.to(fx.Uint64) | (high.to(fx.Uint64) << 32)
            return (word >> shift.to(fx.Uint64)).to(fx.Uint32)
        return low

    if not check_boundary:
        return load_wide()

    # A read that crosses a logical N boundary can feed invalid C columns.
    # Narrow it only near the end of the physical packed allocation.
    wide = first + 8 <= k * n
    edge = scf.IfOp(wide.ir_value(), [fx.Uint32.ir_type], has_else=True)
    with ir.InsertionPoint(edge.then_block):
        scf.yield_([load_wide().ir_value()])
    with ir.InsertionPoint(edge.else_block):
        suffix = column == (n // 8) * 8
        valid = suffix & (row < k)
        offset = valid.select(byte_offset, (k * n + 1) // 2)
        tail = _load_tail_bytes(
            resource, offset, (n % 8 + 1) // 2, fx.Uint32, cache_modifier
        )
        if n % 2:
            tail = tail >> shift
        scf.yield_([tail.ir_value()])
    return fx.Uint32(edge.results[0])


def _load_row_word(resource, row, column, rows, n, cache_modifier, check_boundary=True):
    offset = row * n + column
    if n % 8 == 0 or not check_boundary:
        return _load_buffer_word(resource, offset, fx.Uint64, cache_modifier)

    # Full words may cross logical rows while the physical allocation has room.
    byte_count = rows * n
    wide = offset + 8 <= byte_count
    edge = scf.IfOp(wide.ir_value(), [fx.Uint64.ir_type], has_else=True)
    with ir.InsertionPoint(edge.then_block):
        word = _load_buffer_word(resource, offset, fx.Uint64, cache_modifier)
        scf.yield_([word.ir_value()])
    with ir.InsertionPoint(edge.else_block):
        suffix = column == (n // 8) * 8
        valid = suffix & (row < rows)
        offset = valid.select(offset, byte_count)
        word = _load_tail_bytes(resource, offset, n % 8, fx.Uint64, cache_modifier)
        scf.yield_([word.ir_value()])
    return fx.Uint64(edge.results[0])


def _decode_mxfp4_word(b_packed_word, b_scale, *, shared_scale):
    """Decode eight packed weights to BF16 in nibble order.

    K chunks share one E8M0 byte. N chunks use eight packed scale bytes.
    """
    # The eight little-endian byte entries are twice the E2M1 magnitudes.
    # Each perm_b32 looks up the four even or four odd nibbles in parallel.
    low, high = fx.Uint32(0x03020100), fx.Uint32(0x0C080604)
    even_codes = b_packed_word & 0x07070707
    odd_codes = (b_packed_word >> 4) & 0x07070707
    even_magnitudes = fx.Uint32(fx.rocdl.perm_b32(high, low, even_codes))
    odd_magnitudes = fx.Uint32(fx.rocdl.perm_b32(high, low, odd_codes))

    values = []
    for value_id in range(8):
        magnitudes = even_magnitudes if value_id % 2 == 0 else odd_magnitudes
        doubled_magnitude = (magnitudes >> (8 * (value_id // 2))) & 255
        scale_byte = (
            b_scale
            if shared_scale
            else ((b_scale >> (value_id * 8)) & 255).to(fx.Int32)
        )
        # Subtracting 128 applies the E8M0 scale and cancels the table's factor of two.
        value = doubled_magnitude.to(fx.Int32).to(fx.Float32)
        power = scale_byte - 128
        # LLVM ldexp lowers this E8M0 scaling to v_ldexp_f32.
        scaled = fx.Float32(
            llvm.intr_ldexp(fx.Float32.ir_type, value.ir_value(), power.ir_value())
        )
        # Take the BF16 bits and restore the sign from the original nibble.
        code = (b_packed_word >> (value_id * 4)) & 15
        bits = (scaled.bitcast(fx.Uint32) >> 16) | ((code & 8) << 12)
        bits = (scale_byte == 255).select(fx.Uint32(0x7FC0), bits)
        values.append(bits.to(fx.Uint16).bitcast(fx.BFloat16))
    return tuple(values)


@flyc.kernel
def _mxfp4_w4a16(
    a: fx.Tensor,
    b_packed: fx.Tensor,
    b_scales: fx.Tensor,
    c: fx.Tensor,
    partials,
    counters,
    m: fx.Constexpr,
    n: fx.Constexpr,
    k: fx.Constexpr,
    *,
    GROUP_TILE_M: fx.Constexpr,
    GROUP_TILE_N: fx.Constexpr,
    STAGE_K: fx.Constexpr,
    WAVES_M: fx.Constexpr,
    WAVES_N: fx.Constexpr,
    WAVE_GROUPS_M: fx.Constexpr,
    WAVE_GROUPS_N: fx.Constexpr,
    K_UNROLL: fx.Constexpr,
    RIGHT_CACHE_MODIFIER: fx.Constexpr,
    LEFT_CACHE_MODIFIER: fx.Constexpr,
    LEFT_IN_REGISTERS: fx.Constexpr,
    RIGHT_IN_REGISTERS: fx.Constexpr,
    GROUP_M: fx.Constexpr,
    SPLIT_K: fx.Constexpr,
):
    MMA_M, MMA_N, MMA_K = 16, 16, 16
    WAVE_GROUPS = WAVE_GROUPS_M * WAVE_GROUPS_N
    BLOCK_M = GROUP_TILE_M * WAVE_GROUPS_M
    BLOCK_N = GROUP_TILE_N * WAVE_GROUPS_N
    THREADS_PER_GROUP = 32 * WAVES_M * WAVES_N
    wave_group = fx.thread_idx.x // THREADS_PER_GROUP
    thread_id_in_wave_group = fx.thread_idx.x % THREADS_PER_GROUP
    lds_address_spaces = ir.ArrayAttr.get(
        [ir.Attribute.parse("#gpu.address_space<workgroup>")]
    )
    barrier_scope = (
        gpu.BarrierScope.Subgroup if WAVE_GROUPS > 1 else gpu.BarrierScope.Workgroup
    )

    def lds_barrier():
        # LDS-only ordering; global split-K publication keeps its agent atomics.
        if fx.const_expr(not LEFT_IN_REGISTERS or not RIGHT_IN_REGISTERS):
            fx.barrier(address_spaces=lds_address_spaces, scope=barrier_scope)

    # 1. Bound the physical buffers. Whole rows past M or K land beyond them.
    a_buffer = fx.rocdl.make_buffer_tensor(a, max_size=False)
    # Flat packed nibbles use ceil(K*N/2) bytes. Scales belong to K blocks.
    SCALE_ROWS = (k + MX_BLOCK_K - 1) // MX_BLOCK_K
    b_packed_buffer = fx.rocdl.make_buffer_tensor(b_packed, max_size=False)
    b_scale_buffer = fx.rocdl.make_buffer_tensor(b_scales, max_size=False)
    b_packed_resource = fx.rocdl.get_buffer_rsrc(fx.get_iter(b_packed_buffer))
    b_scale_resource = fx.rocdl.get_buffer_rsrc(fx.get_iter(b_scale_buffer))
    c_buffer = fx.rocdl.make_buffer_tensor(c, max_size=False)
    a_global_view = fx.make_view(
        fx.get_iter(a_buffer),
        fx.make_layout((m, k), (k, 1)),
    )
    c_global_view = fx.make_view(
        fx.get_iter(c_buffer),
        fx.make_layout((m, n), (n, 1)),
    )
    # Coordinate views are needed only where a contiguous axis can cross rows.
    LEFT_TAIL_SEGMENTS = _tail_segments(k)
    REAL_K_STAGES = (k + STAGE_K - 1) // STAGE_K
    HAS_EMPTY_K_STAGES = REAL_K_STAGES % SPLIT_K != 0
    HAS_K_TAIL = k % STAGE_K != 0
    HAS_N_TAIL = n % BLOCK_N != 0
    if fx.const_expr(HAS_K_TAIL):
        a_coordinates = fx.make_view(
            fx.make_int_tuple((0, 0)), fx.make_identity_layout((m, k))
        )
        a_coordinate_tiles = fx.flat_divide(a_coordinates, (BLOCK_M, STAGE_K))
    if fx.const_expr(HAS_N_TAIL):
        c_coordinates = fx.make_view(
            fx.make_int_tuple((0, 0)), fx.make_identity_layout((m, n))
        )
        c_coordinate_tiles = fx.flat_divide(c_coordinates, (BLOCK_M, BLOCK_N))
    # 2. Decode block traversal separately from groups inside the block.
    group_layout = fx.make_layout((WAVE_GROUPS_M, WAVE_GROUPS_N), (WAVE_GROUPS_N, 1))
    group_coords = fx.idx2crd(wave_group, group_layout)
    group_row = fx.get_scalar(group_coords[0])
    group_col = fx.get_scalar(group_coords[1])

    c_block_tiles = fx.flat_divide(c_global_view, (BLOCK_M, BLOCK_N))
    blocks_m = fx.size(c_block_tiles.shape[2]).to_py_value()
    blocks_n = fx.size(c_block_tiles.shape[3]).to_py_value()
    group_m_width = min(GROUP_M, blocks_m)
    # Consecutive splits share an output tile. Decode the output tile ID next.
    output_block_id = fx.block_idx.x // SPLIT_K
    split_id = fx.block_idx.x % SPLIT_K
    full_m_groups, tail_rows = divmod(blocks_m, group_m_width)
    full_rows = full_m_groups * group_m_width
    full_tile_count = full_rows * blocks_n

    # Five tile rows, two columns and group width three give full tile IDs
    # 0–5 and tail tile IDs 6–9. Tail-local IDs 0–3 cover rows 3–4.
    block_row = fx.Int32(0)
    block_col = fx.Int32(0)
    if tail_rows > 0 and output_block_id >= full_tile_count:
        tail_layout = fx.make_ordered_layout((tail_rows, blocks_n), (0, 1))
        tail_coords = fx.idx2crd(output_block_id - full_tile_count, tail_layout)
        block_row = full_rows + fx.get_scalar(tail_coords[0])
        block_col = fx.Int32(fx.get_scalar(tail_coords[1]))
    else:
        full_layout = fx.make_ordered_layout(
            (group_m_width, blocks_n, full_m_groups), (0, 1, 2)
        )
        full_coords = fx.idx2crd(output_block_id, full_layout)
        block_row = fx.get_scalar(full_coords[2]) * group_m_width + fx.get_scalar(
            full_coords[0]
        )
        block_col = fx.Int32(fx.get_scalar(full_coords[1]))

    # Select the complete block output, then this group's output tile.
    c_block_view = fx.slice(c_block_tiles, (None, None, block_row, block_col))
    c_group_tiles = fx.flat_divide(c_block_view, (GROUP_TILE_M, GROUP_TILE_N))
    c_group_view = fx.slice(c_group_tiles, (None, None, group_row, group_col))

    # Workspace IDs name group output tiles independently of launch order.
    output_group_layout = fx.make_layout(
        (blocks_m * WAVE_GROUPS_M, blocks_n * WAVE_GROUPS_N),
        (blocks_n * WAVE_GROUPS_N, 1),
    )
    output_group_row = block_row * WAVE_GROUPS_M + group_row
    output_group_col = block_col * WAVE_GROUPS_N + group_col
    flat_group_id = fx.get_scalar(
        fx.crd2idx((output_group_row, output_group_col), output_group_layout)
    )

    # A block tiles cover the output rows for one K stage.
    a_block_tiles = fx.flat_divide(a_global_view, (BLOCK_M, STAGE_K))

    mma_atom = fx.make_mma_atom(
        fx.rocdl.WMMA(MMA_M, MMA_N, MMA_K, fx.BFloat16, fx.Float32)
    )
    mma = fx.make_tiled_mma(
        mma_atom, fx.make_layout((WAVES_M, WAVES_N, 1), (1, WAVES_M, 0))
    )
    copy_128b = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
    copy_16b = fx.make_copy_atom(fx.UniversalCopy16b(), fx.BFloat16)

    # The first global A tile supplies the full-stage shape (GROUP_TILE_M, STAGE_K).
    # load_stage selects the actual source using block_row, group_row, and tile_k.
    a_block_model = fx.slice(a_block_tiles, (None, None, 0, 0))
    a_group_models = fx.flat_divide(a_block_model, (GROUP_TILE_M, STAGE_K))
    a_group_global_view = fx.slice(a_group_models, (None, None, 0, 0))

    # Direct A uses the global shape model and skips LDS allocation for A.
    a_group_staging_view, b_group_staging_view = _create_staging_views(
        a_group_global_view,
        GROUP_TILE_M,
        GROUP_TILE_N,
        STAGE_K,
        LEFT_IN_REGISTERS,
        WAVE_GROUPS_M,
        WAVE_GROUPS_N,
        group_row,
        group_col,
        RIGHT_IN_REGISTERS,
    )

    thread_mma = mma.thr_slice(thread_id_in_wave_group)
    c_accum = thread_mma.make_fragment_C(c_group_view)
    ACCUMULATOR_VALUES = fx.size(c_accum.shape).to_py_value()
    c_accum.fill(0)

    # Staging copies for A and B.
    staging_value_layout = fx.make_ordered_layout((1, 8), (1, 0))
    a_gmem_load = fx.make_copy_atom(
        fx.rocdl.BufferCopy128b(LEFT_CACHE_MODIFIER), fx.BFloat16
    )
    if fx.const_expr(LEFT_IN_REGISTERS):
        # Allocate per-thread registers for two full A stages in WMMA layout.
        # Staging holds the next load while compute holds the current stage.
        a_staging_fragment = thread_mma.make_fragment_A(a_group_global_view)
        a_compute_fragment = fx.make_fragment_like(a_staging_fragment)
    else:
        # Cooperatively load contiguous K chunks for the padded A tile in LDS.
        a_staging_thread_layout = fx.make_ordered_layout(
            (THREADS_PER_GROUP // (STAGE_K // 8), STAGE_K // 8), (1, 0)
        )
        a_staging_copy_tile, a_staging_copy_tv = fx.make_layout_tv(
            a_staging_thread_layout, staging_value_layout
        )
        a_staging_thread_copy = fx.make_tiled_copy(
            copy_128b, a_staging_copy_tv, a_staging_copy_tile
        ).get_slice(thread_id_in_wave_group)
        a_staging_destination = a_staging_thread_copy.partition_D(a_group_staging_view)
        a_staging_fragment = fx.make_fragment_like(a_staging_destination)

    def make_a_global_copy(atom):
        if fx.const_expr(LEFT_IN_REGISTERS):
            return fx.make_tiled_copy_A(atom, mma).get_slice(thread_id_in_wave_group)
        else:
            return fx.make_tiled_copy(
                atom, a_staging_copy_tv, a_staging_copy_tile
            ).get_slice(thread_id_in_wave_group)

    # Both widths write into views of the same staging fragment.
    a_global_thread_copy = make_a_global_copy(a_gmem_load)
    a_global_destination = a_global_thread_copy.retile(a_staging_fragment)

    # Read eight adjacent N values per lane. Packed weights and scale bytes
    # stay in registers until publish_stage expands them into this group's LDS.
    if fx.const_expr(not RIGHT_IN_REGISTERS):
        RIGHT_N_THREADS = GROUP_TILE_N // 8
        RIGHT_K_THREADS = THREADS_PER_GROUP // RIGHT_N_THREADS
        RIGHT_ROWS_PER_THREAD = STAGE_K // RIGHT_K_THREADS
        b_k_lane = thread_id_in_wave_group // RIGHT_N_THREADS
        b_col = thread_id_in_wave_group % RIGHT_N_THREADS * 8
        global_col = output_group_col * GROUP_TILE_N + b_col

        # Reuse a scale word for rows this lane owns in the same MX block.
        if fx.const_expr(STAGE_K in (16, 32)):
            ROW_SLOTS_PER_SCALE = RIGHT_ROWS_PER_THREAD
        elif fx.const_expr(
            STAGE_K % MX_BLOCK_K == 0 and MX_BLOCK_K % RIGHT_K_THREADS == 0
        ):
            ROW_SLOTS_PER_SCALE = MX_BLOCK_K // RIGHT_K_THREADS
        else:
            ROW_SLOTS_PER_SCALE = 1
        RIGHT_SCALE_SLOTS = RIGHT_ROWS_PER_THREAD // ROW_SLOTS_PER_SCALE
        b_packed_fragment = fx.make_rmem_tensor(RIGHT_ROWS_PER_THREAD, fx.Uint32)
        b_scale_fragment = fx.make_rmem_tensor(RIGHT_SCALE_SLOTS, fx.Uint64)

    # MMA copies for A and B.
    a_group_staging_tiled = fx.flat_divide(a_group_staging_view, (GROUP_TILE_M, MMA_K))
    a_group_staging_mma_initial_view = fx.slice(
        a_group_staging_tiled, (None, None, 0, 0)
    )
    # WMMA consumes one MMA_K step at a time.
    a_r_tile = thread_mma.make_fragment_A(a_group_staging_mma_initial_view)
    if fx.const_expr(LEFT_IN_REGISTERS):
        # Divide the three fragment axes into three inner and three outer axes.
        # For example, (16, 2, 2) becomes (16, 2, 1, 1, 1, 2) for two K steps.
        a_compute_tiled = fx.flat_divide(
            a_compute_fragment, (MMA_K, GROUP_TILE_M // (MMA_M * WAVES_M), 1)
        )
        a_r_destination = a_r_tile
    else:
        a_lds2reg_thread_copy = fx.make_tiled_copy_A(copy_16b, mma).get_slice(
            thread_id_in_wave_group
        )
        a_r_destination = a_lds2reg_thread_copy.retile(a_r_tile)

    if fx.const_expr(RIGHT_IN_REGISTERS):
        # A typed shape model gives the WMMA register layout of (N,K).
        b_group_model = fx.make_view(
            fx.get_iter(a_group_staging_view),
            fx.make_layout((GROUP_TILE_N, STAGE_K), (1, GROUP_TILE_N)),
        )
        b_stage_model = fx.flat_divide(b_group_model, (GROUP_TILE_N, MMA_K))
        b_r_tile = thread_mma.make_fragment_B(
            fx.slice(b_stage_model, (None, None, 0, 0))
        )
        b_r_destination = b_r_tile
        b_compute_fragment = thread_mma.make_fragment_B(b_group_model)
        B_COLUMNS_PER_THREAD = GROUP_TILE_N // (MMA_N * WAVES_N)
        # Use 16-row slots when a stage starts between two scale boundaries.
        B_REGISTER_ROWS_PER_SCALE = MX_BLOCK_K if STAGE_K % MX_BLOCK_K == 0 else MMA_K
        B_REGISTER_SCALE_SLOTS = STAGE_K // B_REGISTER_ROWS_PER_SCALE
        b_register_codes = fx.make_rmem_tensor(
            (STAGE_K // 8) * B_COLUMNS_PER_THREAD, fx.Uint32
        )
        b_register_scale_bytes = fx.make_rmem_tensor(
            B_REGISTER_SCALE_SLOTS * B_COLUMNS_PER_THREAD, fx.Uint8
        )
        b_compute_tiled = fx.flat_divide(
            b_compute_fragment, (MMA_K, B_COLUMNS_PER_THREAD, 1)
        )
        # gfx11 WMMA duplicates each lane's B column in the other half-wave.
        wave_n = (thread_id_in_wave_group // 32) // WAVES_M
        b_first_column = output_group_col * GROUP_TILE_N + wave_n * MMA_N
        b_first_column = b_first_column + thread_id_in_wave_group % MMA_N
    else:
        # Reinterpret B as (N,K) for WMMA without moving its physical data.
        b_group_staging_transposed_view = fx.make_view(
            fx.get_iter(b_group_staging_view),
            fx.make_layout((GROUP_TILE_N, STAGE_K), (1, GROUP_TILE_N)),
        )
        b_group_staging_transposed_tiled = fx.flat_divide(
            b_group_staging_transposed_view, (GROUP_TILE_N, MMA_K)
        )

        # One K=16 slice of B determines the operand register shape.
        b_group_staging_transposed_initial_view = fx.slice(
            b_group_staging_transposed_tiled, (None, None, 0, 0)
        )

        b_group_staging2reg_copy_plan = fx.make_tiled_copy_B(copy_16b, mma)

        b_lds2reg_thread_copy = b_group_staging2reg_copy_plan.get_slice(
            thread_id_in_wave_group
        )

        b_r_tile = thread_mma.make_fragment_B(b_group_staging_transposed_initial_view)

        b_r_destination = b_lds2reg_thread_copy.retile(b_r_tile)

    def load_b_registers(tile_k):
        # Gather each owning lane's column into chunks of eight K codes.
        for column_slot in fx.range_constexpr(B_COLUMNS_PER_THREAD):
            if fx.const_expr(n == 1 and k % 8 == 0):
                # All padded N columns may reuse column zero. Only C[:, 0] is stored.
                b_column = fx.Int32(0)
            else:
                b_column = b_first_column + column_slot * MMA_N * WAVES_N
            if fx.const_expr(HAS_N_TAIL):
                valid_n = b_column < n
            for scale_slot in fx.range_constexpr(B_REGISTER_SCALE_SLOTS):
                first_k = tile_k * STAGE_K + scale_slot * B_REGISTER_ROWS_PER_SCALE
                scale_offset_b = (first_k // MX_BLOCK_K) * n + b_column
                if fx.const_expr(HAS_N_TAIL):
                    scale_offset_b = valid_n.select(scale_offset_b, SCALE_ROWS * n)
                b_register_scale_bytes[
                    column_slot * B_REGISTER_SCALE_SLOTS + scale_slot
                ] = _load_buffer_word(
                    b_scale_resource, scale_offset_b, fx.Uint8, RIGHT_CACHE_MODIFIER
                )
            for chunk in fx.range_constexpr(STAGE_K // 8):
                if fx.const_expr(n == 1 and k % 8 == 0):
                    # Eight consecutive K values form one complete packed word.
                    first_k = tile_k * STAGE_K + chunk * 8
                    word = _load_buffer_word(
                        b_packed_resource, first_k // 2, fx.Uint32, RIGHT_CACHE_MODIFIER
                    )
                else:
                    word = fx.Uint32(0)
                    for value_id in fx.range_constexpr(8):
                        b_row = tile_k * STAGE_K + chunk * 8 + value_id
                        b_position = b_row * n + b_column
                        b_byte_offset = b_position // 2
                        if fx.const_expr(HAS_N_TAIL):
                            b_byte_offset = valid_n.select(
                                b_byte_offset, (k * n + 1) // 2
                            )
                        byte = _load_buffer_word(
                            b_packed_resource,
                            b_byte_offset,
                            fx.Uint8,
                            RIGHT_CACHE_MODIFIER,
                        )
                        code = (byte.to(fx.Uint32) >> ((b_position % 2) * 4)) & 15
                        word = word | (code << (value_id * 4))
                b_register_codes[column_slot * (STAGE_K // 8) + chunk] = word

    def load_b_stage(tile_k, check_boundary):
        first_row = tile_k * STAGE_K + b_k_lane
        for row_slot in fx.range_constexpr(RIGHT_ROWS_PER_THREAD):
            b_row = first_row + row_slot * RIGHT_K_THREADS
            b_packed_fragment[row_slot] = _load_packed_word(
                b_packed_resource,
                b_row,
                global_col,
                k,
                n,
                RIGHT_CACHE_MODIFIER,
                check_boundary,
            )
        for scale_slot in fx.range_constexpr(RIGHT_SCALE_SLOTS):
            b_scale_row = (
                first_row + scale_slot * ROW_SLOTS_PER_SCALE * RIGHT_K_THREADS
            ) // MX_BLOCK_K
            b_scale_fragment[scale_slot] = _load_row_word(
                b_scale_resource,
                b_scale_row,
                global_col,
                SCALE_ROWS,
                n,
                RIGHT_CACHE_MODIFIER,
                check_boundary,
            )

    def load_nonempty_stage(tile_k):
        a_block_view = fx.slice(a_block_tiles, (None, None, block_row, tile_k))
        a_group_tiles = fx.flat_divide(a_block_view, (GROUP_TILE_M, STAGE_K))
        a_group_view = fx.slice(a_group_tiles, (None, None, group_row, 0))
        a_source = a_global_thread_copy.partition_S(a_group_view)

        if fx.const_expr(RIGHT_IN_REGISTERS):
            load_b_registers(tile_k)
        else:
            # A complete N group can stay wide. In an earlier MX block,
            # the next rows of packed values and scales are still allocated.
            if fx.const_expr(n >= 8 and n % 2 != 0):
                full_group = (output_group_col + 1) * GROUP_TILE_N <= n
                before_final_scale = (tile_k + 1) * STAGE_K <= (
                    SCALE_ROWS - 1
                ) * MX_BLOCK_K
                edge = scf.IfOp(
                    (full_group | before_final_scale).ir_value(), [], has_else=True
                )
                with ir.InsertionPoint(edge.then_block):
                    load_b_stage(tile_k, False)
                    scf.yield_([])
                with ir.InsertionPoint(edge.else_block):
                    load_b_stage(tile_k, True)
                    scf.yield_([])
            else:
                load_b_stage(tile_k, True)

        if fx.const_expr(HAS_K_TAIL):
            # A's extra K columns can alias the next row, including nonfinite values.
            is_k_tail = tile_k == REAL_K_STAGES - 1
            tail = scf.IfOp(is_k_tail.ir_value(), [], has_else=True)
            with ir.InsertionPoint(tail.then_block):
                coordinate_block = fx.slice(
                    a_coordinate_tiles, (None, None, block_row, tile_k)
                )
                coordinate_groups = fx.flat_divide(
                    coordinate_block, (GROUP_TILE_M, STAGE_K)
                )
                group_coordinates = fx.slice(
                    coordinate_groups, (None, None, group_row, 0)
                )
                a_global_destination.fill(0)
                if fx.const_expr(k % STAGE_K >= 8):
                    thread_coordinates = a_global_thread_copy.partition_S(
                        group_coordinates
                    )
                    _copy_valid_columns(
                        a_gmem_load,
                        a_source,
                        a_global_destination,
                        thread_coordinates,
                        k,
                        8,
                    )
                _copy_tail_segments(
                    LEFT_TAIL_SEGMENTS,
                    LEFT_CACHE_MODIFIER,
                    make_a_global_copy,
                    a_group_view,
                    group_coordinates,
                    a_staging_fragment,
                )
                scf.yield_([])
            with ir.InsertionPoint(tail.else_block):
                fx.copy(a_gmem_load, a_source, a_global_destination)
                scf.yield_([])
        else:
            fx.copy(a_gmem_load, a_source, a_global_destination)

    def load_stage(tile_k):
        if fx.const_expr(HAS_EMPTY_K_STAGES):
            # Check the stage before slicing. A singleton K-stage stride can be zero.
            stage_exists = tile_k < REAL_K_STAGES
            a_global_destination.fill(0)
            if fx.const_expr(RIGHT_IN_REGISTERS):
                b_register_codes.fill(0)
                b_register_scale_bytes.fill(0)
            else:
                b_packed_fragment.fill(0)
                b_scale_fragment.fill(0)
            if stage_exists:
                load_nonempty_stage(tile_k)
        else:
            load_nonempty_stage(tile_k)

    def publish_b_registers(tile_k):
        for column_slot in fx.range_constexpr(B_COLUMNS_PER_THREAD):
            for scale_slot in fx.range_constexpr(B_REGISTER_SCALE_SLOTS):
                scale_byte = fx.Uint8(
                    b_register_scale_bytes[
                        column_slot * B_REGISTER_SCALE_SLOTS + scale_slot
                    ]
                ).to(fx.Int32)
                for chunk_in_scale in fx.range_constexpr(
                    B_REGISTER_ROWS_PER_SCALE // 8
                ):
                    chunk = (
                        scale_slot * (B_REGISTER_ROWS_PER_SCALE // 8) + chunk_in_scale
                    )
                    word = fx.Uint32(
                        b_register_codes[column_slot * (STAGE_K // 8) + chunk]
                    )
                    values = _decode_mxfp4_word(word, scale_byte, shared_scale=True)
                    for value_id in fx.range_constexpr(8):
                        local_k = chunk * 8 + value_id
                        value = values[value_id]
                        if fx.const_expr(HAS_K_TAIL or HAS_EMPTY_K_STAGES):
                            valid_k = tile_k * STAGE_K + local_k < k
                            value = valid_k.select(value, fx.BFloat16(0.0))
                        b_compute_fragment[
                            (local_k % MMA_K, column_slot, local_k // MMA_K)
                        ] = value

    def publish_stage(tile_k):
        if fx.const_expr(LEFT_IN_REGISTERS):
            # Make the prefetched A stage current within each lane. This frees
            # the staging registers for the next global load.
            fx.copy(copy_16b, a_staging_fragment, a_compute_fragment)
        else:
            fx.copy(copy_128b, a_staging_fragment, a_staging_destination)
        if fx.const_expr(RIGHT_IN_REGISTERS):
            publish_b_registers(tile_k)
        else:
            # Expand this lane's prefetched codes only as the B tile is published.
            for row_slot in fx.range_constexpr(RIGHT_ROWS_PER_THREAD):
                local_row = b_k_lane + row_slot * RIGHT_K_THREADS
                b_packed_word = fx.Uint32(b_packed_fragment[row_slot])
                b_scale_word = fx.Uint64(
                    b_scale_fragment[row_slot // ROW_SLOTS_PER_SCALE]
                )
                if fx.const_expr(HAS_K_TAIL):
                    valid_k = tile_k * STAGE_K + local_row < k
                    b_packed_word = valid_k.select(b_packed_word, fx.Uint32(0))
                    b_scale_word = valid_k.select(b_scale_word, fx.Uint64(0))
                values = _decode_mxfp4_word(
                    b_packed_word, b_scale_word, shared_scale=False
                )
                b_staging_fragment = fx.make_rmem_tensor(8, fx.BFloat16)
                for value_id in fx.range_constexpr(8):
                    b_staging_fragment[value_id] = values[value_id]
                b_lds_destination = fx.make_view(
                    fx.get_iter(b_group_staging_view)
                    + local_row * GROUP_TILE_N
                    + b_col,
                    fx.make_layout(8, 1),
                )
                fx.copy(copy_128b, b_staging_fragment, b_lds_destination)
        lds_barrier()

    def compute_stage():
        for inner_k in fx.range_constexpr(STAGE_K // MMA_K):
            if fx.const_expr(LEFT_IN_REGISTERS):
                # Keep the three inner axes that match a_r_tile. The outer
                # register-slot and M axes have one tile each, so only K varies.
                # The copy below moves this chunk into a_r_tile within the lane.
                a_source = fx.slice(a_compute_tiled, (None, None, None, 0, 0, inner_k))
            else:
                a_mma_view = fx.slice(a_group_staging_tiled, (None, None, 0, inner_k))
                a_source = a_lds2reg_thread_copy.partition_S(a_mma_view)
            if fx.const_expr(RIGHT_IN_REGISTERS):
                b_source = fx.slice(b_compute_tiled, (None, None, None, 0, 0, inner_k))
            else:
                b_mma_view = fx.slice(
                    b_group_staging_transposed_tiled, (None, None, 0, inner_k)
                )
                b_source = b_lds2reg_thread_copy.partition_S(b_mma_view)
            fx.copy(copy_16b, a_source, a_r_destination)
            fx.copy(copy_16b, b_source, b_r_destination)
            fx.gemm(mma_atom, c_accum, a_r_tile, b_r_tile, c_accum)

    # Prime one stage, then prefetch the next stage before consuming the current one.
    # Equal ceiling-counted slots cover K once. Surplus stages load zeros.
    tiles_per_split = (k + STAGE_K * SPLIT_K - 1) // (STAGE_K * SPLIT_K)
    initial_tile_k = split_id * tiles_per_split
    load_stage(initial_tile_k)
    publish_stage(initial_tile_k)

    full_groups, tail_stages = divmod(tiles_per_split - 1, K_UNROLL)
    for group_k in range(full_groups):
        for step in fx.range_constexpr(K_UNROLL):
            next_tile_k = initial_tile_k + group_k * K_UNROLL + step + 1
            load_stage(next_tile_k)
            compute_stage()
            lds_barrier()
            publish_stage(next_tile_k)
    for tail in fx.range_constexpr(tail_stages):
        next_tile_k = initial_tile_k + full_groups * K_UNROLL + tail + 1
        load_stage(next_tile_k)
        compute_stage()
        lds_barrier()
        publish_stage(next_tile_k)

    # Drain the last published stage.
    compute_stage()
    lds_barrier()

    def store_output():
        c_registers = fx.make_rmem_tensor(c_accum.layout, fx.BFloat16)
        for slot in fx.range_constexpr(ACCUMULATOR_VALUES):
            c_registers[slot] = c_accum[slot].to(fx.BFloat16)
        # The MMA C plan stores into this group's selected output view.
        gmem_store_16b = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        reg2gmem_copy_plan = fx.make_tiled_copy_C(gmem_store_16b, mma)
        reg2gmem_thread_copy = reg2gmem_copy_plan.get_slice(thread_id_in_wave_group)
        reg2gmem_source = reg2gmem_thread_copy.retile(c_registers)
        reg2gmem_dest = reg2gmem_thread_copy.partition_D(c_group_view)
        # Complete groups store through C's descriptor, which also handles M tails.
        if fx.const_expr(HAS_N_TAIL):
            complete_n = (output_group_col + 1) * GROUP_TILE_N <= n
            n_tile = scf.IfOp(complete_n.ir_value(), [], has_else=True)
            with ir.InsertionPoint(n_tile.then_block):
                fx.copy(gmem_store_16b, reg2gmem_source, reg2gmem_dest)
                scf.yield_([])
            with ir.InsertionPoint(n_tile.else_block):
                coordinate_block = fx.slice(
                    c_coordinate_tiles, (None, None, block_row, block_col)
                )
                coordinate_groups = fx.flat_divide(
                    coordinate_block, (GROUP_TILE_M, GROUP_TILE_N)
                )
                group_coordinates = fx.slice(
                    coordinate_groups, (None, None, group_row, group_col)
                )
                output_thread_coordinates = reg2gmem_thread_copy.partition_D(
                    group_coordinates
                )
                _copy_valid_columns(
                    gmem_store_16b,
                    reg2gmem_source,
                    reg2gmem_dest,
                    output_thread_coordinates,
                    n,
                    1,
                )
                scf.yield_([])
        else:
            fx.copy(gmem_store_16b, reg2gmem_source, reg2gmem_dest)

    if SPLIT_K == 1:
        store_output()
    else:
        # Publish each thread's FP32 slots to the group's distinct split plane.
        tile_values = GROUP_TILE_M * GROUP_TILE_N
        thread_offset = thread_id_in_wave_group * ACCUMULATOR_VALUES
        partial_offset = (
            flat_group_id * SPLIT_K + split_id
        ) * tile_values + thread_offset
        partial_view = fx.make_view(
            fx.get_iter(partials) + partial_offset, c_accum.layout
        )
        copy_128b_f32 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
        fx.copy(copy_128b_f32, c_accum, partial_view)

        # One counter per (output tile, thread), shared by its split-K peers.
        # AcqRel publishes this thread's stores and acquires all earlier arrivals.
        counter_ptr = (
            fx.get_iter(counters)
            + flat_group_id * THREADS_PER_GROUP
            + thread_id_in_wave_group
        )
        arrived = fx.atomic_add(
            counter_ptr,
            1,
            ordering=fx.AtomicOrdering.AcqRel,
            syncscope=fx.rocdl.SyncScope.Agent,
        )
        last_arrival = arrived == SPLIT_K - 1
        finalizer = scf.IfOp(last_arrival.ir_value(), [], has_else=False)
        with ir.InsertionPoint(finalizer.then_block):
            # No polling: only the last thread for these output slots reduces.
            c_accum.fill(0)
            partial_registers = fx.make_fragment_like(c_accum)
            for part in fx.range_constexpr(SPLIT_K):
                source_offset = (
                    flat_group_id * SPLIT_K + part
                ) * tile_values + thread_offset
                partial_source = fx.make_view(
                    fx.get_iter(partials) + source_offset, c_accum.layout
                )
                fx.copy(copy_128b_f32, partial_source, partial_registers)
                for slot in fx.range_constexpr(ACCUMULATOR_VALUES):
                    c_accum[slot] = c_accum[slot] + partial_registers[slot]
            store_output()
            # Same-stream launches can reuse the workspace without a clear kernel.
            fx.generic_store(
                counter_ptr,
                fx.Int32(0),
                memory_order=fx.AtomicOrdering.Release,
                syncscope=fx.rocdl.SyncScope.Agent,
            )
            scf.yield_([])


@flyc.jit
def launch_mxfp4_w4a16(
    a: fx.Tensor,
    b_packed: fx.Tensor,
    b_scales: fx.Tensor,
    c: fx.Tensor,
    partials,
    counters,
    m: fx.Constexpr,
    n: fx.Constexpr,
    k: fx.Constexpr,
    GROUP_TILE_M: fx.Constexpr,
    GROUP_TILE_N: fx.Constexpr,
    STAGE_K: fx.Constexpr,
    WAVES_M: fx.Constexpr,
    WAVES_N: fx.Constexpr,
    WAVE_GROUPS_M: fx.Constexpr,
    WAVE_GROUPS_N: fx.Constexpr,
    K_UNROLL: fx.Constexpr,
    RIGHT_CACHE_MODIFIER: fx.Constexpr,
    LEFT_CACHE_MODIFIER: fx.Constexpr,
    LEFT_IN_REGISTERS: fx.Constexpr,
    RIGHT_IN_REGISTERS: fx.Constexpr,
    GROUP_M: fx.Constexpr,
    SPLIT_K: fx.Constexpr,
    stream: fx.Stream,
):
    WAVE_GROUPS = WAVE_GROUPS_M * WAVE_GROUPS_N
    BLOCK_M = GROUP_TILE_M * WAVE_GROUPS_M
    BLOCK_N = GROUP_TILE_N * WAVE_GROUPS_N
    THREADS_PER_GROUP = 32 * WAVES_M * WAVES_N
    THREADS_PER_BLOCK = THREADS_PER_GROUP * WAVE_GROUPS
    _mxfp4_w4a16(
        a,
        b_packed,
        b_scales,
        c,
        partials,
        counters,
        m,
        n,
        k,
        GROUP_TILE_M=GROUP_TILE_M,
        GROUP_TILE_N=GROUP_TILE_N,
        STAGE_K=STAGE_K,
        WAVES_M=WAVES_M,
        WAVES_N=WAVES_N,
        WAVE_GROUPS_M=WAVE_GROUPS_M,
        WAVE_GROUPS_N=WAVE_GROUPS_N,
        K_UNROLL=K_UNROLL,
        RIGHT_CACHE_MODIFIER=RIGHT_CACHE_MODIFIER,
        LEFT_CACHE_MODIFIER=LEFT_CACHE_MODIFIER,
        LEFT_IN_REGISTERS=LEFT_IN_REGISTERS,
        RIGHT_IN_REGISTERS=RIGHT_IN_REGISTERS,
        GROUP_M=GROUP_M,
        SPLIT_K=SPLIT_K,
    ).launch(
        grid=(
            fx.ceildiv(m, BLOCK_M) * fx.ceildiv(n, BLOCK_N) * SPLIT_K,
            1,
            1,
        ),
        block=(THREADS_PER_BLOCK, 1, 1),
        stream=stream,
    )
