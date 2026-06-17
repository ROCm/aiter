#!/usr/bin/env python3
"""Introspect the CDNA4 MFMA_Scale 32x32x64 atom TV layouts.

Prints layout_A_tv, layout_B_tv, layout_C_tv and decodes the
(thread, value) -> linear coordinate mapping into (row, col) for each
of A (M x K), B (K x N), C (M x N).
"""

import flydsl.expr as fx
import flydsl.expr.primitive as prim
from flydsl._mlir.dialects import func
from flydsl._mlir.ir import Context, FunctionType, InsertionPoint, Location, Module


def decode_tv(layout, n_threads, n_values, nrows, ncols, name, col_major_coord):
    """layout: TV layout. coord index -> (row,col).

    col_major_coord: if True the linear coord index is column-major over
    (nrows, ncols), i.e. idx = col*nrows + row (typical cute/cutlass
    M-major). If False it is row-major idx = row*ncols + col.
    """
    print(
        f"\n===== {name}  ({nrows} x {ncols})  threads={n_threads} vals/thread={n_values} ====="
    )
    print(f"  raw layout: {layout}")
    # mapping[(t,v)] = (row, col)
    for t in range(n_threads):
        coords = []
        for v in range(n_values):
            idx = prim.crd2idx((t, v), layout).get_static_leaf_int
            if col_major_coord:
                row = idx % nrows
                col = idx // nrows
            else:
                row = idx // ncols
                col = idx % ncols
            coords.append((idx, row, col))
        if t < 8 or t == n_threads - 1:
            pretty = " ".join(f"v{v}=(r{r},c{c})" for v, (i, r, c) in enumerate(coords))
            print(f"  lane {t:2d}: {pretty}")
    return


def _main():
    M, N, K = 32, 32, 64
    atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(M, N, K, fx.Float8E4M3FN))
    print("shape_mnk:", atom.shape_mnk)
    lA = atom.layout_A_tv
    lB = atom.layout_B_tv
    lC = atom.layout_C_tv

    # A/B carry 32 fp8 values per lane (vec<8xi32> = 32 fp8); C carries 16 f32.
    # Coordinate index is column-major over (rows, cols) (cute M-major) for A
    # and C.  NOTE: the col-major decode of B prints (K,N) coords that do NOT
    # match hardware; the empirically verified B map (see rocdl_mfma_fp8.py and
    # the BT1 probe) is B_frag[L][v] = B[row(K)=(L//32)*32+v, col(N)=L%32],
    # i.e. the same lane/value structure as A.
    decode_tv(lA, 64, 32, M, K, "A (M x K) fp8 frag", True)
    decode_tv(lB, 64, 32, K, N, "B (K x N) fp8 frag [col-major decode; see note]", True)
    decode_tv(lC, 64, 16, M, N, "C (M x N) f32 frag", True)


def main():
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown(ctx):
            module = Module.create()
            with InsertionPoint(module.body):
                f = func.FuncOp("introspect", FunctionType.get([], []))
                with InsertionPoint(f.add_entry_block()):
                    _main()
                    func.ReturnOp([])


if __name__ == "__main__":
    main()
