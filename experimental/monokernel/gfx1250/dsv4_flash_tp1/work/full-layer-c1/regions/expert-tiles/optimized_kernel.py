from kernel import *

@g.jit
def routed_matvec(X, W, S, Y, N: gl.constexpr, K: gl.constexpr, FORMAT: gl.constexpr, GRID: gl.constexpr, GROUPED: gl.constexpr=False, FLOAT_OUT: gl.constexpr=False, Gamma=None, Ids=None):
    L: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    rn = gl.arange(0, 8, layout=gl.SliceLayout(1, L))
    rk = gl.arange(0, 256, layout=gl.SliceLayout(0, L))
    for tile in range(gl.program_id(0), gl.cdiv(N * 6, 8), GRID):
        outn = tile * 8 + rn
        expert = dev_load(Ids, outn // N, cache='.cg').to(gl.int64)
        n = outn % N
        acc = gl.full((8, 256), 0, gl.float32, L)
        for kb in range(gl.cdiv(K, 256)):
            k = kb * 256 + rk
            if GROUPED:
                x = dev_load(X, n[:, None] // 1024 * K + k[None, :], cache='.cg')
            else:
                if Gamma is not None:
                    x = dev_load(X, (outn // N)[:, None] * K + k[None, :], cache='.cg')
                else:
                    x = dev_load(X, k, cache='.cg')
                    x = x[None, :]
            if FORMAT == 4:
                packed = dev_load(W, expert[:, None] * N * (K // 2) + n[:, None] * (K // 2) + k[None, :] // 2, cache='.cg').to(gl.uint8)
                v = packed >> k[None, :] % 2 * 4 & 15
                a = v & 7
                w = gl.where(a < 2, a.to(gl.float32) * 0.5, gl.where(a < 4, (a.to(gl.float32) - 2) * 0.5 + 1, gl.where(a < 6, a.to(gl.float32) - 2, (a.to(gl.float32) - 6) * 2 + 4)))
                w = gl.where(v >= 8, -w, w)
                s = dev_load(S, expert[:, None] * N * (K // 32) + n[:, None] * (K // 32) + k[None, :] // 32, cache='.cg').to(gl.uint32)
                w *= (s << 23).to(gl.float32, bitcast=True)
            elif FORMAT == 8:
                w = dev_load(W, n[:, None] * K + k[None, :], cache='.cg').to(gl.float32)
                s = dev_load(S, n[:, None] // 128 * (K // 128) + k[None, :] // 128, cache='.cg').to(gl.uint32)
                w *= (s << 23).to(gl.float32, bitcast=True)
            else:
                w = dev_load(W, n[:, None] * K + k[None, :], cache='.cg').to(gl.float32)
            acc += w * x
        out = gl.sum(acc, 1)
        if Gamma is not None:
            out *= dev_load(Gamma, outn // N, cache='.cg')
        if not FLOAT_OUT:
            out = out.to(gl.bfloat16)
        dev_store(out, Y, outn, cache='.cg')

