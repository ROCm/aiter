import json, torch, sys
grid = int(sys.argv[1]) if len(sys.argv) > 1 else 64
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from kernel import sync

@g.jit
def probe(P, C, H, GRID: gl.constexpr, ROUNDS: gl.constexpr):
    bid = gl.program_id(0)
    i = gl.arange(0, GRID, layout=gl.BlockedLayout([1], [32], [4], [0]))
    for step in range(ROUNDS):
        gl.store(P + bid, bid + step * 128, cache_modifier='.cg')
        sync(C, H, GRID)
        v = gl.load(P + i, cache_modifier='.cg')
        if gl.sum(v, 0) != GRID * (GRID - 1) // 2 + step * 128 * GRID:
            gl.atomic_or(H, 4, sem='relaxed', scope='gpu')
        sync(C, H, GRID)
p = torch.empty(grid, device='cuda', dtype=torch.int32)
c = torch.zeros(1, device='cuda', dtype=torch.int32)
h = torch.zeros_like(c)
k = probe.warmup(p, c, h, grid, 16, grid=(grid,), num_warps=4)
k._init_handles()
print('probe resources', k.n_regs, k.n_spills, k.metadata.shared, flush=True)
for initial in (0, 2147483648 - grid, -grid):
    c.fill_(initial)
    h.zero_()
    probe[grid,](p, c, h, grid, 16, num_warps=4)
    torch.cuda.synchronize()
    assert h.item() == 0, (initial, h.item())
    print('sync pass', initial, c.item(), flush=True)
graph = torch.cuda.CUDAGraph()
c.zero_()
h.zero_()
with torch.cuda.graph(graph):
    probe[grid,](p, c, h, grid, 16, num_warps=4)
for _ in range(1000):
    graph.replay()
torch.cuda.synchronize()
assert h.item() == 0
print('1000 graph replays pass', flush=True)
c.zero_()
h.zero_()
probe[grid // 2,](p, c, h, grid, 1, num_warps=4)
torch.cuda.synchronize()
assert h.item() != 0
print('missing producer fail-stop diagnostic', h.item(), flush=True)
