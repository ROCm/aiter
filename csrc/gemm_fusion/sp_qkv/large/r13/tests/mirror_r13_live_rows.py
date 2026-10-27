# independent mirror of lgc::r13_live_ok (large/r13): for every allowed plan and EVERY m_real in 1..chunks*256 the live rows of the put groups sum to m_real and form a contiguous prefix
# (no live row after a non-full group), so R[:4*m_real] is contiguous; groups without live rows issue no put but still pass their wait and release their arrival (counts unchanged).
def g_first(NP, HG, g, chunks):
    return [0, 2, 3][g] if HG == 2 else g * (chunks // NP)


def g_n(NP, HG, g, chunks):
    return [2, 1, chunks - 3][g] if HG == 2 else chunks // NP


def live(m, c0, cpp):
    lo, hi = c0 * 256, (c0 + cpp) * 256
    e = min(m, hi)
    return e - lo if e > lo else 0


def ok(NP, HG, chunks):
    for m in range(1, chunks * 256 + 1):
        s = 0
        tail = False
        for g in range(NP):
            c0, cpp = g_first(NP, HG, g, chunks), g_n(NP, HG, g, chunks)
            l = live(m, c0, cpp)
            if tail and l != 0:
                return False
            if l != cpp * 256:
                tail = True
            s += l
        if s != m:
            return False
    return True


plans = (
    [(1, 0, c) for c in (1, 2, 4, 8, 16, 32)]
    + [(2, 0, c) for c in (2, 4, 8, 16, 32)]
    + [(4, 0, c) for c in (4, 8, 16, 32)]
    + [(2, 1, c) for c in (2, 4, 8, 16, 32)]
    + [(4, 1, c) for c in (4, 8, 16, 32)]
    + [(3, 2, 8)]
)
bad = [p for p in plans if not ok(*p)]
assert not bad, bad
# the examples of the design: m_real 1025 / 1100 / 1279 / 2049 / 4097 / 4351 in their buckets
for m, chunks, NP, HG in (
    (1025, 5, 1, 0),
    (1100, 5, 1, 0),
    (1279, 5, 1, 0),
    (2049, 9, 1, 0),
    (4097, 17, 1, 0),
    (4351, 17, 1, 0),
    (1100, 8, 2, 0),
    (1100, 8, 2, 1),
    (1100, 8, 3, 2),
):
    pass
assert (
    live(1100, 0, 4) == 1024
    and live(1100, 4, 4) == 76
    and live(1, 0, 4) == 1
    and live(1, 4, 4) == 0
)
tags = {
    ((m - 1) << 19) | (c & 0x7FFFF)
    for m in (1, 255, 256, 257, 8192)
    for c in (0, 1, 0x7FFFF)
}
assert len(tags) == 15 and all(t < 2**32 for t in tags)
print(
    "r13 live-rows mirror OK:",
    len(plans),
    "plans x every m_real; contiguity; tag packing fits 32 bits",
)
