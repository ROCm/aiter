"""Host two-shot SQNR sim: HIP INT6 vs keeper INT4 vs MX 6-bit vs MXFP4.

Matches the QR harness input (N(0, 0.1^2) per rank) and the two-shot
quantize → fp32-sum → requantize → dequant path. Not a GPU kernel.

HIP CodecQ6 numeric: signed INT6 [-32,+31], scale = absmax/32 (fp16 on
the wire; fp32 here is the same to printed digits), group-32. GPU
validate_qr.py INT6 rel MAE ~3.05e-02; this sim lands at 3.30e-02.
"""

from __future__ import annotations

import math

import numpy as np

RNG = np.random.default_rng(1234)
WORLD = 8
BLOCK = 32
N_BLOCKS = 4096  # 131072 values / rank-slice; enough for a stable SQNR
SIGMA = 0.1


def sqnr_db(ref, got):
    err = got - ref
    mse = np.mean(err * err)
    power = np.mean(ref * ref)
    return 10.0 * math.log10(power / mse), float(
        np.mean(np.abs(err)) / np.mean(np.abs(ref))
    )


def e4m3_encode(x):
    """Keeper signed E4M3 of a f32 (qr_int4_kernel._f32_to_e4m3)."""
    x = np.asarray(x, dtype=np.float32)
    out = np.zeros(x.shape, dtype=np.uint8)
    z = x == 0
    sign = np.where(x < 0, np.uint8(0x80), np.uint8(0))
    bits = np.abs(x).view(np.uint32)
    e = (bits >> 23).astype(np.int32) - 127
    mant = bits & 0x7FFFFF
    m3 = (mant + (1 << 19)) >> 20
    carry = m3 == 8
    e = e + carry.astype(np.int32)
    m3 = np.where(carry, 0, m3)
    e4 = np.clip(e + 7, 0, 15)
    byte = sign | ((e4.astype(np.uint8) & 15) << 3) | (np.uint8(m3) & 7)
    out = np.where(z, np.uint8(0), byte)
    return out


def e4m3_decode(b):
    b = np.asarray(b, dtype=np.uint32)
    z = b == 0
    sign = (b & 0x80) != 0
    e4 = (b >> 3) & 15
    m3 = b & 7
    mag_bits = ((e4 + 120) << 23) | (m3 << 20)
    mag = mag_bits.astype(np.uint32).view(np.float32)
    signed = np.where(sign, -mag, mag)
    return np.where(z, np.float32(0.0), signed).astype(np.float32)


def quant_int4_e4m3_group16(x):
    """c16q4: group-16 signed extremum → E4M3, INT4 nibbles, decode scale × -1/8."""
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[-1]
    g = 16
    assert n % g == 0
    blk = x.reshape(*x.shape[:-1], n // g, g)
    amax = np.max(blk, axis=-1)
    amin = np.min(blk, axis=-1)
    ext = np.where(np.abs(amax) >= np.abs(amin), amax, amin)
    e = e4m3_encode(ext)
    d = e4m3_decode(e) * np.float32(-0.125)
    inv = np.float32(1.0) / (d + np.float32(1e-7))
    q = np.round(blk * inv[..., None]).clip(-8, 7)
    rec = q * d[..., None]
    return rec.reshape(x.shape)


def quant_int6_group32(x):
    """HIP CodecQ6 numeric: signed INT6, scale = absmax/32, group-32."""
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[-1]
    g = 32
    assert n % g == 0
    blk = x.reshape(*x.shape[:-1], n // g, g)
    absmax = np.max(np.abs(blk), axis=-1)
    scale = np.maximum(absmax / np.float32(32.0), np.float32(1e-7))
    q = np.rint(blk / scale[..., None]).clip(-32, 31)
    rec = q * scale[..., None]
    return rec.reshape(x.shape)


def quant_int6_e8m0_group32(x):
    """INT6 codebook with E8M0 scale (PK32 constraint, not HIP)."""
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[-1]
    g = 32
    blk = x.reshape(*x.shape[:-1], n // g, g)
    absmax = np.max(np.abs(blk), axis=-1)
    need = np.maximum(absmax / np.float32(32.0), np.float32(1e-30))
    e = np.clip(np.ceil(np.log2(need)) + 127.0, 0, 254).astype(np.int32)
    scale = np.exp2(e.astype(np.float32) - 127.0)
    q = np.rint(blk / scale[..., None]).clip(-32, 31)
    rec = q * scale[..., None]
    return rec.reshape(x.shape)


def fp6_e2m3_from_f32(y):
    """Round y to OCP E2M3 (bias 1, max ±7.5, e=0 denorms)."""
    y = np.asarray(y, dtype=np.float32)
    sign = np.signbit(y)
    mag = np.abs(y)
    maxv = np.float32(7.5)
    mag = np.minimum(mag, maxv)
    # normals: (1 + m/8) * 2^(e-1), e in 1..3
    # denorms: (m/8) * 2^(1-1) = m/8; rounding may carry to 1.0.
    out = np.zeros(y.shape, dtype=np.float32)
    # denorm / zero
    den = mag < 1.0
    m_den = np.rint(mag * 8.0)
    out = np.where(den, m_den.astype(np.float32) / 8.0, out)
    # normals
    # find e such that 2^(e-1) <= mag < 2^e  for e=1,2,3  i.e. [1,2), [2,4), [4,8)
    log2 = np.floor(np.log2(np.maximum(mag, np.float32(1e-30))))
    e = np.clip(log2 + 1, 1, 3).astype(np.int32)
    scale = np.exp2(e.astype(np.float32) - 1.0)
    m = np.rint((mag / scale - 1.0) * 8.0)
    # carry into exp
    carry = m == 8
    m = np.where(carry, 0, m)
    e = np.clip(e + carry.astype(np.int32), 1, 3)
    scale = np.exp2(e.astype(np.float32) - 1.0)
    rec = (1.0 + m.astype(np.float32) / 8.0) * scale
    rec = np.minimum(rec, maxv)
    out = np.where(den, out, rec)
    out = np.where(sign, -out, out)
    return out.astype(np.float32)


def fp6_e3m2_from_f32(y):
    """Round y to OCP E3M2 (bias 3, max ±28, min normal 0.25)."""
    y = np.asarray(y, dtype=np.float32)
    sign = np.signbit(y)
    mag = np.abs(y)
    maxv = np.float32(28.0)
    mag = np.minimum(mag, maxv)
    out = np.zeros(y.shape, dtype=np.float32)
    den = mag < 0.25
    # denorm: (m/4) * 2^(1-3) = m/16, m in 1..3
    m_den = np.rint(mag * 16.0)
    out = np.where(den, m_den.astype(np.float32) / 16.0, out)
    log2 = np.floor(np.log2(np.maximum(mag, np.float32(1e-30))))
    e = np.clip(log2 + 3, 1, 7).astype(np.int32)
    scale = np.exp2(e.astype(np.float32) - 3.0)
    m = np.rint((mag / scale - 1.0) * 4.0)
    carry = m == 4
    m = np.where(carry, 0, m)
    e = np.clip(e + carry.astype(np.int32), 1, 7)
    scale = np.exp2(e.astype(np.float32) - 3.0)
    rec = (1.0 + m.astype(np.float32) / 4.0) * scale
    rec = np.minimum(rec, maxv)
    out = np.where(den, out, rec)
    out = np.where(sign, -out, out)
    return out.astype(np.float32)


def fp4_e2m1_from_f32(y):
    y = np.asarray(y, dtype=np.float32)
    sign = np.signbit(y)
    mag = np.abs(y)
    maxv = np.float32(6.0)
    mag = np.minimum(mag, maxv)
    # E2M1 bias 1, max 6, min normal 1. Codes: 0, 0.5 denorm; 1,1.5, 2,3, 4,6
    codes = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    idx = np.abs(mag[..., None] - codes).argmin(axis=-1)
    rec = codes[idx]
    rec = np.where(sign, -rec, rec)
    rec = np.where(mag == 0, 0.0, rec)
    return rec.astype(np.float32)


def e8m0_from_absmax(absmax, fmt_max):
    """Ceil power-of-two scale so absmax / 2^(e-127) <= fmt_max."""
    absmax = np.maximum(np.asarray(absmax, dtype=np.float32), np.float32(1e-30))
    # want 2^(e-127) >= absmax / fmt_max
    need = absmax / np.float32(fmt_max)
    log2 = np.ceil(np.log2(need))
    e = np.clip(log2 + 127.0, 0, 254).astype(np.int32)
    scale = np.exp2(e.astype(np.float32) - 127.0)
    return scale


def quant_mx(x, block, fmt_max, quant_fn):
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[-1]
    assert n % block == 0
    blk = x.reshape(*x.shape[:-1], n // block, block)
    absmax = np.max(np.abs(blk), axis=-1)
    scale = e8m0_from_absmax(absmax, fmt_max)
    y = blk / scale[..., None]
    rec = quant_fn(y) * scale[..., None]
    return rec.reshape(x.shape)


def two_shot(codec, xs):
    """xs: (WORLD, N) per-rank shards of one dest slice. Return reconstructed AR."""
    q = np.stack([codec(x) for x in xs], axis=0)
    reduced = q.sum(axis=0)
    q2 = codec(reduced)
    return q2


def e3m2_levels():
    lv = [0.0]
    for md in range(1, 4):
        lv.append(md / 16.0)
    for e in range(1, 8):
        scale = 2.0 ** (e - 3)
        for md in range(4):
            lv.append((1.0 + md / 4.0) * scale)
    return np.sort(np.unique(np.clip(np.array(lv, dtype=np.float32), 0, 28.0)))


def e2m3_levels():
    lv = [0.0]
    for md in range(1, 8):
        lv.append(md / 8.0)
    for e in range(1, 4):
        scale = 2.0 ** (e - 1)
        for md in range(8):
            lv.append((1.0 + md / 8.0) * scale)
    return np.sort(np.unique(np.clip(np.array(lv, dtype=np.float32), 0, 7.5)))


L_E3M2 = e3m2_levels()
L_E2M3 = e2m3_levels()


def quant_sr(y, levels, rng):
    """Unbiased SR onto adjacent codebook values. Host stand-in for SR_PK32."""
    y = np.asarray(y, dtype=np.float32)
    sign = np.signbit(y)
    mag = np.minimum(np.abs(y), levels[-1])
    idx = np.clip(np.searchsorted(levels, mag, side="left"), 1, len(levels) - 1)
    hi = levels[idx]
    lo = levels[idx - 1]
    on = mag == hi
    lo = np.where(on, hi, lo)
    span = hi - lo
    p = np.divide(mag - lo, span, out=np.zeros_like(mag), where=span > 0)
    rec = np.where(rng.random(mag.shape, dtype=np.float32) < p, hi, lo).astype(
        np.float32
    )
    rec = np.where(sign, -rec, rec)
    return np.where(np.abs(y) == 0, 0.0, rec)


def quant_mx_sr(x, block, fmt_max, levels, rng):
    x = np.asarray(x, dtype=np.float32)
    blk = x.reshape(*x.shape[:-1], x.shape[-1] // block, block)
    absmax = np.max(np.abs(blk), axis=-1)
    scale = e8m0_from_absmax(absmax, fmt_max)
    rec = quant_sr(blk / scale[..., None], levels, rng) * scale[..., None]
    return rec.reshape(x.shape)


def two_shot_sr(xs, fmt_max, levels, rng):
    q = np.stack([quant_mx_sr(x, 32, fmt_max, levels, rng) for x in xs], axis=0)
    return quant_mx_sr(q.sum(axis=0), 32, fmt_max, levels, rng)


def report_sr(name, xs, ref, fmt_max, levels, trials=16, seed=999):
    sqs, rmas, biases = [], [], []
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        got = two_shot_sr(xs, fmt_max, levels, rng)
        s, r = sqnr_db(ref, got)
        sqs.append(s)
        rmas.append(r)
        biases.append(float(np.mean(got - ref)))
    print(
        f"{name:<28} {np.mean(sqs):6.2f}±{np.std(sqs):.2f} "
        f"{np.mean(rmas):12.4e}  bias={np.mean(biases):.3e}±{np.std(biases):.1e}"
    )


def peakiness(xs, block=BLOCK):
    """Median over MX blocks of (median |x| / absmax). 1=flat, 0=one spike."""
    blk = np.asarray(xs).reshape(-1, block)
    amax = np.max(np.abs(blk), axis=-1)
    med = np.median(np.abs(blk), axis=-1)
    r = np.divide(med, amax, out=np.zeros_like(med), where=amax > 0)
    return float(np.median(r))


def make_shards(kind, rng, n=N_BLOCKS * BLOCK):
    if kind == "gauss σ=0.1":
        return rng.standard_normal((WORLD, n), dtype=np.float32) * 0.1
    if kind == "gauss σ=10":
        return rng.standard_normal((WORLD, n), dtype=np.float32) * 10.0
    if kind == "bf16-gauss":
        return (
            (rng.standard_normal((WORLD, n), dtype=np.float32) * 0.1)
            .astype(np.float16)
            .astype(np.float32)
        )
    if kind == "laplace":
        return rng.laplace(0.0, 0.1, size=(WORLD, n)).astype(np.float32)
    if kind == "student-t df=3":
        return rng.standard_t(3, size=(WORLD, n)).astype(np.float32) * 0.1
    if kind == "uniform":
        return rng.uniform(-0.2, 0.2, size=(WORLD, n)).astype(np.float32)
    if kind == "flat-block":
        scale = rng.uniform(0.05, 0.2, size=(WORLD, n // BLOCK, 1)).astype(np.float32)
        mag = scale * rng.uniform(0.8, 1.0, size=(WORLD, n // BLOCK, BLOCK)).astype(
            np.float32
        )
        sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=mag.shape)
        return (mag * sign).reshape(WORLD, n)
    if kind == "spike":
        xs = rng.standard_normal((WORLD, n), dtype=np.float32) * 0.005
        blk = xs.reshape(WORLD, n // BLOCK, BLOCK)
        spike = rng.standard_normal((WORLD, n // BLOCK), dtype=np.float32) * 0.2
        idx = rng.integers(0, BLOCK, size=(WORLD, n // BLOCK))
        blk[np.arange(WORLD)[:, None], np.arange(n // BLOCK)[None, :], idx] = spike
        return blk.reshape(WORLD, n)
    if kind == "rmsnorm":
        x = rng.standard_normal((WORLD, n), dtype=np.float32)
        g = 128
        t = x.reshape(WORLD, n // g, g)
        rms = np.sqrt(np.mean(t * t, axis=-1, keepdims=True) + 1e-6)
        return (t / rms).reshape(WORLD, n)
    if kind == "90% zero":
        xs = rng.standard_normal((WORLD, n), dtype=np.float32) * 0.1
        xs[rng.random((WORLD, n)) < 0.9] = 0.0
        return xs
    raise KeyError(kind)


def sr_mean_sqnr(xs, ref, fmt_max, levels, trials, seed=999):
    rng = np.random.default_rng(seed)
    sqs = [
        sqnr_db(ref, two_shot_sr(xs, fmt_max, levels, rng))[0] for _ in range(trials)
    ]
    return float(np.mean(sqs)), float(np.std(sqs))


def sweep_peakiness(trials=8):
    kinds = [
        "gauss σ=0.1",
        "gauss σ=10",
        "bf16-gauss",
        "laplace",
        "student-t df=3",
        "uniform",
        "flat-block",
        "spike",
        "rmsnorm",
        "90% zero",
    ]

    def bf6(x):
        return quant_mx(x, 32, 28.0, fp6_e3m2_from_f32)

    def fp6(x):
        return quant_mx(x, 32, 7.5, fp6_e2m3_from_f32)

    print()
    print(f"peakiness sweep, SR {trials} trials. pk = median_block(median|x|/absmax)")
    print(
        f"{'dist':<16} {'pk':>5} {'INT4':>6} {'INT6':>6} {'BF6R':>6} {'BF6S':>10} "
        f"{'FP6R':>6} {'FP6S':>10}  mx-winner"
    )
    for kind in kinds:
        xs = make_shards(kind, np.random.default_rng(1234))
        ref = xs.sum(axis=0)
        pk = peakiness(xs)
        int4 = sqnr_db(ref, two_shot(quant_int4_e4m3_group16, xs))[0]
        int6 = sqnr_db(ref, two_shot(quant_int6_group32, xs))[0]
        bf6r = sqnr_db(ref, two_shot(bf6, xs))[0]
        fp6r = sqnr_db(ref, two_shot(fp6, xs))[0]
        bf6s = sr_mean_sqnr(xs, ref, 28.0, L_E3M2, trials)
        fp6s = sr_mean_sqnr(xs, ref, 7.5, L_E2M3, trials)
        scores = {
            "BF6 RNE": bf6r,
            "BF6 SR": bf6s[0],
            "FP6 RNE": fp6r,
            "FP6 SR": fp6s[0],
        }
        winner = max(scores, key=scores.get)
        print(
            f"{kind:<16} {pk:5.2f} {int4:6.2f} {int6:6.2f} {bf6r:6.2f} "
            f"{bf6s[0]:5.2f}±{bf6s[1]:.2f} {fp6r:6.2f} "
            f"{fp6s[0]:5.2f}±{fp6s[1]:.2f}  {winner}"
        )


def main():
    n = N_BLOCKS * BLOCK
    xs = RNG.standard_normal((WORLD, n), dtype=np.float32) * SIGMA
    ref = xs.sum(axis=0)

    codecs = {
        "c16q4 INT4+E4M3/16": lambda x: quant_int4_e4m3_group16(x),
        "HIP INT6 absmax/32 g32": quant_int6_group32,
        "INT6 E8M0/32": quant_int6_e8m0_group32,
        "MXFP6 E2M3+E8M0/32": lambda x: quant_mx(x, 32, 7.5, fp6_e2m3_from_f32),
        "MXBF6 E3M2+E8M0/32": lambda x: quant_mx(x, 32, 28.0, fp6_e3m2_from_f32),
        "MXFP4 E2M1+E8M0/32": lambda x: quant_mx(x, 32, 6.0, fp4_e2m1_from_f32),
        "MXFP6 E2M3+E8M0/16": lambda x: quant_mx(x, 16, 7.5, fp6_e2m3_from_f32),
    }
    print(f"world={WORLD} n={n} sigma={SIGMA} two-shot vs fp32 sum")
    print(f"{'codec':<28} {'SQNR dB':>10} {'rel MAE':>12}")
    for name, fn in codecs.items():
        got = two_shot(fn, xs)
        sqnr, rmae = sqnr_db(ref, got)
        print(f"{name:<28} {sqnr:10.2f} {rmae:12.4e}")

    print()
    print("SR vs RNE, 16 trials, host codebook SR (stand-in for SR_PK32)")
    print(f"{'codec':<28} {'SQNR dB':>14} {'rel MAE':>12}  mean(got-ref)")
    for name, fn in (
        ("MXBF6 RNE", codecs["MXBF6 E3M2+E8M0/32"]),
        ("MXFP6 RNE", codecs["MXFP6 E2M3+E8M0/32"]),
    ):
        got = two_shot(fn, xs)
        sqnr, rmae = sqnr_db(ref, got)
        print(
            f"{name:<28} {sqnr:14.2f} {rmae:12.4e}  bias={float(np.mean(got - ref)):.3e}"
        )
    report_sr("MXBF6 SR E3M2+E8M0/32", xs, ref, 28.0, L_E3M2)
    report_sr("MXFP6 SR E2M3+E8M0/32", xs, ref, 7.5, L_E2M3)
    sweep_peakiness()


if __name__ == "__main__":
    main()
