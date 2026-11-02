from kernel import *

@g.jit
def split_stage(R, Out, Token, Cache, Cos, Sin, HAF, HAS, HAB, HFN, HFS, HFB, AN, FN, QN, KN, Sink, QA, QAS, QB, QBS, KVW, KVS, OA, OAS, OB, OBS, SW1, SS1, SW3, SS3, SW2, SS2, Gate, Tid, EW1, ES1, EW3, ES3, EW2, ES2, MixA, MixF, R2, X, XQ, QL, QLQ, Q, KV, QRot, KVRot, AO, AR, AQ, OL, OLQ, Attn, FF, FFQ, FFMX, SG, SU, SA, SAQ, SO, Logits, Ids, Gamma, EG, EU, EA, EAQ, EO, Sum, Counter, Health, GRID: gl.constexpr=64, STAGE: gl.constexpr=0):
    if STAGE == 0:
        mhc_mix(R, HAF, HAS, HAB, MixA, GRID)
    if STAGE == 1:
        sinkhorn(MixA)
    if STAGE == 2:
        mhc_pre(R, MixA, X, GRID)
    if STAGE == 3:
        norm(X, AN, X, 4096, 1, True, GRID)
    if STAGE == 4:
        quant(X, XQ, 4096, 128, GRID)
    if STAGE == 5:
        matvec(XQ, QA, QAS, QL, 1024, 4096, 8, GRID)
        matvec(XQ, KVW, KVS, KV, 512, 4096, 8, GRID)
    if STAGE == 6:
        norm(QL, QN, QL, 1024, 1, True, GRID, False)
        norm(KV, KN, KV, 512, 1, True, GRID, False)
    if STAGE == 7:
        quant(QL, QLQ, 1024, 128, GRID)
    if STAGE == 8:
        matvec(QLQ, QB, QBS, Q, 32768, 1024, 8, GRID)
    if STAGE == 9:
        norm(Q, QN, Q, 512, 64, False, GRID, False)
    if STAGE == 10:
        rope(Q, Cos, Sin, QRot, 64, False, GRID)
        rope(KV, Cos, Sin, KVRot, 1, False, GRID)
    if STAGE == 11:
        if gl.program_id(0) == 0:
            i = gl.arange(0, 512, layout=gl.BlockedLayout([1], [32], [4], [0]))
            gl.store(Cache + 127 * 512 + i, gl.load(KVRot + i, cache_modifier='.cg'), cache_modifier='.cg')
    if STAGE == 12:
        attention(QRot, KVRot, Cache, Sink, AO, GRID)
    if STAGE == 13:
        rope(AO, Cos, Sin, AR, 64, True, GRID)
    if STAGE == 14:
        quant(AR, AQ, 32768, 128, GRID)
    if STAGE == 15:
        matvec(AQ, OA, OAS, OL, 8192, 4096, 8, GRID, True)
    if STAGE == 16:
        quant(OL, OLQ, 8192, 128, GRID)
    if STAGE == 17:
        matvec(OLQ, OB, OBS, Attn, 4096, 8192, 8, GRID)
    if STAGE == 18:
        mhc_post(Attn, R, MixA, R2, GRID)
    if STAGE == 19:
        mhc_mix(R2, HFN, HFS, HFB, MixF, GRID)
    if STAGE == 20:
        sinkhorn(MixF)
    if STAGE == 21:
        mhc_pre(R2, MixF, FF, GRID)
    if STAGE == 22:
        norm(FF, FN, FF, 4096, 1, True, GRID)
    if STAGE == 23:
        quant(FF, FFQ, 4096, 128, GRID)
        quant(FF, FFMX, 4096, 32, GRID)
        matvec(FF, Gate, Gate, Logits, 256, 4096, 16, GRID)
    if STAGE == 24:
        route(Logits, Tid, Token, Ids, Gamma)
        matvec(FFQ, SW1, SS1, SG, 2048, 4096, 8, GRID)
        matvec(FFQ, SW3, SS3, SU, 2048, 4096, 8, GRID)
    if STAGE == 25:
        activation(SG, SU, SA, 2048, GRID)
    if STAGE == 26:
        quant(SA, SAQ, 2048, 128, GRID)
    if STAGE == 27:
        matvec(SAQ, SW2, SS2, SO, 4096, 2048, 8, GRID)
        for slot in range(6):
            expert = gl.load(Ids + slot, cache_modifier='.cg').to(gl.int64)
            matvec(FFMX, EW1 + expert * 2048 * 2048, ES1 + expert * 2048 * 128, EG + slot * 2048, 2048, 4096, 4, GRID, FLOAT_OUT=True)
            matvec(FFMX, EW3 + expert * 2048 * 2048, ES3 + expert * 2048 * 128, EU + slot * 2048, 2048, 4096, 4, GRID, FLOAT_OUT=True)
    if STAGE == 28:
        activation(EG, EU, EA, 6 * 2048, GRID, False)
    if STAGE == 29:
        quant(EA, EAQ, 6 * 2048, 32, GRID)
    if STAGE == 30:
        for slot in range(6):
            expert = gl.load(Ids + slot, cache_modifier='.cg').to(gl.int64)
            matvec(EAQ + slot * 2048, EW2 + expert * 4096 * 1024, ES2 + expert * 4096 * 64, EO + slot * 4096, 4096, 2048, 4, GRID, Gamma=Gamma + slot)
    if STAGE == 31:
        i = gl.arange(0, 256, layout=gl.BlockedLayout([1], [32], [4], [0]))
        for tile in range(gl.program_id(0), 16, GRID):
            c = tile * 256 + i
            value = gl.full((256,), 0.0, gl.float32, gl.BlockedLayout([1], [32], [4], [0]))
            for reduce_slot in gl.static_range(6):
                value += gl.load(EO + reduce_slot * 4096 + c, cache_modifier='.cg').to(gl.float32)
            value = value.to(gl.bfloat16).to(gl.float32) + gl.load(SO + c, cache_modifier='.cg').to(gl.float32)
            gl.store(Sum + c, value.to(gl.bfloat16), cache_modifier='.cg')
    if STAGE == 32:
        mhc_post(Sum, R2, MixF, Out, GRID)

NUM_STAGES=33
