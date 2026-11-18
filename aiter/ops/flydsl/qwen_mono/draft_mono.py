"""Host side of the drafter's fused post-attention kernel (post.py): weights from the checkpoint files, scratch
buffers, the pointer table, launch. One instance a rank (the current CUDA device)."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import post as P, stage as S, head8 as H8
from safetensors import safe_open


def _q4(w):
    """[E][N][K] bf16 -> MXFP4 1x32 (aiter's reference quantizer): fp4 bytes [E][N][K/2], e8m0 [E][N][K/32]."""
    from aiter.ops.quant import per_1x32_f4_quant
    q, s = per_1x32_f4_quant(w.float().reshape(-1, w.shape[-1]))
    return q.view(torch.uint8).reshape(*w.shape[:-1], -1), s.view(torch.uint8).reshape(*w.shape[:-1], -1)


def _pk4(q, s):
    E, N, K2 = q.shape
    st = K2 // 64
    return (q.view(E, N // 16, 16, st, 2, 2, 16).permute(0, 1, 3, 5, 2, 4, 6).contiguous(),
            torch.cat([s.view(E, N // 16, 16, st, 4).permute(0, 1, 3, 2, 4).reshape(-1), torch.zeros(1024, dtype=torch.uint8, device=q.device)]))


class DraftMono:
    def __init__(self, mtp_ckpt, head_ckpt, device="cuda", stamps=False):
        dev = self.dev = device
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev)
        with safe_open(os.path.join(mtp_ckpt, "mtp.safetensors"), "pt") as f:
            g = lambda k: f.get_tensor("mtp." + k).to(dev)
            L = "layers.0."
            Wo, Wg = g(L + "self_attn.o_proj.weight"), g(L + "mlp.gate.weight")
            sgp, sup, sdn = (g(L + "mlp.shared_expert.%s_proj.weight" % n) for n in ("gate", "up", "down"))
            seg = g(L + "mlp.shared_expert_gate.weight")
            w1, wf = g(L + "post_attention_layernorm.weight").float(), g("norm.weight").float()
            E = [torch.stack([g(L + "mlp.experts.%d.%s_proj.weight" % (e, n)) for e in range(256)]) for n in ("gate", "up", "down")]
        with safe_open(os.path.join(head_ckpt, "outside.safetensors"), "pt") as f:
            w8, ws = f.get_tensor("lm_head.weight").to(dev), f.get_tensor("lm_head.weight_scale").to(dev)
        self.ref = dict(Wo=Wo, Wg=Wg, sgp=sgp, sup=sup, sdn=sdn, seg=seg, w1=w1, wf=wf, E=E, w8=w8, ws=ws) if stamps else None
        qg, sg = _q4(E[0]); qu, su = _q4(E[1]); qd, sd = _q4(E[2])
        EW, ES = _pk4(torch.cat([qg, qu], 1), torch.cat([sg, su], 1))
        ED, EDS = _pk4(qd, sd)
        bf = torch.bfloat16
        B = self.B = dict(
            Wo=S.pack_w(Wo), Wr=S.pack_w(torch.cat([Wg, sgp, sup, seg, z(15, 2048, dt=bf)])), EW=EW, ES=ES, ED=ED, EDS=EDS, SD=S.pack_w(sdn),
            h1=z(16, 2048, dt=bf), xg1=z(64, 32, 16, dt=bf), xg1p=z(16, 2048, dt=bf), part1=z(16, 256), lg=z(16, 256),
            gsu=z(16, 32, 16, dt=bf), sgate=z(16), sel=z(16, 8, dt=torch.int32), wt=z(16, 8),
            GE=z(128, 4, 32, 64, dt=torch.uint8), GS=z(128, 4, 32, 4, dt=torch.int32), xf=z(16, 2048, dt=bf), part2=z(16, 256),
            g1=1 + w1, gf=1 + wf, out=z(16, dt=torch.int64))
        B["w8"], B["ws8"] = H8.pack_w(w8.view(torch.uint8), ws, P.HEAD["TW"])
        self.tab = torch.tensor([B[k].data_ptr() for k in P.TAB], dtype=torch.int64, device=dev)
        self.cnt = torch.zeros(4096, dtype=torch.int32, device=dev)
        self.st = torch.zeros(256, P.NSTAMP, dtype=torch.int64, device=dev)
        self.ids = torch.zeros(16, dtype=torch.int64, device=dev)
        self.hout = torch.zeros(16, 2048, dtype=bf, device=dev)
        self.arange = torch.arange(16, dtype=torch.int64, device=dev)
        self.launch = P.build()

    def run(self, x0, resid, rows=None):
        """x0 [R][4096] bf16 (gated attention output), resid [*][2048] bf16, rows [R] i64 or None (row i).
        -> (hidden [R][2048] bf16, ids [R] i64): views of this object's buffers, valid until the next run."""
        n = x0.shape[0]
        assert 1 <= n <= 16 and x0.dtype == torch.bfloat16 and resid.dtype == torch.bfloat16 and x0.is_contiguous() and resid.is_contiguous()
        rows = self.arange if rows is None else rows
        assert rows.dtype == torch.int64
        self.cnt.zero_()
        self.launch(self.tab.data_ptr(), self.cnt.data_ptr(), self.st.data_ptr(), x0.data_ptr(), resid.data_ptr(), rows.data_ptr(), n,
                    self.ids.data_ptr(), self.hout.data_ptr(), stream=torch.cuda.current_stream())
        return self.hout[:n], self.ids[:n]


class DraftPre:
    """Host side of pre.build_qkv: the drafter pass up to the q/k/v projection, on the caller's weights (bf16):
    emb [V][2048], w_e / w_h / w_in [2048] RMS-norm weights (the norm is x rstd (1 + w)), fc [2048][4096],
    qkv [9216][2048] (any row order: the output follows it)."""

    def __init__(self, emb, w_e, w_h, fc, w_in, qkv, device="cuda"):
        import pre as PR
        self.PR = PR
        dev = device
        bf = torch.bfloat16
        assert emb.dtype == bf and fc.dtype == bf and qkv.dtype == bf and fc.shape == (2048, 4096) and qkv.shape == (9216, 2048), (emb.dtype, fc.shape, qkv.shape)
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev)
        B = self.B = dict(emb=emb, ge=(1 + w_e.float()).contiguous(), gh=(1 + w_h.float()).contiguous(), Wfc=S.pack_w(fc), Wqkv=S.pack_w(qkv),
                          gin=(1 + w_in.float()).contiguous(), xfc=z(128, 32, 16, dt=bf), resid=z(16, 2048, dt=bf), xg=z(64, 32, 16, dt=bf),
                          part=z(16, 256), qkv=z(16, 9216, dt=bf))
        self.tab = torch.tensor([B[k].data_ptr() for k in PR.TABQ], dtype=torch.int64, device=dev)
        self.cnt = torch.zeros(4096, dtype=torch.int32, device=dev)
        self.st = torch.zeros(256, PR.NSTAMP, dtype=torch.int64, device=dev)
        self.launch = PR.build_qkv()

    def run(self, ids, hid):
        """ids [T] i64, hid [T][2048] bf16 -> (resid [T][2048], qkv [T][9216]) bf16 views, valid until the next run."""
        n = ids.shape[0]
        assert 1 <= n <= 16 and ids.dtype in (torch.int64, torch.int32) and hid.dtype == torch.bfloat16 and hid.is_contiguous() and ids.is_contiguous()
        self.cnt.zero_()
        self.launch(self.tab.data_ptr(), self.cnt.data_ptr(), self.st.data_ptr(), ids.data_ptr(), hid.data_ptr(), n, ids.element_size(), stream=torch.cuda.current_stream())
        return self.B["resid"][:n], self.B["qkv"][:n]
