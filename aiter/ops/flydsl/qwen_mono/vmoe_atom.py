"""The verifier's sparse MoE block (router, top-8, MXFP4 routed experts, shared expert, gated add) as one launch of
vmoe2's dataflow kernel, driven from ATOM (ATOM_VMOE=1, atom/models/qwen3_5_vmoe.py).

Input: ATOM's normed x [rows <= 16][2048] bf16 (quantized to MXFP8 1x32 inside the kernel). Output: the block's sum
[rows][2048] bf16 (no residual: ATOM's next norm adds it). Weights are read from a checkpoint directory in the kernel's
own packed layouts, a layer at a time (experts, router and shared gate are the served tensors; the shared expert is the
MXFP8 1x32 one of the -MXFP8-gemm checkpoints)."""
import os
import sys

os.environ.setdefault("ATOMIO", "1")
os.environ.setdefault("STAMPS", "0")
os.environ.setdefault("ENTRYH", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from safetensors import safe_open

import draft_mono as DM
import stage as S
import vmoe2 as V

MAX_ROWS = 16


def _pk8(w8, ws):
    """[N][K] fp8, [N][K/32] codes -> [N/16][K/128][lane][64 B], [N/16][K/128][16][4] (+ 1 KB of slack)."""
    n, k = w8.shape
    return (w8.view(torch.uint8).view(n // 16, 16, k // 128, 4, 2, 16).permute(0, 2, 4, 1, 3, 5).contiguous(),
            torch.cat([ws.view(n // 16, 16, k // 128, 4).permute(0, 2, 1, 3).reshape(-1), torch.zeros(1024, dtype=torch.uint8, device=w8.device)]))


class VMoE:
    def __init__(self, ckpt: str, device):
        assert V.ATOMIO and (not V.STAMPS or os.environ.get("ALLOW_STAMPS")) and not V.ENTRYH and not V.SPEC, "vmoe2 must be built for ATOM (ATOMIO=1 STAMPS=0 ENTRYH=0)"
        self.ckpt, self.dev = ckpt.rstrip("/") + "/", device
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=device)
        self.z = z
        # scratch shared by every layer (the launches run one after another on the stream)
        self.buf = dict(lg=z(16, 256), sgate=z(16), sel=z(16, 8, dt=torch.int32), topv=z(16, 8), sdn=z(16, 2048), slot=z(16, 8, 2048),
                        GE=z(256, 4, 32, 64, dt=torch.uint8), GS=z(256, 4, 32, 4, dt=torch.int32), GSU=z(32768, dt=torch.uint8),
                        GSS=z(4, 32, 4, dt=torch.int32))
        self.dummy = z(16, 2048, dt=torch.bfloat16)                 # the table's unused entries (h1 xg1 xq8 xqs8 part1 h2)
        self.buf["xq8"] = z(16 * 2048 + 1024, dt=torch.uint8)       # A4: the MXFP4 rows of x published by 16 CTAs (XQ layout)
        self.buf["xqs8"] = z(2048, dt=torch.uint8)                  # their scale codes (XQS layout)
        # MB: the routing mailboxes (64-bit words: launch tag << 32 | data) and the launch tag (starts at 1: zeros never match)
        self.buf.update(lgm=z(16, 256, dt=torch.int64), bm=z(16, 8, dt=torch.int64), selm=z(16, 8, dt=torch.int64), topm=z(16, 8, dt=torch.int64),
                        gen=torch.ones(8, dtype=torch.int32, device=device))
        self.cnt = torch.zeros(8192, dtype=torch.int32, device=device)       # two 16 KB banks (BANK=1: the kernel zeroes the next launch's)
        self.st = torch.zeros(2048, V.NSTAMP, dtype=torch.int64, device=device)
        self.out = [z(16, 2048, dt=torch.bfloat16) for _ in range(2)]
        self.layers = {}
        self.ln = V.build()

    def load_layer(self, i: int):
        """Layer i's weights in the kernel's layouts and its pointer table. Not inside a graph capture."""
        if i in self.layers:
            return self.layers[i]
        dev = self.dev
        with safe_open(self.ckpt + "layers-%d.safetensors" % i, "pt") as f:
            g = lambda k: f.get_tensor("model.language_model.layers.%d.%s" % (i, k)).to(dev)
            Wg, seg = g("mlp.gate.weight"), g("mlp.shared_expert_gate.weight")
            if V.SHB:                                       # the served bf16 shared expert
                SH = {n: (g("mlp.shared_expert.%s_proj.weight" % n), None) for n in ("gate", "up", "down")}
                assert SH["gate"][0].dtype == torch.bfloat16, "SHB needs a checkpoint with a bf16 shared expert: " + self.ckpt
            else:
                SH = {n: (g("mlp.shared_expert.%s_proj.weight" % n), g("mlp.shared_expert.%s_proj.weight_scale" % n)) for n in ("gate", "up", "down")}
                assert SH["gate"][0].dtype == torch.float8_e4m3fn, "the shared expert must be MXFP8 1x32 in " + self.ckpt
            EX = {n: (torch.stack([g("mlp.experts.%d.%s_proj.weight" % (e, n)) for e in range(256)]),
                      torch.stack([g("mlp.experts.%d.%s_proj.weight_scale" % (e, n)) for e in range(256)])) for n in ("gate", "up", "down")}
        if V.SHB:
            SW, SD = S.pack_w(torch.cat([SH["gate"][0], SH["up"][0]])), S.pack_w(SH["down"][0])
            SWS = SDS = self.dummy
        else:
            SW, SWS = _pk8(torch.cat([SH["gate"][0], SH["up"][0]]), torch.cat([SH["gate"][1], SH["up"][1]]))
            SD, SDS = _pk8(*SH["down"])
        EW, ES = DM._pk4(torch.cat([EX["gate"][0], EX["up"][0]], 1), torch.cat([EX["gate"][1], EX["up"][1]], 1))
        ED, EDS = DM._pk4(*EX["down"])
        del EX, SH
        B = dict(self.buf, Wg=S.pack_w(Wg), Wsg=S.pack_w(torch.cat([seg, self.z(15, 2048, dt=torch.bfloat16)])), SW=SW, SWS=SWS,
                 EW=EW, ES=ES, ED=ED, EDS=EDS, SD=SD, SDS=SDS)
        tab = torch.tensor([B[k].data_ptr() if k in B else self.dummy.data_ptr() for k in V.TAB], dtype=torch.int64, device=dev)
        self.layers[i] = (tab, B)
        return self.layers[i]

    def run(self, i: int, x: torch.Tensor) -> torch.Tensor:
        """x [rows <= 16][2048] bf16 contiguous -> the MoE block's output rows (the first rows of a new tensor)."""
        rows = x.shape[0]
        tab, _ = self.layers[i]
        # a fresh tensor every call (16 rows: the kernel writes every token lane): ATOM keeps a block's output alive past
        # the next blocks (a persistent buffer overwritten two layers later broke draft acceptance, 11-16)
        out = torch.empty(16, 2048, dtype=torch.bfloat16, device=self.dev)
        if not V.BANK:
            self.cnt.zero_()
        self.ln(tab.data_ptr(), self.cnt.data_ptr(), self.st.data_ptr(), x.data_ptr(), out.data_ptr(), rows, stream=torch.cuda.current_stream())
        return out[:rows]
