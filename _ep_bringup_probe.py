"""Single-rank EP MegaMoE bring-up: run the one-launch kernel end-to-end and compare
to the pure-torch oracle (ep_reference). Natural (unshuffled) MXFP4 weights are fed to
BOTH the device engine and the oracle, so the device _wfrag natural read and the oracle
dequant agree. relL2 is expected to be large until the GEMM numeric layout is nailed;
the point here is to get a RUNNING kernel + a relL2 number to iterate against.
"""

import os

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29537")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

import aiter  # noqa: E402
from aiter import dtypes  # noqa: E402
from aiter.ops.flydsl.kernels.mega_moe_ep.mega_moe_ep import MegaMoeEPEngine  # noqa: E402

torch.cuda.set_device(0)
if not dist.is_initialized():
    dist.init_process_group("gloo", rank=0, world_size=1)

H, I, E, K = 7168, 3072, 384, 6
WORLD = 1
MTPR = 1024
BS = 8
DEV = torch.device("cuda", 0)
torch.manual_seed(0)

quantize = aiter.get_torch_quant(aiter.QuantType.per_1x32)
w1 = (torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device=DEV) * (H**-0.25))
w1_q, w1_scale = quantize(w1, quant_dtype=dtypes.fp4x2)
w1_q = w1_q.view(E, 2 * I, H // 2).contiguous()
w1_scale = w1_scale.view(E, 2 * I, H // 32).contiguous()
del w1
w2 = (torch.randn(E, H, I, dtype=torch.bfloat16, device=DEV) * (I**-0.25))
w2_q, w2_scale = quantize(w2, quant_dtype=dtypes.fp4x2)
w2_q = w2_q.view(E, H, I // 2).contiguous()
w2_scale = w2_scale.view(E, H, I // 32).contiguous()
del w2
torch.cuda.empty_cache()

QUANT = os.environ.get("EP_QUANT", "a4w4")
TOL = 0.25 if QUANT == "a4w4" else 0.10
engine = MegaMoeEPEngine(
    rank=0,
    world_size=WORLD,
    model_dim=H,
    inter_dim=I,
    experts=E,
    topk=K,
    max_tok_per_rank=MTPR,
    w1=w1_q,
    w1_scale=w1_scale,
    w2=w2_q,
    w2_scale=w2_scale,
    quant=QUANT,
    act="silu",
    swiglu_limit=10.0,
    device=DEV,
)
print(f"engine built; n_cta={engine.n_cta} epr={engine.epr} quant={QUANT}", flush=True)


def run(bs):
    x = torch.randn(bs, H, dtype=torch.bfloat16, device=DEV) * 0.1
    logits = torch.randn(bs, E, device=DEV)
    wts, ids = torch.topk(torch.softmax(logits, dim=-1), K, dim=-1)
    ids = ids.to(torch.int32).contiguous()
    wts = wts.to(torch.float32).contiguous()
    out = engine.forward(x, wts, ids)
    torch.cuda.synchronize()
    ref = engine.reference_forward(x, wts, ids, (w1_q, w1_scale, w2_q, w2_scale))
    o, r = out.float(), ref.float()
    rel = (o - r).norm() / r.norm().clamp_min(1e-9)
    mt = engine.config(bs).mt
    ok = "PASS" if rel.item() < TOL else "FAIL"
    print(
        f"  bs={bs:<5d} mt={mt} err={engine.poll_errors()} "
        f"relL2={rel.item():.4f} (tol {TOL}) {ok}  out|{o.abs().mean():.4f} ref|{r.abs().mean():.4f}",
        flush=True,
    )


for bs in (1, 8, 32, 64, 128, 512):
    run(bs)
dist.destroy_process_group()
