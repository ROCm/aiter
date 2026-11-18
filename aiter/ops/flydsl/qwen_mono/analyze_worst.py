import sys, torch
from safetensors import safe_open
for p in sys.argv[1:]:
    d = torch.load(p, map_location="cpu"); LY, T = int(d["meta"][0]), int(d["meta"][1])
    x, kr, nr = d["x"][:T], d["kr"][:T], d["nr"][:T]
    rr = ((kr - nr).norm(dim=1) / nr.norm(dim=1))
    i = int(rr.argmax())
    with safe_open("/data/Qwen3.6-35B-A3B-MXFP4/layers-%d.safetensors" % LY, "pt") as f:
        Wg = f.get_tensor("model.language_model.layers.%d.mlp.gate.weight" % LY).float()
    lg = (x @ Wg.T).to(torch.bfloat16).float(); order = torch.sort(-lg, dim=1, stable=True).indices
    ksel = d["sel"][:T].long()
    print("%s: worst err %.3f at layer %d, T %d, row %d; rows > 0.2: %s" % (p.split("/")[-1], float(d["err"]), LY, T, i, [(j, round(float(v), 3)) for j, v in enumerate(rr) if v > 0.2]))
    print("   row %d: kernel top-8 %s" % (i, sorted(ksel[i].tolist())))
    print("           torch  top-8 %s  (logits at ranks 7, 8, 9: %s)" % (sorted(order[i, :8].tolist()), [round(float(lg[i, order[i, r]]), 4) for r in (6, 7, 8)]))
    print("   kernel logits vs torch bf16 logits max abs diff %.2e; sets equal on %d/%d rows" % (float((d["lg"][:T] - lg).abs().max()), sum(set(ksel[r].tolist()) == set(order[r, :8].tolist()) for r in range(T)), T))
