"""Report same-wave cycle deltas; never align clocks from different waves."""

import json
from pathlib import Path
import sys
import numpy as np


def summarize(values):
    x = np.asarray(values, dtype=float)
    assert len(x) and np.all(x >= 0)
    return dict(
        n=len(x),
        mean=float(x.mean()),
        median=float(np.median(x)),
        p90=float(np.percentile(x, 90)),
        p99=float(np.percentile(x, 99)),
        max=float(x.max()),
    )


def analyze(run):
    data = {}

    def add(key, x):
        data.setdefault(key, []).extend(np.asarray(x).reshape(-1).tolist())

    for rank in range(4):
        a = np.load(run / f"cycles_rank{rank}.npz")
        meta = json.loads((run / f"cycles_rank{rank}.json").read_text())
        flags, mcap = meta["flags"], meta["m_tiles"]
        p = a["trace"][: flags * 16].reshape(flags, 16)
        c = a["trace"][flags * 16 :].reshape(mcap, 18, 4, 20)
        gen = c[..., 14].max()
        live = np.flatnonzero((a["info"] >> 16) & 65535)
        assert np.all(p[live, 5] == gen), (rank, "producer generations")
        for tile in live:
            v = p[tile]
            rows = int(v[13])
            owners = a["rowmap"][tile * 64 : tile * 64 + rows, 0] // meta["max_routes"]
            category = (
                "local"
                if np.all(owners == rank)
                else ("remote" if np.all(owners != rank) else "mixed")
            )
            for key, lo, hi in [
                ("m64_copy", 0, 1),
                ("m64_store_wait", 1, 2),
                ("m64_publish", 2, 3),
                ("m16_metadata_issue", 6, 7),
                ("m16_descriptor_issue", 7, 8),
                ("m16_tensor_wait", 8, 9),
                ("m16_scale_issue", 9, 10),
                ("m16_payload_issue", 10, 11),
                ("m16_total", 6, 11),
            ]:
                add("producer_" + category + "_" + key, v[hi] - v[lo])
        valid = c[..., 14] == gen
        active = np.flatnonzero(valid.any((1, 2)))
        for m in active:
            assert np.all(valid[m]), (rank, m, "consumer missing")
        cw = c[active, :, 0, :].reshape(-1, 20)
        pairs = [
            ("task_total", 0, 1),
            ("before_ready", 0, 2),
            ("a_ready", 2, 3),
            ("mainloop_to_drain", 3, 4),
            ("claim_issue", 4, 5),
            ("drain", 5, 6),
            ("claim_resolve", 6, 7),
            ("metadata_prefetch_issue", 7, 8),
            ("epilogue", 8, 9),
            ("prefetched_ready_check", 9, 10),
            ("steal_publish", 10, 11),
            ("closing", 11, 1),
        ]
        for key, lo, hi in pairs:
            delta = cw[:, hi] - cw[:, lo]
            assert np.all(delta >= 0), (rank, key, delta.min())
            add("consumer_w0_" + key, delta)
            if key == "a_ready":
                for label, mask in [
                    ("first", cw[:, 13] == 0),
                    ("later", cw[:, 13] > 0),
                    ("prechecked", cw[:, 15] == 1),
                    ("needs_check", cw[:, 15] == 0),
                ]:
                    if np.any(mask):
                        add("consumer_w0_a_ready_" + label, delta[mask])
        for wave in range(4):
            v = c[active, :, wave, :].reshape(-1, 20)
            add(f"consumer_wave{wave}_task_total", v[:, 1] - v[:, 0])
    result = {key: summarize(v) for key, v in data.items()}
    (run / "fine_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for key, v in result.items():
        print(
            f"{key}: n={v['n']} median={v['median']:.0f} mean={v['mean']:.0f} p90={v['p90']:.0f} p99={v['p99']:.0f}"
        )


if __name__ == "__main__":
    analyze(Path(sys.argv[1]))
