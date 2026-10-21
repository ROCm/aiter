import hashlib
import json
import sqlite3
import statistics
from pathlib import Path

w = Path("/home/jun_chen2_qle/yzhou_aiter/fp8-qk-opt-20261021")
result = {}
for tag in ["baseline", "qkpipe"]:
    result[tag] = {}
    for i in range(2):
        p = w / f"profile_{tag}_{i}/counters_results.db"
        db = sqlite3.connect(p)
        tables = [
            x[0]
            for x in db.execute("select name from sqlite_master where type='table'")
        ]
        pmc = next(x for x in tables if x.startswith("rocpd_info_pmc_"))
        event = next(x for x in tables if x.startswith("rocpd_pmc_event_"))
        dispatch = next(x for x in tables if x.startswith("rocpd_kernel_dispatch_"))
        rows = db.execute(
            f'SELECT p.name,e.event_id,SUM(e.value),COUNT(*) FROM "{event}" e JOIN "{pmc}" p ON p.id=e.pmc_id GROUP BY p.name,e.event_id'
        ).fetchall()
        grouped = {}
        for name, e, value, count in rows:
            grouped.setdefault(name, []).append(
                {"event_id": e, "value": value, "instances": count}
            )
        durations = [
            x[0] for x in db.execute(f'SELECT (end-start)/1000.0 FROM "{dispatch}"')
        ]
        for name, items in grouped.items():
            result[tag][name] = {
                "median": statistics.median(x["value"] for x in items),
                "samples": items,
            }
        result[tag][f"pass{i}"] = {
            "db": str(p),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "durations_us": durations,
        }
(w / "profile_summary.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
