import glob
import os
import re
import shlex
import sqlite3
import subprocess
import sys

out_dir = sys.argv[1]
env_path = sys.argv[2]
summary_path = sys.argv[3]


def q(name):
    return '"' + name.replace('"', '""') + '"'


def table_columns(cur, table):
    cur.execute(f"PRAGMA table_info({q(table)})")
    return [row[1] for row in cur.fetchall()]


def first_existing(columns, names):
    for name in names:
        if name in columns:
            return name
    return None


def strip_flydsl_kernel_suffix(name):
    if name.endswith(".kd"):
        return name[:-3]
    return name


def demangle_kernel_name(name):
    stripped = strip_flydsl_kernel_suffix(name)
    if not stripped.startswith("_Z"):
        return stripped
    try:
        completed = subprocess.run(
            ["c++filt", stripped],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return stripped
    demangled = completed.stdout.strip()
    return demangled if completed.returncode == 0 and demangled else stripped


db_files = sorted(glob.glob(os.path.join(out_dir, "**", "*results.db"), recursive=True))
if not db_files:
    db_files = sorted(glob.glob(os.path.join(out_dir, "**", "*.db"), recursive=True))

rows = []
normalized_resources_by_name = {}
db_path = db_files[0] if db_files else ""
db_error = ""

if db_path:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]
        dispatch_table = next((t for t in tables if "kernel_dispatch" in t.lower()), None)
        symbol_table = next((t for t in tables if "kernel_symbol" in t.lower()), None)
        if not dispatch_table or not symbol_table:
            raise RuntimeError(f"missing dispatch/symbol tables; available={tables}")

        dcols = table_columns(cur, dispatch_table)
        scols = table_columns(cur, symbol_table)
        start_col = first_existing(dcols, ["start", "start_ns", "start_timestamp"])
        end_col = first_existing(dcols, ["end", "end_ns", "end_timestamp"])
        kernel_id_col = first_existing(dcols, ["kernel_id"])
        symbol_id_col = first_existing(scols, ["id", "kernel_id"])
        kernel_name_col = first_existing(scols, ["kernel_name", "name"])
        if not all([start_col, end_col, kernel_id_col, symbol_id_col, kernel_name_col]):
            raise RuntimeError(
                "unexpected kernel trace schema: "
                f"dispatch={dcols}, symbol={scols}"
            )

        resource_cols = [
            "arch_vgpr_count",
            "accum_vgpr_count",
            "sgpr_count",
            "group_segment_size",
        ]
        resource_exprs = []
        for col in resource_cols:
            if col in scols:
                resource_exprs.append(f"MAX(s.{q(col)}) AS {q(col)}")
            else:
                resource_exprs.append(f"NULL AS {q(col)}")

        query = f"""
            SELECT
                s.{q(kernel_name_col)} AS kernel_name,
                COUNT(*) AS dispatches,
                AVG(d.{q(end_col)} - d.{q(start_col)}) AS avg_ns,
                MIN(d.{q(end_col)} - d.{q(start_col)}) AS min_ns,
                MAX(d.{q(end_col)} - d.{q(start_col)}) AS max_ns,
                {", ".join(resource_exprs)}
            FROM {q(dispatch_table)} d
            JOIN {q(symbol_table)} s
                ON d.{q(kernel_id_col)} = s.{q(symbol_id_col)}
            GROUP BY s.{q(kernel_name_col)}
            ORDER BY avg_ns DESC
        """
        cur.execute(query)
        rows = cur.fetchall()
        for row in rows:
            normalized_resources_by_name[demangle_kernel_name(row[0])] = row[5:]
        conn.close()
    except Exception as exc:
        db_error = str(exc)

rows_by_name = {}
raw_names_by_name = {}
for row in rows:
    normalized = demangle_kernel_name(row[0])
    rows_by_name[normalized] = row
    raw_names_by_name[normalized] = row[0]

# Discover the two grouped-GEMM kernels by shape instead of hardcoding one tile
# config: the names carry tile_m/n/k and num_buffers, so any AITER_TDM_* override
# (or a future tuned CSV row) is picked up without editing this script.
# gemm1 is the silu/swiglu stage, gemm2 the noact down-projection.
GEMM_RE = re.compile(r"^gemm_a8w4_tdm_t(\d+)x(\d+)x(\d+)_w\d+x\d+_b(\d+)_e\d+_")

candidates = [n for n in rows_by_name if GEMM_RE.match(n)]
acted = [n for n in candidates if "_silu_" in n or "_swiglu_" in n]
plain = [n for n in candidates if n not in acted]

expected = {}
if acted:
    # Normal case: stage1 fuses the activation, stage2 is noact.
    expected["GEMM1"] = max(acted, key=lambda n: rows_by_name[n][1])
    rest = [n for n in candidates if n != expected["GEMM1"]]
    if rest:
        expected["GEMM2"] = max(rest, key=lambda n: rows_by_name[n][1])
elif len(plain) >= 2:
    # Both stages noact: stage1 is the one whose K matches model_dim (the larger
    # tile_k side); fall back to dispatch order via total time.
    ordered = sorted(plain, key=lambda n: -rows_by_name[n][2])
    expected["GEMM1"], expected["GEMM2"] = ordered[0], ordered[1]

missing = [label for label in ("GEMM1", "GEMM2") if label not in expected]

with open(summary_path, "w", encoding="utf-8") as out:
    out.write(f"db_path={db_path or 'not_found'}\n")
    if db_error:
        out.write(f"db_error={db_error}\n")
    out.write("\ntop kernel trace rows:\n")
    out.write(
        "kernel_name,dispatches,avg_us,min_us,max_us,"
        "arch_vgpr,accum_vgpr,sgpr,lds\n"
    )
    for row in rows[:30]:
        name, dispatches, avg_ns, min_ns, max_ns, *resources = row
        out.write(
            f"{name},{dispatches},{avg_ns / 1000.0:.3f},"
            f"{min_ns / 1000.0:.3f},{max_ns / 1000.0:.3f},"
            + ",".join("" if value is None else str(value) for value in resources)
            + "\n"
        )
    out.write("\nselected grouped GEMM kernels:\n")
    for label, name in expected.items():
        out.write(f"{label.lower()}={name if name in rows_by_name else 'not_found'}\n")
        raw_name = raw_names_by_name.get(name)
        if raw_name and raw_name != name:
            out.write(f"{label.lower()}_raw={raw_name}\n")
        if name in normalized_resources_by_name:
            resource_labels = ["arch_vgpr", "accum_vgpr", "sgpr", "lds"]
            out.write(f"{label.lower()}_resources:\n")
            for resource_label, value in zip(
                resource_labels, normalized_resources_by_name[name]
            ):
                out.write(f"  {resource_label}={value}\n")

if missing:
    sys.exit(
        "Unable to find expected grouped GEMM kernel(s): " + ", ".join(missing)
    )

with open(env_path, "w", encoding="utf-8") as env:
    for label, name in expected.items():
        env.write(f"{label}_KERNEL_NAME={shlex.quote(name)}\n")
        env.write(f"{label}_KERNEL_REGEX={shlex.quote(re.escape(name))}\n")
