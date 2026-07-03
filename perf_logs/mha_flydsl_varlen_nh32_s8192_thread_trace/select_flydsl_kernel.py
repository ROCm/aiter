import glob
import os
import re
import shlex
import sqlite3
import sys

out_dir = sys.argv[1]
flydsl_dump_dir = sys.argv[2]
env_path = sys.argv[3]
summary_path = sys.argv[4]


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


dump_symbols = []
for llvm_ir in sorted(glob.glob(os.path.join(flydsl_dump_dir, "**", "20_llvm_ir.ll"), recursive=True)):
    try:
        with open(llvm_ir, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        continue
    for match in re.finditer(r"define\s+amdgpu_kernel\s+void\s+@([A-Za-z0-9_.$]+)\(", text):
        dump_symbols.append(match.group(1))

db_files = sorted(glob.glob(os.path.join(out_dir, "**", "*results.db"), recursive=True))
if not db_files:
    db_files = sorted(glob.glob(os.path.join(out_dir, "**", "*.db"), recursive=True))

rows = []
resources_by_name = {}
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
            resources_by_name[row[0]] = row[5:]
        conn.close()
    except Exception as exc:
        db_error = str(exc)

selected = None
for symbol in dump_symbols:
    if symbol.startswith("fmha_fwd_kernel"):
        selected = symbol
        break
if not selected and dump_symbols:
    selected = dump_symbols[0]
if not selected:
    preferred = [
        row[0]
        for row in rows
        if re.search(r"(fmha|flash.*attn|mha)", row[0], re.IGNORECASE)
        and not re.search(r"(triton|aten|elementwise|copy|memset)", row[0], re.IGNORECASE)
    ]
    if preferred:
        selected = preferred[0]

with open(summary_path, "w", encoding="utf-8") as out:
    out.write(f"db_path={db_path or 'not_found'}\n")
    if db_error:
        out.write(f"db_error={db_error}\n")
    out.write("dump_symbols:\n")
    for symbol in dump_symbols:
        out.write(f"  {symbol}\n")
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
    out.write(f"\nselected_kernel={selected or 'not_found'}\n")
    if selected and selected in resources_by_name:
        labels = ["arch_vgpr", "accum_vgpr", "sgpr", "lds"]
        out.write("selected_resources:\n")
        for label, value in zip(labels, resources_by_name[selected]):
            out.write(f"  {label}={value}\n")

if not selected:
    sys.exit("Unable to select FlyDSL kernel symbol")

with open(env_path, "w", encoding="utf-8") as env:
    env.write(f"KERNEL_NAME={shlex.quote(selected)}\n")
    env.write(f"KERNEL_REGEX={shlex.quote(re.escape(selected))}\n")
