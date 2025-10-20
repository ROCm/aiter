#!/usr/bin/env python


# Imports
# ------------------------------------------------------------------------------

# Python standard library.
import argparse
import ast
import functools
import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

# Third party libraries.
import networkx as nx

# Small utility functions.
# ------------------------------------------------------------------------------


def log_file_list(log_level: int, files: list[Path]) -> None:
    if not logging.getLogger().isEnabledFor(log_level):
        return
    for f in files:
        logging.log(log_level, "* %s", f)


# Structure of Triton source files.
# ------------------------------------------------------------------------------


def check_dir(p: Path) -> Path:
    if not p.exists():
        logging.critical("Required directory [%s] doesn't exist.", p)
        sys.exit(1)
    if not p.is_dir():
        logging.critical("Required directory [%s] isn't a directory.", p)
        sys.exit(1)
    return p


@functools.cache
def root_dir() -> Path:
    return check_dir(Path(__file__).parent.parent.parent)


@functools.cache
def triton_op_dir() -> Path:
    return check_dir(root_dir() / "aiter" / "ops" / "triton")


@functools.cache
def triton_config_dir() -> Path:
    return check_dir(triton_op_dir() / "configs")


@functools.cache
def triton_gemm_config_dir() -> Path:
    return check_dir(triton_config_dir() / "gemm")


def list_files(dir: Path, suffix: str = "") -> set[Path]:
    return {p.relative_to(root_dir()) for p in dir.glob(f"**/*{suffix}") if p.is_file()}


def list_triton_op_files() -> set[Path]:
    files = list_files(triton_op_dir(), suffix=".py")
    logging.debug("Found %d Triton operator source files.", len(files))
    return files


def list_triton_kernel_files(kernel_dir: Path) -> set[Path]:
    files = list_files(kernel_dir, suffix=".py")
    logging.debug("Found %d Triton kernel source files.", len(files))
    return files


def list_triton_config_files() -> set[Path]:
    files = list_files(triton_config_dir(), suffix=".json")
    logging.debug("Found %d Triton kernel config files.", len(files))
    return files


def list_triton_test_files(test_dir: Path) -> set[Path]:
    files = list_files(test_dir, suffix=".py")
    logging.debug("Found %d Triton test source files.", len(files))
    return files


def list_triton_bench_files(bench_dir: Path) -> set[Path]:
    files = list_files(bench_dir, suffix=".py")
    logging.debug("Found %d Triton benchmark source files.", len(files))
    return files


def list_triton_source_files() -> (
    tuple[set[Path], list[Path], list[Path], list[Path], list[Path]]
):
    kernels_dir = check_dir(triton_op_dir() / "_triton_kernels")
    op_test_dir = check_dir(root_dir() / "op_tests")
    test_dir = check_dir(op_test_dir / "triton_tests")
    bench_dir = check_dir(op_test_dir / "op_benchmarks" / "triton")
    op_files = list_triton_op_files()
    kernel_files = list_triton_kernel_files(kernels_dir)
    config_files = list_triton_config_files()
    test_files = list_triton_test_files(test_dir)
    bench_files = list_triton_bench_files(bench_dir)
    all_files = op_files | kernel_files | config_files | test_files | bench_files
    return (
        all_files,
        sorted(kernel_files),
        sorted(config_files),
        sorted(test_files),
        sorted(bench_files),
    )


# Matching of kernel config files.
# ------------------------------------------------------------------------------


DEVICES: frozenset[str] = frozenset(["gfx942", "gfx950"])


def resolve_json_string_interpolation(json_string: str) -> list[str]:
    resolved_strings = [json_string]
    if json_string.startswith("f'") or json_string.startswith('f"'):
        # f-string with variable interpolation.
        # Resolve {AITER_TRITON_CONFIGS_PATH} interpolation:
        if r"{AITER_TRITON_CONFIGS_PATH}" in json_string:
            json_string = json_string.replace(
                r"{AITER_TRITON_CONFIGS_PATH}",
                str(triton_config_dir().relative_to(root_dir()).as_posix()),
            )
            logging.debug(
                r"Resolved {AITER_TRITON_CONFIGS_PATH} in JSON string: [%s]",
                json_string,
            )
        # Resolve {dev} interpolation:
        if r"{dev}" in json_string:
            resolved_strings = [json_string.replace(r"{dev}", dev) for dev in DEVICES]
            logging.debug(r"Resolved {dev} in JSON string: %s", str(resolved_strings))
        # Resolve MOE data type interpolation:
        resolved_strings = [
            replaced
            for resolved_string in resolved_strings
            for replaced in (
                [
                    resolved_string.replace(r"{dtype_str}", moe_dtype)
                    for moe_dtype in [
                        "DEFAULT",
                        "FP8_W8A8",
                        "INT8_W8A16",
                        "INT8_W8A8",
                        "INT4_W4A16",
                        "MX_FP4",
                    ]
                ]
                if r"MOE-{dtype_str}" in resolved_string
                else [resolved_string]
            )
        ]
        # Remove f-string delimiters if there's no more variable interpolation.
        resolved_strings = [
            s[2:-1] if not any(c in s for c in "{}") else s for s in resolved_strings
        ]
    return resolved_strings


def resolve_json_strings(json_strings: list[str]) -> list[Path]:
    json_strings = sorted(
        resolved_json_strings
        for json_string in json_strings
        for resolved_json_strings in resolve_json_string_interpolation(json_string)
    )
    resolved: list[Path] = []
    unresolved: list[str] = []
    for json_string in json_strings:
        p = Path(json_string)
        if p.is_absolute() and p.exists() and p.is_file():
            resolved.append(p.relative_to(root_dir()))
        elif (p := root_dir() / json_string).exists() and p.is_file():
            resolved.append(p.relative_to(root_dir()))
        else:
            unresolved.append(json_string)
    if resolved:
        logging.debug("Resolved JSON config files:")
        log_file_list(logging.DEBUG, resolved)
    log_level = logging.DEBUG
    if unresolved and logging.getLogger().isEnabledFor(log_level):
        logging.log(log_level, "Unresolved JSON strings:")
        for s in unresolved:
            logging.log(log_level, "* %s", s)
    return resolved


def resolve_gemm_config_names(gemm_config_names: list[str]) -> list[Path]:
    gemm_configs = [
        p.relative_to(root_dir())
        for dev in DEVICES
        for gemm_config_name in gemm_config_names
        for p in triton_gemm_config_dir().glob(f"{dev}-{gemm_config_name}*.json")
        if p.is_file()
    ]
    if gemm_configs:
        logging.debug("Resolved JSON GEMM config files:")
        log_file_list(logging.DEBUG, gemm_configs)
    return gemm_configs


def resolve_configs(
    json_strings: list[str], gemm_config_names: list[str]
) -> list[Path]:
    return sorted(
        set(resolve_json_strings(json_strings)).union(
            resolve_gemm_config_names(gemm_config_names)
        )
    )


# Git commands.
# ------------------------------------------------------------------------------


def git(args: str, check: bool = True) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git"] + shlex.split(args),
            capture_output=True,
            text=True,
            check=check,
            cwd=root_dir(),  # always run git commands from repo root
        )
    except FileNotFoundError:
        logging.critical("Git not found.")
        sys.exit(1)
    except subprocess.CalledProcessError:
        logging.critical("Malformed Git command: [git %s].", args)
        sys.exit(1)


def git_current_branch() -> str:
    return git("rev-parse --abbrev-ref HEAD").stdout.rstrip()


def git_check_branch(branch: str) -> None:
    if git(f"rev-parse --verify --quiet {branch}", check=False).returncode != 0:
        logging.critical("Branch [%s] doesn't exist.", branch)
        sys.exit(1)


def git_filename_diff(source_branch: str, target_branch: str) -> set[Path]:
    diff_output = git(
        f"diff --name-only {target_branch} {source_branch}"
    ).stdout.rstrip()
    if not diff_output:
        return set()
    files = set()
    for diff_p in diff_output.splitlines():
        abs_path = root_dir() / diff_p
        if abs_path.exists() and abs_path.is_file():
            # Add path relative to root_dir() - this will match paths from `list_files` function.
            files.add(abs_path.relative_to(root_dir()))
    logging.debug(
        "There %s %d file%s in the diff from [%s] to [%s].",
        "is" if len(files) == 1 else "are",
        len(files),
        "" if len(files) == 1 else "s",
        source_branch,
        target_branch,
    )
    return files


def get_filename_diff(source_branch: str | None, target_branch: str) -> set[Path]:
    if source_branch is None:
        source_branch = git_current_branch()
        logging.info(
            "Source branch wasn't provided, using current branch [%s] as source branch.",
            source_branch,
        )
    else:
        git_check_branch(source_branch)

    git_check_branch(target_branch)

    if target_branch == source_branch:
        logging.error("Source and target branches must be different.")
        sys.exit(1)

    return git_filename_diff(source_branch, target_branch)


# Source file parsing.
# ------------------------------------------------------------------------------


class Visitor(ast.NodeVisitor):
    imports_to_ignore: frozenset[str] = frozenset(
        [
            "einops",
            "iris",
            "jax",
            "jinja2",
            "matplotlib",
            "mori",
            "numpy",
            "packaging",
            "pandas",
            "prettytable",
            "psutil",
            "pybind11",
            "pytest",
            "setuptools",
            "torch",
            "triton",
            "vllm",
            "yaml",
            "zmq",
        ]
    )
    json_strings_to_ignore: frozenset[str] = frozenset(
        [
            "./utils/model_configs.json",
            ".json",
            "empty_kernel.json",
            "f'{kernel_name}.json'",
            "utils/model_configs.json",
        ]
    )

    @classmethod
    def is_import_of_interest(cls, import_: str) -> bool:
        return (
            import_ not in sys.stdlib_module_names
            and import_ not in cls.imports_to_ignore
            and not any(
                import_.startswith(module + ".") for module in cls.imports_to_ignore
            )
            and not any(
                import_.startswith(module + ".") for module in sys.stdlib_module_names
            )
        )

    @classmethod
    def is_json_string_of_interest(cls, json_string: str) -> bool:
        return json_string not in cls.json_strings_to_ignore

    def __init__(self, source_file: Path) -> None:
        # Remove extension from source file, and split directories into module parts.
        self.source_file: Path = source_file
        self.source_module_parts: tuple[str, ...] = self.source_file.with_suffix(
            ""
        ).parts
        self.dependencies: set[Path] = set()
        self.json_strings: set[str] = set()
        self.gemm_config_names: set[str] = set()

    def add_dependency(self, import_: str, suppress_warning: bool = False) -> None:
        if not import_ or not self.__class__.is_import_of_interest(import_):
            return
        import_py_file = import_.replace(".", os.sep) + ".py"
        # Check if it's a module file (e.g., `foo/bar.py`).
        p = root_dir() / import_py_file
        if p.exists() and p.is_file():
            self.dependencies.add(p.relative_to(root_dir()))
            return
        # Check if it's a package directory with `__init__.py`.
        p_dir = (root_dir() / import_py_file).with_suffix("")
        p_init = p_dir / "__init__.py"
        if p_init.exists() and p_init.is_file():
            # Add dependency as the `__init__.py` file of the package.
            self.dependencies.add(p_init.relative_to(root_dir()))
            return
        # Check for relative imports within the same package.
        p = root_dir() / self.source_file.parent / import_py_file
        if p.exists() and p.is_file():
            self.dependencies.add(p.relative_to(root_dir()))
            return
        # Check for relative package imports.
        p_dir = (root_dir() / self.source_file.parent / import_py_file).with_suffix("")
        p_init = p_dir / "__init__.py"
        if p_init.exists() and p_init.is_file():
            self.dependencies.add(p_init.relative_to(root_dir()))
            return
        if not suppress_warning:
            # Check if it's a directory without `__init__.py`` (namespace package or external)
            p_dir = (root_dir() / import_py_file).with_suffix("")
            if p_dir.exists() and p_dir.is_dir():
                logging.debug(
                    "Directory [%s] exists but has no '__init__.py', skipping dependency [%s] of [%s].",
                    p_dir.relative_to(root_dir()),
                    import_,
                    self.source_file,
                )
            else:
                logging.warning(
                    "Unable to find [%s] dependency of [%s] on filesystem.",
                    import_,
                    self.source_file,
                )

    def add_json_string(self, json_string: str) -> None:
        if not json_string or not self.__class__.is_json_string_of_interest(
            json_string
        ):
            return
        self.json_strings.add(json_string)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.add_dependency(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module_name = node.module if node.module else ""
        if node.level > 0:
            # Resolve absolute import from relative import.
            full_module_name = ".".join(self.source_module_parts[: -node.level])
            if module_name:
                full_module_name += "." + module_name
        else:
            full_module_name = module_name
        self.add_dependency(full_module_name)
        # For "from X import Y", also try to add Y as a potential module
        # (in case Y is a submodule rather than a name defined in X).
        if full_module_name and node.names[0].name != "*":
            for alias in node.names:
                potential_submodule = f"{full_module_name}.{alias.name}"
                self.add_dependency(potential_submodule, suppress_warning=True)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        # Plain string literals.
        if isinstance(node.value, str) and node.value.lower().endswith(".json"):
            self.add_json_string(node.value)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        # f-strings.
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                # Unparse the inner expression to a readable form.
                expr_str = ast.unparse(value.value)
                parts.append(f"{{{expr_str}}}")
        joined = "".join(parts)
        if joined.lower().endswith(".json"):
            self.add_json_string(f"f{joined!r}")

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        is_get_gemm_config = (
            isinstance(func, ast.Name)
            and func.id == "get_gemm_config"
            or isinstance(func, ast.Attribute)
            and func.attr == "get_gemm_config"
        )
        if is_get_gemm_config and node.args:
            gemm_config_name = node.args[0]
            if isinstance(gemm_config_name, ast.Constant) and isinstance(
                gemm_config_name.value, str
            ):
                self.gemm_config_names.add(gemm_config_name.value)
        self.generic_visit(node)


def parse_source_file(source_file: Path) -> tuple[list[Path], list[str], list[str]]:
    try:
        logging.debug("Parsing source file [%s]...", str(source_file))
        source = (root_dir() / source_file).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_file))
    except Exception:
        logging.exception("Skipping source file [%s].", source_file)
        return [], [], []

    visitor = Visitor(source_file)
    visitor.visit(tree)

    dependecies = sorted(visitor.dependencies)
    if not dependecies:
        logging.debug("No dependecies of interest in [%s].", source_file)
    else:
        logging.debug("Dependecies of interest in [%s]:", source_file)
        log_file_list(logging.DEBUG, dependecies)

    json_strings = sorted(visitor.json_strings)
    if not json_strings:
        logging.debug("No JSON strings in [%s].", source_file)
    else:
        logging.debug("JSON strings in [%s]: %s", source_file, str(json_strings))

    gemm_config_names = sorted(visitor.gemm_config_names)
    if not gemm_config_names:
        logging.debug("No GEMM config names in [%s].", source_file)
    else:
        logging.debug(
            "GEMM config names in [%s]: %s", source_file, str(gemm_config_names)
        )

    return dependecies, json_strings, gemm_config_names


def parse_source_file_recursively(
    graph: nx.DiGraph,
    source_file: Path,
    visited: set[Path],
) -> None:
    stack = [source_file]

    while stack:
        current = stack.pop()
        if current in visited:
            continue

        dependencies, json_strings, gemm_config_names = parse_source_file(current)
        configs = resolve_configs(json_strings, gemm_config_names)

        # Add current node to the graph.
        current_str = str(current)
        graph.add_node(current_str)
        logging.debug("Added graph node [%s].", current_str)

        # Add dependencies of current node, and respective edges, to the graph.
        for d in dependencies:
            d_str = str(d)
            graph.add_node(d_str)
            logging.debug("Added graph node [%s].", d_str)
            graph.add_edge(d_str, current_str)
            logging.debug("Added graph edge [%s]->[%s].", d_str, current_str)

        # Add configs of current node, and respective edges, to the graph.
        for c in configs:
            c_str = str(c)
            graph.add_node(c_str)
            logging.debug("Added graph node [%s].", c_str)
            graph.add_edge(c_str, current_str)
            logging.debug("Added graph edge [%s]->[%s].", c_str, current_str)

        stack.extend(d for d in dependencies if (root_dir() / d).is_file())
        visited.add(current)


# Dependency graph.
# ------------------------------------------------------------------------------


def tag_node(graph: nx.DiGraph, file: Path, tag: str) -> None:
    file_str = str(file)
    if file_str in graph.nodes:
        graph.nodes[file_str]["type"] = tag
        logging.debug("Tagged file [%s] as a '%s' in the graph.", file_str, tag)
    else:
        logging.warning(
            "Couldn't find file [%s] in the graph, unable to tag it as '%s'.",
            file_str,
            tag,
        )


def add_files_to_dependency_graph(
    graph: nx.DiGraph,
    files: list[Path],
    file_type: str,
    visited: set[Path],
) -> None:
    for f in files:
        parse_source_file_recursively(graph, f, visited)
        tag_node(graph, f, file_type)


def build_dependency_graph(
    kernel_files: list[Path],
    cofig_files: list[Path],
    test_files: list[Path],
    bench_files: list[Path],
) -> nx.DiGraph:
    graph: nx.DiGraph = nx.DiGraph()
    visited: set[Path] = set()
    # Add files that tests depends on.
    add_files_to_dependency_graph(graph, test_files, "test", visited)
    # Add files that benchmarks depends on.
    add_files_to_dependency_graph(graph, bench_files, "bench", visited)
    logging.debug(
        "Built dependency graph of Triton source files with %d nodes and %d edges.",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    # Tag kernel files.
    for kernel_file in kernel_files:
        tag_node(graph, kernel_file, "kernel")
    # Tag config files.
    for cofig_file in cofig_files:
        tag_node(graph, cofig_file, "config")
    return graph


def find_tests_to_run(graph: nx.DiGraph, diff_inter_triton: list[Path]) -> list[Path]:
    tests_to_run: set[Path] = set()

    for p in diff_inter_triton:
        p_str = str(p)
        logging.debug(
            "Searching for tests related to Triton source file [%s]...", p_str
        )
        if p_str not in graph:
            logging.warning(
                "Triton source file [%s] isn't in the dependency graph, it's unreachable.",
                p_str,
            )
            continue

        is_bench = graph.nodes[p_str].get("type") == "bench"
        if not is_bench:
            # Forward traversal for non-benchmarks.
            reachable_files = nx.descendants(graph, p_str) | {p_str}
        else:
            # Backward traversal for benchmarks, filtering only dependencies on kernel files. After
            # that, perform a forward traversal from kernels. This strategy isn't perfect but it's
            # an attempt to take benchmark files into account. It may fail if a benchmark utility is
            # changed in isolation.
            reachable_files = {
                kernel_descendant
                for p_ancestor in nx.ancestors(graph, p_str)
                if graph.nodes[p_ancestor].get("type") == "kernel"
                for kernel_descendant in nx.descendants(graph, p_ancestor)
            }
        logging.debug(
            "There %s %d file%s reachable from [%s].",
            "is" if len(reachable_files) == 1 else "are",
            len(reachable_files),
            "" if len(reachable_files) == 1 else "s",
            p_str,
        )

        test_files = {
            Path(f) for f in reachable_files if graph.nodes[f].get("type") == "test"
        }
        if test_files:
            logging.debug(
                "There %s %d test%s reachable from [%s].",
                "is" if len(test_files) == 1 else "are",
                len(test_files),
                "" if len(test_files) == 1 else "s",
                p_str,
            )
            tests_to_run.update(test_files)
        else:
            logging.warning(
                "Couldn't find test files related to [%s] Triton source.", p_str
            )

    if tests_to_run:
        sorted_tests_to_run = sorted(tests_to_run)
        logging.info(
            "There %s %d test%s reachable from the Triton diff:",
            "is" if len(sorted_tests_to_run) == 1 else "are",
            len(sorted_tests_to_run),
            "" if len(sorted_tests_to_run) == 1 else "s",
        )
        log_file_list(logging.INFO, sorted_tests_to_run)
        return sorted_tests_to_run
    else:
        logging.warning("Couldn't find any test file related to Triton diff.")
        logging.warning(
            "Please check test selection script, there might be a bug in it."
        )
        logging.warning("Please check Triton code base, there may be untested kernels.")
        return []


# Command line interface parsing.
# ------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="select which Triton tests to run based on git diff"
    )
    parser.add_argument(
        "-s", "--source", help="source branch, defaults to current branch"
    )
    parser.add_argument(
        "-t", "--target", default="main", help="target branch, defaults to main"
    )
    parser.add_argument(
        "-l",
        "--log-level",
        type=str.lower,
        choices=["critical", "error", "warning", "info", "debug", "off"],
        default="info",
        help="log level to enable (default: info)",
    )
    args = parser.parse_args()
    args.log_level = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "off": logging.CRITICAL + 1000,
    }[args.log_level]
    return args


# Main logic and script entry point.
# ------------------------------------------------------------------------------


def main_logic(args: argparse.Namespace) -> int:
    diff_files = get_filename_diff(args.source, args.target)
    all_files, kernel_files, config_files, test_files, bench_files = (
        list_triton_source_files()
    )
    diff_inter_triton = diff_files & all_files
    del diff_files, all_files

    if not diff_inter_triton:
        logging.info(
            "There are no Triton source files in diff, there's no need to run Triton tests."
        )
        return 0

    logging.info(
        "There %s %d Triton source file%s in the diff:",
        "is" if len(diff_inter_triton) == 1 else "are",
        len(diff_inter_triton),
        "" if len(diff_inter_triton) == 1 else "s",
    )
    sorted_diff_inter_triton = sorted(diff_inter_triton)
    del diff_inter_triton
    log_file_list(logging.INFO, sorted_diff_inter_triton)

    graph = build_dependency_graph(kernel_files, config_files, test_files, bench_files)
    tests_to_run = find_tests_to_run(graph, sorted_diff_inter_triton)

    if not tests_to_run:
        return 1

    return 0


def main() -> int:
    start_timestamp = time.perf_counter()
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s|%(levelname)s|%(message)s", level=args.log_level
    )
    exit_code = main_logic(args)
    end_timestamp = time.perf_counter()
    elapsed_time_s = end_timestamp - start_timestamp
    logging.info("Finished, execution took %.2f seconds.", elapsed_time_s)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
