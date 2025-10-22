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


def list_files(root_dir: Path, dir: Path, suffix: str = "") -> set[Path]:
    return {p.relative_to(root_dir) for p in dir.glob(f"**/*{suffix}") if p.is_file()}


def list_triton_op_files(root_dir: Path, op_dir: Path) -> set[Path]:
    files = list_files(root_dir, op_dir, suffix=".py")
    logging.debug("Found %d Triton operator source files.", len(files))
    return files


def list_triton_kernel_files(root_dir: Path, kernel_dir: Path) -> set[Path]:
    files = list_files(root_dir, kernel_dir, suffix=".py")
    logging.debug("Found %d Triton kernel source files.", len(files))
    return files


def list_triton_config_files(root_dir: Path, config_dir: Path) -> set[Path]:
    files = list_files(root_dir, config_dir, suffix=".json")
    logging.debug("Found %d Triton kernel config files.", len(files))
    return files


def list_triton_test_files(root_dir: Path, test_dir: Path) -> set[Path]:
    files = list_files(root_dir, test_dir, suffix=".py")
    logging.debug("Found %d Triton test source files.", len(files))
    return files


def list_triton_bench_files(root_dir: Path, bench_dir: Path) -> set[Path]:
    # TODO: How to deal with these files?
    #       `op_tests/op_benchmarks/triton/utils/model_configs.json`
    #       `op_tests/op_benchmarks/triton/bench_schema.yaml`
    files = list_files(root_dir, bench_dir, suffix=".py")
    logging.debug("Found %d Triton benchmark source files.", len(files))
    return files


def list_triton_source_files(
    root_dir: Path,
) -> tuple[set[Path], list[Path], list[Path], list[Path], list[Path]]:
    op_dir = check_dir(root_dir / "aiter" / "ops" / "triton")
    kernels_dir = check_dir(op_dir / "_triton_kernels")
    config_dir = check_dir(op_dir / "configs")
    test_dir = check_dir(root_dir / "op_tests" / "triton_tests")
    bench_dir = check_dir(root_dir / "op_tests" / "op_benchmarks" / "triton")
    op_files = list_triton_op_files(root_dir, op_dir)
    kernel_files = list_triton_kernel_files(root_dir, kernels_dir)
    config_files = list_triton_config_files(root_dir, config_dir)
    test_files = list_triton_test_files(root_dir, test_dir)
    bench_files = list_triton_bench_files(root_dir, bench_dir)
    all_files = op_files | kernel_files | config_files | test_files | bench_files
    return (
        all_files,
        sorted(kernel_files),
        sorted(config_files),
        sorted(test_files),
        sorted(bench_files),
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
    files = {
        p
        for diff_p in git(f"diff --name-only {target_branch} {source_branch}")
        .stdout.rstrip()
        .splitlines()
        if (p := Path(diff_p)).exists() and p.is_file()
    }
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
            "os.path",
            "numpy",
            "pandas",
            "matplotlib.pyplot",
            "torch",
            "einops",
            "triton",
            "pytest",
            "prettytable",
        ]
    )
    import_prefixes_to_ignore: tuple[str, ...] = (
        "torch.",
        "triton.",
        "jax.",
    )
    json_strings_to_ignore: frozenset[str] = frozenset(
        [".json", "empty_kernel.json", "f'{kernel_name}.json'"]
    )
    json_string_suffixes_to_ignore: tuple[str, ...] = ("utils/model_configs.json",)

    @classmethod
    def is_import_of_interest(cls, import_: str) -> bool:
        return (
            import_ not in sys.stdlib_module_names
            and import_ not in cls.imports_to_ignore
            and not any(
                import_.startswith(prefix) for prefix in cls.import_prefixes_to_ignore
            )
        )

    @classmethod
    def is_json_string_of_interest(cls, json_string: str) -> bool:
        return json_string not in cls.json_strings_to_ignore and not any(
            json_string.endswith(suffix)
            for suffix in cls.json_string_suffixes_to_ignore
        )

    def __init__(self, root_dir: Path, source_file: Path) -> None:
        self.root_dir: Path = root_dir
        # Remove extension from source file, and split directories into module parts.
        self.source_file: Path = source_file
        self.source_module_parts: tuple[str, ...] = self.source_file.with_suffix(
            ""
        ).parts
        self.dependencies: set[Path] = set()
        self.json_strings: set[str] = set()

    def add_dependency(self, import_: str) -> None:
        if not import_ or not self.__class__.is_import_of_interest(import_):
            return
        import_py_file = import_.replace(".", os.sep) + ".py"
        p = self.root_dir / import_py_file
        if p.exists() and p.is_file():
            # Add dependency as Python module / source file, imported with project scope.
            self.dependencies.add(p.relative_to(self.root_dir))
            return
        p = (self.root_dir / import_py_file).with_suffix("")
        if p.exists() and p.is_dir():
            # Add dependency as Python package / directory.
            self.dependencies.add(p.relative_to(self.root_dir))
            return
        p = self.root_dir / self.source_file.parent / import_py_file
        if p.exists() and p.is_file():
            # Add dependency as Python module / source file, imported with local package scope.
            self.dependencies.add(p.relative_to(self.root_dir))
            return
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


def parse_source_file(
    root_dir: Path, source_file: Path
) -> tuple[list[Path], list[str]]:
    try:
        source = (root_dir / source_file).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_file))
    except Exception:
        logging.exception("Skipping source file [%s].", source_file)
        return [], []

    visitor = Visitor(root_dir, source_file)
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

    return dependecies, json_strings


# TODO: How to deal with package imports?
# TODO: How to deal with `__init__.py` files?
def parse_source_file_recursively(
    graph: nx.DiGraph,
    root_dir: Path,
    source_file: Path,
    visited: set[Path],
) -> None:
    stack = [source_file]

    while stack:
        current = stack.pop()
        if current in visited:
            continue

        dependencies, json_strings = parse_source_file(root_dir, current)

        current_str = str(current)
        graph.add_node(current_str)
        logging.debug("Added graph node [%s].", current_str)
        for d in dependencies:
            d_str = str(d)
            graph.add_node(d_str)
            logging.debug("Added graph node [%s].", d_str)
            graph.add_edge(d_str, current_str)
            logging.debug("Added graph edge [%s]->[%s].", d_str, current_str)

        stack.extend(d for d in dependencies if d.is_file())
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
    root_dir: Path,
    files: list[Path],
    file_type: str,
    visited: set[Path],
) -> None:
    for f in files:
        parse_source_file_recursively(graph, root_dir, f, visited)
        tag_node(graph, f, file_type)


def build_dependency_graph(
    root_dir: Path,
    kernel_files: list[Path],
    test_files: list[Path],
    bench_files: list[Path],
) -> nx.DiGraph:
    graph: nx.DiGraph = nx.DiGraph()
    visited: set[Path] = set()
    # Add files that tests depends on.
    add_files_to_dependency_graph(graph, root_dir, test_files, "test", visited)
    # Add files that benchmarks depends on.
    add_files_to_dependency_graph(graph, root_dir, bench_files, "bench", visited)
    logging.debug(
        "Built dependency graph of Triton source files with %d nodes and %d edges.",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    # Tag kernel files.
    for kernel_file in kernel_files:
        tag_node(graph, kernel_file, "kernel")
    return graph


def find_tests_to_run(graph: nx.DiGraph, diff_inter_triton: list[Path]) -> list[Path]:
    tests_to_run: set[Path] = set()

    for p in diff_inter_triton:
        p_str = str(p)
        logging.debug(
            "Searching for tests related to Triton source file [%s]...", p_str
        )

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
        return []


# Command line interface parsing.
# ------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="select which Triton tests to run based on git diff"
    )
    parser.add_argument("-s", "--source", required=True, help="source branch")
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


# Script entry point.
# ------------------------------------------------------------------------------


def main() -> None:
    start_timestamp = time.perf_counter()

    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s|%(levelname)s|%(message)s", level=args.log_level
    )

    diff_files = get_filename_diff(args.source, args.target)
    all_files, kernel_files, config_files, test_files, bench_files = (
        list_triton_source_files(root_dir())
    )
    diff_inter_triton = diff_files & all_files
    del diff_files, all_files

    if not diff_inter_triton:
        logging.info(
            "There are no Triton source files in diff, there's no need to run Triton tests."
        )
        sys.exit(0)

    logging.info(
        "There %s %d Triton source file%s in the diff:",
        "is" if len(diff_inter_triton) == 1 else "are",
        len(diff_inter_triton),
        "" if len(diff_inter_triton) == 1 else "s",
    )
    sorted_diff_inter_triton = sorted(diff_inter_triton)
    del diff_inter_triton
    log_file_list(logging.INFO, sorted_diff_inter_triton)

    graph = build_dependency_graph(root_dir(), kernel_files, test_files, bench_files)
    _ = find_tests_to_run(graph, sorted_diff_inter_triton)

    end_timestamp = time.perf_counter()
    elapsed_time_s = end_timestamp - start_timestamp
    logging.info("Finished, execution took %.2f seconds.", elapsed_time_s)


if __name__ == "__main__":
    main()
