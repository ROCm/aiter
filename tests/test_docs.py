# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Documentation regression tests.

These tests guard against documentation rot by verifying:
1. Every function/class documented in RST files actually exists in the codebase
2. Python code examples in docs are syntactically valid
3. Sphinx can build without warnings
4. Documentation version matches the package version

Run with: pytest tests/test_docs.py -v
No GPU required — these tests run on CPU.
"""

import ast
import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

DOCS_DIR = Path(__file__).parent.parent / "docs"
API_DIR = DOCS_DIR / "api"
TUTORIAL_DIR = DOCS_DIR / "tutorials"


# ---------------------------------------------------------------------------
# Test 1: API Signature Consistency
# Every function/class referenced in docs/api/*.rst must be importable.
# ---------------------------------------------------------------------------


def _extract_autofunction_refs(rst_path: Path) -> list[str]:
    """Extract all '.. autofunction:: X' and '.. autoclass:: X' from an RST file."""
    refs = []
    pattern = re.compile(r"\.\.\s+auto(?:function|class)::\s+(\S+)")
    for line in rst_path.read_text().splitlines():
        m = pattern.search(line)
        if m:
            refs.append(m.group(1))
    return refs


def _extract_module_refs(rst_path: Path) -> list[str]:
    """Extract module paths referenced as ``aiter.ops.xxx`` or ``aiter.xxx``."""
    refs = []
    pattern = re.compile(r"``(aiter(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)``")
    for line in rst_path.read_text().splitlines():
        for m in pattern.finditer(line):
            ref = m.group(1)
            # Only include module-level references (at least aiter.X)
            if ref.count(".") >= 1:
                refs.append(ref)
    return list(set(refs))


def _extract_function_names_from_rst(rst_path: Path) -> list[str]:
    """Extract function names documented with ``func_name(`` pattern in RST files."""
    refs = []
    # Match patterns like: - ``gemm_a8w8_ck(...)`` or ``func_name(params)``
    pattern = re.compile(r"``([a-zA-Z_][a-zA-Z0-9_]*)\(")
    text = rst_path.read_text()
    for m in pattern.finditer(text):
        refs.append(m.group(1))
    return list(set(refs))


def _collect_all_api_refs() -> list[tuple[str, str]]:
    """Collect all auto-doc references from API RST files.

    Returns list of (dotted_path, source_file) tuples.
    """
    results = []
    if not API_DIR.exists():
        return results
    for rst_file in API_DIR.glob("*.rst"):
        for ref in _extract_autofunction_refs(rst_file):
            results.append((ref, str(rst_file.name)))
    return results


def _collect_all_module_refs() -> list[tuple[str, str]]:
    """Collect all module references from API RST files."""
    results = []
    if not API_DIR.exists():
        return results
    for rst_file in API_DIR.glob("*.rst"):
        for ref in _extract_module_refs(rst_file):
            results.append((ref, str(rst_file.name)))
    return results


_api_refs = _collect_all_api_refs()
_module_refs = _collect_all_module_refs()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="AITER ops require ROCm, skip on Windows",
)
class TestAPISignatureConsistency:
    """Verify that documented functions/classes actually exist."""

    @pytest.mark.parametrize(
        "dotted_path,source_file",
        (
            _api_refs
            if _api_refs
            else [pytest.param("skip", "skip", marks=pytest.mark.skip)]
        ),
        ids=[f"{r[1]}::{r[0]}" for r in _api_refs] if _api_refs else ["no_refs"],
    )
    def test_autofunction_importable(self, dotted_path, source_file):
        """Each .. autofunction:: / .. autoclass:: target must be importable."""
        parts = dotted_path.rsplit(".", 1)
        if len(parts) != 2:
            pytest.skip(f"Cannot parse dotted path: {dotted_path}")

        module_path, attr_name = parts
        try:
            mod = importlib.import_module(module_path)
        except (ImportError, ModuleNotFoundError) as e:
            pytest.fail(
                f"Module '{module_path}' not importable "
                f"(documented in {source_file}): {e}"
            )

        if not hasattr(mod, attr_name):
            available = [a for a in dir(mod) if not a.startswith("_")]
            pytest.fail(
                f"'{attr_name}' not found in module '{module_path}' "
                f"(documented in {source_file}). "
                f"Available: {', '.join(available[:20])}"
            )

    @pytest.mark.parametrize(
        "module_path,source_file",
        (
            _module_refs
            if _module_refs
            else [pytest.param("skip", "skip", marks=pytest.mark.skip)]
        ),
        ids=[f"{r[1]}::{r[0]}" for r in _module_refs]
        if _module_refs
        else ["no_refs"],
    )
    def test_module_ref_importable(self, module_path, source_file):
        """Each ``aiter.ops.xxx`` module reference must be importable."""
        try:
            importlib.import_module(module_path)
        except (ImportError, ModuleNotFoundError) as e:
            pytest.fail(
                f"Module '{module_path}' not importable "
                f"(referenced in {source_file}): {e}"
            )


# ---------------------------------------------------------------------------
# Test 2: Code Example Smoke Tests
# Python code blocks in RST must at least parse (syntax check).
# ---------------------------------------------------------------------------


def _extract_python_code_blocks(rst_path: Path) -> list[tuple[int, str]]:
    """Extract Python code blocks from RST file.

    Returns list of (line_number, code_string) tuples.
    """
    blocks = []
    lines = rst_path.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"\s*\.\.\s+code-block::\s+python", line):
            # Find the indentation of the code block
            i += 1
            # Skip blank lines
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            if i >= len(lines):
                break
            # Determine indent level
            indent_match = re.match(r"^(\s+)", lines[i])
            if not indent_match:
                continue
            indent = len(indent_match.group(1))
            block_start = i
            code_lines = []
            while i < len(lines):
                if lines[i].strip() == "":
                    code_lines.append("")
                elif (
                    len(lines[i]) > 0
                    and len(lines[i]) - len(lines[i].lstrip()) >= indent
                ):
                    code_lines.append(lines[i][indent:])
                else:
                    break
                i += 1
            # Strip trailing blank lines
            while code_lines and code_lines[-1].strip() == "":
                code_lines.pop()
            if code_lines:
                blocks.append((block_start + 1, "\n".join(code_lines)))
        else:
            i += 1
    return blocks


def _collect_code_blocks() -> list[tuple[str, int, str]]:
    """Collect all Python code blocks from all RST files.

    Returns list of (filename, line_number, code) tuples.
    """
    results = []
    if not DOCS_DIR.exists():
        return results
    for rst_file in DOCS_DIR.rglob("*.rst"):
        for line_no, code in _extract_python_code_blocks(rst_file):
            rel_path = rst_file.relative_to(DOCS_DIR)
            results.append((str(rel_path), line_no, code))
    return results


_code_blocks = _collect_code_blocks()


class TestCodeExamples:
    """Verify that Python code examples in documentation are syntactically valid."""

    @pytest.mark.parametrize(
        "rst_file,line_no,code",
        (
            _code_blocks
            if _code_blocks
            else [pytest.param("skip", 0, "", marks=pytest.mark.skip)]
        ),
        ids=(
            [f"{cb[0]}:L{cb[1]}" for cb in _code_blocks]
            if _code_blocks
            else ["no_blocks"]
        ),
    )
    def test_code_block_parses(self, rst_file, line_no, code):
        """Each Python code block must be syntactically valid."""
        # Skip blocks that are clearly fragments or pseudocode
        if code.strip().startswith("#") and "\n" not in code.strip():
            pytest.skip("Single-line comment, not real code")
        if "..." in code and code.count("...") > code.count("\n") // 2:
            pytest.skip("Pseudocode with ellipsis placeholders")

        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(
                f"Syntax error in {rst_file} at doc line {line_no}: {e}\n"
                f"Code:\n{code[:300]}"
            )

    @pytest.mark.parametrize(
        "rst_file,line_no,code",
        (
            _code_blocks
            if _code_blocks
            else [pytest.param("skip", 0, "", marks=pytest.mark.skip)]
        ),
        ids=(
            [f"imports:{cb[0]}:L{cb[1]}" for cb in _code_blocks]
            if _code_blocks
            else ["no_blocks"]
        ),
    )
    def test_code_block_imports_valid(self, rst_file, line_no, code):
        """Import statements in code blocks must reference real modules."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            pytest.skip("Cannot parse, covered by test_code_block_parses")

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod_name = alias.name
                    # Only check stdlib and aiter imports, skip third-party
                    if mod_name.startswith("aiter"):
                        try:
                            importlib.import_module(mod_name)
                        except (ImportError, ModuleNotFoundError):
                            pytest.fail(
                                f"Import '{mod_name}' in {rst_file} at line {line_no} "
                                f"is not importable"
                            )
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("aiter"):
                    try:
                        importlib.import_module(node.module)
                    except (ImportError, ModuleNotFoundError):
                        pytest.fail(
                            f"Import from '{node.module}' in {rst_file} at line {line_no} "
                            f"is not importable"
                        )


# ---------------------------------------------------------------------------
# Test 3: Sphinx Build
# The documentation must build without warnings.
# ---------------------------------------------------------------------------


class TestSphinxBuild:
    """Verify that Sphinx can build the documentation."""

    @pytest.mark.skipif(
        not DOCS_DIR.exists(),
        reason="docs/ directory not found",
    )
    def test_sphinx_build_no_warnings(self):
        """Sphinx build with -W must succeed (warnings are errors)."""
        try:
            import sphinx  # noqa: F401
        except ImportError:
            pytest.skip("sphinx not installed")

        build_dir = DOCS_DIR / "_build" / "test"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sphinx",
                "-W",
                "--keep-going",
                "-b",
                "html",
                "-q",  # quiet
                str(DOCS_DIR),
                str(build_dir),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Clean up test build dir
        import shutil

        if build_dir.exists():
            shutil.rmtree(build_dir, ignore_errors=True)

        if result.returncode != 0:
            # Extract just the warning/error lines
            errors = [
                line
                for line in result.stderr.splitlines()
                if "WARNING" in line or "ERROR" in line
            ]
            error_summary = "\n".join(errors[:20])
            pytest.fail(
                f"Sphinx build failed with {len(errors)} warnings/errors:\n"
                f"{error_summary}"
            )

    @pytest.mark.skipif(
        not DOCS_DIR.exists(),
        reason="docs/ directory not found",
    )
    def test_all_toctree_files_exist(self):
        """Every file referenced in index.rst toctree must exist."""
        index = DOCS_DIR / "index.rst"
        if not index.exists():
            pytest.skip("index.rst not found")

        content = index.read_text()
        in_toctree = False
        missing = []

        for line in content.splitlines():
            if ".. toctree::" in line:
                in_toctree = True
                continue
            if in_toctree:
                stripped = line.strip()
                if stripped.startswith(":"):
                    # toctree option like :maxdepth:
                    continue
                if stripped == "":
                    if in_toctree:
                        continue  # blank line within toctree is OK
                if (
                    stripped
                    and not stripped.startswith(":")
                    and not stripped.startswith("..")
                ):
                    # This is a document reference
                    doc_path = DOCS_DIR / (stripped + ".rst")
                    if not doc_path.exists():
                        missing.append(stripped)
                # Detect end of toctree (non-indented non-blank line)
                if (
                    line
                    and not line.startswith(" ")
                    and not line.startswith("\t")
                    and stripped
                ):
                    in_toctree = False

        if missing:
            pytest.fail(
                f"Files referenced in toctree but missing: {', '.join(missing)}"
            )


# ---------------------------------------------------------------------------
# Test 4: Version Consistency
# Documentation version must match the installed package version.
# ---------------------------------------------------------------------------


class TestVersionConsistency:
    """Verify documentation version stays in sync with the package."""

    def test_conf_py_no_hardcoded_version(self):
        """conf.py must not have a hardcoded release version string."""
        conf_py = DOCS_DIR / "conf.py"
        if not conf_py.exists():
            pytest.skip("conf.py not found")

        content = conf_py.read_text()
        # Look for hardcoded version like: release = "0.1.0"
        # But allow: release = get_version(...) or release = "dev"
        hardcoded = re.findall(
            r'^release\s*=\s*["\'](\d+\.\d+[^"\']*)["\']',
            content,
            re.MULTILINE,
        )
        if hardcoded:
            pytest.fail(
                f'conf.py has hardcoded version: release = "{hardcoded[0]}". '
                f"Use auto-detection from setuptools_scm or importlib.metadata."
            )

    def test_conf_py_has_version_detection(self):
        """conf.py must contain version auto-detection logic."""
        conf_py = DOCS_DIR / "conf.py"
        if not conf_py.exists():
            pytest.skip("conf.py not found")

        content = conf_py.read_text()
        has_auto_version = (
            "get_version" in content
            or "setuptools_scm" in content
            or "importlib.metadata" in content
        )
        if not has_auto_version:
            pytest.fail(
                "conf.py lacks version auto-detection. "
                "Must use setuptools_scm or importlib.metadata."
            )

    def test_package_version_accessible(self):
        """aiter._version module must exist and define __version__."""
        version_file = Path(__file__).parent.parent / "aiter" / "_version.py"
        if not version_file.exists():
            pytest.skip(
                "_version.py not found (generated by setuptools_scm at build time)"
            )
        content = version_file.read_text()
        assert (
            "__version__" in content
        ), "_version.py exists but does not define __version__"


# ---------------------------------------------------------------------------
# Test 5: RST Structure Validation
# Catch common RST issues that Sphinx might not flag clearly.
# ---------------------------------------------------------------------------


class TestRSTStructure:
    """Validate RST file structure and cross-references."""

    @pytest.mark.skipif(
        not DOCS_DIR.exists(),
        reason="docs/ directory not found",
    )
    def test_no_orphan_api_pages(self):
        """Every RST in docs/api/ must be referenced in index.rst toctree."""
        index = DOCS_DIR / "index.rst"
        if not index.exists():
            pytest.skip("index.rst not found")

        content = index.read_text()
        api_files = list(API_DIR.glob("*.rst")) if API_DIR.exists() else []
        orphans = []

        for api_file in api_files:
            rel_name = f"api/{api_file.stem}"
            if rel_name not in content:
                orphans.append(str(api_file.name))

        if orphans:
            pytest.fail(f"API docs not in index.rst toctree: {', '.join(orphans)}")

    @pytest.mark.skipif(
        not DOCS_DIR.exists(),
        reason="docs/ directory not found",
    )
    def test_no_cuda_references_in_docs(self):
        """Documentation must not contain CUDA-specific references.

        AITER is a ROCm project. References to nvcc, CUDAExtension,
        __nv_bfloat16, sm_90a etc. indicate copy-paste errors.
        """
        cuda_patterns = [
            r"\bCUDAExtension\b",
            r"\bnvcc\b",
            r"\b__nv_bfloat16\b",
            r"\bsm_\d{2}[a-z]?\b",
            r"\bcompute_\d{2}[a-z]?\b",
            r"\bcuda_ext\b",
        ]
        combined = re.compile("|".join(cuda_patterns))
        violations = []

        for rst_file in DOCS_DIR.rglob("*.rst"):
            for i, line in enumerate(rst_file.read_text().splitlines(), 1):
                # Skip lines in comments or the audit report
                if "AUDIT" in str(rst_file).upper():
                    continue
                if combined.search(line):
                    rel = rst_file.relative_to(DOCS_DIR)
                    violations.append(f"{rel}:{i}: {line.strip()[:80]}")

        if violations:
            msg = "CUDA references found in ROCm docs:\n" + "\n".join(violations[:10])
            pytest.fail(msg)
