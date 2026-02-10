#!/usr/bin/env python3
"""
GPU Kernel Development Documentation MCP Server

Provides tools for AI agents to fetch and consume documentation about:
- Triton language and API
- AMD GPU architecture (CDNA3/CDNA4)
- HIP/ROCm programming
- Composable Kernel (CK) library
- Research papers (FlashAttention, etc.)
- General GPU kernel optimization
"""

import re
import sys
from typing import Any

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as html_to_md
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "gpu-docs",
    version="1.0.0",
)

USER_AGENT = "gpu-docs-mcp/1.0 (AI kernel development assistant)"
TIMEOUT = 30.0

# ──────────────────────────────────────────────────────────────────────
# Known documentation URLs organized by topic
# ──────────────────────────────────────────────────────────────────────

KNOWN_DOCS = {
    # Triton
    "triton-api": "https://triton-lang.org/main/python-api/triton.language.html",
    "triton-tutorials": "https://triton-lang.org/main/getting-started/tutorials/index.html",
    "triton-matmul": "https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html",
    "triton-softmax": "https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html",
    "triton-attention": "https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html",
    "triton-layernorm": "https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html",
    "triton-group-gemm": "https://triton-lang.org/main/getting-started/tutorials/07-group-gemm.html",
    "triton-persistent-matmul": "https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html",
    "triton-block-scaled-matmul": "https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html",
    "triton-dot": "https://triton-lang.org/main/python-api/generated/triton.language.dot.html",
    "triton-load": "https://triton-lang.org/main/python-api/generated/triton.language.load.html",
    "triton-store": "https://triton-lang.org/main/python-api/generated/triton.language.store.html",
    # AMD Triton optimization
    "amd-triton-optimization": "https://github-wiki-see.page/m/ROCm/triton/wiki/General-Guide-of-AMD-Triton-Performance-Optimization",
    "amd-triton-blog": "https://rocm.blogs.amd.com/software-tools-optimization/kernel-development-optimizations-with-triton-on-/README.html",
    "amd-triton-kernel-dev": "https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/triton_kernel_dev.html",
    # AMD GPU architecture
    "amd-mi300-arch": "https://instinct.docs.amd.com/latest/gpu-arch/mi300.html",
    "amd-gpu-arch": "https://instinct.docs.amd.com/latest/gpu-arch/gpu-arch.html",
    # HIP
    "hip-performance": "https://rocm.docs.amd.com/projects/HIP/en/develop/how-to/performance_guidelines.html",
    "hip-programming": "https://rocm.docs.amd.com/projects/HIP/en/develop/how-to/programming_manual.html",
    "hip-gpu-perf-theory": "https://rocm.docs.amd.com/projects/HIP/en/develop/understand/performance_optimization.html",
    # Composable Kernel
    "ck-docs": "https://rocm.docs.amd.com/projects/composable_kernel/en/develop/",
    "ck-hello-world": "https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.2.0/tutorial/tutorial_hello_world.html",
    # ROCm
    "rocm-programming": "https://rocm.docs.amd.com/en/develop/how-to/programming_guide.html",
}

KNOWN_PAPERS = {
    "flash-attention-1": "https://arxiv.org/abs/2205.14135",
    "flash-attention-2": "https://arxiv.org/abs/2307.08691",
    "flash-attention-3": "https://arxiv.org/abs/2407.08610",
    "online-softmax": "https://arxiv.org/abs/1805.02867",
    "rope": "https://arxiv.org/abs/2104.09864",
    "rmsnorm": "https://arxiv.org/abs/1910.07467",
    "switch-transformers-moe": "https://arxiv.org/abs/2101.03961",
    "gptq": "https://arxiv.org/abs/2210.17323",
    "awq": "https://arxiv.org/abs/2306.00978",
    "deepseek-v3": "https://arxiv.org/abs/2412.19437",
    "mla-deepseek-v2": "https://arxiv.org/abs/2405.04434",
}

KNOWN_PDFS = {
    "cdna3-isa": "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf",
    "cdna4-whitepaper": "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf",
    "mi300x-datasheet": "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf",
}


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

async def _fetch_url(url: str) -> str:
    """Fetch a URL and return the raw HTML/text."""
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(follow_redirects=True, timeout=TIMEOUT) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.text


async def _fetch_html_as_markdown(url: str, max_chars: int = 80000) -> str:
    """Fetch a web page and convert to readable markdown."""
    html = await _fetch_url(url)
    soup = BeautifulSoup(html, "html.parser")

    # Remove nav, footer, scripts, styles
    for tag in soup(["nav", "footer", "script", "style", "header", "aside"]):
        tag.decompose()

    # Try to find main content
    main = soup.find("main") or soup.find("article") or soup.find("div", {"role": "main"})
    if main:
        md = html_to_md(str(main), heading_style="ATX", strip=["img"])
    else:
        md = html_to_md(str(soup.body or soup), heading_style="ATX", strip=["img"])

    # Clean up excessive whitespace
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = md.strip()

    if len(md) > max_chars:
        md = md[:max_chars] + "\n\n[... content truncated ...]"

    return md


async def _fetch_arxiv_abstract(arxiv_id: str) -> str:
    """Fetch an arXiv paper abstract and metadata."""
    url = f"https://arxiv.org/abs/{arxiv_id}"
    html = await _fetch_url(url)
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("h1", class_="title")
    title = title_tag.get_text(strip=True).replace("Title:", "").strip() if title_tag else "Unknown"

    authors_tag = soup.find("div", class_="authors")
    authors = authors_tag.get_text(strip=True).replace("Authors:", "").strip() if authors_tag else "Unknown"

    abstract_tag = soup.find("blockquote", class_="abstract")
    abstract = abstract_tag.get_text(strip=True).replace("Abstract:", "").strip() if abstract_tag else "No abstract available"

    return f"# {title}\n\n**Authors:** {authors}\n\n**arXiv:** {arxiv_id}\n\n## Abstract\n\n{abstract}\n\n**PDF:** https://arxiv.org/pdf/{arxiv_id}"


# ──────────────────────────────────────────────────────────────────────
# MCP Tools
# ──────────────────────────────────────────────────────────────────────

@mcp.tool()
async def list_available_docs() -> str:
    """List all available documentation sources organized by topic.
    Call this first to see what documentation is available."""

    lines = ["# Available GPU Kernel Development Documentation\n"]

    lines.append("## Triton Language & Tutorials")
    for key, url in KNOWN_DOCS.items():
        if key.startswith("triton"):
            lines.append(f"- **{key}**: {url}")

    lines.append("\n## AMD GPU Optimization")
    for key, url in KNOWN_DOCS.items():
        if key.startswith("amd"):
            lines.append(f"- **{key}**: {url}")

    lines.append("\n## HIP / ROCm")
    for key, url in KNOWN_DOCS.items():
        if key.startswith(("hip", "rocm")):
            lines.append(f"- **{key}**: {url}")

    lines.append("\n## Composable Kernel (CK)")
    for key, url in KNOWN_DOCS.items():
        if key.startswith("ck"):
            lines.append(f"- **{key}**: {url}")

    lines.append("\n## Research Papers")
    for key, url in KNOWN_PAPERS.items():
        lines.append(f"- **{key}**: {url}")

    lines.append("\n## Architecture PDFs")
    for key, url in KNOWN_PDFS.items():
        lines.append(f"- **{key}**: {url}")

    return "\n".join(lines)


@mcp.tool()
async def fetch_doc(doc_id: str) -> str:
    """Fetch a documentation page by its ID and return as markdown.

    Args:
        doc_id: The documentation ID from list_available_docs()
                (e.g. 'triton-api', 'hip-performance', 'amd-mi300-arch')
    """
    url = KNOWN_DOCS.get(doc_id)
    if not url:
        available = ", ".join(sorted(KNOWN_DOCS.keys()))
        return f"Unknown doc_id: '{doc_id}'. Available: {available}"

    try:
        content = await _fetch_html_as_markdown(url)
        return f"# Documentation: {doc_id}\n**Source:** {url}\n\n{content}"
    except Exception as e:
        return f"Error fetching {doc_id}: {e}"


@mcp.tool()
async def fetch_paper(paper_id: str) -> str:
    """Fetch a research paper's abstract and metadata from arXiv.

    Args:
        paper_id: Either a known paper ID (e.g. 'flash-attention-2', 'rope')
                  or a raw arXiv ID (e.g. '2307.08691')
    """
    if paper_id in KNOWN_PAPERS:
        url = KNOWN_PAPERS[paper_id]
        arxiv_id = url.split("/abs/")[-1]
    elif re.match(r"\d{4}\.\d{4,5}", paper_id):
        arxiv_id = paper_id
    else:
        available = ", ".join(sorted(KNOWN_PAPERS.keys()))
        return f"Unknown paper_id: '{paper_id}'. Known papers: {available}\nOr provide a raw arXiv ID like '2307.08691'"

    try:
        return await _fetch_arxiv_abstract(arxiv_id)
    except Exception as e:
        return f"Error fetching paper {arxiv_id}: {e}"


@mcp.tool()
async def fetch_url(url: str) -> str:
    """Fetch any URL and return its content as markdown.
    Use this for documentation pages, blog posts, or other web resources
    not in the known docs list.

    Args:
        url: The full URL to fetch (e.g. 'https://triton-lang.org/...')
    """
    try:
        content = await _fetch_html_as_markdown(url)
        return f"# Content from: {url}\n\n{content}"
    except Exception as e:
        return f"Error fetching URL: {e}"


@mcp.tool()
async def fetch_triton_api(function_name: str) -> str:
    """Fetch detailed documentation for a specific triton.language function.

    Args:
        function_name: The function name without 'tl.' prefix
                       (e.g. 'dot', 'load', 'store', 'reduce', 'atomic_add',
                        'make_block_ptr', 'dot_scaled', 'associative_scan')
    """
    url = f"https://triton-lang.org/main/python-api/generated/triton.language.{function_name}.html"
    try:
        content = await _fetch_html_as_markdown(url, max_chars=20000)
        return f"# triton.language.{function_name}\n**Source:** {url}\n\n{content}"
    except Exception as e:
        return f"Error fetching triton.language.{function_name}: {e}\nCheck function name. Common functions: dot, load, store, reduce, sum, max, arange, zeros, where, atomic_add, make_block_ptr, dot_scaled"


@mcp.tool()
async def search_gpu_docs(query: str) -> str:
    """Search for GPU kernel development documentation using DuckDuckGo.
    Use for finding specific optimization techniques, API details, or
    troubleshooting GPU kernel issues.

    Args:
        query: Search query (e.g. 'triton persistent kernel AMD',
               'HIP shared memory bank conflict', 'MFMA instruction latency')
    """
    search_url = "https://html.duckduckgo.com/html/"
    params = {"q": f"{query} site:triton-lang.org OR site:rocm.docs.amd.com OR site:github.com/ROCm OR site:instinct.docs.amd.com"}

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=TIMEOUT) as client:
            resp = await client.post(search_url, data=params, headers={
                "User-Agent": USER_AGENT,
            })
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for r in soup.find_all("div", class_="result", limit=8):
            title_tag = r.find("a", class_="result__a")
            snippet_tag = r.find("a", class_="result__snippet")
            if title_tag:
                title = title_tag.get_text(strip=True)
                href = title_tag.get("href", "")
                snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                results.append(f"**{title}**\n{href}\n{snippet}\n")

        if not results:
            return f"No results found for: {query}\nTry a broader search or use fetch_url with a specific URL."

        return f"# Search results for: {query}\n\n" + "\n---\n".join(results)
    except Exception as e:
        return f"Search error: {e}. Try using fetch_url with a direct documentation URL instead."


@mcp.tool()
async def get_amd_arch_specs(gpu: str = "mi300x") -> str:
    """Get AMD GPU architecture specifications and key numbers
    for kernel optimization decisions.

    Args:
        gpu: GPU model - 'mi300x' (CDNA3) or 'mi350x' (CDNA4)
    """
    if gpu.lower() in ("mi300x", "mi300", "cdna3", "gfx942"):
        return """# AMD Instinct MI300X (CDNA3 / gfx942) Specifications

## Compute
- **Compute Units (CUs):** 304 (38 active per XCD × 8 XCDs)
- **SIMDs per CU:** 4
- **Wavefront size:** 64 threads
- **Max threads per workgroup:** 1024
- **Clock speed:** ~2.1 GHz (boost)

## Registers & On-Chip Memory
- **VGPRs per SIMD:** 512 (allocated in units of 16)
- **SGPRs per CU:** 800
- **LDS per CU:** 64 KB
- **L1 Cache per CU:** 32 KB
- **L2 Cache per XCD:** 4 MB (shared across 40 CUs on same XCD)
- **Total L2:** 32 MB (8 XCDs × 4 MB, NOT shared across XCDs)

## VGPR Occupancy Table
| VGPRs/wave | Max waves/SIMD |
|-----------|----------------|
| 1-128     | 4              |
| 129-170   | 3              |
| 171-256   | 2              |
| 257-512   | 1              |

## Memory
- **HBM3 Capacity:** 192 GB
- **HBM3 Bandwidth:** 5.3 TB/s peak
- **Infinity Fabric:** 7 links for 8-GPU full mesh

## Peak Performance
| Data Type    | FLOPS/clock/CU | Peak TFLOPS |
|-------------|----------------|-------------|
| FP64 matrix  | 256            | 163.4       |
| FP32 matrix  | 256            | 163.4       |
| TF32         | 1024           | 653.7       |
| FP16/BF16    | 2048           | 1,307.4     |
| FP8/INT8     | 4096           | 2,614.9     |

## MFMA Instructions
| Instruction                        | M×N×K   | Cycles |
|------------------------------------|---------|--------|
| v_mfma_f32_16x16x16_f16/bf16      | 16×16×16| 16     |
| v_mfma_f32_32x32x8_f16/bf16       | 32×32×8 | 32     |
| v_mfma_f32_16x16x32_fp8/bf8       | 16×16×32| 16     |
| v_mfma_f32_32x32x16_fp8/bf8       | 32×32×16| 32     |
| v_mfma_i32_16x16x32_i8            | 16×16×32| 16     |
| v_mfma_i32_32x32x16_i8            | 32×32×16| 32     |

## Roofline Ridge Points
- FP16/BF16: 1307 TFLOPS / 5.3 TB/s ≈ **246 FLOPs/byte**
- FP8/INT8:  2615 TFLOPS / 5.3 TB/s ≈ **493 FLOPs/byte**

## Triton Optimization Defaults
- `matrix_instr_nonkdim=16` for GEMM, `=32` for fused GEMM
- `num_stages=2` for GEMM, `=1` for fused GEMM (prefill), `=3` for indirect loads
- `waves_per_eu=2` default occupancy target
- `kpack=2` for GEMM shared memory optimization
- `NUM_XCDS=8` for XCD-aware scheduling
"""

    elif gpu.lower() in ("mi350x", "mi350", "cdna4", "gfx950"):
        return """# AMD Instinct MI350X (CDNA4 / gfx950) Specifications

## Key Differences from MI300X (CDNA3)
- **XCDs:** 8 (3D vertically stacked)
- **Memory:** HBM3E (higher bandwidth and capacity)
- **TDP:** 1000W (air-cooled)
- **New:** Scaled MFMA instructions for OCP microscaling formats (MXFP4, MXFP8)
- **Triton:** `tl.dot_scaled()` support for native scaled matrix multiplication

## Scaled MFMA (New in CDNA4)
Natively supports microscaling formats with per-block scale factors:
- MXFP4 (4-bit mantissa, shared exponent per 32 elements)
- MXFP8 (8-bit microscaling)
- In Triton: `tl.dot_scaled(a, b, a_scale, b_scale, ...)`

## Architecture Documentation
- CDNA4 Whitepaper: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
"""
    else:
        return f"Unknown GPU: {gpu}. Supported: mi300x (CDNA3/gfx942), mi350x (CDNA4/gfx950)"


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
