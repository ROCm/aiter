# AITER Documentation

This directory contains the source files for AITER's documentation, built with [Sphinx](https://www.sphinx-doc.org/).

## Quick Start

### Build Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Build HTML documentation
make html

# Open in browser
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
```

### Live Preview (Recommended)

```bash
# Install sphinx-autobuild
pip install sphinx-autobuild

# Start live server (auto-rebuilds on changes)
make livehtml

# Open http://127.0.0.1:8000 in your browser
```

## Documentation Structure

```
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Homepage
├── installation.rst     # Installation guide
├── quickstart.rst       # Quick start tutorial
├── api/                 # API reference
│   ├── attention.rst    # Attention operations
│   ├── gemm.rst         # GEMM operations
│   ├── operators.rst    # Core operators
│   └── ...
├── tutorials/           # Tutorials
│   ├── index.rst
│   ├── basic_usage.rst
│   ├── attention_tutorial.rst
│   └── ...
├── _static/             # Static files (images, CSS, JS)
└── _build/              # Built documentation (generated)
```

## Writing Documentation

### Adding a New Page

1. Create a new `.rst` file in the appropriate directory
2. Add it to the `toctree` in `index.rst` or relevant section index
3. Build and verify: `make html`

### reStructuredText Syntax

#### Headers

```rst
Page Title
==========

Section
-------

Subsection
^^^^^^^^^^
```

#### Code Blocks

```rst
.. code-block:: python

   import aiter
   output = aiter.flash_attn_func(q, k, v)
```

#### Links

```rst
:doc:`installation`              # Link to another document
:ref:`my-label`                  # Link to a label
`External Link <https://...>`_   # External URL
```

#### API Documentation

```rst
.. autofunction:: aiter.flash_attn_func
.. autoclass:: aiter.FlashAttention
   :members:
```

### Style Guide

- **Headings**: Use sentence case (not title case)
- **Code**: Use inline code for function names: ``` ``aiter.flash_attn_func()`` ```
- **Examples**: Always include runnable code examples
- **Links**: Use relative links for internal references
- **Line length**: Keep lines under 100 characters when possible

## Building Options

### Check for Warnings

```bash
make html SPHINXOPTS="-W --keep-going"
```

This treats warnings as errors and shows all issues.

### Check Links

```bash
make linkcheck
```

Validates all external links (may take a few minutes).

### Clean Build

```bash
make clean
make html
```

### PDF Output

```bash
make latexpdf
```

Requires LaTeX installation.

## Deployment

Documentation is automatically deployed to `doc.aiter.amd.com` via GitHub Actions on every push to `main`.

See [DEPLOYMENT.md](DEPLOYMENT.md) for details.

## Contributing

### Before Submitting

1. ✅ Build locally and check for warnings
2. ✅ Verify all links work (`make linkcheck`)
3. ✅ Test code examples
4. ✅ Check spelling and grammar
5. ✅ Follow the style guide

### Pull Request

Documentation changes should be submitted via PR with:
- Clear description of what's changed
- Screenshots if adding new pages
- Link to preview build (GitHub Actions provides artifacts)

## Troubleshooting

### "Module not found" errors

Install AITER in development mode:

```bash
cd ..  # Go to repository root
pip install -e .
```

### Missing dependencies

```bash
pip install -r requirements.txt
```

### Broken links in API docs

Ensure the module is importable:

```python
import aiter
print(dir(aiter))
```

### Build is slow

Use `make html` instead of `make clean html` for incremental builds.

## Tools

### Useful Sphinx Extensions

Already included:
- `sphinx.ext.autodoc` - Auto-generate API docs from docstrings
- `sphinx.ext.napoleon` - Support Google/NumPy docstring styles
- `sphinx.ext.viewcode` - Add links to source code
- `sphinx.ext.intersphinx` - Link to PyTorch docs
- `sphinx_copybutton` - Copy button for code blocks

### Theme

We use `sphinx_rtd_theme` (Read the Docs theme) with AMD branding:
- Primary color: AMD Red (#C00000)
- Custom logo in `_static/`

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [Read the Docs Theme](https://sphinx-rtd-theme.readthedocs.io/)
- [Example: FlashInfer Docs](https://docs.flashinfer.ai/)

## Support

- **Documentation issues**: Open issue with `documentation` label
- **Build problems**: Check GitHub Actions logs
- **Content questions**: Ask in GitHub Discussions
