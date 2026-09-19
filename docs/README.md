# Building AITER documentation

Use Python 3.10+ in an isolated environment. The build requires no GPU, ROCm,
PyTorch, AITER installation or initialized kernel submodules.

```bash
python -m venv .venv-docs
. .venv-docs/bin/activate
python -m pip install -r docs/requirements.txt
python docs/check_examples.py
sphinx-build -n -W --keep-going -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`. Both reStructuredText and Markdown are
published. Add user-facing pages to a toctree; repository-only maintenance
notes are explicitly excluded in `conf.py`.

## Source-backed API references

Use `.. aiter-function:: aiter.ops.module.function` to render a signature and
revision-pinned source link. The directive parses Python syntax without
executing imports and fails on a missing function. It deliberately does not
pretend to validate runtime exports or kernel support. Check those on a matched
ROCm stack. Handwritten explanations must document layouts, dtype restrictions,
return values and mutation behavior alongside the generated signature.

The quickstart includes regions from `docs/examples/quickstart.py`. Its static
check verifies direct AITER imports and call signatures without importing GPU
packages. On a supported CDNA ROCm stack, run:

```bash
python docs/examples/quickstart.py
```

This validates actual imports, dispatch and numerical results. Record the script
output, source commit and environment with the result; a successful CPU docs
build is not a GPU validation result.

## CI and deployment

`.github/workflows/docs.yml` builds pull requests and publishes successful main
builds to GitHub Pages. Strict Sphinx warnings fail the job. No package install
failure is swallowed. Public API and dependency changes trigger docs checks as
well as edits under `docs/`.

The reference currently needs no external Sphinx inventories, so builds do not
contact third-party inventory endpoints. External hyperlinks remain explicit
links; verify changed links separately rather than suppressing local warnings.
The footer shows the source version and build date, and API links use the
checkout's exact commit.
