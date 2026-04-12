Contributing to AITER
=====================

Setup
-----

Clone the repository with submodules:

.. code-block:: bash

   git clone --recursive https://github.com/ROCm/aiter.git
   cd aiter

Install development dependencies:

.. code-block:: bash

   pip install -r requirements.txt
   pip install ninja

Running Tests
-------------

.. code-block:: bash

   pytest tests/

Tests require a ROCm-capable GPU. Some tests are architecture-specific and
will be skipped automatically on unsupported hardware.

Code Style
----------

AITER uses `ruff <https://docs.astral.sh/ruff/>`_ for linting and formatting:

.. code-block:: bash

   ruff check .
   ruff format .

Run both checks before submitting a pull request. CI will reject PRs with
lint or format violations.

Pull Request Workflow
----------------------

1. Create a branch from ``main``:

   .. code-block:: bash

      git checkout -b my-feature main

2. Make your changes and add tests where applicable.

3. Run linting, formatting, and tests locally:

   .. code-block:: bash

      ruff check . && ruff format --check . && pytest tests/

4. Push your branch and open a pull request against ``main``.

5. CI will run the ``aiter-test`` and ``triton-test`` pipelines on your PR.
   All checks must pass before merge.
