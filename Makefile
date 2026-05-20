# Makefile for aiter development on MI355X (gfx950)
#
# All targets run inside the instinct-dev Docker container via run.sh.
# Run from the host (no need to enter the container manually).
#
# Usage:
#   make setup          — create .venv with system-site-packages
#   make build          — compile all HIP kernels (PREBUILD_KERNELS=1)
#   make install        — register aiter in .venv (editable install)
#   make test-gemm      — run GEMM bf16 tests
#   make test-mla       — run MLA bf16 tests (default params)
#   make test-mla-fp8   — run MLA fp8 tests (co-worker's config)
#   make shell          — open interactive shell inside container
#   make clean          — remove .venv and compiled build artifacts

INFRA        := $(HOME)/infra
RUN          := bash $(INFRA)/docker/run.sh instinct-dev
AITER        := $(HOME)/aiter
GPU_ARCHS    ?= gfx950
VENV         := $(AITER)/.venv
ACTIVATE     := source $(VENV)/bin/activate

# Helper: run a command inside the container with the venv active
define IN_VENV
	$(RUN) bash -c "$(ACTIVATE) && cd $(AITER) && $(1)"
endef

.PHONY: setup build install test-gemm test-mla test-mla-fp8 shell clean help

help:
	@echo ""
	@echo "  make setup         create .venv (system-site-packages)"
	@echo "  make build         compile all HIP kernels (hours, run in tmux)"
	@echo "  make install       editable install into .venv (fast)"
	@echo "  make test-gemm     run op_tests/test_gemm_a16w16.py"
	@echo "  make test-mla      run op_tests/test_mla.py (bf16, default params)"
	@echo "  make test-mla-fp8  run op_tests/test_mla.py --dtype fp8 --kv_dtype fp8 -n 8,1 -b 1 -c 1024 -splits 4"
	@echo "  make shell         interactive container shell"
	@echo "  make clean         remove .venv and jit/build artifacts"
	@echo ""

setup:
	$(RUN) bash -c "python3 -m venv --system-site-packages $(VENV) && echo 'venv created at $(VENV)'"

build:
	@echo "Starting kernel build in tmux session 'aiter-build' (host-side, survives SSH disconnect)"
	@echo "Monitor with: tmux attach -t aiter-build"
	@echo "Log at: $(HOME)/aiter-build.log"
	tmux new-session -d -s aiter-build \
		'$(RUN) bash -c "$(ACTIVATE) && cd $(AITER) && \
		HIP_VISIBLE_DEVICES=\"\" PREBUILD_KERNELS=1 GPU_ARCHS=$(GPU_ARCHS) \
		python3 setup.py build_ext --inplace 2>&1 | tee $(HOME)/aiter-build.log \
		&& echo DONE >> $(HOME)/aiter-build.log \
		|| echo FAILED >> $(HOME)/aiter-build.log"'

install:
	$(call IN_VENV, pip install -e . --no-build-isolation -q)

test-gemm:
	$(call IN_VENV, python3 op_tests/test_gemm_a16w16.py)

test-mla:
	$(call IN_VENV, python3 op_tests/test_mla.py)

test-mla-fp8:
	$(call IN_VENV, python3 op_tests/test_mla.py --dtype fp8 --kv_dtype fp8 -n 8,1 -b 1 -c 1024 -splits 4)

shell:
	$(RUN) bash -c "$(ACTIVATE) && cd $(AITER) && exec bash"

clean:
	$(RUN) bash -c "rm -rf $(VENV) $(AITER)/aiter/jit/build $(AITER)/build && echo 'Cleaned'"
