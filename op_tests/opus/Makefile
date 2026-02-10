# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Makefile for OPUS unit tests

# Use hipcc for OPUS tests (requires HIP/ROCm for full functionality)
# For systems without HIP, we provide a mock environment

HIPCC ?= hipcc
CXX ?= g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -I../../csrc/include

# Detect if hipcc is available
HAVE_HIPCC := $(shell which $(HIPCC) 2>/dev/null)

# Test targets
TESTS := test_opus_basic

.PHONY: all clean test help check-hip

all: check-hip $(TESTS)

check-hip:
ifdef HAVE_HIPCC
	@echo "Using $(HIPCC) for compilation"
else
	@echo "Warning: $(HIPCC) not found. Tests require ROCm/HIP to compile."
	@echo "Please install ROCm or set HIPCC to your hipcc path."
endif

# Main test - requires HIP
$(TESTS): test_opus_basic.cpp
ifdef HAVE_HIPCC
	$(HIPCC) $(CXXFLAGS) $< -o $@
else
	@echo "Skipping $@ (no hipcc)"
endif

# Run all tests
test: $(TESTS)
	@echo "======================================"
	@echo "Running OPUS Unit Tests"
	@echo "======================================"
	@for test in $(TESTS); do \
		echo ""; \
		echo "Running $$test..."; \
		./$$test || exit 1; \
	done
	@echo ""
	@echo "======================================"
	@echo "All tests passed!"
	@echo "======================================"

# Clean build artifacts
clean:
	rm -f $(TESTS)

# Help
help:
	@echo "OPUS Unit Tests - Makefile Targets"
	@echo "======================================"
	@echo "make all      - Build all tests"
	@echo "make test     - Build and run all tests"
	@echo "make clean    - Remove build artifacts"
	@echo "make help     - Show this help message"
