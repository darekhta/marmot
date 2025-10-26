.PHONY: help build build-debug build-release test test-verbose clean clean-all
.PHONY: build-asan build-ubsan build-tsan test-asan test-ubsan test-tsan
.PHONY: test-ci test-ci-verbose test-ci-asan test-ci-ubsan test-ci-tsan
.PHONY: test-ci-metal-matrix
.PHONY: build-release-asan build-release-ubsan
.PHONY: build-lm build-lm-release test-lm
.PHONY: install install-lib install-lm
.PHONY: format format-check

help:
	@echo "Marmot ML Framework - Build & Test"
	@echo ""
	@echo "Quick Start:"
	@echo "  make build          - Debug build (libmarmot)"
	@echo "  make build-release  - Release build (libmarmot)"
	@echo "  make build-lm       - Build marmot-lm CLI (debug, requires libmarmot)"
	@echo "  make test           - Run all tests"
	@echo ""
	@echo "Debug Builds:"
	@echo "  make build-debug    - Debug build (default)"
	@echo "  make build-asan     - Debug build with AddressSanitizer"
	@echo "  make build-ubsan    - Debug build with UndefinedBehaviorSanitizer"
	@echo "  make build-tsan     - Debug build with ThreadSanitizer"
	@echo ""
	@echo "Release Builds:"
	@echo "  make build-release  - Release build (optimized)"
	@echo "  make build-release-asan   - Release + AddressSanitizer"
	@echo "  make build-release-ubsan  - Release + UBSanitizer"
	@echo ""
	@echo "marmot-lm (Rust CLI):"
	@echo "  make build-lm           - Debug build"
	@echo "  make build-lm-release   - Release build"
	@echo "  make test-lm            - Run marmot-lm e2e tests"
	@echo ""
	@echo "Installation:"
	@echo "  make install            - Install libmarmot + marmot-lm to PREFIX"
	@echo "  make install-lib        - Install libmarmot only"
	@echo "  make install-lm         - Install marmot-lm only"
	@echo "  PREFIX=/usr/local make install  (default: /usr/local)"
	@echo ""
	@echo "Testing (fast - for development):"
	@echo "  make test           - Run fast tests (debug build)"
	@echo "  make test-verbose   - Run fast tests with verbose output"
	@echo "  make test-asan      - Run fast tests with AddressSanitizer"
	@echo "  make test-ubsan     - Run fast tests with UBSanitizer"
	@echo "  make test-tsan      - Run fast tests with ThreadSanitizer"
	@echo ""
	@echo "Testing (CI - all tests including slow/LLM):"
	@echo "  make test-ci        - Run all tests including slow ones"
	@echo "  make test-ci-verbose - Run all tests with verbose output"
	@echo "  make test-ci-asan   - Run all tests with AddressSanitizer"
	@echo "  make test-ci-ubsan  - Run all tests with UBSanitizer"
	@echo "  make test-ci-tsan   - Run all tests with ThreadSanitizer"
	@echo "  make test-ci-metal-matrix - Run CI suite with Metal env matrix"
	@echo ""
	@echo "Cleaning:"
	@echo "  make clean          - Remove debug build"
	@echo "  make clean-all      - Remove all build directories (including marmot-lm)"
	@echo ""
	@echo "Formatting:"
	@echo "  make format         - Format all C/C++/ObjC++ code"
	@echo "  make format-check   - Check if code needs formatting"
	@echo ""
	@echo "Examples:"
	@echo "  make format && make build-asan && make test-asan"
	@echo "  make build-release && make build-lm-release"
	@echo "  PREFIX=/opt/marmot make install"

# --- libmarmot builds ---

build: build-debug

build-debug: format
	./scripts/build.sh --type debug

build-release: format
	./scripts/build.sh --type release

build-asan: format
	./scripts/build.sh --type debug --sanitizer asan

build-ubsan: format
	./scripts/build.sh --type debug --sanitizer ubsan

build-tsan: format
	./scripts/build.sh --type debug --sanitizer tsan

build-release-asan: format
	./scripts/build.sh --type release --sanitizer asan

build-release-ubsan: format
	./scripts/build.sh --type release --sanitizer ubsan

# --- marmot-lm builds ---

build-lm:
	cd apps/marmot-lm && cargo build

build-lm-release:
	cd apps/marmot-lm && cargo build --release

test-lm:
	@if [ -f apps/marmot-lm/tests/e2e_test.sh ]; then \
		bash apps/marmot-lm/tests/e2e_test.sh; \
	else \
		echo "No marmot-lm e2e tests found"; \
	fi

# --- Installation ---

PREFIX ?= /usr/local

install: install-lib install-lm

install-lib:
	./scripts/install.sh --prefix $(PREFIX) --lib-only

install-lm:
	./scripts/install.sh --prefix $(PREFIX) --lm-only

# --- libmarmot tests ---

# Fast tests (default suite - for development iteration)
test:
	./scripts/test.sh --type debug --suite default

test-verbose:
	./scripts/test.sh --type debug --suite default --verbose

test-asan:
	./scripts/test.sh --type debug --sanitizer asan --suite default --verbose

test-ubsan:
	./scripts/test.sh --type debug --sanitizer ubsan --suite default --verbose

test-tsan:
	./scripts/test.sh --type debug --sanitizer tsan --suite default --verbose

# CI tests (all tests including slow/LLM tests)
test-ci:
	./scripts/test.sh --type debug --suite ci

test-ci-verbose:
	./scripts/test.sh --type debug --suite ci --verbose

test-ci-asan:
	./scripts/test.sh --type debug --sanitizer asan --suite ci --verbose

test-ci-ubsan:
	./scripts/test.sh --type debug --sanitizer ubsan --suite ci --verbose

test-ci-tsan:
	./scripts/test.sh --type debug --sanitizer tsan --suite ci --verbose

test-ci-metal-matrix:
	METAL_MATRIX=1 ./scripts/test.sh --type debug --suite ci

# --- Cleaning ---

clean:
	rm -rf build-debug

clean-all:
	rm -rf build-*
	cd apps/marmot-lm && cargo clean 2>/dev/null || true

# --- Formatting ---

format:
	./scripts/format.sh

format-check:
	./scripts/format-check.sh
