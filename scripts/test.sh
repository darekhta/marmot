#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

BUILD_TYPE="${BUILD_TYPE:-debug}"
SANITIZER="${SANITIZER:-none}"
VERBOSE="${VERBOSE:-0}"
SUITE="${SUITE:-}"
METAL_MATRIX="${METAL_MATRIX:-0}"

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Run Marmot test suite.

OPTIONS:
    -t, --type TYPE         Build type: debug, release, debugoptimized (default: debug)
    -s, --sanitizer SAN     Sanitizer used in build: asan, ubsan, tsan, none (default: none)
    -S, --suite SUITE       Test suite: default (fast), ci (all tests) (default: all)
    -v, --verbose           Verbose test output
    --metal-matrix          Run Metal env matrix (simdgroup_mm=0, decode_simd_groups=4)
    -h, --help              Show this help message

ENVIRONMENT VARIABLES:
    BUILD_TYPE              Same as --type
    SANITIZER               Same as --sanitizer
    SUITE                   Same as --suite
    VERBOSE                 1 for verbose output
    METAL_MATRIX            1 to enable Metal env matrix

EXAMPLES:
    # Run fast tests (default suite)
    $0 --suite default

    # Run all tests including slow/LLM tests (CI suite)
    $0 --suite ci

    # Run fast tests with verbose output
    $0 --suite default --verbose

    # Run all tests with address sanitizer build
    $0 --sanitizer asan --suite ci

    # Run release build tests
    $0 --type release

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -s|--sanitizer)
            SANITIZER="$2"
            shift 2
            ;;
        -S|--suite)
            SUITE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        --metal-matrix)
            METAL_MATRIX=1
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

if [[ "$SANITIZER" != "none" ]]; then
    BUILD_DIR="build-${BUILD_TYPE}-${SANITIZER}"
else
    BUILD_DIR="build-${BUILD_TYPE}"
fi

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "Error: Build directory '$BUILD_DIR' does not exist."
    echo "Run './scripts/build.sh --type $BUILD_TYPE --sanitizer $SANITIZER' first"
    exit 1
fi

echo "========================================="
echo "Marmot Test Suite"
echo "========================================="
echo "Build Type:  $BUILD_TYPE"
echo "Sanitizer:   $SANITIZER"
echo "Test Suite:  ${SUITE:-all}"
echo "Build Dir:   $BUILD_DIR"
echo "========================================="
echo

TEST_FAILED=0

# Build meson test command
MESON_ARGS="-C $BUILD_DIR"
if [[ -n "$SUITE" ]]; then
    MESON_ARGS="$MESON_ARGS --suite $SUITE"
fi
if [[ $VERBOSE -eq 1 ]]; then
    MESON_ARGS="$MESON_ARGS --verbose"
fi

echo "Running tests..."
meson test $MESON_ARGS || TEST_FAILED=1
echo

if [[ "$METAL_MATRIX" -eq 1 ]]; then
    echo "Running Metal matrix: simdgroup_mm=0..."
    MARMOT_METAL_SIMDGROUP_MM=0 meson test $MESON_ARGS || TEST_FAILED=1
    echo
    echo "Running Metal matrix: metal_decode_simd_groups=4..."
    MARMOT_METAL_DECODE_SIMD_GROUPS=4 meson test $MESON_ARGS || TEST_FAILED=1
    echo
fi

if [[ $TEST_FAILED -eq 0 ]]; then
    echo "========================================="
    echo "✅ All tests passed!"
    echo "========================================="
    exit 0
else
    echo "========================================="
    echo "❌ Some tests failed"
    echo "========================================="
    exit 1
fi
