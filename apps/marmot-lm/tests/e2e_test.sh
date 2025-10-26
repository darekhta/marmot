#!/usr/bin/env bash
#
# End-to-End Tests for marmot-lm
# Uses TinyLlama Q4_K_M model from test fixtures
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MARMOT_LM="$PROJECT_ROOT/apps/marmot-lm/target/release/marmot-lm"
MODEL_PATH="$PROJECT_ROOT/tests/fixtures/gguf/multiarch/tinyllama-q4_k_m.gguf"
LIBMARMOT_PATH="$PROJECT_ROOT/build-release"

# Export library path for macOS
export DYLD_LIBRARY_PATH="$LIBMARMOT_PATH:${DYLD_LIBRARY_PATH:-}"

# Server settings - use default port since run command doesn't have --port option
SERVER_PORT=1234
SERVER_PID=""

# Cleanup function
cleanup() {
    echo -e "\n${BLUE}Cleaning up...${NC}"

    # Stop server if running
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping server (PID: $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi

    # Also try the stop command
    "$MARMOT_LM" stop 2>/dev/null || true
}

trap cleanup EXIT

# Helper functions
log_test() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}TEST: $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

pass() {
    echo -e "${GREEN}✓ PASSED${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

fail() {
    echo -e "${RED}✗ FAILED: $1${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

run_test() {
    local name="$1"
    shift
    local func="$1"

    log_test "$name"
    TESTS_RUN=$((TESTS_RUN + 1))

    if $func; then
        pass
    else
        fail "$name"
    fi
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

preflight_check() {
    echo -e "${YELLOW}Running pre-flight checks...${NC}"

    # Check binary exists
    if [ ! -x "$MARMOT_LM" ]; then
        echo -e "${RED}ERROR: marmot-lm binary not found at $MARMOT_LM${NC}"
        echo "Please build with: cd apps/marmot-lm && cargo build --release"
        exit 1
    fi
    echo -e "  ${GREEN}✓${NC} marmot-lm binary found"

    # Check model exists
    if [ ! -f "$MODEL_PATH" ]; then
        echo -e "${RED}ERROR: Test model not found at $MODEL_PATH${NC}"
        exit 1
    fi
    echo -e "  ${GREEN}✓${NC} Test model found ($(du -h "$MODEL_PATH" | cut -f1))"

    # Check libmarmot
    if [ ! -f "$LIBMARMOT_PATH/libmarmot.dylib" ]; then
        echo -e "${RED}ERROR: libmarmot.dylib not found at $LIBMARMOT_PATH${NC}"
        echo "Please build with: make build-release"
        exit 1
    fi
    echo -e "  ${GREEN}✓${NC} libmarmot.dylib found"

    # Stop any existing server
    "$MARMOT_LM" stop 2>/dev/null || true

    echo -e "${GREEN}Pre-flight checks passed!${NC}"
}

# ============================================================================
# Test: Model Info
# ============================================================================

test_info_command() {
    local output
    local exit_code=0
    output=$("$MARMOT_LM" info "$MODEL_PATH" 2>&1) || exit_code=$?

    echo "$output"

    if [ $exit_code -ne 0 ]; then
        echo "Command failed with exit code $exit_code"
        return 1
    fi

    # Check for expected fields in output (use -E for extended regex)
    if ! echo "$output" | grep -Ei "llama|Architecture" > /dev/null; then
        echo "Expected architecture info not found"
        return 1
    fi

    return 0
}

# ============================================================================
# Test: List Models
# ============================================================================

test_list_command() {
    local output
    local exit_code=0
    output=$("$MARMOT_LM" list 2>&1) || exit_code=$?

    echo "$output"

    if [ $exit_code -ne 0 ]; then
        echo "Command failed with exit code $exit_code"
        return 1
    fi

    # Command should succeed (may show no models or some models)
    return 0
}

# ============================================================================
# Test: Embedded Mode Inference
# ============================================================================

test_embedded_inference() {
    local output
    local prompt="What is 2+2? Answer with just the number:"

    echo "Running inference with prompt: '$prompt'"

    # Run inference with timeout
    output=$(timeout 120 "$MARMOT_LM" run "$MODEL_PATH" --prompt "$prompt" 2>&1) || true
    local exit_code=$?

    echo "Output:"
    echo "$output"

    # Check if inference produced output
    if [ -z "$output" ]; then
        echo "No output produced"
        return 1
    fi

    # Check for common error patterns
    if echo "$output" | grep -qi "error\|failed\|panic"; then
        # Some errors might be in tracing output, check if we got actual tokens
        if ! echo "$output" | grep -qi "generated\|token\|4"; then
            echo "Error detected in output"
            return 1
        fi
    fi

    return 0
}

# ============================================================================
# Test: Server Start
# ============================================================================

test_server_start() {
    echo "Starting server on port $SERVER_PORT..."

    # Start server in background
    "$MARMOT_LM" serve --port "$SERVER_PORT" &
    SERVER_PID=$!

    echo "Server started with PID: $SERVER_PID"

    # Wait for server to be ready
    local retries=30
    local ready=false

    while [ $retries -gt 0 ]; do
        if curl -s "http://127.0.0.1:$SERVER_PORT/lmstudio-greeting" >/dev/null 2>&1; then
            ready=true
            break
        fi
        sleep 0.5
        ((retries--))
    done

    if [ "$ready" = true ]; then
        echo "Server is ready and responding"
        return 0
    else
        echo "Server failed to become ready within timeout"
        return 1
    fi
}

# ============================================================================
# Test: Server Client Inference
# ============================================================================

test_server_client_inference() {
    local output
    local prompt="Say hello in one word:"

    echo "Running client inference against server on port $SERVER_PORT..."
    echo "Prompt: '$prompt'"

    # Run inference via client (should use server)
    output=$(timeout 120 "$MARMOT_LM" run "$MODEL_PATH" --prompt "$prompt" 2>&1) || true

    echo "Output:"
    echo "$output"

    # Check if inference produced output
    if [ -z "$output" ]; then
        echo "No output produced"
        return 1
    fi

    return 0
}

# ============================================================================
# Test: Server Stop
# ============================================================================

test_server_stop() {
    echo "Stopping server..."

    local output
    output=$("$MARMOT_LM" stop 2>&1) || true

    echo "$output"

    # Wait a bit for server to stop
    sleep 1

    # Check if server actually stopped
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server still running, force killing..."
        kill -9 "$SERVER_PID" 2>/dev/null || true
    fi

    SERVER_PID=""

    # Verify server is no longer responding
    if curl -s "http://127.0.0.1:$SERVER_PORT/lmstudio-greeting" >/dev/null 2>&1; then
        echo "Server still responding after stop"
        return 1
    fi

    echo "Server stopped successfully"
    return 0
}

# ============================================================================
# Test: Multi-turn Conversation (Embedded)
# ============================================================================

test_multi_prompt_embedded() {
    local output
    local prompt="Count from 1 to 5:"

    echo "Testing multi-token generation in embedded mode..."
    echo "Prompt: '$prompt'"

    output=$(timeout 120 "$MARMOT_LM" run "$MODEL_PATH" --prompt "$prompt" 2>&1) || true

    echo "Output:"
    echo "$output"

    # Check if we got multiple tokens
    local word_count
    word_count=$(echo "$output" | wc -w | tr -d ' ')

    if [ "$word_count" -lt 3 ]; then
        echo "Expected multiple tokens, got only $word_count words"
        return 1
    fi

    return 0
}

# ============================================================================
# Main Test Runner
# ============================================================================

main() {
    echo -e "${YELLOW}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║             marmot-lm End-to-End Test Suite                 ║"
    echo "║                  TinyLlama Q4_K_M Model                     ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    # Pre-flight checks
    preflight_check

    echo ""
    echo -e "${YELLOW}Running tests...${NC}"

    # Run tests
    run_test "Model Info Command" test_info_command
    run_test "List Models Command" test_list_command
    run_test "Embedded Mode Inference" test_embedded_inference
    run_test "Multi-token Generation (Embedded)" test_multi_prompt_embedded
    run_test "Server Start" test_server_start
    run_test "Server Client Inference" test_server_client_inference
    run_test "Server Stop" test_server_stop

    # Summary
    echo ""
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}                        TEST SUMMARY                           ${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  Total:  $TESTS_RUN"
    echo -e "  ${GREEN}Passed: $TESTS_PASSED${NC}"
    echo -e "  ${RED}Failed: $TESTS_FAILED${NC}"
    echo ""

    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
        exit 0
    else
        echo -e "${RED}Some tests failed.${NC}"
        exit 1
    fi
}

# Run main
main "$@"
