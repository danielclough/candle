#!/bin/bash
# Test script for Qwen2.5-VL CLI flags
# Usage: ./test_cli.sh [--quick] [--skip-model-download]
#
# This script tests all CLI flags documented in README.md
# Uses the included dice.png test image

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TEST_IMAGE="$SCRIPT_DIR/dice.png"
MAX_LENGTH=32  # Short for fast testing
QUICK_MODE=false
SKIP_DOWNLOAD=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --quick)
            QUICK_MODE=true
            MAX_LENGTH=16
            shift
            ;;
        --skip-model-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--quick] [--skip-model-download]"
            echo ""
            echo "Options:"
            echo "  --quick              Use shorter generation length (16 tokens)"
            echo "  --skip-model-download  Skip tests that require model download"
            echo ""
            exit 0
            ;;
    esac
done

# Counters
PASSED=0
FAILED=0
SKIPPED=0

# Helper functions
log_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    PASSED=$((PASSED + 1))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    FAILED=$((FAILED + 1))
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
    SKIPPED=$((SKIPPED + 1))
}

log_section() {
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  $1${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Run a test command
run_test() {
    local name="$1"
    shift
    local cmd="$@"

    log_test "$name"
    echo "  Command: $cmd"

    if eval "$cmd" > /tmp/qwen_test_output.txt 2>&1; then
        log_pass "$name"
        if [ "$VERBOSE" = true ]; then
            cat /tmp/qwen_test_output.txt
        fi
        return 0
    else
        log_fail "$name"
        echo "  Error output:"
        tail -20 /tmp/qwen_test_output.txt | sed 's/^/    /'
        return 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log_section "Checking Prerequisites"

    # Check test image exists
    if [ ! -f "$TEST_IMAGE" ]; then
        echo -e "${RED}Error: Test image not found at $TEST_IMAGE${NC}"
        exit 1
    fi
    log_pass "Test image found: $TEST_IMAGE"

    # Check we're in the right directory
    if [ ! -f "$REPO_ROOT/Cargo.toml" ]; then
        echo -e "${RED}Error: Cannot find Cargo.toml in $REPO_ROOT${NC}"
        exit 1
    fi
    log_pass "Repository root found: $REPO_ROOT"

    # Build the example first
    log_test "Building qwen2_5_vl example..."
    cd "$REPO_ROOT"
    if cargo build --example qwen2_5_vl --release 2>&1 | tail -5; then
        log_pass "Build successful"
    else
        log_fail "Build failed"
        exit 1
    fi
}

# Base command
BASE_CMD="cargo run --features metal,accelerate --example qwen2_5_vl --release --"

# ============================================================================
# TEST SECTIONS
# ============================================================================

test_help_and_version() {
    log_section "Help and Basic Flags"

    run_test "Help flag" "$BASE_CMD --help" || true
}

test_basic_inference() {
    log_section "Basic Inference"

    if [ "$SKIP_DOWNLOAD" = true ]; then
        log_skip "Basic inference (model download required)"
        return
    fi

    run_test "Basic image inference" \
        "$BASE_CMD --image '$TEST_IMAGE' --prompt 'What do you see?' --max-length $MAX_LENGTH" || true
}

test_prompt_variations() {
    log_section "Prompt Variations"

    if [ "$SKIP_DOWNLOAD" = true ]; then
        log_skip "Prompt variations (model download required)"
        return
    fi

    run_test "Custom prompt" \
        "$BASE_CMD --image '$TEST_IMAGE' --prompt 'Count the objects' --max-length $MAX_LENGTH" || true

    run_test "Long prompt" \
        "$BASE_CMD --image '$TEST_IMAGE' --prompt 'Please describe this image in detail, including colors, shapes, and any text visible' --max-length $MAX_LENGTH" || true
}

test_sampling_strategies() {
    log_section "Sampling Strategies"

    if [ "$SKIP_DOWNLOAD" = true ]; then
        log_skip "Sampling strategies (model download required)"
        return
    fi

    run_test "Temperature sampling (0.5)" \
        "$BASE_CMD --image '$TEST_IMAGE' --prompt 'Describe this' --temperature 0.5 --max-length $MAX_LENGTH" || true

    run_test "Temperature sampling (1.0)" \
        "$BASE_CMD --image '$TEST_IMAGE' --prompt 'Describe this' --temperature 1.0 --max-length $MAX_LENGTH" || true

    run_test "Top-p sampling" \
        "$BASE_CMD --image '$TEST_IMAGE' --prompt 'Describe this' --temperature 0.8 --top-p 0.9 --max-length $MAX_LENGTH" || true

    run_test "Top-k sampling" \
        "$BASE_CMD --image '$TEST_IMAGE' --prompt 'Describe this' --temperature 0.8 --top-k 50 --max-length $MAX_LENGTH" || true

    run_test "Combined top-k + top-p" \
        "$BASE_CMD --image '$TEST_IMAGE' --prompt 'Describe this' --temperature 0.7 --top-k 40 --top-p 0.95 --max-length $MAX_LENGTH" || true

    run_test "Greedy (no temperature)" \
        "$BASE_CMD --image '$TEST_IMAGE' --prompt 'Describe this' --max-length $MAX_LENGTH" || true
}

test_repeat_penalty() {
    log_section "Repeat Penalty"

    if [ "$SKIP_DOWNLOAD" = true ]; then
        log_skip "Repeat penalty (model download required)"
        return
    fi

    run_test "Repeat penalty 1.1" \
        "$BASE_CMD --image '$TEST_IMAGE' --prompt 'Describe this' --repeat-penalty 1.1 --max-length $MAX_LENGTH" || true

    run_test "Repeat penalty 1.5" \
        "$BASE_CMD --image '$TEST_IMAGE' --prompt 'Describe this' --repeat-penalty 1.5 --max-length $MAX_LENGTH" || true

    run_test "Repeat penalty with custom window" \
        "$BASE_CMD --image '$TEST_IMAGE' --prompt 'Describe this' --repeat-penalty 1.2 --repeat-last-n 32 --max-length $MAX_LENGTH" || true
}

test_streaming() {
    log_section "Streaming Output"

    if [ "$SKIP_DOWNLOAD" = true ]; then
        log_skip "Streaming output (model download required)"
        return
    fi

    run_test "Streaming mode" \
        "$BASE_CMD --image '$TEST_IMAGE' --prompt 'Describe this' --stream --max-length $MAX_LENGTH" || true

    run_test "Streaming with temperature" \
        "$BASE_CMD --image '$TEST_IMAGE' --prompt 'Describe this' --stream --temperature 0.7 --max-length $MAX_LENGTH" || true
}

test_seed_reproducibility() {
    log_section "Seed Reproducibility"

    if [ "$SKIP_DOWNLOAD" = true ]; then
        log_skip "Seed reproducibility (model download required)"
        return
    fi

    run_test "Custom seed" \
        "$BASE_CMD --image '$TEST_IMAGE' --prompt 'Describe this' --seed 42 --temperature 0.8 --max-length $MAX_LENGTH" || true

    # Test that same seed gives same output
    log_test "Seed reproducibility check"
    OUTPUT1=$($BASE_CMD --image "$TEST_IMAGE" --prompt 'What?' --seed 12345 --temperature 0.8 --max-length 16 2>&1 | tail -5)
    OUTPUT2=$($BASE_CMD --image "$TEST_IMAGE" --prompt 'What?' --seed 12345 --temperature 0.8 --max-length 16 2>&1 | tail -5)
    if [ "$OUTPUT1" = "$OUTPUT2" ]; then
        log_pass "Seed reproducibility check"
    else
        log_fail "Seed reproducibility check (outputs differ)"
    fi
}

test_generation_length() {
    log_section "Generation Length"

    if [ "$SKIP_DOWNLOAD" = true ]; then
        log_skip "Generation length (model download required)"
        return
    fi

    run_test "Short generation (16 tokens)" \
        "$BASE_CMD --image '$TEST_IMAGE' --prompt 'Describe this' --max-length 16" || true

    run_test "Medium generation (64 tokens)" \
        "$BASE_CMD --image '$TEST_IMAGE' --prompt 'Describe this' --max-length 64" || true
}

test_multi_image() {
    log_section "Multi-Image Input"

    if [ "$SKIP_DOWNLOAD" = true ]; then
        log_skip "Multi-image input (model download required)"
        return
    fi

    # Use the same image twice for testing
    run_test "Two images without placeholders" \
        "$BASE_CMD --image '$TEST_IMAGE' --image '$TEST_IMAGE' --prompt 'Compare these images' --max-length $MAX_LENGTH" || true

    run_test "Two images with placeholders" \
        "$BASE_CMD --image '$TEST_IMAGE' --image '$TEST_IMAGE' --prompt 'What is in {image1}? What about {image2}?' --max-length $MAX_LENGTH" || true
}

test_precision_options() {
    log_section "Precision Options"

    if [ "$SKIP_DOWNLOAD" = true ]; then
        log_skip "Precision options (model download required)"
        return
    fi

    run_test "BFloat16 precision" \
        "$BASE_CMD --image '$TEST_IMAGE' --prompt 'Describe this' --bf16 --max-length $MAX_LENGTH" || true
}

test_attention_options() {
    log_section "Attention Options"

    if [ "$SKIP_DOWNLOAD" = true ]; then
        log_skip "Attention options (model download required)"
        return
    fi

    run_test "Sliding window attention" \
        "$BASE_CMD --image '$TEST_IMAGE' --prompt 'Describe this' --sliding-window --sliding-window-size 2048 --max-length $MAX_LENGTH" || true

    run_test "Sliding window with custom layers" \
        "$BASE_CMD --image '$TEST_IMAGE' --prompt 'Describe this' --sliding-window --max-window-layers 10 --max-length $MAX_LENGTH" || true
}

test_cpu_mode() {
    log_section "CPU Mode"

    if [ "$SKIP_DOWNLOAD" = true ]; then
        log_skip "CPU mode (model download required)"
        return
    fi

    if [ "$QUICK_MODE" = true ]; then
        log_skip "CPU mode (slow, skipped in quick mode)"
        return
    fi

    run_test "CPU inference" \
        "$BASE_CMD --image '$TEST_IMAGE' --prompt 'What is this?' --cpu --max-length 8" || true
}

# Print summary
print_summary() {
    log_section "Test Summary"

    echo ""
    echo -e "  ${GREEN}Passed:${NC}  $PASSED"
    echo -e "  ${RED}Failed:${NC}  $FAILED"
    echo -e "  ${YELLOW}Skipped:${NC} $SKIPPED"
    echo ""

    if [ $FAILED -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
        return 0
    else
        echo -e "${RED}Some tests failed.${NC}"
        return 1
    fi
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║        Qwen2.5-VL CLI Test Suite                         ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""

    if [ "$QUICK_MODE" = true ]; then
        echo -e "${YELLOW}Running in quick mode (max_length=$MAX_LENGTH)${NC}"
    fi
    if [ "$SKIP_DOWNLOAD" = true ]; then
        echo -e "${YELLOW}Skipping tests that require model download${NC}"
    fi

    check_prerequisites

    # Run all test sections
    test_help_and_version
    test_basic_inference
    test_prompt_variations
    test_sampling_strategies
    test_repeat_penalty
    test_streaming
    test_seed_reproducibility
    test_generation_length
    test_multi_image
    test_precision_options
    test_attention_options
    test_cpu_mode

    print_summary
}

main "$@"
