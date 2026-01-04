#!/bin/bash
# Test script for Qwen2.5-VL Video CLI flags
# Usage: ./test_video.sh [--quick] [--skip-model-download]
#
# This script tests video-specific CLI flags for Qwen2.5-VL
# Uses Big Buck Bunny (CC-BY 3.0) - an open source animated film by Blender Foundation

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
TEST_VIDEO="$SCRIPT_DIR/bunny_10s.mp4"
TEST_VIDEO_URL="https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"
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

# Download test video if not present
download_test_video() {
    if [ -f "$TEST_VIDEO" ]; then
        log_pass "Test video already exists: $TEST_VIDEO"
        return 0
    fi

    log_test "Downloading test video..."
    echo "  URL: $TEST_VIDEO_URL"

    if command -v curl &> /dev/null; then
        if curl -L -o "$TEST_VIDEO" "$TEST_VIDEO_URL" 2>/dev/null; then
            log_pass "Downloaded test video"
            return 0
        fi
    elif command -v wget &> /dev/null; then
        if wget -O "$TEST_VIDEO" "$TEST_VIDEO_URL" 2>/dev/null; then
            log_pass "Downloaded test video"
            return 0
        fi
    fi

    log_fail "Could not download test video. Please download manually:"
    echo "  curl -L -o '$TEST_VIDEO' '$TEST_VIDEO_URL'"
    return 1
}

# Check prerequisites
check_prerequisites() {
    log_section "Checking Prerequisites"

    # Check ffmpeg is installed (required for video processing)
    if ! command -v ffmpeg &> /dev/null; then
        echo -e "${RED}Error: ffmpeg not found. Install with:${NC}"
        echo "  macOS:  brew install ffmpeg"
        echo "  Ubuntu: sudo apt install ffmpeg"
        exit 1
    fi
    log_pass "ffmpeg found: $(ffmpeg -version 2>&1 | head -1)"

    # Download test video
    download_test_video || exit 1

    # Check test video exists and is valid
    if [ ! -f "$TEST_VIDEO" ]; then
        echo -e "${RED}Error: Test video not found at $TEST_VIDEO${NC}"
        exit 1
    fi

    # Verify video is valid using ffprobe
    if command -v ffprobe &> /dev/null; then
        if ffprobe -v error -select_streams v:0 -show_entries stream=duration -of csv=p=0 "$TEST_VIDEO" &>/dev/null; then
            local duration=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$TEST_VIDEO" 2>/dev/null)
            log_pass "Test video verified: $TEST_VIDEO (${duration}s)"
        else
            echo -e "${RED}Error: Test video appears to be invalid${NC}"
            rm -f "$TEST_VIDEO"
            exit 1
        fi
    else
        log_pass "Test video found: $TEST_VIDEO"
    fi

    # Check we're in the right directory
    if [ ! -f "$REPO_ROOT/Cargo.toml" ]; then
        echo -e "${RED}Error: Cannot find Cargo.toml in $REPO_ROOT${NC}"
        exit 1
    fi
    log_pass "Repository root found: $REPO_ROOT"

    # Build the example first
    log_test "Building qwen2_5_vl example..."
    cd "$REPO_ROOT"
    if cargo build --features metal,accelerate --example qwen2_5_vl --release 2>&1 | tail -5; then
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

test_basic_video() {
    log_section "Basic Video Inference"

    if [ "$SKIP_DOWNLOAD" = true ]; then
        log_skip "Basic video inference (model download required)"
        return
    fi

    run_test "Basic video inference" \
        "$BASE_CMD --video '$TEST_VIDEO' --prompt 'What happens in this video?' --max-length $MAX_LENGTH" || true
}

test_video_fps() {
    log_section "Video FPS Settings"

    if [ "$SKIP_DOWNLOAD" = true ]; then
        log_skip "Video FPS settings (model download required)"
        return
    fi

    run_test "Video with 1 FPS" \
        "$BASE_CMD --video '$TEST_VIDEO' --video-fps 1.0 --prompt 'Describe this video' --max-length $MAX_LENGTH" || true

    run_test "Video with 2 FPS (default)" \
        "$BASE_CMD --video '$TEST_VIDEO' --video-fps 2.0 --prompt 'Describe this video' --max-length $MAX_LENGTH" || true

    run_test "Video with 4 FPS" \
        "$BASE_CMD --video '$TEST_VIDEO' --video-fps 4.0 --prompt 'Describe this video' --max-length $MAX_LENGTH" || true
}

test_video_max_frames() {
    log_section "Video Max Frames"

    if [ "$SKIP_DOWNLOAD" = true ]; then
        log_skip "Video max frames (model download required)"
        return
    fi

    run_test "Max 8 frames" \
        "$BASE_CMD --video '$TEST_VIDEO' --max-frames 8 --prompt 'Describe this video' --max-length $MAX_LENGTH" || true

    run_test "Max 16 frames" \
        "$BASE_CMD --video '$TEST_VIDEO' --max-frames 16 --prompt 'Describe this video' --max-length $MAX_LENGTH" || true

    run_test "Combined: 4 FPS with max 10 frames" \
        "$BASE_CMD --video '$TEST_VIDEO' --video-fps 4.0 --max-frames 10 --prompt 'Describe this video' --max-length $MAX_LENGTH" || true
}

test_video_streaming() {
    log_section "Video Streaming Output"

    if [ "$SKIP_DOWNLOAD" = true ]; then
        log_skip "Video streaming output (model download required)"
        return
    fi

    run_test "Streaming mode with video" \
        "$BASE_CMD --video '$TEST_VIDEO' --prompt 'Describe this' --stream --max-length $MAX_LENGTH" || true

    run_test "Streaming with temperature" \
        "$BASE_CMD --video '$TEST_VIDEO' --prompt 'Describe this' --stream --temperature 0.7 --max-length $MAX_LENGTH" || true
}

test_video_sampling() {
    log_section "Video Sampling Strategies"

    if [ "$SKIP_DOWNLOAD" = true ]; then
        log_skip "Video sampling strategies (model download required)"
        return
    fi

    run_test "Video with temperature 0.5" \
        "$BASE_CMD --video '$TEST_VIDEO' --prompt 'Describe this video' --temperature 0.5 --max-length $MAX_LENGTH" || true

    run_test "Video with top-p sampling" \
        "$BASE_CMD --video '$TEST_VIDEO' --prompt 'Describe this video' --temperature 0.8 --top-p 0.9 --max-length $MAX_LENGTH" || true

    run_test "Video with top-k sampling" \
        "$BASE_CMD --video '$TEST_VIDEO' --prompt 'Describe this video' --temperature 0.8 --top-k 50 --max-length $MAX_LENGTH" || true
}

test_video_repeat_penalty() {
    log_section "Video Repeat Penalty"

    if [ "$SKIP_DOWNLOAD" = true ]; then
        log_skip "Video repeat penalty (model download required)"
        return
    fi

    run_test "Video with repeat penalty 1.1" \
        "$BASE_CMD --video '$TEST_VIDEO' --prompt 'Describe what happens in this video' --repeat-penalty 1.1 --max-length $MAX_LENGTH" || true

    run_test "Video with repeat penalty and custom window" \
        "$BASE_CMD --video '$TEST_VIDEO' --prompt 'Describe what happens in this video' --repeat-penalty 1.2 --repeat-last-n 32 --max-length $MAX_LENGTH" || true
}

test_video_prompts() {
    log_section "Video Prompt Variations"

    if [ "$SKIP_DOWNLOAD" = true ]; then
        log_skip "Video prompt variations (model download required)"
        return
    fi

    run_test "Action question" \
        "$BASE_CMD --video '$TEST_VIDEO' --prompt 'What actions are happening in this video?' --max-length $MAX_LENGTH" || true

    run_test "Character question" \
        "$BASE_CMD --video '$TEST_VIDEO' --prompt 'Describe the characters or objects in this video' --max-length $MAX_LENGTH" || true

    run_test "Timeline question" \
        "$BASE_CMD --video '$TEST_VIDEO' --prompt 'What happens at the beginning, middle, and end of this video?' --max-length $MAX_LENGTH" || true
}

test_video_precision() {
    log_section "Video Precision Options"

    if [ "$SKIP_DOWNLOAD" = true ]; then
        log_skip "Video precision options (model download required)"
        return
    fi

    run_test "Video with BFloat16" \
        "$BASE_CMD --video '$TEST_VIDEO' --prompt 'Describe this video' --bf16 --max-length $MAX_LENGTH" || true
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
    echo -e "${BLUE}║        Qwen2.5-VL Video Test Suite                       ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Test video: Big Buck Bunny (CC-BY 3.0) by Blender Foundation"
    echo ""

    if [ "$QUICK_MODE" = true ]; then
        echo -e "${YELLOW}Running in quick mode (max_length=$MAX_LENGTH)${NC}"
    fi
    if [ "$SKIP_DOWNLOAD" = true ]; then
        echo -e "${YELLOW}Skipping tests that require model download${NC}"
    fi

    check_prerequisites

    # Run all test sections
    test_basic_video
    test_video_fps
    test_video_max_frames
    test_video_streaming
    test_video_sampling
    test_video_repeat_penalty
    test_video_prompts
    test_video_precision

    print_summary
}

main "$@"