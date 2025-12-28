#!/bin/bash
# Test script for Parakeet ASR transcriptions across all model variants
# Compares WAV file transcriptions against expected TXT files
# Tests: TDT v2, TDT v3, RNN-T 1.1B, CTC 1.1B

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEXT_DIR="${SCRIPT_DIR}/text"
CANDLE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RESULTS_FILE="${SCRIPT_DIR}/candle_results.json"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Model variants (must match clap enum values)
MODEL_VARIANTS=("tdt-v2" "tdt-v3" "rnnt1b" "ctc1b")
MODEL_DESCRIPTIONS=(
    "TDT v2 (600M, English)"
    "TDT v3 (600M, 25 langs)"
    "RNN-T 1.1B (English)"
    "CTC 1.1B (English)"
)

# Parse arguments
MODELS_TO_TEST=()
COMPARE_NEMO=false
SKIP_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODELS_TO_TEST+=("$2")
            shift 2
            ;;
        --all)
            MODELS_TO_TEST=("${MODEL_VARIANTS[@]}")
            shift
            ;;
        --compare-nemo)
            COMPARE_NEMO=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model VARIANT   Test specific model variant (can be repeated)"
            echo "                    Variants: tdt-v2, tdt-v3, rnnt1b, ctc1b"
            echo "  --all             Test all model variants (default if no --model)"
            echo "  --compare-nemo    Compare results with NeMo (requires nemo_results.json)"
            echo "  --skip-build      Skip cargo build step"
            echo "  --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                        # Test default model (tdt-v3)"
            echo "  $0 --all                  # Test all models"
            echo "  $0 --model tdt-v2         # Test only TDT v2"
            echo "  $0 --all --compare-nemo   # Test all and compare with NeMo"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default to tdt-v3 if no models specified
if [[ ${#MODELS_TO_TEST[@]} -eq 0 ]]; then
    MODELS_TO_TEST=("tdt-v3")
fi

echo "=============================================="
echo "  Parakeet ASR Multi-Model Test Suite"
echo "=============================================="
echo ""
echo "Text directory: ${TEXT_DIR}"
echo "Candle root: ${CANDLE_ROOT}"
echo "Models to test: ${MODELS_TO_TEST[*]}"
echo ""

# Build the example first
if [[ "${SKIP_BUILD}" == "false" ]]; then
    echo "Building parakeet example..."
    cd "${CANDLE_ROOT}"
    cargo build --example parakeet --release --features parakeet 2>&1 | tail -5
    echo ""
fi

# Initialize results JSON
echo '{' > "${RESULTS_FILE}"
echo '  "tool": "candle",' >> "${RESULTS_FILE}"
echo '  "models": {' >> "${RESULTS_FILE}"

# Track overall statistics
declare -A MODEL_PASS_COUNT
declare -A MODEL_FAIL_COUNT
declare -A MODEL_TOTAL_COUNT

FIRST_MODEL=true

# Test each model variant
for i in "${!MODELS_TO_TEST[@]}"; do
    MODEL="${MODELS_TO_TEST[$i]}"

    # Find description
    DESC=""
    for j in "${!MODEL_VARIANTS[@]}"; do
        if [[ "${MODEL_VARIANTS[$j]}" == "${MODEL}" ]]; then
            DESC="${MODEL_DESCRIPTIONS[$j]}"
            break
        fi
    done

    echo ""
    echo -e "${CYAN}==============================================  ${NC}"
    echo -e "${CYAN}  Testing: ${MODEL} - ${DESC}${NC}"
    echo -e "${CYAN}==============================================${NC}"
    echo ""

    MODEL_PASS_COUNT["${MODEL}"]=0
    MODEL_FAIL_COUNT["${MODEL}"]=0
    MODEL_TOTAL_COUNT["${MODEL}"]=0

    # Add model to JSON
    if [[ "${FIRST_MODEL}" == "true" ]]; then
        FIRST_MODEL=false
    else
        echo ',' >> "${RESULTS_FILE}"
    fi
    echo "    \"${MODEL}\": {" >> "${RESULTS_FILE}"
    echo "      \"transcriptions\": {" >> "${RESULTS_FILE}"

    FIRST_FILE=true

    # Loop through all WAV files in the text directory
    for wav_file in "${TEXT_DIR}"/*.wav; do
        if [[ ! -f "${wav_file}" ]]; then
            continue
        fi

        basename=$(basename "${wav_file}" .wav)
        txt_file="${TEXT_DIR}/${basename}.txt"

        if [[ ! -f "${txt_file}" ]]; then
            echo -e "${YELLOW}SKIP${NC}: ${basename} - no matching .txt file"
            continue
        fi

        MODEL_TOTAL_COUNT["${MODEL}"]=$((MODEL_TOTAL_COUNT["${MODEL}"] + 1))

        echo "----------------------------------------------"
        echo "Testing: ${basename}"
        echo "----------------------------------------------"

        # Get expected transcription
        expected=$(cat "${txt_file}" | tr -d '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

        # Run parakeet with the specific model variant and capture output
        output=$(cargo run --example parakeet --release --features parakeet -- \
            --input "${wav_file}" \
            --model-variant "${MODEL}" 2>&1) || {
            echo -e "${RED}ERROR${NC}: Model failed to run"
            MODEL_FAIL_COUNT["${MODEL}"]=$((MODEL_FAIL_COUNT["${MODEL}"] + 1))
            continue
        }

        # Extract transcription (between --- Transcription --- and -----)
        actual=$(echo "${output}" | sed -n '/--- Transcription ---/,/---------------------/p' | grep -v "^---" | tr -d '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

        echo "Expected: ${expected}"
        echo "Actual:   ${actual}"
        echo ""

        # Compare (case-insensitive, normalize whitespace)
        expected_norm=$(echo "${expected}" | tr '[:upper:]' '[:lower:]' | tr -s ' ')
        actual_norm=$(echo "${actual}" | tr '[:upper:]' '[:lower:]' | tr -s ' ')

        # Calculate word accuracy
        expected_words=$(echo "${expected_norm}" | tr ' ' '\n' | sort -u)
        actual_words=$(echo "${actual_norm}" | tr ' ' '\n' | sort -u)
        common=$(comm -12 <(echo "${expected_words}") <(echo "${actual_words}") | wc -l)
        total_expected=$(echo "${expected_norm}" | wc -w)

        if [[ ${total_expected} -gt 0 ]]; then
            accuracy=$((common * 100 / total_expected))
        else
            accuracy=0
        fi

        if [[ "${expected_norm}" == "${actual_norm}" ]]; then
            match="EXACT"
            echo -e "Result: ${GREEN}PASS${NC} (exact match)"
            MODEL_PASS_COUNT["${MODEL}"]=$((MODEL_PASS_COUNT["${MODEL}"] + 1))
        elif [[ ${accuracy} -ge 90 ]]; then
            match="PARTIAL"
            echo -e "Result: ${YELLOW}PARTIAL${NC} (${accuracy}% word match)"
            MODEL_PASS_COUNT["${MODEL}"]=$((MODEL_PASS_COUNT["${MODEL}"] + 1))
        else
            match="DIFFERENT"
            echo -e "Result: ${RED}FAIL${NC} (${accuracy}% word match)"
            MODEL_FAIL_COUNT["${MODEL}"]=$((MODEL_FAIL_COUNT["${MODEL}"] + 1))
        fi
        echo ""

        # Add to JSON
        if [[ "${FIRST_FILE}" == "true" ]]; then
            FIRST_FILE=false
        else
            echo ',' >> "${RESULTS_FILE}"
        fi

        # Escape quotes in strings for JSON
        expected_json=$(echo "${expected}" | sed 's/"/\\"/g')
        actual_json=$(echo "${actual}" | sed 's/"/\\"/g')

        cat >> "${RESULTS_FILE}" << EOF
        "${basename}": {
          "expected": "${expected_json}",
          "transcription": "${actual_json}",
          "match": "${match}",
          "word_accuracy": ${accuracy}
        }
EOF
    done

    # Close model JSON
    echo '' >> "${RESULTS_FILE}"
    echo '      }' >> "${RESULTS_FILE}"
    echo '    }' >> "${RESULTS_FILE}"
done

# Close JSON
echo '  }' >> "${RESULTS_FILE}"
echo '}' >> "${RESULTS_FILE}"

# Print summary
echo ""
echo "=============================================="
echo "  Summary: Candle Parakeet Results"
echo "=============================================="
echo ""

printf "${BLUE}%-12s${NC} ${GREEN}%-8s${NC} ${RED}%-8s${NC} %-8s\n" "Model" "Passed" "Failed" "Total"
echo "----------------------------------------"

OVERALL_PASS=0
OVERALL_FAIL=0
OVERALL_TOTAL=0

for MODEL in "${MODELS_TO_TEST[@]}"; do
    pass=${MODEL_PASS_COUNT["${MODEL}"]}
    fail=${MODEL_FAIL_COUNT["${MODEL}"]}
    total=${MODEL_TOTAL_COUNT["${MODEL}"]}

    OVERALL_PASS=$((OVERALL_PASS + pass))
    OVERALL_FAIL=$((OVERALL_FAIL + fail))
    OVERALL_TOTAL=$((OVERALL_TOTAL + total))

    if [[ ${fail} -eq 0 ]]; then
        status="${GREEN}OK${NC}"
    else
        status="${RED}FAIL${NC}"
    fi

    printf "%-12s ${GREEN}%-8s${NC} ${RED}%-8s${NC} %-8s %b\n" "${MODEL}" "${pass}" "${fail}" "${total}" "${status}"
done

echo "----------------------------------------"
printf "%-12s ${GREEN}%-8s${NC} ${RED}%-8s${NC} %-8s\n" "TOTAL" "${OVERALL_PASS}" "${OVERALL_FAIL}" "${OVERALL_TOTAL}"
echo ""

# Compare with NeMo if requested
if [[ "${COMPARE_NEMO}" == "true" ]]; then
    NEMO_FILE="${SCRIPT_DIR}/nemo_results.json"

    if [[ ! -f "${NEMO_FILE}" ]]; then
        echo -e "${YELLOW}Warning: NeMo results file not found: ${NEMO_FILE}${NC}"
        echo "Run test_official_nemo.py first to generate NeMo results."
    else
        echo ""
        echo "=============================================="
        echo "  Comparison: Candle vs NeMo"
        echo "=============================================="
        echo ""

        # Use Python for JSON comparison with detailed diff
        python3 << 'PYTHON_EOF'
import json
from pathlib import Path

def word_error_rate(ref_words, hyp_words):
    """Calculate WER using dynamic programming (Levenshtein distance)."""
    r, h = len(ref_words), len(hyp_words)
    d = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1):
        d[i][0] = i
    for j in range(h + 1):
        d[0][j] = j
    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    return d[r][h] / max(r, 1) if r > 0 else (1.0 if h > 0 else 0.0)

def find_first_diff(a, b):
    """Find first differing position and return context."""
    words_a = a.split()
    words_b = b.split()
    for i, (wa, wb) in enumerate(zip(words_a, words_b)):
        if wa != wb:
            return i, wa, wb
    if len(words_a) != len(words_b):
        return min(len(words_a), len(words_b)), "(end)", "(end)"
    return None

script_dir = Path(".")
candle_file = script_dir / "candle_results.json"
nemo_file = script_dir / "nemo_results.json"

with open(candle_file) as f:
    candle_data = json.load(f)
with open(nemo_file) as f:
    nemo_data = json.load(f)

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
NC = "\033[0m"

exact_matches = 0
case_diffs = 0
word_diffs = 0
total = 0
differences = []

print(f"\n{'Model':<12} {'File':<22} {'WER':<8} {'Status':<12} {'Detail'}")
print("-" * 90)

for model_name, candle_model in candle_data.get("models", {}).items():
    nemo_model = nemo_data.get("models", {}).get(model_name, {})
    candle_trans = candle_model.get("transcriptions", {})
    nemo_trans = nemo_model.get("transcriptions", {})

    for filename in sorted(candle_trans.keys()):
        c_text = candle_trans[filename].get("transcription", "").strip()
        n_text = nemo_trans.get(filename, {}).get("transcription", "").strip()
        total += 1

        c_lower = c_text.lower()
        n_lower = n_text.lower()

        c_words = c_lower.split()
        n_words = n_lower.split()
        wer = word_error_rate(n_words, c_words)
        wer_pct = f"{wer*100:.1f}%"

        if c_text == n_text:
            status = f"{GREEN}EXACT{NC}"
            detail = ""
            exact_matches += 1
        elif c_lower == n_lower:
            status = f"{YELLOW}CASE{NC}"
            diff_info = find_first_diff(c_text, n_text)
            if diff_info:
                pos, ca, ne = diff_info
                detail = f"word {pos}: '{ca}' vs '{ne}'"
            else:
                detail = "case difference"
            case_diffs += 1
            differences.append((model_name, filename, "CASE", c_text[:60], n_text[:60]))
        else:
            status = f"{RED}WORDS{NC}"
            diff_info = find_first_diff(c_lower, n_lower)
            if diff_info:
                pos, ca, ne = diff_info
                detail = f"word {pos}: '{ca}' vs '{ne}'"
            else:
                detail = "length differs"
            word_diffs += 1
            differences.append((model_name, filename, "WORDS", c_text[:60], n_text[:60]))

        print(f"{model_name:<12} {filename:<22} {wer_pct:<8} {status:<20} {detail}")

print("-" * 90)
print(f"\n{CYAN}Summary:{NC}")
print(f"  Exact matches: {GREEN}{exact_matches}/{total}{NC}")
print(f"  Case only:     {YELLOW}{case_diffs}/{total}{NC} (same words, different capitalization)")
print(f"  Word diff:     {RED}{word_diffs}/{total}{NC} (different words)")

if differences:
    print(f"\n{CYAN}Differences Detail:{NC}")
    for model, fname, dtype, candle, nemo in differences:
        print(f"\n  {model}/{fname} ({dtype}):")
        print(f"    Candle: {candle}{'...' if len(candle)==60 else ''}")
        print(f"    NeMo:   {nemo}{'...' if len(nemo)==60 else ''}")
PYTHON_EOF
    fi
fi

echo ""
echo "Results saved to: ${RESULTS_FILE}"
echo ""

if [[ ${OVERALL_FAIL} -eq 0 ]]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi
