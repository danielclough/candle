#!/bin/bash
# Test Candle against NeMo results from test_willinsky.py
# Compares Candle transcriptions with NeMo outputs stored in willinsky_results.json

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANDLE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
AUDIO_FILE="${SCRIPT_DIR}/willinsky_150s.wav"
NEMO_RESULTS="${SCRIPT_DIR}/willinsky_results.json"
CANDLE_RESULTS="${SCRIPT_DIR}/willinsky_candle_results.json"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Check prerequisites
if [[ ! -f "${AUDIO_FILE}" ]]; then
    echo -e "${RED}ERROR: Audio file not found: ${AUDIO_FILE}${NC}"
    echo "Convert the OGG file first:"
    echo "  ffmpeg -i 060123-John.Willinsky-The.Economics.of.Knowledge.as.a.Public.Good.ogg -ar 16000 -ac 1 willinsky_150s.wav"
    exit 1
fi

if [[ ! -f "${NEMO_RESULTS}" ]]; then
    echo -e "${YELLOW}WARNING: NeMo results not found: ${NEMO_RESULTS}${NC}"
    echo "Run test_willinsky.py first to generate NeMo results."
    echo "Continuing with Candle-only test..."
    HAS_NEMO=false
else
    HAS_NEMO=true
fi

echo "=============================================="
echo "  Candle Parakeet - Willinsky Test"
echo "=============================================="
echo "Audio: ${AUDIO_FILE}"
echo ""

# Model variants
declare -A MODELS
MODELS["tdt-v2"]="tdt-v2"
MODELS["tdt-v3"]="tdt-v3"
MODELS["rnnt-1b"]="rnnt1b"
MODELS["ctc-1b"]="ctc1b"

# Build first
echo "Building Candle parakeet..."
cd "${CANDLE_ROOT}"
cargo build --example parakeet --release --features parakeet 2>&1 | tail -3

# Initialize results JSON
echo '{' > "${CANDLE_RESULTS}"
echo '  "audio": "willinsky_150s.wav",' >> "${CANDLE_RESULTS}"
echo '  "models": {' >> "${CANDLE_RESULTS}"

FIRST_MODEL=true

for MODEL_NAME in "tdt-v2" "tdt-v3" "rnnt-1b" "ctc-1b"; do
    VARIANT="${MODELS[$MODEL_NAME]}"

    echo ""
    echo -e "${CYAN}=============================================${NC}"
    echo -e "${CYAN}  Model: ${MODEL_NAME} (${VARIANT})${NC}"
    echo -e "${CYAN}=============================================${NC}"

    # Run Candle
    echo "Running Candle..."
    START_TIME=$(date +%s.%N)

    OUTPUT=$(cargo run --example parakeet --release --features parakeet -- \
        --model-variant "${VARIANT}" \
        --input "${AUDIO_FILE}" 2>&1) || {
        echo -e "${RED}ERROR: Candle failed${NC}"
        continue
    }

    END_TIME=$(date +%s.%N)
    DURATION=$(echo "$END_TIME - $START_TIME" | bc)

    # Extract transcription
    CANDLE_TEXT=$(echo "${OUTPUT}" | sed -n '/--- Transcription ---/,/---------------------/p' | grep -v "^---" | tr '\n' ' ' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    CANDLE_WORDS=$(echo "${CANDLE_TEXT}" | wc -w)

    echo "  Candle: ${CANDLE_WORDS} words in ${DURATION}s"
    echo "  First 100 chars: ${CANDLE_TEXT:0:100}..."

    # Get NeMo result if available
    if [[ "${HAS_NEMO}" == "true" ]]; then
        NEMO_TEXT=$(python3 -c "
import json
with open('${NEMO_RESULTS}') as f:
    data = json.load(f)
print(data.get('${MODEL_NAME}', {}).get('nemo', ''))
" 2>/dev/null || echo "")
        NEMO_WORDS=$(echo "${NEMO_TEXT}" | wc -w)

        if [[ -n "${NEMO_TEXT}" && ! "${NEMO_TEXT}" =~ ^ERROR ]]; then
            echo "  NeMo:   ${NEMO_WORDS} words"

            # Calculate WER using Python
            WER=$(python3 -c "
def wer(ref, hyp):
    ref_words = ref.lower().split()
    hyp_words = hyp.lower().split()
    r, h = len(ref_words), len(hyp_words)
    if r == 0: return 1.0 if h > 0 else 0.0
    d = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1): d[i][0] = i
    for j in range(h + 1): d[0][j] = j
    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    return d[r][h] / r

nemo = '''${NEMO_TEXT}'''
candle = '''${CANDLE_TEXT}'''
print(f'{wer(nemo, candle)*100:.2f}')
" 2>/dev/null || echo "-1")

            if [[ "${WER}" == "0.00" ]]; then
                echo -e "  Match:  ${GREEN}EXACT${NC}"
                MATCH="EXACT"
            else
                echo -e "  Match:  ${YELLOW}WER=${WER}%${NC}"
                MATCH="WER=${WER}%"
            fi
        else
            MATCH="NO_NEMO"
            WER="-1"
        fi
    else
        MATCH="NO_NEMO"
        WER="-1"
        NEMO_TEXT=""
        NEMO_WORDS=0
    fi

    # Add to JSON
    if [[ "${FIRST_MODEL}" == "true" ]]; then
        FIRST_MODEL=false
    else
        echo ',' >> "${CANDLE_RESULTS}"
    fi

    # Escape quotes for JSON
    CANDLE_JSON=$(echo "${CANDLE_TEXT}" | sed 's/"/\\"/g' | head -c 5000)

    cat >> "${CANDLE_RESULTS}" << EOF
    "${MODEL_NAME}": {
      "candle_words": ${CANDLE_WORDS},
      "candle_first_100": "${CANDLE_JSON:0:100}",
      "duration_s": ${DURATION},
      "match": "${MATCH}",
      "wer": ${WER}
    }
EOF

done

# Close JSON
echo '' >> "${CANDLE_RESULTS}"
echo '  }' >> "${CANDLE_RESULTS}"
echo '}' >> "${CANDLE_RESULTS}"

# Summary
echo ""
echo "=============================================="
echo "  SUMMARY"
echo "=============================================="
echo ""
printf "${CYAN}%-12s${NC} %-14s %-12s %-15s\n" "Model" "Candle Words" "Duration" "Match"
echo "------------------------------------------------------"

python3 << 'PYTHON_EOF'
import json
with open("willinsky_candle_results.json") as f:
    data = json.load(f)

for model, info in data.get("models", {}).items():
    words = info.get("candle_words", 0)
    duration = info.get("duration_s", 0)
    match = info.get("match", "?")

    # Color based on match
    if match == "EXACT":
        color = "\033[32m"  # green
    elif match.startswith("WER"):
        wer = float(match.split("=")[1].replace("%", ""))
        if wer < 1:
            color = "\033[32m"  # green
        elif wer < 5:
            color = "\033[33m"  # yellow
        else:
            color = "\033[31m"  # red
    else:
        color = "\033[0m"

    print(f"{model:<12} {words:<14} {duration:<12.1f}s {color}{match}\033[0m")
PYTHON_EOF

echo ""
echo "Results saved to: ${CANDLE_RESULTS}"