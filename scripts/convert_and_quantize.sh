#!/usr/bin/env bash
# Convert a HuggingFace model to GGUF and quantize to all vram-bench levels.
#
# This script handles the full pipeline:
#   1. Download a HuggingFace model (if not already local)
#   2. Convert to GGUF BF16 (lossless)
#   3. Quantize to all standard precision levels
#
# Prerequisites:
#   - llama.cpp built locally (cmake -B build && cmake --build build)
#   - Python with huggingface-cli installed (pip install huggingface-hub)
#   - The convert_hf_to_gguf.py script from llama.cpp
#
# Usage:
#   bash scripts/convert_and_quantize.sh HuggingFaceTB/SmolLM2-1.7B-Instruct
#   bash scripts/convert_and_quantize.sh google/gemma-3-1b-it --q4-only
#   bash scripts/convert_and_quantize.sh ./local/model/path --skip-download
#
# Output goes to models/ with standard vram-bench naming:
#   models/{ModelName}-BF16.gguf
#   models/{ModelName}-Q8_0.gguf
#   models/{ModelName}-Q4_K_M.gguf
#   ...etc

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────

MODELS_DIR="${MODELS_DIR:-models}"
HF_CACHE_DIR="${HF_CACHE_DIR:-models/hf-staging}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-}"

# Try to find llama.cpp tools
find_tool() {
    local name="$1"
    # Check explicit path first
    if [[ -n "$LLAMA_CPP_DIR" ]] && [[ -x "$LLAMA_CPP_DIR/build/bin/$name" ]]; then
        echo "$LLAMA_CPP_DIR/build/bin/$name"
        return
    fi
    # Check PATH
    if command -v "$name" &>/dev/null; then
        command -v "$name"
        return
    fi
    # Check common build locations
    for dir in llama.cpp llama-cpp-python/vendor/llama.cpp ../llama.cpp; do
        if [[ -x "$dir/build/bin/$name" ]]; then
            echo "$dir/build/bin/$name"
            return
        fi
    done
    return 1
}

find_convert_script() {
    # Check explicit path first
    if [[ -n "$LLAMA_CPP_DIR" ]] && [[ -f "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" ]]; then
        echo "$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
        return
    fi
    # Check common locations
    for dir in llama.cpp llama-cpp-python/vendor/llama.cpp ../llama.cpp; do
        if [[ -f "$dir/convert_hf_to_gguf.py" ]]; then
            echo "$dir/convert_hf_to_gguf.py"
            return
        fi
    done
    return 1
}

# ── Parse arguments ──────────────────────────────────────────────────

MODEL_ID=""
Q4_ONLY=false
SKIP_DOWNLOAD=false
UPLOAD=false
HF_ORG="${HF_ORG:-vram-bench}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --q4-only)       Q4_ONLY=true; shift ;;
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        --upload)        UPLOAD=true; shift ;;
        --hf-org)        HF_ORG="$2"; shift 2 ;;
        --llama-cpp)     LLAMA_CPP_DIR="$2"; shift 2 ;;
        -*)              echo "Unknown option: $1"; exit 1 ;;
        *)               MODEL_ID="$1"; shift ;;
    esac
done

if [[ -z "$MODEL_ID" ]]; then
    echo "Usage: $0 <model-id-or-path> [options]"
    echo ""
    echo "Arguments:"
    echo "  model-id-or-path    HuggingFace model ID (e.g. HuggingFaceTB/SmolLM2-1.7B-Instruct)"
    echo "                      or path to a local HuggingFace model directory"
    echo ""
    echo "Options:"
    echo "  --q4-only           Only produce Q4_K_M (skip other quant levels)"
    echo "  --skip-download     Model is already local; skip HuggingFace download"
    echo "  --upload            Upload produced GGUFs to HuggingFace (requires hf auth)"
    echo "  --hf-org NAME       HuggingFace org for uploads (default: vram-bench)"
    echo "  --llama-cpp PATH    Path to llama.cpp directory"
    exit 1
fi

# ── Validate tools ───────────────────────────────────────────────────

CONVERT_SCRIPT=$(find_convert_script) || {
    echo "ERROR: Cannot find convert_hf_to_gguf.py"
    echo "  Set LLAMA_CPP_DIR or clone llama.cpp alongside this repo:"
    echo "  git clone https://github.com/ggml-org/llama.cpp"
    exit 1
}

QUANTIZE_BIN=$(find_tool "llama-quantize") || {
    echo "ERROR: Cannot find llama-quantize binary"
    echo "  Build llama.cpp first:"
    echo "  cd llama.cpp && cmake -B build && cmake --build build --config Release"
    exit 1
}

echo "Using convert script: $CONVERT_SCRIPT"
echo "Using quantize binary: $QUANTIZE_BIN"

# ── Derive model name ────────────────────────────────────────────────

if [[ -d "$MODEL_ID" ]]; then
    # Local path — use directory name
    MODEL_PATH="$MODEL_ID"
    MODEL_NAME=$(basename "$MODEL_PATH")
else
    # HuggingFace ID — derive name from the repo
    MODEL_NAME=$(echo "$MODEL_ID" | sed 's|.*/||')
    MODEL_PATH="$HF_CACHE_DIR/$MODEL_NAME"
fi

mkdir -p "$MODELS_DIR"

echo ""
echo "========================================"
echo "vram-bench: Convert & Quantize"
echo "========================================"
echo "Model:  $MODEL_ID"
echo "Name:   $MODEL_NAME"
echo "Output: $MODELS_DIR/"
echo "========================================"
echo ""

# ── Step 1: Download from HuggingFace ────────────────────────────────

if $SKIP_DOWNLOAD; then
    echo "── Skipping download (--skip-download) ──"
    if [[ ! -d "$MODEL_PATH" ]]; then
        echo "ERROR: Model path does not exist: $MODEL_PATH"
        exit 1
    fi
else
    echo "── Step 1: Download from HuggingFace ──"
    if [[ -d "$MODEL_PATH" ]] && [[ -f "$MODEL_PATH/config.json" ]]; then
        echo "  SKIP (exists): $MODEL_PATH"
    else
        mkdir -p "$HF_CACHE_DIR"
        echo "  Downloading $MODEL_ID ..."
        huggingface-cli download "$MODEL_ID" \
            --local-dir "$MODEL_PATH" \
            --local-dir-use-symlinks False
    fi
fi
echo ""

# ── Step 2: Convert to GGUF BF16 ────────────────────────────────────

BF16_FILE="$MODELS_DIR/${MODEL_NAME}-BF16.gguf"

echo "── Step 2: Convert to GGUF BF16 ──"
if [[ -f "$BF16_FILE" ]]; then
    echo "  SKIP (exists): $BF16_FILE"
else
    echo "  Converting $MODEL_PATH -> $BF16_FILE ..."
    python3 "$CONVERT_SCRIPT" "$MODEL_PATH" \
        --outfile "$BF16_FILE" \
        --outtype bf16
    echo "  Done: $(du -h "$BF16_FILE" | cut -f1)"
fi
echo ""

# ── Step 3: Quantize ─────────────────────────────────────────────────

if $Q4_ONLY; then
    QUANT_LEVELS=("Q4_K_M")
else
    QUANT_LEVELS=("Q8_0" "Q6_K" "Q5_K_M" "Q4_K_M" "Q4_0" "Q3_K_M" "Q2_K")
fi

echo "── Step 3: Quantize to ${#QUANT_LEVELS[@]} precision levels ──"
for q in "${QUANT_LEVELS[@]}"; do
    QUANT_FILE="$MODELS_DIR/${MODEL_NAME}-${q}.gguf"
    if [[ -f "$QUANT_FILE" ]]; then
        echo "  SKIP (exists): ${MODEL_NAME}-${q}.gguf"
    else
        echo "  Quantizing: ${MODEL_NAME}-${q}.gguf ..."
        "$QUANTIZE_BIN" "$BF16_FILE" "$QUANT_FILE" "$q"
        echo "  Done: $(du -h "$QUANT_FILE" | cut -f1)"
    fi
done
echo ""

# ── Step 4 (optional): Upload to HuggingFace ─────────────────────────

if $UPLOAD; then
    REPO_NAME="${HF_ORG}/${MODEL_NAME}-GGUF"
    echo "── Step 4: Upload to HuggingFace ($REPO_NAME) ──"

    # Create repo if it doesn't exist
    huggingface-cli repo create "$REPO_NAME" --type model 2>/dev/null || true

    # Upload all GGUF files for this model
    for f in "$MODELS_DIR/${MODEL_NAME}"-*.gguf; do
        if [[ -f "$f" ]]; then
            fname=$(basename "$f")
            echo "  Uploading: $fname"
            huggingface-cli upload "$REPO_NAME" "$f" "$fname"
        fi
    done
    echo "  Done: https://huggingface.co/$REPO_NAME"
    echo ""
fi

# ── Summary ──────────────────────────────────────────────────────────

echo "========================================"
echo "COMPLETE"
echo "========================================"
echo "Produced files:"
for f in "$MODELS_DIR/${MODEL_NAME}"-*.gguf; do
    if [[ -f "$f" ]]; then
        echo "  $(du -h "$f" | cut -f1)  $(basename "$f")"
    fi
done
echo ""

# Clean up staging directory hint
if [[ -d "$MODEL_PATH" ]] && [[ "$MODEL_PATH" == "$HF_CACHE_DIR"/* ]]; then
    SIZE=$(du -sh "$MODEL_PATH" | cut -f1)
    echo "Tip: The HuggingFace download ($SIZE) is cached at $MODEL_PATH"
    echo "     Delete it after conversion to reclaim disk space:"
    echo "     rm -rf $MODEL_PATH"
fi
