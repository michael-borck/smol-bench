#!/usr/bin/env bash
# Download vram-bench models in GGUF format at multiple precision levels.
#
# All models use GGUF format — including full-precision (BF16/F16) variants.
# GGUF BF16 is lossless relative to the original HuggingFace weights; it's
# the same precision in a different container format optimised for local inference.
#
# Designed to be run in a tmux session. Safe to restart — only downloads
# files that don't already exist locally (huggingface-cli skips existing).
#
# Usage:
#   bash scripts/download_models.sh              # download all precision levels
#   bash scripts/download_models.sh --q4-only    # just Q4_K_M (fast start)
#   bash scripts/download_models.sh --tier 4gb   # only models that fit in 4GB VRAM
#   bash scripts/download_models.sh --tier 8gb   # only models that fit in 8GB VRAM
#
# Estimated total size (all levels): ~250+ GB
# Estimated Q4_K_M only: ~25 GB

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-models}"
mkdir -p "$MODELS_DIR"

# Parse arguments
Q4_ONLY=false
TIER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --q4-only) Q4_ONLY=true; shift ;;
        --tier)    TIER="$2"; shift 2 ;;
        *)         echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Precision levels to download (order: smallest to largest)
if $Q4_ONLY; then
    QUANTS=("Q4_K_M")
else
    QUANTS=("Q2_K" "Q3_K_M" "Q4_0" "Q4_K_M" "Q5_K_M" "Q6_K" "Q8_0" "BF16")
fi

# Track stats
TOTAL=0
SKIPPED=0
DOWNLOADED=0
FAILED=0
FAILURES=""

download() {
    local repo="$1"
    local filename="$2"
    local local_name="${3:-$filename}"

    local dest="$MODELS_DIR/$local_name"

    TOTAL=$((TOTAL + 1))

    if [[ -f "$dest" ]]; then
        echo "  SKIP (exists): $local_name"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    echo "  Downloading: $repo -> $local_name"
    if huggingface-cli download "$repo" "$filename" \
        --local-dir "$MODELS_DIR" \
        --local-dir-use-symlinks False 2>&1; then

        # huggingface-cli downloads to the original filename; rename if needed
        if [[ "$filename" != "$local_name" ]] && [[ -f "$MODELS_DIR/$filename" ]]; then
            mv "$MODELS_DIR/$filename" "$dest"
        fi
        DOWNLOADED=$((DOWNLOADED + 1))
    else
        echo "  FAILED: $local_name"
        FAILED=$((FAILED + 1))
        FAILURES="$FAILURES\n  - $local_name ($repo)"
    fi
}

# ─────────────────────────────────────────────────────────────────────
# VRAM tier filtering
#
# Approximate Q4_K_M sizes determine tier membership:
#   1B model:   Q4=~0.6GB   BF16=~2GB
#   1.7B model: Q4=~1.1GB   BF16=~3.4GB
#   3B model:   Q4=~1.9GB   BF16=~6GB
#   4B model:   Q4=~2.5GB   BF16=~8GB
#   7B model:   Q4=~4.4GB   BF16=~14GB
#
# --tier filters by max parameter count that fits at Q4_K_M.
# BF16 variants of larger models may exceed the tier; this is expected
# — they serve as baselines, and the tier comparison focuses on what
# actually fits at each VRAM level.
# ─────────────────────────────────────────────────────────────────────

should_download() {
    local params_b="$1"
    case "$TIER" in
        4gb)  awk "BEGIN{exit !($params_b <= 4.0)}" ;;
        6gb)  awk "BEGIN{exit !($params_b <= 7.0)}" ;;
        8gb)  awk "BEGIN{exit !($params_b <= 7.0)}" ;;
        12gb) awk "BEGIN{exit !($params_b <= 14.0)}" ;;
        24gb) return 0 ;;
        "")   return 0 ;;
        *)    echo "Unknown tier: $TIER (use 4gb, 6gb, 8gb, 12gb, 24gb)"; exit 1 ;;
    esac
}

# ─────────────────────────────────────────────────────────────────────
# Model definitions
#
# Naming convention for local files: {model-name}-{QUANT}.gguf
# ─────────────────────────────────────────────────────────────────────

download_smollm2_135m() {
    should_download 0.135 || return 0
    echo "=== SmolLM2-135M-Instruct (0.135B) ==="
    local repo="bartowski/SmolLM2-135M-Instruct-GGUF"
    for q in "${QUANTS[@]}"; do
        if [[ "$q" == "BF16" ]]; then
            download "$repo" "SmolLM2-135M-Instruct-f16.gguf" "SmolLM2-135M-Instruct-BF16.gguf"
        else
            download "$repo" "SmolLM2-135M-Instruct-${q}.gguf" "SmolLM2-135M-Instruct-${q}.gguf"
        fi
    done
    echo
}

download_smollm2_360m() {
    should_download 0.36 || return 0
    echo "=== SmolLM2-360M-Instruct (0.36B) ==="
    local repo="bartowski/SmolLM2-360M-Instruct-GGUF"
    for q in "${QUANTS[@]}"; do
        if [[ "$q" == "BF16" ]]; then
            download "$repo" "SmolLM2-360M-Instruct-f16.gguf" "SmolLM2-360M-Instruct-BF16.gguf"
        else
            download "$repo" "SmolLM2-360M-Instruct-${q}.gguf" "SmolLM2-360M-Instruct-${q}.gguf"
        fi
    done
    echo
}

download_smollm2_1_7b() {
    should_download 1.7 || return 0
    echo "=== SmolLM2-1.7B-Instruct (1.7B) ==="
    local repo="bartowski/SmolLM2-1.7B-Instruct-GGUF"
    for q in "${QUANTS[@]}"; do
        if [[ "$q" == "BF16" ]]; then
            download "$repo" "SmolLM2-1.7B-Instruct-f16.gguf" "SmolLM2-1.7B-Instruct-BF16.gguf"
        else
            download "$repo" "SmolLM2-1.7B-Instruct-${q}.gguf" "SmolLM2-1.7B-Instruct-${q}.gguf"
        fi
    done
    echo
}

download_tinyllama() {
    should_download 1.1 || return 0
    echo "=== TinyLlama-1.1B-Chat (1.1B) ==="
    local repo="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    for q in "${QUANTS[@]}"; do
        local lq
        lq=$(echo "$q" | tr '[:upper:]' '[:lower:]')
        if [[ "$q" == "BF16" ]]; then
            download "$repo" "tinyllama-1.1b-chat-v1.0.fp16.gguf" "TinyLlama-1.1B-Chat-BF16.gguf"
        else
            download "$repo" "tinyllama-1.1b-chat-v1.0.${lq}.gguf" "TinyLlama-1.1B-Chat-${q}.gguf"
        fi
    done
    echo
}

download_gemma_1b() {
    should_download 1.0 || return 0
    echo "=== Gemma-3-1B-it (1B) ==="
    local repo="unsloth/gemma-3-1b-it-GGUF"
    for q in "${QUANTS[@]}"; do
        download "$repo" "gemma-3-1b-it-${q}.gguf" "gemma-3-1b-it-${q}.gguf"
    done
    echo
}

download_llama_1b() {
    should_download 1.0 || return 0
    echo "=== Llama-3.2-1B-Instruct (1B) ==="
    local repo="unsloth/Llama-3.2-1B-Instruct-GGUF"
    for q in "${QUANTS[@]}"; do
        download "$repo" "Llama-3.2-1B-Instruct-${q}.gguf" "Llama-3.2-1B-Instruct-${q}.gguf"
    done
    echo
}

download_deepseek_1_5b() {
    should_download 1.5 || return 0
    echo "=== DeepSeek-R1-Distill-Qwen-1.5B (1.5B) ==="
    local repo="unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF"
    for q in "${QUANTS[@]}"; do
        download "$repo" "DeepSeek-R1-Distill-Qwen-1.5B-${q}.gguf" "DeepSeek-R1-Distill-Qwen-1.5B-${q}.gguf"
    done
    echo
}

download_qwen3_1_7b() {
    should_download 1.7 || return 0
    echo "=== Qwen3-1.7B (1.7B) ==="
    local repo="unsloth/Qwen3-1.7B-GGUF"
    for q in "${QUANTS[@]}"; do
        download "$repo" "Qwen3-1.7B-${q}.gguf" "Qwen3-1.7B-${q}.gguf"
    done
    echo
}

download_ministral_3b() {
    should_download 3.0 || return 0
    echo "=== Ministral-3-3B-Instruct (3B) ==="
    local repo="bartowski/mistralai_Ministral-3-3B-Instruct-2512-GGUF"
    for q in "${QUANTS[@]}"; do
        download "$repo" \
            "mistralai_Ministral-3-3B-Instruct-2512-${q}.gguf" \
            "Ministral-3-3B-Instruct-${q}.gguf"
    done
    echo
}

download_qwen25_coder_3b() {
    should_download 3.0 || return 0
    echo "=== Qwen2.5-Coder-3B (3B) ==="
    local repo="bartowski/Qwen2.5-Coder-3B-GGUF"
    for q in "${QUANTS[@]}"; do
        if [[ "$q" == "BF16" ]]; then
            download "$repo" "Qwen2.5-Coder-3B-f16.gguf" "Qwen2.5-Coder-3B-BF16.gguf"
        else
            download "$repo" "Qwen2.5-Coder-3B-${q}.gguf" "Qwen2.5-Coder-3B-${q}.gguf"
        fi
    done
    echo
}

download_llama_3b() {
    should_download 3.2 || return 0
    echo "=== Llama-3.2-3B-Instruct (3.2B) ==="
    local repo="unsloth/Llama-3.2-3B-Instruct-GGUF"
    for q in "${QUANTS[@]}"; do
        download "$repo" "Llama-3.2-3B-Instruct-${q}.gguf" "Llama-3.2-3B-Instruct-${q}.gguf"
    done
    echo
}

download_phi4_mini() {
    should_download 3.8 || return 0
    echo "=== Phi-4-Mini-Instruct (3.8B) ==="
    local repo="unsloth/Phi-4-mini-instruct-GGUF"
    for q in "${QUANTS[@]}"; do
        download "$repo" "Phi-4-mini-instruct-${q}.gguf" "Phi-4-mini-instruct-${q}.gguf"
    done
    echo
}

download_phi4_mini_reasoning() {
    should_download 3.8 || return 0
    echo "=== Phi-4-Mini-Reasoning (3.8B) ==="
    local repo="bartowski/microsoft_Phi-4-mini-reasoning-GGUF"
    for q in "${QUANTS[@]}"; do
        download "$repo" "microsoft_Phi-4-mini-reasoning-${q}.gguf" "Phi-4-mini-reasoning-${q}.gguf"
    done
    echo
}

download_gemma_4b() {
    should_download 4.0 || return 0
    echo "=== Gemma-3-4B-it (4B) ==="
    local repo="unsloth/gemma-3-4b-it-GGUF"
    for q in "${QUANTS[@]}"; do
        download "$repo" "gemma-3-4b-it-${q}.gguf" "gemma-3-4b-it-${q}.gguf"
    done
    echo
}

download_qwen3_4b() {
    should_download 4.0 || return 0
    echo "=== Qwen3-4B (4B) ==="
    local repo="unsloth/Qwen3-4B-GGUF"
    for q in "${QUANTS[@]}"; do
        download "$repo" "Qwen3-4B-${q}.gguf" "Qwen3-4B-${q}.gguf"
    done
    echo
}

download_deepseek_7b() {
    should_download 7.0 || return 0
    echo "=== DeepSeek-R1-Distill-Qwen-7B (7B) ==="
    local repo="unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF"
    for q in "${QUANTS[@]}"; do
        download "$repo" "DeepSeek-R1-Distill-Qwen-7B-${q}.gguf" "DeepSeek-R1-Distill-Qwen-7B-${q}.gguf"
    done
    echo
}

# ─────────────────────────────────────────────────────────────────────

echo "========================================"
echo "vram-bench Model Downloader"
echo "========================================"
echo "Destination: $MODELS_DIR/"
if [[ -n "$TIER" ]]; then
    echo "VRAM Tier:  $TIER"
fi
if $Q4_ONLY; then
    echo "Mode:       Q4_K_M only"
else
    echo "Mode:       All precision levels (${#QUANTS[@]} per model)"
fi
echo "Models:     16 (0.135B to 7B)"
echo "Format:     GGUF (all precision levels including lossless BF16)"
echo "========================================"
echo

# Download in order: smallest models first
download_smollm2_135m
download_smollm2_360m
download_gemma_1b
download_llama_1b
download_tinyllama
download_deepseek_1_5b
download_qwen3_1_7b
download_smollm2_1_7b
download_ministral_3b
download_qwen25_coder_3b
download_llama_3b
download_phi4_mini
download_phi4_mini_reasoning
download_gemma_4b
download_qwen3_4b
download_deepseek_7b

echo "========================================"
echo "DOWNLOAD COMPLETE"
echo "========================================"
echo "Total files:  $TOTAL"
echo "Downloaded:   $DOWNLOADED"
echo "Skipped:      $SKIPPED"
echo "Failed:       $FAILED"
if [[ $FAILED -gt 0 ]]; then
    echo -e "\nFailed downloads:$FAILURES"
    echo -e "\nSome failures are expected (not all quants exist for every model)."
    echo "Re-run this script to retry failed downloads."
fi
echo
echo "Models saved to: $MODELS_DIR/"
du -sh "$MODELS_DIR/" 2>/dev/null || true
