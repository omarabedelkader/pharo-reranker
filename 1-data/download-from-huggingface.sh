#!/bin/bash
set -e

# Download JSONL datasets from Hugging Face repositories
# Usage: ./download-from-huggingface.sh [hf_username] [output_directory]

HF_USERNAME="${1:-}"
OUTPUT_DIR="${2:-./downloaded-datasets}"

# Check if username is provided
if [ -z "$HF_USERNAME" ]; then
    echo "Usage: $0 <huggingface-username> [output_directory]"
    echo ""
    echo "Example: $0 myusername"
    echo "Example: $0 myusername ./my-datasets"
    echo ""
    echo "This script will download:"
    echo "  1. FIM files from 'fim-pharo-reranker' repository"
    echo "  2. Reranker files from 'reranker-pharo-re-ranker' repository"
    exit 1
fi

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli is not installed."
    echo "Install it with: pip install huggingface_hub[cli]"
    exit 1
fi

# Create output directories
FIM_DIR="$OUTPUT_DIR/fim-pharo-reranker"
RERANKER_DIR="$OUTPUT_DIR/reranker-pharo-re-ranker"

mkdir -p "$FIM_DIR"
mkdir -p "$RERANKER_DIR"

echo "=== Downloading from Hugging Face ==="
echo "Username: $HF_USERNAME"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Download FIM dataset
echo "=== Downloading FIM dataset ==="
echo "Repository: https://huggingface.co/datasets/$HF_USERNAME/fim-pharo-reranker"

if huggingface-cli download "$HF_USERNAME/fim-pharo-reranker" --repo-type dataset --local-dir "$FIM_DIR"; then
    echo "✓ FIM dataset downloaded to: $FIM_DIR"
    echo "Files downloaded:"
    ls -1 "$FIM_DIR"/*.jsonl 2>/dev/null | while read file; do
        echo "  - $(basename "$file")"
    done
else
    echo "⚠ Failed to download FIM dataset. Repository may not exist or you don't have access."
fi

echo ""

# Download Reranker dataset
echo "=== Downloading Reranker dataset ==="
echo "Repository: https://huggingface.co/datasets/$HF_USERNAME/reranker-pharo-re-ranker"

if huggingface-cli download "$HF_USERNAME/reranker-pharo-re-ranker" --repo-type dataset --local-dir "$RERANKER_DIR"; then
    echo "✓ Reranker dataset downloaded to: $RERANKER_DIR"
    echo "Files downloaded:"
    ls -1 "$RERANKER_DIR"/*.jsonl 2>/dev/null | while read file; do
        echo "  - $(basename "$file")"
    done
else
    echo "⚠ Failed to download Reranker dataset. Repository may not exist or you don't have access."
fi

echo ""
echo "=== Summary ==="
echo "FIM files: $FIM_DIR"
FIM_COUNT=$(find "$FIM_DIR" -name "*.jsonl" 2>/dev/null | wc -l | tr -d ' ')
echo "  Total: $FIM_COUNT files"

echo ""
echo "Reranker files: $RERANKER_DIR"
RERANKER_COUNT=$(find "$RERANKER_DIR" -name "*.jsonl" 2>/dev/null | wc -l | tr -d ' ')
echo "  Total: $RERANKER_COUNT files"

echo ""
echo "Done!"
