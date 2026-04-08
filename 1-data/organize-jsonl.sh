#!/bin/bash
set -e

# Organize JSONL files from baseimage directory into a structured directory
# Usage: ./organize-jsonl.sh [output_directory]

OUTPUT_DIR="${1:-./jsonl-files}"

echo "Organizing JSONL files from baseimage directory..."

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Find all JSONL files in baseimage and copy them to output directory
JSONL_COUNT=0
while IFS= read -r -d '' file; do
    filename=$(basename "$file")
    cp "$file" "$OUTPUT_DIR/$filename"
    echo "✓ Copied: $filename"
    JSONL_COUNT=$((JSONL_COUNT + 1))
done < <(find ./baseimage -name "*.jsonl" -type f -print0)

echo ""
echo "Total JSONL files copied: $JSONL_COUNT"
echo "Files saved to: $OUTPUT_DIR"

if [ "$JSONL_COUNT" -eq 0 ]; then
    echo "⚠ No JSONL files found in baseimage directory"
    exit 1
fi

# List files by pattern
echo ""
echo "Files organized:"
echo "  FIM files: $(find "$OUTPUT_DIR" -name "*fim*.jsonl" -o -name "*FIM*.jsonl" | wc -l | tr -d ' ')"
echo "  Reranker files: $(find "$OUTPUT_DIR" -name "*reranker*.jsonl" -o -name "*Reranker*.jsonl" | wc -l | tr -d ' ')"
echo ""
echo "Done!"
