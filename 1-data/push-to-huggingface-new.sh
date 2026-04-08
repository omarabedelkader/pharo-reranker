#!/bin/bash
set -e

# Push JSONL files to Hugging Face repositories
# Creates repositories if they don't exist
# Usage: ./push-to-huggingface.sh [hf_username]

HF_USERNAME="${1:-}"
DATASET_DIR="./jsonl-files"

# Check if username is provided
if [ -z "$HF_USERNAME" ]; then
    echo "Usage: $0 <huggingface-username>"
    echo ""
    echo "Example: $0 myusername"
    echo ""
    echo "This script will:"
    echo "  1. Create repositories on Hugging Face if they don't exist"
    echo "  2. Push FIM files to 'fim-pharo-reranker' repository"
    echo "  3. Push reranker files to 'reranker-pharo-re-ranker' repository"
    exit 1
fi

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli is not installed."
    echo "Install it with: pip install huggingface_hub[cli]"
    exit 1
fi

# Check if logged in
if ! huggingface-cli whoami &> /dev/null; then
    echo "Error: Not logged in to Hugging Face."
    echo "Login with: huggingface-cli login"
    exit 1
fi

# Check if dataset directory exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory '$DATASET_DIR' does not exist."
    echo "Run ./organize-jsonl.sh first to organize your JSONL files."
    exit 1
fi

# Function to create repo if it doesn't exist
create_repo_if_not_exists() {
    local repo_name="$1"
    local repo_type="$2"
    
    echo "Checking if repository '$repo_name' exists..."
    
    # Try to get repo info, if it fails, create it
    if ! huggingface-cli repo info "$HF_USERNAME/$repo_name" --repo-type dataset &> /dev/null; then
        echo "Repository '$repo_name' does not exist. Creating..."
        huggingface-cli repo create "$HF_USERNAME/$repo_name" --repo-type dataset --yes
        
        # Initialize local repo
        mkdir -p "/tmp/$repo_name"
        cd "/tmp/$repo_name"
        git init
        git lfs install
        git lfs track "*.jsonl"
        
        # Add remote
        git remote add origin "https://huggingface.co/datasets/$HF_USERNAME/$repo_name"
        echo "Repository created: https://huggingface.co/datasets/$HF_USERNAME/$repo_name"
        cd - > /dev/null
    else
        echo "Repository '$repo_name' already exists."
        echo "URL: https://huggingface.co/datasets/$HF_USERNAME/$repo_name"
        
        # Clone if not already cloned
        if [ ! -d "/tmp/$repo_name/.git" ]; then
            mkdir -p "/tmp/$repo_name"
            cd "/tmp/$repo_name"
            git init
            git lfs install
            git lfs track "*.jsonl"
            git remote add origin "https://huggingface.co/datasets/$HF_USERNAME/$repo_name"
            git pull origin main || true
            cd - > /dev/null
        fi
    fi
}

# Create repositories
echo "=== Setting up Hugging Face repositories ==="
create_repo_if_not_exists "fim-pharo-reranker" "dataset"
create_repo_if_not_exists "reranker-pharo-re-ranker" "dataset"

# Push FIM files
echo ""
echo "=== Pushing FIM files ==="
FIM_FILES=$(find "$DATASET_DIR" -name "*fim*.jsonl" -o -name "*FIM*.jsonl" | head -20)
if [ -n "$FIM_FILES" ]; then
    cd "/tmp/fim-pharo-reranker"
    
    for file in $FIM_FILES; do
        filename=$(basename "$file")
        cp "$file" .
        git add "$filename"
        echo "✓ Added: $filename"
    done
    
    git add .gitattributes
    git commit -m "Update FIM dataset files" || echo "No changes to commit"
    git push origin main
    echo "✓ FIM files pushed to: https://huggingface.co/datasets/$HF_USERNAME/fim-pharo-reranker"
    
    cd - > /dev/null
else
    echo "⚠ No FIM files found in $DATASET_DIR"
fi

# Push Reranker files
echo ""
echo "=== Pushing Reranker files ==="
RERANKER_FILES=$(find "$DATASET_DIR" -name "*reranker*.jsonl" -o -name "*Reranker*.jsonl" -o -name "*data*.jsonl" -o -name "*Data*.jsonl" | head -20)
if [ -n "$RERANKER_FILES" ]; then
    cd "/tmp/reranker-pharo-re-ranker"
    
    for file in $RERANKER_FILES; do
        filename=$(basename "$file")
        cp "$file" .
        git add "$filename"
        echo "✓ Added: $filename"
    done
    
    git add .gitattributes
    git commit -m "Update reranker dataset files" || echo "No changes to commit"
    git push origin main
    echo "✓ Reranker files pushed to: https://huggingface.co/datasets/$HF_USERNAME/reranker-pharo-re-ranker"
    
    cd - > /dev/null
else
    echo "⚠ No reranker files found in $DATASET_DIR"
fi

echo ""
echo "=== Summary ==="
echo "FIM Repository: https://huggingface.co/datasets/$HF_USERNAME/fim-pharo-reranker"
echo "Reranker Repository: https://huggingface.co/datasets/$HF_USERNAME/reranker-pharo-re-ranker"
echo ""
echo "Done!"
