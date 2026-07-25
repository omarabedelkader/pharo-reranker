#!/usr/bin/env bash
set -euo pipefail

source_folder="${SOURCE_FOLDER:-image}"
destination_folder="${RESULTS_DIR:-results}"

mkdir -p "$destination_folder"

shopt -s nullglob
txt_files=("$source_folder"/*.txt)

if [ "${#txt_files[@]}" -eq 0 ]; then
  echo "No TXT files found in $source_folder"
  exit 0
fi

mv "${txt_files[@]}" "$destination_folder"/

echo "ALL TXT FILES MOVED OK"
