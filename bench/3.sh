#!/bin/sh

source_folder="image"
destination_folder="results"

mkdir -p "$destination_folder"

mv "$source_folder"/*.txt "$destination_folder" 2>/dev/null

echo "ALL TXT FILES MOVED OK"