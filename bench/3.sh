#!/bin/sh

source_file="image/nec.txt"
destination_folder="resutls"

mkdir -p "$destination_folder"
mv "$source_file" "$destination_folder"

echo "FILE MOVED OK"
