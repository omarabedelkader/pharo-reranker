#!/usr/bin/env bash
set -euo pipefail

source venv/bin/activate
echo "Plot input directory: ${PLOT_INPUT_DIR:-results}"
python plot.py


echo "PLOT OK"

# rm -rf image/
# rm -rf venv/


echo "CLEAN OK"
