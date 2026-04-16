#!/usr/bin/env bash
set -euo pipefail

source venv/bin/activate
python plot.py


echo "PLOT OK"

# rm -rf image/
# rm -rf venv/


echo "CLEAN OK"