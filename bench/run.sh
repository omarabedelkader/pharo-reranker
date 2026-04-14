#!/usr/bin/env bash
set -euo pipefail

# server side - model preparation
chmod +x 1.sh
bash 1.sh

# pharo-side - bench
chmod +x 2.sh
bash 2.sh

# backend side
chmod +x 3.sh
bash 3.sh

# results side
chmod +x 4.sh
bash 4.sh