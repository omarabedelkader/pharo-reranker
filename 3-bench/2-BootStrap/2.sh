#!/usr/bin/env bash

# =========================
# Shell safety & logging
# =========================
set -euo pipefail

# Log every command
set -x

# Save logs to file + stdout
exec > >(tee build.log) 2>&1

echo "[INFO] Starting build"

# =========================
# Setup base image
# =========================
echo "[INFO] Creating baseimage directory"
mkdir -p baseimage

echo "[INFO] Entering baseimage directory"
cd baseimage || { echo "[ERROR] Failed to enter baseimage"; exit 1; }

# =========================
# Download Pharo
# =========================
echo "[INFO] Downloading Pharo 14 + VM"
wget --quiet -O - get.pharo.org/140+vm | bash
echo "[INFO] Pharo downloaded successfully"

# =========================
# Load AISorter baseline
# =========================
echo "[INFO] Loading AISorter baseline"

./pharo Pharo.image evaluate "
Metacello new
  githubUser: 'omarabedelkader' project: 'AI-Sorter' commitish: 'main' path: 'src';
  baseline: 'AISorter';
  load.

Smalltalk snapshot: true andQuit: true
"

echo "[INFO] AISorter baseline loaded"

# =========================
# Run AST benchmark
# =========================
echo "[INFO] Running AST benchmark"

./pharo Pharo.image eval --no-quit "
Transcript show: 'Starting AST benchmark'; cr.

[
  CooStaticBenchmarksMessageSorter nec.
  Transcript show: 'AST benchmark finished OK'; cr.
  Smalltalk snapshot: false andQuit: true
] on: Error do: [ :e |
  Transcript show: 'ERROR: ', e printString; cr.
  e signal
].
"


echo "[INFO] AST benchmark completed"

# =========================
# Done
# =========================
echo "[INFO] Build finished successfully"



