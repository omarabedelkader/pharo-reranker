#!/bin/bash
set -e

# Create directory and download Pharo image
mkdir -p baseimage && cd baseimage
wget -q -O - https://get.pharo.org/140+vm | bash

# Clone the pharo-dataset repository
echo "Cloning pharo-dataset repository..."
git clone https://github.com/pharo-llm/pharo-dataset.git

# Load the pharo-dataset package into the image
echo "Loading HeuristicCompletionGenerator package..."
./pharo Pharo.image eval --save "
Metacello new
    githubUser: 'omarabedelkader'
    project: 'HeuristicCompletion-Generator'
    commitish: 'main'
    path: 'src';
    baseline: 'HeuristicCompletionGenerator';
    load."

# Launch the FIM JSONL exporter
echo "Running CooCompletionFineTuningDatasetExporter exportAllFIMJsonl..."
./pharo Pharo.image eval --save "CooCompletionFineTuningDatasetExporter exportAllFIMJsonl."

echo "Export completed!"
