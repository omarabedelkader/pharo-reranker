#!/usr/bin/env sh
set -eu

mkdir -p image
cd image

# --------------------------------------------------
# 1) Install Pharo image
# --------------------------------------------------
curl -fsSL https://get.pharo.org/140+vm | bash
curl -fLO https://files.pharo.org/image/140/Pharo14.0-SNAPSHOT.build.415.sha.359c9be46a.arch.64bit.zip
unzip -o Pharo14.0-SNAPSHOT.build.415.sha.359c9be46a.arch.64bit.zip

./pharo Pharo14.0-SNAPSHOT-64bit-359c9be46a.image eval "2+2"
echo "Done image installation"

# --------------------------------------------------
# 2) Load your Smalltalk project into the image
# --------------------------------------------------
./pharo Pharo14.0-SNAPSHOT-64bit-359c9be46a.image eval --save "
Metacello new
  baseline: 'AISorter';
  repository: 'github://omarabedelkader/AI-Sorter:main/src';
  load.
."
echo "Done AI installation"

# --------------------------------------------------
# 3) Prepare Python reranker service
# --------------------------------------------------
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install fastapi 'uvicorn[standard]' sentence-transformers torch

# --------------------------------------------------
# 4) Start reranker API in background
# --------------------------------------------------
nohup uvicorn qwen_reranker_api:app --host 127.0.0.1 --port 8000 > reranker.log 2>&1 &
echo $! > reranker.pid

echo "Reranker API started"

# optional quick check
sleep 5
curl -fsS http://127.0.0.1:8000/health || {
  echo "Reranker API failed to start"
  exit 1
}

echo "Everything is ready"