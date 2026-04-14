#!/usr/bin/env bash
set -euo pipefail

PORT=8000
VENV_DIR="venv"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r requirements.txt

if lsof -ti tcp:"$PORT" >/dev/null 2>&1; then
  echo "Killing existing process(es) on port $PORT"
  lsof -ti tcp:"$PORT" | xargs -r kill -9
fi

if [ -f reranker.pid ]; then
  OLD_PID="$(cat reranker.pid || true)"
  if [ -n "${OLD_PID:-}" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Killing old reranker pid $OLD_PID"
    kill -9 "$OLD_PID" || true
  fi
fi

rm -f reranker.pid reranker.log

nohup python -m uvicorn qwen_reranker_api:app \
  --host 127.0.0.1 \
  --port "$PORT" \
  > reranker.log 2>&1 &

echo $! > reranker.pid

for i in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$PORT/health" > /dev/null; then
    echo "Reranker is up:"
    curl -fsS "http://127.0.0.1:$PORT/health"
    echo
    exit 0
  fi
  sleep 1
done

echo "FAILED"
cat reranker.log || true
exit 1