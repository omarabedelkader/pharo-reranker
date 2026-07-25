#!/usr/bin/env bash
set -euo pipefail

PORT="${RERANKER_PORT:-8000}"
VENV_DIR="${VENV_DIR:-venv}"
PID_FILE="${RERANKER_PID_FILE:-reranker.pid}"
LOG_FILE="${RERANKER_LOG_FILE:-reranker.log}"
MODEL="${QWEN_RERANKER_MODEL:-tomaarsen/Qwen3-Reranker-0.6B-seq-cls}"
STARTUP_TIMEOUT="${RERANKER_STARTUP_TIMEOUT:-1800}"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Starting reranker model: $MODEL"

if lsof -ti tcp:"$PORT" >/dev/null 2>&1; then
  echo "Killing existing process(es) on port $PORT"
  lsof -ti tcp:"$PORT" | xargs -r kill -9
fi

if [ -f "$PID_FILE" ]; then
  OLD_PID="$(cat "$PID_FILE" || true)"
  if [ -n "${OLD_PID:-}" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Killing old reranker pid $OLD_PID"
    kill -9 "$OLD_PID" || true
  fi
fi

rm -f "$PID_FILE" "$LOG_FILE"

nohup python -m uvicorn qwen_reranker_api:app \
  --host 127.0.0.1 \
  --port "$PORT" \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"

for i in $(seq 1 "$STARTUP_TIMEOUT"); do
  if curl -fsS "http://127.0.0.1:$PORT/health" > /dev/null; then
    echo "Reranker is up:"
    curl -fsS "http://127.0.0.1:$PORT/health"
    echo
    exit 0
  fi
  sleep 1
done

echo "FAILED"
cat "$LOG_FILE" || true
exit 1
