#!/bin/bash

set -e

log() {
  echo "[`date '+%Y-%m-%d %H:%M:%S'`] $1"
}


log "Starting FastAPI"
cd Bench/1-FastAPI
uvicorn app:app --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &
UVICORN_PID=$!

sleep 5


log "Download Pharo + Executing Benchmark"
cd ../../
cd Bench/2-BootStrap
bash 2.sh


log "Showing Results"
cd ../../
python3 Bench/3-Results/results.py

log "Cleaning up"
kill $UVICORN_PID || true

log "All tasks completed."