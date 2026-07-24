#!/usr/bin/env bash
#OAR -q default
#OAR -p chirop
#OAR -l host=3,walltime=30:00:00
#OAR -n pharo-reranker-qwen3
#OAR -O pharo-reranker-qwen3.%jobid%.out
#OAR -E pharo-reranker-qwen3.%jobid%.err

set -euo pipefail

MODELS=(
  "0.6b=tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
  "4b=tomaarsen/Qwen3-Reranker-4B-seq-cls"
  "8b=tomaarsen/Qwen3-Reranker-8B-seq-cls"
)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

stage_files() {
  local target_dir="$1"

  mkdir -p "$target_dir"
  cp \
    "$ROOT_DIR/1.sh" \
    "$ROOT_DIR/2.sh" \
    "$ROOT_DIR/3.sh" \
    "$ROOT_DIR/4.sh" \
    "$ROOT_DIR/plot.py" \
    "$ROOT_DIR/qwen_reranker_api.py" \
    "$ROOT_DIR/requirements.txt" \
    "$ROOT_DIR/run.sh" \
    "$target_dir"/
  chmod +x "$target_dir"/*.sh
}

stop_reranker() {
  local pid_file="${RERANKER_PID_FILE:-reranker.pid}"

  if [ -f "$pid_file" ]; then
    local pid
    pid="$(cat "$pid_file" || true)"
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      log "Stopping reranker pid $pid"
      kill "$pid" || true
    fi
  fi
}

worker() {
  local label="$1"
  local model="$2"
  local model_run_dir="$3"
  local cores="${4:-}"
  local work_dir="$model_run_dir/work"
  local logs_dir="$model_run_dir/logs"
  local results_dir="$model_run_dir/results"

  mkdir -p "$logs_dir" "$results_dir"
  stage_files "$work_dir"

  cd "$work_dir"

  export QWEN_RERANKER_LABEL="$label"
  export QWEN_RERANKER_MODEL="$model"
  export RERANKER_PORT="${RERANKER_PORT:-8000}"
  export RERANKER_PID_FILE="$logs_dir/reranker.pid"
  export RERANKER_LOG_FILE="$logs_dir/reranker.log"
  export RERANKER_STARTUP_TIMEOUT="${RERANKER_STARTUP_TIMEOUT:-1800}"
  export RESULTS_DIR="$results_dir"
  export PLOT_INPUT_DIR="$results_dir"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

  if [ -n "$cores" ] && [ "$cores" != "0" ]; then
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$cores}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$cores}"
    export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$cores}"
  fi

  trap stop_reranker EXIT

  log "[$label] Worker started on $(hostname -f 2>/dev/null || hostname)"
  log "[$label] Model: $model"
  log "[$label] Results: $results_dir"

  bash ./run.sh > "$logs_dir/pipeline.log" 2>&1

  log "[$label] Worker finished"
}

cores_for_host() {
  local host="$1"

  if [ -n "${OAR_NODEFILE:-}" ] && [ -f "$OAR_NODEFILE" ]; then
    awk -v h="$host" '$0 == h { n++ } END { print n }' "$OAR_NODEFILE"
  else
    getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1
  fi
}

collect_hosts() {
  HOSTS=()

  if [ -n "${OAR_NODEFILE:-}" ] && [ -f "$OAR_NODEFILE" ]; then
    while IFS= read -r host; do
      [ -n "$host" ] && HOSTS+=("$host")
    done < <(sort -u "$OAR_NODEFILE")
  fi

  if [ "${#HOSTS[@]}" -eq 0 ]; then
    HOSTS+=("$(hostname -f 2>/dev/null || hostname)")
  fi
}

record_status() {
  local summary_file="$1"
  local label="$2"
  local model="$3"
  local status="$4"
  local model_run_dir="$5"

  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$label" \
    "$model" \
    "$status" \
    "$model_run_dir/results" \
    "$model_run_dir/logs" \
    >> "$summary_file"
}

launch_remote_worker() {
  local host="$1"
  local label="$2"
  local model="$3"
  local model_run_dir="$4"
  local cores="$5"
  local stdout_file="$6"
  local stderr_file="$7"
  local current_host
  local current_short
  local host_short

  current_host="$(hostname -f 2>/dev/null || hostname)"
  current_short="${current_host%%.*}"
  host_short="${host%%.*}"

  if [ "$host" = "$current_host" ] || [ "$host_short" = "$current_short" ]; then
    bash "$ROOT_DIR/job.oar.sh" --worker "$label" "$model" "$model_run_dir" "$cores" \
      > "$stdout_file" 2> "$stderr_file"
  else
    local remote_cmd
    remote_cmd="$(printf 'cd %q && bash %q --worker %q %q %q %q' \
      "$ROOT_DIR" \
      "$ROOT_DIR/job.oar.sh" \
      "$label" \
      "$model" \
      "$model_run_dir" \
      "$cores")"

    oarsh "$host" "$remote_cmd" > "$stdout_file" 2> "$stderr_file"
  fi
}

main() {
  local run_id="${OAR_JOB_ID:-manual-$(date '+%Y%m%d-%H%M%S')}"
  local run_dir="${RUN_DIR:-$ROOT_DIR/oar-runs/$run_id}"
  local summary_file="$run_dir/summary.tsv"
  local failed=0

  mkdir -p "$run_dir/logs" "$run_dir/models"
  printf 'label\tmodel\tstatus\tresults_dir\tlogs_dir\n' > "$summary_file"

  log "Job started"
  log "Root directory: $ROOT_DIR"
  log "Run directory: $run_dir"

  if command -v nvidia-smi >/dev/null 2>&1; then
    log "Detected $(nvidia-smi -L | wc -l | tr -d ' ') NVIDIA GPU(s)"
  else
    log "nvidia-smi not found; running on CPU"
  fi

  collect_hosts
  printf '%s\n' "${HOSTS[@]}" > "$run_dir/allocated-hosts.txt"
  log "Allocated host(s): ${HOSTS[*]}"

  if [ "${FORCE_SEQUENTIAL:-0}" != "1" ] &&
    [ "${#HOSTS[@]}" -ge "${#MODELS[@]}" ] &&
    command -v oarsh >/dev/null 2>&1; then
    log "Launching one model per host"

    pids=()
    labels=()
    model_ids=()
    model_dirs=()

    for index in "${!MODELS[@]}"; do
      spec="${MODELS[$index]}"
      label="${spec%%=*}"
      model="${spec#*=}"
      host="${HOSTS[$index]}"
      cores="$(cores_for_host "$host")"
      model_run_dir="$run_dir/models/$label"

      log "Launching $label on $host with ${cores:-unknown} core slot(s)"
      launch_remote_worker \
        "$host" \
        "$label" \
        "$model" \
        "$model_run_dir" \
        "$cores" \
        "$run_dir/logs/$label.worker.out" \
        "$run_dir/logs/$label.worker.err" &

      pids+=("$!")
      labels+=("$label")
      model_ids+=("$model")
      model_dirs+=("$model_run_dir")
    done

    for index in "${!pids[@]}"; do
      if wait "${pids[$index]}"; then
        log "${labels[$index]} completed"
        record_status "$summary_file" "${labels[$index]}" "${model_ids[$index]}" "ok" "${model_dirs[$index]}"
      else
        log "${labels[$index]} failed"
        record_status "$summary_file" "${labels[$index]}" "${model_ids[$index]}" "failed" "${model_dirs[$index]}"
        failed=1
      fi
    done
  else
    log "Launching models sequentially"

    for spec in "${MODELS[@]}"; do
      label="${spec%%=*}"
      model="${spec#*=}"
      host="${HOSTS[0]}"
      cores="$(cores_for_host "$host")"
      model_run_dir="$run_dir/models/$label"

      log "Running $label on $host with ${cores:-unknown} core slot(s)"
      if worker "$label" "$model" "$model_run_dir" "$cores" \
        > "$run_dir/logs/$label.worker.out" 2> "$run_dir/logs/$label.worker.err"; then
        log "$label completed"
        record_status "$summary_file" "$label" "$model" "ok" "$model_run_dir"
      else
        log "$label failed"
        record_status "$summary_file" "$label" "$model" "failed" "$model_run_dir"
        failed=1
      fi
    done
  fi

  log "Summary: $summary_file"

  if [ "$failed" -ne 0 ]; then
    log "One or more model runs failed"
    exit 1
  fi

  log "Job finished"
}

if [ "${1:-}" = "--worker" ]; then
  shift
  worker "$@"
else
  main "$@"
fi
