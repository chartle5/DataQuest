#!/usr/bin/env bash
set -euo pipefail

# DataQuest project one-command starter.
# Usage: ./start.sh [--no-install] [--no-preprocess] [--static-frontend] [--train]

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

activate_venv() {
  if [ -f ".venv/bin/activate" ] && [ -z "${VIRTUAL_ENV-}" ]; then
    echo "Activating .venv"
    # shellcheck disable=SC1091
    source .venv/bin/activate
  fi
}

echo "Starting DataQuest launcher"
activate_venv

# Defaults
SKIP_INSTALL=false
SKIP_PREPROCESS=false
STATIC_FRONTEND=false
DO_TRAIN=false
USE_CONDA=false


check_python_version() {
  PYVER=$(python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')")
  if [ "$PYVER" != "3.11" ]; then
    echo "Current Python version: $PYVER (recommended: 3.11)"
    return 1
  fi
  return 0
}

conda_setup_and_activate() {
  # ensure conda is available
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found on PATH. Install Miniconda/Anaconda or activate a Python 3.11 env manually." >&2
    return 1
  fi
  CONDA_BASE=$(conda info --base)
  # shellcheck disable=SC1090
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  ENV_NAME=dataquest
  if conda env list | awk '{print $1}' | grep -xq "$ENV_NAME"; then
    echo "Activating existing conda env '$ENV_NAME'"
    conda activate "$ENV_NAME"
    return 0
  fi
  echo "Creating conda env '$ENV_NAME' with Python 3.11 (this may take a few minutes)..."
  conda create -n "$ENV_NAME" -c conda-forge python=3.11 -y || return 1
  conda activate "$ENV_NAME"
  echo "Created and activated '$ENV_NAME'. You may want to run './start.sh --no-install' now to avoid re-installing packages." 
  return 0
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --no-install) SKIP_INSTALL=true; shift ;;
    --no-preprocess) SKIP_PREPROCESS=true; shift ;;
    --static-frontend) STATIC_FRONTEND=true; shift ;;
    --train) DO_TRAIN=true; shift ;;
    --use-conda) USE_CONDA=true; shift ;;
    --help) echo "Usage: ./start.sh [--no-install] [--no-preprocess] [--static-frontend] [--train]"; exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# If requested, try to set up/activate conda env (dataquest)
if [ "$USE_CONDA" = true ]; then
  if ! conda_setup_and_activate; then
    echo "--use-conda requested but failed. Aborting." >&2
    exit 1
  fi
fi

# Verify python version (recommend 3.11)
if ! check_python_version; then
  echo "Python 3.11 is recommended. Re-run with --use-conda to create/activate a conda env, or activate a Python 3.11 venv." >&2
  exit 1
fi

if ! $SKIP_INSTALL; then
  echo "Installing requirements (pip may attempt to compile native packages)..."
  pip install -r requirements.txt || true
  pip install Flask-Cors six || true
fi

if ! $SKIP_PREPROCESS; then
  if [ -f data/processed/patients.parquet ]; then
    echo "Processed data exists — skipping preprocessing."
  else
    echo "Running preprocessing..."
    python -c "from src.data.loader import process_all; process_all()"
  fi
fi

if $DO_TRAIN; then
  echo "Training models (this may take a while)..."
  python -c "from src.matching.train_models import train_for_condition; print(train_for_condition('type 2 diabetes', trials_limit=50))"
  python -c "from src.matching.train_models import train_for_condition; print(train_for_condition('oncology', trials_limit=50))"
fi

echo "Starting Flask server on http://localhost:8080"
python -m src.server &
SERVER_PID=$!

if $STATIC_FRONTEND; then
  echo "Starting static frontend on http://localhost:8000"
  python -m http.server 8000 --directory frontend &
  STATIC_PID=$!
fi

echo "Flask PID: $SERVER_PID"
echo "Press Ctrl-C to stop."
wait $SERVER_PID
