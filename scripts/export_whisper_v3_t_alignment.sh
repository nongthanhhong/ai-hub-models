#!/usr/bin/env bash

set -euo pipefail

# ------------------------------------------------------------------------------
# Whisper Large V3 Turbo - Alignment Export Script (Ubuntu)
# - Creates/uses a conda env
# - Installs local ai-hub-models in editable mode (or clones from Git if provided)
# - Exports float16-ready assets for multiple Snapdragon chipsets
# ------------------------------------------------------------------------------
#
# Prereqs:
# - Ubuntu with bash, curl, git
# - Optional: Set QAI_HUB_API_TOKEN for non-interactive hub access:
#     export QAI_HUB_API_TOKEN="xxxxx"
# - Optional: Set HUGGINGFACE_TOKEN or HUGGINGFACE_HUB_TOKEN for HF auth
# - This script assumes it is run from any directory; it will resolve paths.
#
# Usage:
#   chmod +x scripts/export_whisper_v3_t_alignment.sh
#   scripts/export_whisper_v3_t_alignment.sh [--qai-token TOKEN] [--hf-token TOKEN]
#                                           [--chipsets "Snapdragon 8 Gen 3,Snapdragon 8 Elite"]
#                                           [--output-root DIR] [--env-name NAME] [--python VER]
#                                           [--skip-profiling] [--skip-inferencing]
#                                           [--git-clone-url URL] [--git-branch BRANCH] [--workdir DIR]
#
# Environment variables (optional overrides):
#   ENV_NAME          Conda environment name (default: whisperkit)
#   PYTHON_VERSION    Python version for env (default: 3.11)
#   SKIP_PROFILING    Set to "1" to skip profiling (default: 1)
#   SKIP_INFERENCING  Set to "1" to skip on-device inference (default: 1)
#   OUTPUT_ROOT       Output dir for compiled assets (default: <repo>/build)
# ------------------------------------------------------------------------------

log() { echo -e "[export] $*"; }
fail() { echo -e "[export][ERROR] $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="${ENV_NAME:-whisperkit}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
SKIP_PROFILING="${SKIP_PROFILING:-1}"
SKIP_INFERENCING="${SKIP_INFERENCING:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/build}"

# Optional tokens via CLI
QAI_TOKEN=""
HF_TOKEN=""

# Optional Git clone support
GIT_CLONE_URL=""
GIT_BRANCH=""
WORKDIR="${HOME}/ai-hub-models"

CHIPSETS=(
  "Snapdragon 8 Gen 2"
  "Snapdragon 8 Gen 3"
  "Snapdragon 8 Elite"
  "Snapdragon 8 Elite Gen 5"
)

# ------------------------------------------------------------------------------
# Parse CLI args
# ------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --qai-token)
      QAI_TOKEN="$2"; shift 2;;
    --hf-token)
      HF_TOKEN="$2"; shift 2;;
    --chipset)
      CHIPSETS=("$2"); shift 2;;
    --chipsets)
      IFS=',' read -r -a CHIPSETS <<< "$2"; shift 2;;
    --output-root)
      OUTPUT_ROOT="$2"; shift 2;;
    --env-name)
      ENV_NAME="$2"; shift 2;;
    --python)
      PYTHON_VERSION="$2"; shift 2;;
    --skip-profiling)
      SKIP_PROFILING="1"; shift;;
    --skip-inferencing)
      SKIP_INFERENCING="1"; shift;;
    --git-clone-url)
      GIT_CLONE_URL="$2"; shift 2;;
    --git-branch)
      GIT_BRANCH="$2"; shift 2;;
    --workdir)
      WORKDIR="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo "  --qai-token TOKEN          QAI Hub API token"
      echo "  --hf-token TOKEN           Hugging Face token"
      echo "  --chipset NAME             Single chipset name"
      echo "  --chipsets CSV             Comma-separated list of chipsets"
      echo "  --output-root DIR          Output directory"
      echo "  --env-name NAME            Conda environment name"
      echo "  --python VER               Python version for env"
      echo "  --skip-profiling           Skip profiling on hosted device"
      echo "  --skip-inferencing         Skip inferencing on hosted device"
      echo "  --git-clone-url URL        Git repository to clone if source is not present"
      echo "  --git-branch BRANCH        Branch to checkout (used with --git-clone-url)"
      echo "  --workdir DIR              Directory to clone into (default: ${HOME}/ai-hub-models)"
      exit 0;;
    *)
      echo "Unknown option: $1"; exit 1;;
  esac
done

# If tokens passed via CLI, export them to the environment for downstream tools
if [[ -n "$QAI_TOKEN" ]]; then
  export QAI_HUB_API_TOKEN="$QAI_TOKEN"
fi
if [[ -n "$HF_TOKEN" ]]; then
  export HUGGINGFACE_TOKEN="$HF_TOKEN"
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

# If a git clone URL is provided (or source missing), clone/update and set REPO_ROOT
if [[ -n "$GIT_CLONE_URL" ]]; then
  log "Git clone requested: $GIT_CLONE_URL"
  mkdir -p "$WORKDIR"
  if [[ -d "$WORKDIR/.git" ]]; then
    log "Existing git repo found at $WORKDIR, updating..."
    pushd "$WORKDIR" >/dev/null
    git remote set-url origin "$GIT_CLONE_URL" || true
    git fetch --all --tags || true
    if [[ -n "$GIT_BRANCH" ]]; then
      git checkout "$GIT_BRANCH" || git checkout -b "$GIT_BRANCH"
      git pull --rebase origin "$GIT_BRANCH" || true
    fi
    popd >/dev/null
  else
    if [[ -n "$GIT_BRANCH" ]]; then
      git clone -b "$GIT_BRANCH" "$GIT_CLONE_URL" "$WORKDIR"
    else
      git clone "$GIT_CLONE_URL" "$WORKDIR"
    fi
  fi
  REPO_ROOT="$WORKDIR"
  OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/build}"
  log "Using REPO_ROOT=$REPO_ROOT"
fi

# ------------------------------------------------------------------------------
# Conda bootstrap - ALWAYS use Miniconda at ${HOME}/miniconda
# ------------------------------------------------------------------------------
if [[ ! -d "${HOME}/miniconda" ]]; then
  log "Miniconda not found at ${HOME}/miniconda. Installing (non-interactive)..."
  TMP_DIR="$(mktemp -d)"
  pushd "${TMP_DIR}" >/dev/null
  curl -fsSL -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash miniconda.sh -b -p "${HOME}/miniconda"
  popd >/dev/null
  rm -rf "${TMP_DIR}"
fi
# Initialize Miniconda for this shell
# shellcheck disable=SC1091
source "${HOME}/miniconda/etc/profile.d/conda.sh" || fail "Failed to source Miniconda conda.sh"
# best-effort hook init (does nothing if already initialized)
"${HOME}/miniconda/bin/conda" >/dev/null 2>&1 || true

# ------------------------------------------------------------------------------
# Conda env setup
# ------------------------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  log "Using existing conda env: ${ENV_NAME}"
else
  log "Creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

# ------------------------------------------------------------------------------
# Python deps: install local package in editable mode
# ------------------------------------------------------------------------------
log "Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel >/dev/null

log "Installing local ai-hub-models (editable)..."
python -m pip install -e "${REPO_ROOT}" >/dev/null

# Ensure required CLIs/libs present
python - <<'PYCHK' || true
import importlib, sys, subprocess
for pkg in ("qai_hub", "huggingface_hub"):
    try:
        importlib.import_module(pkg)
    except Exception:
        print(f"[export] Installing missing package: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
# Install CLI wrappers when available
try:
    import shutil
    if shutil.which("qai-hub") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "qai-hub"])  # CLI
except Exception:
    pass
PYCHK

# ------------------------------------------------------------------------------
# Configure tokens (if provided)
# ------------------------------------------------------------------------------
if [[ -n "${QAI_HUB_API_TOKEN:-}" ]]; then
  if command -v qai-hub >/dev/null 2>&1; then
    log "Configuring QAI Hub CLI with provided token"
    qai-hub configure --api_token "${QAI_HUB_API_TOKEN}" || true
  else
    log "qai-hub CLI not found; proceeding with Python client using env token"
  fi
else
  log "QAI_HUB_API_TOKEN not set; exports may require interactive auth"
fi

# Hugging Face login if token provided
if [[ -n "${HUGGINGFACE_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_TOKEN:-${HUGGINGFACE_HUB_TOKEN}}"
  if command -v huggingface-cli >/dev/null 2>&1; then
    log "Configuring Hugging Face CLI with provided token"
    huggingface-cli login --token "${HUGGINGFACE_HUB_TOKEN}" --add-to-git-credential --non-interactive || true
  else
    log "huggingface-cli not found; attempting Python login"
    python - <<'HFLOGIN' || true
import os
from huggingface_hub import login
 tok = os.environ.get("HUGGINGFACE_HUB_TOKEN")
 if tok:
     try:
         login(token=tok, add_to_git_credential=True)
     except Exception:
         pass
HFLOGIN
  fi
else
  log "HUGGINGFACE_TOKEN/HUGGINGFACE_HUB_TOKEN not set; will rely on public access or prior login"
fi

# ------------------------------------------------------------------------------
# Export options
# ------------------------------------------------------------------------------
SKIP_PROFILING_FLAG=""
SKIP_INFERENCING_FLAG=""
[[ "${SKIP_PROFILING}" == "1" ]] && SKIP_PROFILING_FLAG="--skip_profiling"
[[ "${SKIP_INFERENCING}" == "1" ]] && SKIP_INFERENCING_FLAG="--skip_inferencing"

mkdir -p "${OUTPUT_ROOT}"

log "Starting exports (float16 for edge inference) to: ${OUTPUT_ROOT}"
log "If authentication is required, ensure QAI_HUB_API_TOKEN and HUGGINGFACE_TOKEN are set or passed via CLI."

# ------------------------------------------------------------------------------
# Loop over chipsets and export
# ------------------------------------------------------------------------------
for CHIPSET in "${CHIPSETS[@]}"; do
  # Safe dir tag
  SAFE_TAG="$(echo "${CHIPSET}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_*//;s/_*$//')"
  OUT_DIR="${OUTPUT_ROOT}/whisper_large_v3_turbo_${SAFE_TAG}"
  log "Exporting for chipset: '${CHIPSET}' -> ${OUT_DIR}"

  # Float precision with QNN compilers will internally use --quantize_full_type float16 --quantize_io
  python -m qai_hub_models.models.whisper_large_v3_turbo.export \
    --chipset "${CHIPSET}" \
    --precision float \
    --target_runtime PRECOMPILED_QNN_ONNX \
    --output_dir "${OUT_DIR}" \
    --skip_summary \
    ${SKIP_PROFILING_FLAG} \
    ${SKIP_INFERENCING_FLAG}

  log "Completed export for '${CHIPSET}'. Output at: ${OUT_DIR}"
done

log "All exports completed successfully."
