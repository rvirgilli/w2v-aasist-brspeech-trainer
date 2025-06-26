#!/bin/bash
set -e

# --- Configuration ---
GPU_ID=""
ASVSPOOF_DATASET_PATH=""
OUTPUT_PATH="./outputs_asvspoof"
DOCKER_COMPOSE_FILE="docker-compose.yml"
DOCKER_ARGS=""

# --- Helper Functions ---
log() {
    echo "[$1] $2"
}

show_usage() {
    echo "Usage: $0 --gpu=<id> --data-path=<path> [options]"
    echo ""
    echo "Required:"
    echo "  --gpu=<id>              GPU ID for training."
    echo "  --data-path=<path>    Absolute path to the ASVspoof 2019 LA dataset root."
    echo ""
    echo "Optional:"
    echo "  --output=<path>         Output directory for models (default: ./outputs_asvspoof)."
    echo "  --build                 Build the Docker image before running."
    echo "  --help                  Show this help message."
}

# --- Argument Parsing ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu=*) GPU_ID="${1#*=}"; shift ;;
        --data-path=*) ASVSPOOF_DATASET_PATH="${1#*=}"; shift ;;
        --output=*) OUTPUT_PATH="${1#*=}"; shift ;;
        --build) DOCKER_ARGS="--build"; shift ;;
        --help) show_usage; exit 0 ;;
        *) log "ERROR" "Unknown argument: $1"; show_usage; exit 1 ;;
    esac
done

# --- Validation ---
if [ -z "$GPU_ID" ] || [ -z "$ASVSPOOF_DATASET_PATH" ]; then
    log "ERROR" "GPU ID and data path are required."
    show_usage
    exit 1
fi

if [ ! -d "$ASVSPOOF_DATASET_PATH" ]; then
    log "ERROR" "Dataset path not found: $ASVSPOOF_DATASET_PATH"
    exit 1
fi

# Convert to absolute path to be safe
ASVSPOOF_DATASET_PATH=$(realpath "$ASVSPOOF_DATASET_PATH")
OUTPUT_PATH=$(realpath "$OUTPUT_PATH")

# Export variables for docker-compose
export ASVSPOOF_DATASET_PATH
export OUTPUT_PATH
export GPU_ID

log "INFO" "GPU ID: $GPU_ID"
log "INFO" "Dataset Path: $ASVSPOOF_DATASET_PATH"
log "INFO" "Output Path: $OUTPUT_PATH"

# --- Docker Execution ---
log "INFO" "Detecting Docker Compose command..."

# Check if we need sudo for docker
NEED_SUDO=""
if ! docker ps &> /dev/null; then
    if sudo -E docker ps &> /dev/null; then
        NEED_SUDO="sudo -E"
        log "INFO" "Using sudo for Docker commands"
    else
        log "ERROR" "Cannot access Docker. Please check Docker installation and permissions."
        exit 1
    fi
fi

# Prioritize docker compose (v2) over docker-compose (v1)
if command -v 'docker' &> /dev/null && $NEED_SUDO docker compose version &> /dev/null; then
    log "INFO" "Found 'docker compose' (v2)"
    DOCKER_CMD="$NEED_SUDO docker compose"
elif command -v 'docker-compose' &> /dev/null; then
    log "INFO" "Found 'docker-compose' (v1)"
    DOCKER_CMD="$NEED_SUDO docker-compose"
else
    log "ERROR" "Could not find 'docker compose' or 'docker-compose'. Please install Docker Compose."
    exit 1
fi

log "INFO" "Starting Docker..."
if [[ "$DOCKER_ARGS" == "--build" ]]; then
  log "INFO" "Building image with no cache..."
  # Build service without cache
  $DOCKER_CMD -f $DOCKER_COMPOSE_FILE build --no-cache training-asv
  # Run containers
  $DOCKER_CMD -f $DOCKER_COMPOSE_FILE up --remove-orphans
else
  # Run containers
  $DOCKER_CMD -f $DOCKER_COMPOSE_FILE up $DOCKER_ARGS --remove-orphans
fi

log "INFO" "Training finished. Check outputs in $OUTPUT_PATH" 