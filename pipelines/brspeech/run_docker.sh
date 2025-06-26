#!/bin/bash

# BrSpeech Training Docker Runner
# Usage: ./run_docker.sh --gpu=<id> --real-data=<path> --spoof-data=<path> [options]

set -e

# Initialize variables
GPU_ID=""
REAL_DATASET_PATH=""
SPOOF_DATASET_PATH=""
OUTPUT_PATH="./outputs"
DOCKER_ARGS=""
PYTHON_ARGS=""

# Function to log messages
log() {
    echo "[$1] $2"
}

# Function to show usage
show_usage() {
    echo "❌ Usage: $0 --gpu=<id> --real-data=<path> --spoof-data=<path> [options]"
    echo ""
    echo "🚨 GPU ID is REQUIRED to prevent conflicts on shared servers!"
    echo ""
    echo "Required parameters:"
    echo "  --gpu=<id>              GPU ID to use (e.g., --gpu=6)"
    echo "  --real-data=<path>      Path to real BrSpeech dataset"
    echo "  --spoof-data=<path>     Path to synthetic speech dataset"
    echo ""
    echo "Optional parameters:"
    echo "  --output=<path>         Output directory (default: ./outputs)"
    echo "  --build                 Build Docker image before running"
    echo "  --test_only             Run only the test phase"
    echo "  --checkpoint_path=<path> Path to checkpoint"
    echo "  --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --gpu=6 --real-data=/path/to/BRSpeech_CML_TTS_v04012024 --spoof-data=/path/to/brspeech_df"
    echo "  $0 --gpu=6 --real-data=/data/real --spoof-data=/data/spoof --output=/outputs --build"
    echo ""
    echo "Dataset structure expected:"
    echo "  Real dataset: <real_dataset_path>/{train,dev,test}.csv and train/audio/*.flac"
    echo "  Spoof dataset: <spoof_dataset_path>/{f5tts,fish-speech,toucantts,xtts,yourtts}/{train,dev,test}/audio/*.flac"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu=*)
            GPU_ID="${1#*=}"
            shift
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --real-data=*)
            REAL_DATASET_PATH="${1#*=}"
            shift
            ;;
        --real-data)
            REAL_DATASET_PATH="$2"
            shift 2
            ;;
        --spoof-data=*)
            SPOOF_DATASET_PATH="${1#*=}"
            shift
            ;;
        --spoof-data)
            SPOOF_DATASET_PATH="$2"
            shift 2
            ;;
        --output=*)
            OUTPUT_PATH="${1#*=}"
            shift
            ;;
        --output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --build)
            DOCKER_ARGS="$DOCKER_ARGS --build"
            shift
            ;;
        --test_only)
            PYTHON_ARGS="$PYTHON_ARGS --test_only"
            shift
            ;;
        --checkpoint_path=*)
            PYTHON_ARGS="$PYTHON_ARGS --checkpoint_path ${1#*=}"
            shift
            ;;
        --checkpoint_path)
            PYTHON_ARGS="$PYTHON_ARGS --checkpoint_path $2"
            shift 2
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            log "WARN" "Passing unknown argument '$1' to docker-compose"
            DOCKER_ARGS="$DOCKER_ARGS $1"
            shift
            ;;
    esac
done

# Check required arguments
if [ -z "$GPU_ID" ] || [ -z "$REAL_DATASET_PATH" ] || [ -z "$SPOOF_DATASET_PATH" ]; then
    show_usage
    exit 1
fi

# --- Path and Structure Validation ---
log "INFO" "🚀 Validating input paths and structure..."

# Validate real data structure
for f in train.csv dev.csv test.csv; do
    if [ ! -f "$REAL_DATASET_PATH/$f" ]; then
        log "ERROR" "Real data validation failed: Missing file '$f' in $REAL_DATASET_PATH"
        exit 1
    fi
done
log "INFO" "✅ Real data structure looks good."

# Validate spoof data structure
found_spoof_model=false
for model_dir in f5tts fish-speech toucantts xtts yourtts; do
    if [ -d "$SPOOF_DATASET_PATH/$model_dir" ]; then
        found_spoof_model=true
        break
    fi
done

if [ "$found_spoof_model" = false ]; then
    log "ERROR" "Spoof data validation failed: Could not find any expected model subdirectories (e.g., 'xtts', 'yourtts') in $SPOOF_DATASET_PATH"
    exit 1
fi
log "INFO" "✅ Spoof data structure looks good."
# --- End Validation ---

# --- Auto-find latest checkpoint for test-only mode ---
if [[ "$PYTHON_ARGS" == *"--test_only"* ]] && [[ "$PYTHON_ARGS" != *"--checkpoint_path"* ]]; then
    log "INFO" "Test-only mode enabled with no checkpoint specified. Finding the latest..."
    
    # Find the most recently modified .pth file in the output directory
    latest_checkpoint_host=$(find "$OUTPUT_PATH" -name "*.pth" -type f -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
    
    if [ -z "$latest_checkpoint_host" ]; then
        log "ERROR" "Could not find any checkpoint files (*.pth) in '$OUTPUT_PATH'."
        exit 1
    fi
    
    log "INFO" "Found latest checkpoint on host: $latest_checkpoint_host"
    
    # The path inside the container is relative to the mount point.
    # We strip the host OUTPUT_PATH to get the relative path for the container.
    checkpoint_container_path="fine_tuned_models/${latest_checkpoint_host#$OUTPUT_PATH/}"
    
    PYTHON_ARGS="$PYTHON_ARGS --checkpoint_path=$checkpoint_container_path"
    log "INFO" "Automatically using checkpoint for container: $checkpoint_container_path"
fi
# --- End auto-find ---

# Convert to absolute paths
REAL_DATASET_PATH=$(realpath "$REAL_DATASET_PATH")
SPOOF_DATASET_PATH=$(realpath "$SPOOF_DATASET_PATH")
OUTPUT_PATH=$(realpath "$OUTPUT_PATH")

# Validate paths exist
if [ ! -d "$REAL_DATASET_PATH" ]; then
    echo "❌ Real dataset path does not exist: $REAL_DATASET_PATH"
    exit 1
fi

if [ ! -d "$SPOOF_DATASET_PATH" ]; then
    echo "❌ Spoof dataset path does not exist: $SPOOF_DATASET_PATH"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_PATH"

echo "🚀 Starting BrSpeech Training with:"
echo "   Real dataset: $REAL_DATASET_PATH"
echo "   Spoof dataset: $SPOOF_DATASET_PATH"
echo "   Output: $OUTPUT_PATH"
echo ""

# Validate GPU ID is numeric
if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "❌ GPU ID must be a number (0, 1, 2, etc.)"
    exit 1
fi

echo "🎮 Using GPU: $GPU_ID"

# Export environment variables for docker-compose
export REAL_DATASET_PATH
export SPOOF_DATASET_PATH
export OUTPUT_PATH
export GPU_ID
export PYTHON_ARGS

# Detect which docker compose command is available
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker compose"
else
    echo "❌ Neither 'docker-compose' nor 'docker compose' is available"
    exit 1
fi

echo "🐳 Using: $DOCKER_COMPOSE_CMD"
echo "🐳 Running docker compose..."
$DOCKER_COMPOSE_CMD up $DOCKER_ARGS 