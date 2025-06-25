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
        --real-data=*)
            REAL_DATASET_PATH="${1#*=}"
            shift
            ;;
        --spoof-data=*)
            SPOOF_DATASET_PATH="${1#*=}"
            shift
            ;;
        --output=*)
            OUTPUT_PATH="${1#*=}"
            shift
            ;;
        --build)
            DOCKER_ARGS="$DOCKER_ARGS --build"
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            # Pass unknown arguments to docker-compose
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