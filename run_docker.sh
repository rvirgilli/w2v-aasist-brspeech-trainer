#!/bin/bash

# BrSpeech Training Docker Runner
# Usage: ./run_docker.sh <real_dataset_path> <spoof_dataset_path> [output_path] [additional_docker_compose_args]

set -e

# Check for required arguments
if [ $# -lt 2 ]; then
    echo "❌ Usage: $0 <real_dataset_path> <spoof_dataset_path> [output_path] [additional_args...]"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/BRSpeech_CML_TTS_v04012024 /path/to/brspeech_df"
    echo "  $0 /path/to/BRSpeech_CML_TTS_v04012024 /path/to/brspeech_df /path/to/outputs"
    echo "  $0 /path/to/BRSpeech_CML_TTS_v04012024 /path/to/brspeech_df ./outputs --build"
    echo ""
    echo "Dataset structure expected:"
    echo "  Real dataset: <real_dataset_path>/{train,dev,test}.csv and train/audio/*.flac"
    echo "  Spoof dataset: <spoof_dataset_path>/{f5tts,fish-speech,toucantts,xtts,yourtts}/{train,dev,test}/audio/*.flac"
    exit 1
fi

REAL_DATASET_PATH="$1"
SPOOF_DATASET_PATH="$2"
OUTPUT_PATH="${3:-./outputs}"

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

# Shift the first 3 arguments so remaining args go to docker-compose
shift 3

# Export environment variables for docker-compose
export REAL_DATASET_PATH
export SPOOF_DATASET_PATH
export OUTPUT_PATH

# Run docker compose with any additional arguments
echo "🐳 Running docker compose..."
docker compose up "$@" 