#!/bin/bash
#
# Dataset Structure Validator
#
# This script checks if the provided dataset paths for real and spoofed
# audio data conform to the expected structure for the BrSpeech training pipeline.
#
# Usage:
#   ./validate_datasets.sh --real-data <path_to_real> --spoof-data <path_to_spoof>

set -e

# --- Helper Functions ---
log() {
    local level=$1
    local msg=$2
    echo "[$level] $2"
}

show_usage() {
    echo "Usage: $0 --real-data <path> --spoof-data <path>"
    echo ""
    echo "Parameters:"
    echo "  --real-data   Path to the real speech dataset (must contain train.csv, dev.csv, test.csv)."
    echo "  --spoof-data  Path to the spoofed speech dataset (must contain model subdirectories like 'xtts')."
    echo ""
    echo "Example:"
    echo "  ./validate_datasets.sh --real-data /path/to/BRSpeech --spoof-data /path/to/brspeech_df"
}

# --- Argument Parsing ---
REAL_DATASET_PATH=""
SPOOF_DATASET_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --real-data)
            REAL_DATASET_PATH="$2"
            shift 2
            ;;
        --spoof-data)
            SPOOF_DATASET_PATH="$2"
            shift 2
            ;;
        *)
            log "ERROR" "Unknown parameter: $1"
            show_usage
            exit 1
            ;;
    esac
done

if [ -z "$REAL_DATASET_PATH" ] || [ -z "$SPOOF_DATASET_PATH" ]; then
    log "ERROR" "Both --real-data and --spoof-data parameters are required."
    show_usage
    exit 1
fi

# --- Path and Structure Validation ---
log "INFO" "🚀 Starting dataset validation..."

# 1. Validate top-level paths exist
if [ ! -d "$REAL_DATASET_PATH" ]; then
    log "ERROR" "Real data path does not exist: $REAL_DATASET_PATH"
    exit 1
fi

if [ ! -d "$SPOOF_DATASET_PATH" ]; then
    log "ERROR" "Spoof data path does not exist: $SPOOF_DATASET_PATH"
    exit 1
fi
log "INFO" "✅ Top-level directories exist."

# 2. Validate real data structure
log "INFO" "Validating real data structure..."
for f in train.csv dev.csv test.csv; do
    if [ ! -f "$REAL_DATASET_PATH/$f" ]; then
        log "ERROR" "Real data validation failed: Missing file '$f' in $REAL_DATASET_PATH"
        exit 1
    fi
done
log "INFO" "✅ Real data CSVs found."

# 3. Validate spoof data structure
log "INFO" "Validating spoof data structure..."
found_spoof_model=false
expected_models=("f5tts" "fish-speech" "toucantts" "xtts" "yourtts")
for model_dir in "${expected_models[@]}"; do
    if [ -d "$SPOOF_DATASET_PATH/$model_dir" ]; then
        found_spoof_model=true
        log "INFO" " -> Found spoof model directory: $model_dir"
    fi
done

if [ "$found_spoof_model" = false ]; then
    log "ERROR" "Spoof data validation failed: Could not find any expected model subdirectories (e.g., ${expected_models[*]}) in $SPOOF_DATASET_PATH"
    exit 1
fi
log "INFO" "✅ Spoof data structure looks good."

log "INFO" "🎉 All dataset validations passed successfully!"
exit 0 