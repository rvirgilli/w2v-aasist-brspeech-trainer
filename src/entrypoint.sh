#!/bin/bash
set -e

echo "🚀 Starting BrSpeech training pipeline..."

# Step 1: Generate metadata
echo "📊 Generating metadata CSVs..."
python /app/prepare_brspeech_metadata.py

# Step 2: Validate generated metadata
echo "🔎 Validating generated metadata..."
for f in metadata/train.csv metadata/val.csv metadata/test.csv; do
    if [ ! -f "$f" ] || [ ! -s "$f" ]; then
        echo "❌ Metadata validation failed: File '$f' was not created or is empty."
        exit 1
    fi
done
echo "✅ Metadata validation successful."

# Step 3: Execute the main training command
echo "🎯 Starting model training or testing..."
exec "$@" $PYTHON_ARGS 