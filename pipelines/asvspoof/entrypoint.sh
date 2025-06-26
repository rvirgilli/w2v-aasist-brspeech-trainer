#!/bin/bash
set -e

echo "🚀 ASVspoof 2019 LA Training Pipeline Started"

# Define the environment variables that train.py expects
export PATH_TO_ASV="/data/ASVspoof2019_LA"
export PATH_TO_ASV_PROTOCOL="/data/ASVspoof2019_LA/ASVspoof2019_LA_cm_protocols"

echo "Environment variables set:"
echo "  PATH_TO_ASV=$PATH_TO_ASV"
echo "  PATH_TO_ASV_PROTOCOL=$PATH_TO_ASV_PROTOCOL"

# Debug: Check if the paths exist
echo "🔍 Debug - Checking if paths exist:"
echo "  $PATH_TO_ASV exists: $(test -d "$PATH_TO_ASV" && echo "YES" || echo "NO")"
echo "  $PATH_TO_ASV_PROTOCOL exists: $(test -d "$PATH_TO_ASV_PROTOCOL" && echo "YES" || echo "NO")"
echo "  Protocol file exists: $(test -f "$PATH_TO_ASV_PROTOCOL/ASVspoof2019.LA.cm.train.trn.txt" && echo "YES" || echo "NO")"

# List mounted data to verify volume mapping
echo "🔍 Debug - Contents of /data:"
ls -la /data/ || echo "Failed to list /data"

# Create a default configuration file (still needed for model parameters)
CONFIG_PATH="/app/configs/aasist_w2v_asv.yaml"
echo "Creating default training configuration at $CONFIG_PATH"
mkdir -p /app/configs
cat > $CONFIG_PATH <<- EOM
model:
  name: w2v_aasist
  parameters: {} 

training:
  batch_size: 4
  learning_rate: 5e-6
  weight_decay: 5e-7
  epochs: 25
  optimizer: Adam
EOM

# Execute the main training command passed from docker-compose
echo "🎯 Starting model training..."
exec python train.py --config $CONFIG_PATH --model_out_dir /app/outputs "$@" $PYTHON_ARGS 