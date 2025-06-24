FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libsox-dev \
    libsox3 \
    && rm -rf /var/lib/apt/lists/*

# Clone the base repository
WORKDIR /app
RUN git clone https://github.com/bartlomiejmarek/are_audio_df_polyglots.git .

# Install Python dependencies
COPY requirements_additions.txt .
RUN pip install -r requirements.txt
RUN pip install -r requirements_additions.txt

# Copy our custom files
COPY src/prepare_brspeech_metadata.py .
COPY src/brspeech_dataset.py src/datasets/
COPY configs/aasist_w2v_brspeech.yaml configs/
COPY src/train_brspeech.py .

# Set environment variables
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training pipeline
CMD ["bash", "-c", "\
    echo '🚀 Starting BrSpeech training pipeline...' && \
    echo '📊 Generating metadata CSVs...' && \
    python prepare_brspeech_metadata.py && \
    if [ ! -f 'metadata/train.csv' ]; then echo '❌ Failed to generate metadata files'; exit 1; fi && \
    echo '✅ Metadata generation complete' && \
    echo '🎯 Starting model training...' && \
    python train_brspeech.py \
        --config configs/aasist_w2v_brspeech.yaml \
        --batch_size ${BATCH_SIZE:-4} \
        --epochs ${EPOCHS:-25} \
        --lr ${LEARNING_RATE:-5e-6} \
        --weight_decay ${WEIGHT_DECAY:-5e-7} && \
    echo '🎉 Training complete!'"] 