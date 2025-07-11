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

# Copy our custom files and configurations
COPY src/prepare_brspeech_metadata.py .
COPY src/brspeech_dataset.py src/datasets/
COPY src/train_brspeech.py .
COPY configs/aasist_w2v_brspeech.yaml configs/

# Set environment variables
ENV PYTHONPATH "${PYTHONPATH}:/app"
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Make entrypoint script executable and set as entrypoint
COPY src/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]

# The command to run when the container starts
CMD ["python", "train_brspeech.py", "--config", "configs/aasist_w2v_brspeech.yaml"] 