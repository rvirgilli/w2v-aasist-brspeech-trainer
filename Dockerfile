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
COPY src/ /app/src/
COPY src/brspeech_dataset.py src/datasets/
COPY configs/aasist_w2v_brspeech.yaml configs/

# Set environment variables
ENV PYTHONPATH "${PYTHONPATH}:/app"
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Make entrypoint script executable and set as entrypoint
RUN chmod +x /app/src/entrypoint.sh
ENTRYPOINT ["/app/src/entrypoint.sh"]

# The command to run when the container starts
CMD ["python", "src/train_brspeech.py", "--config", "configs/aasist_w2v_brspeech.yaml"] 