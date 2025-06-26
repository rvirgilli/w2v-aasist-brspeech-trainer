FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libsox-dev \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

# Clone the base repository
WORKDIR /app
RUN git clone https://github.com/bartlomiejmarek/are_audio_df_polyglots.git .

# Install Python dependencies
RUN pip install -r requirements.txt

# Set environment variables
ENV PYTHONPATH="${PYTHONPATH}:/app"
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Make entrypoint script executable and set as entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]

# The command to run when the container starts
CMD ["python", "train.py", "--config", "/app/configs/aasist_w2v_asv.yaml"] 