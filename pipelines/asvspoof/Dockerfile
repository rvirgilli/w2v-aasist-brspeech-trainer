FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libsox-dev \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone the repository
RUN git clone https://github.com/bartlomiejmarek/are_audio_df_polyglots.git .

# Install Python dependencies with increased timeout and retries
RUN pip install --no-cache-dir --timeout 300 --retries 3 -r requirements.txt

# Set environment variables for consistency
ENV PYTHONPATH "${PYTHONPATH}:/app"
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Copy the entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"] 