FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libsox-dev \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

# Clone the original repository
WORKDIR /app
RUN git clone https://github.com/bartlomiejmarek/are_audio_df_polyglots.git .

# Install Python dependencies from the original repository
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables for consistency
ENV PYTHONPATH "${PYTHONPATH}:/app"
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Copy entrypoint and make it executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Default command to ensure container runs, can be overridden
CMD ["/bin/bash"] 