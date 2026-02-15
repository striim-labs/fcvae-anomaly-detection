FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (Java for Spark, librdkafka for confluent-kafka)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    librdkafka-dev \
    curl \
    openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME (detect architecture dynamically)
RUN JAVA_PATH=$(find /usr/lib/jvm -name "java-21-openjdk-*" -type d | head -1) && \
    echo "JAVA_HOME=$JAVA_PATH" >> /etc/environment
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy pyproject.toml and install dependencies
COPY pyproject.toml .
RUN uv pip install --system -r pyproject.toml

# Copy FCVAE application code
COPY main.py .
COPY base_detector.py .

# Core FCVAE model files
COPY fcvae_model.py .
COPY fcvae_registry.py .
COPY fcvae_scorer.py .
COPY fcvae_augment.py .
COPY fcvae_streaming_detector.py .
COPY attention.py .

# Training and evaluation (optional, for development)
COPY train_fcvae.py .
COPY evaluate_fcvae.py .

# Create models directory (will be mounted as volume)
RUN mkdir -p models

# Expose Dash port
EXPOSE 8050

# Default environment variables for FCVAE
ENV DETECTOR_TYPE=fcvae
ENV WINDOW_SIZE=24
ENV MIN_SAMPLES=24
ENV MODEL_PATH=models/fcvae_60_kl
ENV N_SAMPLES=16

# Run the application
CMD ["python", "-u", "main.py"]
