# Multi-stage Dockerfile for complete Peptide Profiler Pipeline
FROM python:3.11-slim

# Install system dependencies (including NCBI BLAST+ for hybrid models)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ncbi-blast+ \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy repository code and models
COPY src/ ./src/
COPY models/ ./models/
COPY examples/ ./examples/
COPY app.py .

# Expose Streamlit dashboard port
EXPOSE 8501

# Default entrypoint runs CLI; can also run streamlit
ENTRYPOINT ["python", "src/cli.py"]
CMD ["-h"]
