# Stage 1: Builder
# Use a Python base image with build tools
FROM python:3.9-slim-bullseye as builder

# Set working directory
WORKDIR /app

# Install build dependencies and libraries required by opencv-python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libfontconfig1 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
# Use a slim Python base image for the final image
FROM python:3.9-slim-bullseye

# Set working directory
WORKDIR /app

# Install runtime dependencies and libraries required by opencv-python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from the builder stage
COPY --from=builder /root/.local /root/.local

# Copy application source code and tests
COPY ./src /app/src

# Add /app/src to the PYTHONPATH
ENV PYTHONPATH=/app/src

# Set environment variable for Python to find user-installed packages
ENV PATH="/root/.local/bin:$PATH"

# Expose the port the FastAPI application runs on
EXPOSE 8000

# Command to run the FastAPI application using uvicorn
CMD ["uvicorn", "ingest_service.app:app", "--host", "0.0.0.0", "--port", "8000"]