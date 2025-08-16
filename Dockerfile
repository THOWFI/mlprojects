# Use Python 3.11 slim base image
FROM python:3.11-slim-bullseye

# Set working directory
WORKDIR /app

# Copy project files into container
COPY . /app

# Update & install required system dependencies (including AWS CLI if needed)
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    awscli \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Command to run your app
CMD ["python", "application.py"]
