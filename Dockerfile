FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Labels required for Hugging Face Spaces integration
LABEL maintainer="OpenEnv"
LABEL tags="openenv"

# Validate environment during build phase
RUN python validate_env.py

# Expose port for the server
EXPOSE 7860

# Run the FastAPI server via uvicorn
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]