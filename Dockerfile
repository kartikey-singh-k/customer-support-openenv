FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Labels required for Hugging Face Spaces integration
LABEL maintainer="OpenEnv"
LABEL tags="openenv"

# Validate environment during build phase by testing imports
RUN python -c "from src.env import CustomerSupportEnv, Action, Observation, Reward, Info; print('Environment validation passed')"

# Expose port for the server
EXPOSE 7860

# Run the FastAPI server
CMD ["python", "server.py"]