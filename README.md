---
title: Customer Support Triage OpenEnv
emoji: 🎫
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - agent-evaluation
  - customer-support
pinned: false
---
# OpenEnv: Customer Support Triage

A real-world OpenEnv-compliant environment simulating a customer support agent. The agent must process a queue of incoming text tickets, correctly categorize their domain, and assign business priority.

## Overview
Customer support triaging is a high-volume real-world task. This environment trains and evaluates agents on their ability to:
1. Parse noisy human text.
2. Resolve conflicting information (e.g., a billing issue caused by a severe system outage should be prioritized as Urgent Tech Support).
3. Avoid looping actions and correctly format outputs to interface with enterprise APIs.

## Architecture

This environment is deployed as a **FastAPI HTTP server** that provides RESTful endpoints for environment interaction, following the OpenEnv specification.

### HTTP API Endpoints

- `GET /health` - Health check
- `GET /info` - Environment metadata and specifications
- `POST /reset` - Reset environment to initial state
- `POST /step` - Execute an action and get next state
- `GET /state` - Get current environment state

## Environment Details

### Action Space (Pydantic Model)
* `ticket_id` (str): Target ticket to process.
* `category` (str): ['Billing', 'Tech Support', 'Refund', 'General']
* `priority` (str): ['Low', 'Medium', 'High', 'Urgent']

### Observation Space (Pydantic Model)
* `open_tickets`: List of tickets awaiting triage.
* `resolved_tickets`: History of actions taken.
* `last_feedback`: Immediate environmental feedback (success, errors).
* `current_step`: Tracks trajectory length to prevent infinite loops.

### Reward Function
The environment provides **dense, incremental rewards**:
* `+0.5` for correctly identifying the Category.
* `+0.5` for correctly identifying the Priority.
* `-0.1` for invalid parameters or attempting to operate on closed/non-existent tickets.
* `-0.2` for completely misclassifying a ticket.

## Tasks
* **Easy**: 1 clearly written ticket. Tests basic formatting and API compliance.
* **Medium**: 3 tickets. Includes frustrated users and ambiguous phrasing.
* **Hard**: 5 tickets. Requires deep contextual reasoning (e.g., prioritizing severe technical outages over angry billing complaints). May require retry logic.

## Setup and Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Server Locally

```bash
# Start the FastAPI server
python server.py
```

The server will start on `http://localhost:7860`

**Test the endpoints:**
```bash
# Health check
curl http://localhost:7860/health

# Reset environment
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy"}'

# Execute a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"ticket_id": "T001", "category": "Billing", "priority": "High"}}'

# Get current state
curl http://localhost:7860/state
```

### 3. Run via Docker

**Build and run the server:**
```bash
docker build -t customer-support-env .
docker run -p 7860:7860 customer-support-env
```
**With Hugging Face models:**
```bash
docker build -t openenv-triage .
docker run -e HF_TOKEN="your_hf_token" \
           -e API_BASE_URL="https://router.huggingface.co/v1" \
           -e MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct" \
           openenv-triage
```
The server will be available at `http://localhost:7860`

**Test the running container:**
```bash
curl http://localhost:7860/health
```

### 4. Run Baseline Inference Script (Optional)

For standalone LLM evaluation without the server:
```bash
export HF_TOKEN="your_huggingface_token"
# Optional: export API_BASE_URL="https://api.openai.com/v1"
# Optional: export MODEL_NAME="gpt-4o-mini"
python inference.py
```

## API Reference

### POST /reset

Reset the environment to initial state.

**Request Body:**
```json
{
  "task": "easy"  // Options: "easy", "medium", "hard"
}
```

**Response:**
```json
{
  "observation": {
    "open_tickets": [...],
    "resolved_tickets": [],
    "last_feedback": "Environment initialized. Awaiting actions.",
    "current_step": 0
  },
  "task": "easy",
  "status": "reset_successful"
}
```

### POST /step

Execute an action in the environment.

**Request Body:**
```json
{
  "action": {
    "ticket_id": "T001",
    "category": "Billing",
    "priority": "High"
  }
}
```

**Response:**
```json
{
  "observation": {...},
  "reward": {
    "value": 1.0,
    "feedback": "Successfully triaged ticket T001. Reward: 1.0"
  },
  "done": true,
  "info": {
    "error": null,
    "reason": null
  }
}
```

### GET /state

Get the current environment state.

**Response:**
```json
{
  "observation": {...},
  "task": "easy"
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "customer-support-triage-env"
}
```

### GET /info

Get environment metadata.

**Response:**
```json
{
  "name": "customer-support-triage",
  "version": "1.0.0",
  "description": "Real-world customer support ticket triage environment",
  "tasks": ["easy", "medium", "hard"],
  "action_space": {...},
  "observation_space": {...}
}
```

### Environment Variables

**For the Server:**
- `PORT` (optional): Server port, defaults to 7860

**For Baseline Inference Script (inference.py):**
- `HF_TOKEN` (required): Hugging Face API token or OpenAI API key
- `API_BASE_URL` (optional): API endpoint, defaults to "https://api.openai.com/v1"
  - For Hugging Face: `https://router.huggingface.co/v1`
- `MODEL_NAME` (optional): Model identifier, defaults to "gpt-4o-mini"
  - Tested with: `meta-llama/Meta-Llama-3-8B-Instruct`, `gpt-4o-mini`

## Validation

### Test the HTTP Server

Start the server and test endpoints:
```bash
# Start server
python server.py

# In another terminal, test endpoints
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task": "easy"}'
```

### Test Environment Code

Validate the OpenEnv implementation:
```bash
# Test environment imports and initialization
python -c "from src.env import CustomerSupportEnv, Action, Observation, Reward, Info; print('✓ Environment validation passed')"

# Test with sample task
python -c "from src.env import CustomerSupportEnv; from src.tasks import EASY_TASK; env = CustomerSupportEnv(EASY_TASK['tickets'], EASY_TASK['ground_truth']); print('✓ Environment initialized successfully')"
```

## Baseline Performance

Baseline inference uses the OpenAI API client with configurable model. The output follows the hackathon-required format:

```
[START] task=<task_name> env=customer-support-triage model=<model_name>
[STEP] step=<n> action=<action_json> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> rewards=<r1,r2,...>
```

Results are reproducible when using the same seed and model configuration.

| Task | Steps | Score (Avg Reward) | Notes |
|------|-------|-------------------|-------|
| Easy | 1 | 1.00 | Perfect execution. |
| Medium | 3 | 0.50 | Consistent partial credit on priority/category. |
| Hard | 6 | 0.42 | Handled duplicate action error and partial logic correctly. |

*Run `python inference.py` with your HF_TOKEN to reproduce baseline scores.*

## Deployment

This environment is designed to be deployed as a containerized Hugging Face Space:
- Tagged with `openenv`
- Fully containerized execution via Dockerfile
- All dependencies specified in requirements.txt


