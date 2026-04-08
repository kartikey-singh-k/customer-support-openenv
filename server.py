"""
FastAPI server for Customer Support Triage OpenEnv environment.
Provides HTTP endpoints for reset, step, and state operations.
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
import os

from src.env import CustomerSupportEnv, Action, Observation, Reward, Info
from src.tasks import get_task

app = FastAPI(
    title="Customer Support Triage OpenEnv",
    description="Real-world environment for customer support ticket triage",
    version="1.0.0"
)

# Global environment instance
env_instance: Optional[CustomerSupportEnv] = None
current_task: str = "easy"


class ResetRequest(BaseModel):
    task: str = "easy"  # easy, medium, or hard


class StepRequest(BaseModel):
    action: Dict[str, Any]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "customer-support-triage-env",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "customer-support-triage-env"
    }


@app.post("/reset")
async def reset(request: ResetRequest = None):
    """
    Reset the environment to initial state.
    
    Args:
        request: Optional ResetRequest with task level (easy, medium, hard)
    
    Returns:
        Initial observation
    """
    global env_instance, current_task
    
    try:
        # Get task level from request or use default
        task_level = request.task if request else "easy"
        current_task = task_level
        
        # Load task data
        task_data = get_task(task_level)
        
        # Create new environment instance
        env_instance = CustomerSupportEnv(
            initial_tickets=task_data["tickets"],
            ground_truth=task_data["ground_truth"]
        )
        
        # Reset and get initial observation
        observation = env_instance.reset()
        
        return JSONResponse(content={
            "observation": observation.model_dump(),
            "task": task_level,
            "status": "reset_successful"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step")
async def step(request: StepRequest):
    """
    Execute one step in the environment.
    
    Args:
        request: StepRequest containing the action
    
    Returns:
        observation, reward, done, info
    """
    global env_instance
    
    if env_instance is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )
    
    try:
        # Parse action
        action = Action(**request.action)
        
        # Execute step
        observation, reward, done, info = env_instance.step(action)
        
        return JSONResponse(content={
            "observation": observation.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info.model_dump()
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Step failed: {str(e)}")


@app.get("/state")
async def state():
    """
    Get current environment state.
    
    Returns:
        Current observation
    """
    global env_instance
    
    if env_instance is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )
    
    try:
        current_state = env_instance.state()
        return JSONResponse(content={
            "observation": current_state.model_dump(),
            "task": current_task
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get state: {str(e)}")


@app.get("/info")
async def info():
    """Get environment information and metadata."""
    return {
        "name": "customer-support-triage",
        "version": "1.0.0",
        "description": "Real-world customer support ticket triage environment",
        "tasks": ["easy", "medium", "hard"],
        "action_space": {
            "ticket_id": "str",
            "category": ["Billing", "Tech Support", "Refund", "General"],
            "priority": ["Low", "Medium", "High", "Urgent"]
        },
        "observation_space": {
            "open_tickets": "List[Ticket]",
            "resolved_tickets": "List[Ticket]",
            "last_feedback": "str",
            "current_step": "int"
        }
    }


if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", 7860))
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
