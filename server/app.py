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
import sys
import traceback

# Add current directory to Python path
sys.path.insert(0, '/app')

app = FastAPI(
    title="Customer Support Triage OpenEnv",
    description="Real-world environment for customer support ticket triage",
    version="1.0.0"
)

# Global environment instance
env_instance: Optional[Any] = None
current_task: str = "easy"

# Lazy import to avoid hanging
_env_module = None
_tasks_module = None

def get_env_module():
    """Lazy import of environment module."""
    global _env_module
    if _env_module is None:
        try:
            from src.env import CustomerSupportEnv, Action, Observation, Reward, Info
            _env_module = {
                'CustomerSupportEnv': CustomerSupportEnv,
                'Action': Action,
                'Observation': Observation, 
                'Reward': Reward,
                'Info': Info
            }
        except Exception as e:
            print(f"Error importing env module: {e}")
            traceback.print_exc()
            raise
    return _env_module

def get_tasks_module():
    """Lazy import of tasks module.""" 
    global _tasks_module
    if _tasks_module is None:
        try:
            from src.tasks import get_task
            _tasks_module = {'get_task': get_task}
        except Exception as e:
            print(f"Error importing tasks module: {e}")
            traceback.print_exc()
            raise
    return _tasks_module


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
        
        # Load task data using lazy import
        tasks_module = get_tasks_module()
        task_data = tasks_module['get_task'](task_level)
        
        # Create new environment instance using lazy import
        env_module = get_env_module()
        env_instance = env_module['CustomerSupportEnv'](
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
        import traceback
        print(f"Reset error: {e}")
        traceback.print_exc()
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
        # Parse action using lazy import
        env_module = get_env_module()
        action = env_module['Action'](**request.action)
        
        # Execute step
        observation, reward, done, info = env_instance.step(action)
        
        return JSONResponse(content={
            "observation": observation.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info.model_dump()
        })
        
    except Exception as e:
        import traceback
        print(f"Step error: {e}")
        traceback.print_exc()
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
def main():
    """Entry point for the CLI script."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()