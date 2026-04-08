"""
Customer Support Triage Environment
OpenEnv-compliant environment for agent evaluation
"""

from src.env import (
    CustomerSupportEnv,
    Action,
    Observation,
    Reward,
    Info,
    Ticket
)

from src.tasks import (
    get_task,
    score_episode,
    EASY_TASK,
    MEDIUM_TASK,
    HARD_TASK
)

__all__ = [
    # Environment
    "CustomerSupportEnv",
    # Models
    "Action",
    "Observation",
    "Reward",
    "Info",
    "Ticket",
    # Tasks
    "get_task",
    "score_episode",
    "EASY_TASK",
    "MEDIUM_TASK",
    "HARD_TASK",
]

__version__ = "1.0.0"
