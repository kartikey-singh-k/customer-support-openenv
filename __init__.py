"""
Customer Support Triage OpenEnv Environment

A real-world OpenEnv-compliant environment for evaluating AI agents on
customer support ticket triage tasks.
"""

__version__ = "1.0.0"
__author__ = "Kartikey Singh"
__email__ = "kartikey@example.com"

from src.env import CustomerSupportEnv, Action, Observation, Reward, Info, Ticket
from src.tasks import get_task, score_episode, EASY_TASK, MEDIUM_TASK, HARD_TASK

__all__ = [
    "CustomerSupportEnv",
    "Action", 
    "Observation",
    "Reward",
    "Info", 
    "Ticket",
    "get_task",
    "score_episode", 
    "EASY_TASK",
    "MEDIUM_TASK", 
    "HARD_TASK",
]