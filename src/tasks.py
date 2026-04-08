from typing import Dict, Any

def score_episode(env_state: Any, ground_truth: Dict[str, Dict[str, str]]) -> float:
    """
    Grader: Returns a score strictly between 0.0 and 1.0 based on final state correctness.
    Ensures score is never exactly 0.0 or 1.0 as required by hackathon validation.
    """
    resolved = {t.id: t for t in env_state.resolved_tickets}
    
    total_points = 0.0
    max_points = len(ground_truth) * 2.0 # 1 point for cat, 1 for priority per ticket

    for t_id, truth in ground_truth.items():
        if t_id in resolved:
            ticket = resolved[t_id]
            if ticket.assigned_category == truth["category"]:
                total_points += 1.0
            if ticket.assigned_priority == truth["priority"]:
                total_points += 1.0

    # Calculate base score
    base_score = total_points / max_points
    
    # Ensure score is strictly between 0 and 1 (exclusive)
    # Map [0, 1] to (0.01, 0.99) to satisfy hackathon requirements
    adjusted_score = 0.01 + (base_score * 0.98)
    
    return round(adjusted_score, 3)

# --- TASKS ---

EASY_TASK = {
    "tickets": [
        {"id": "T001", "text": "I was double charged for my subscription this month. Please help!"}
    ],
    "ground_truth": {
        "T001": {"category": "Billing", "priority": "High"}
    }
}

MEDIUM_TASK = {
    "tickets": [
        {"id": "T101", "text": "How do I update my profile picture?"},
        {"id": "T102", "text": "The app keeps crashing when I try to export my report. Deadline is tomorrow!"},
        {"id": "T103", "text": "I want a refund, your product is terrible."}
    ],
    "ground_truth": {
        "T101": {"category": "General", "priority": "Low"},
        "T102": {"category": "Tech Support", "priority": "Urgent"},
        "T103": {"category": "Refund", "priority": "Medium"}
    }
}

HARD_TASK = {
    "tickets": [
        {"id": "T201", "text": "I need help with my bill, but also the website is down for me right now."},
        {"id": "T202", "text": "Can you extend my trial? Also, how do I integrate the API?"},
        {"id": "T203", "text": "Cancel my account immediately. You billed me after I sent an email last week."},
        {"id": "T204", "text": "Hey, just wanted to say I love the new feature update!"},
        {"id": "T205", "text": "URGENT: Entire database deleted after clicking 'sync'. Need engineers NOW."}
    ],
    "ground_truth": {
        "T201": {"category": "Tech Support", "priority": "Urgent"}, # Site down overrides bill
        "T202": {"category": "Tech Support", "priority": "Medium"}, # API integration is tech support
        "T203": {"category": "Refund", "priority": "High"},
        "T204": {"category": "General", "priority": "Low"},
        "T205": {"category": "Tech Support", "priority": "Urgent"}
    }
}

def get_task(level: str):
    if level == "easy":
        return EASY_TASK
    elif level == "medium":
        return MEDIUM_TASK
    elif level == "hard":
        return HARD_TASK
    raise ValueError("Invalid task level.")