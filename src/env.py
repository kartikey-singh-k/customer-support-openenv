from pydantic import BaseModel, Field
from typing import List, Dict, Any, Tuple, Optional
import copy

# --- SPACES ---

class Ticket(BaseModel):
    id: str
    text: str
    status: str = "open" # open, resolved
    assigned_category: Optional[str] = None
    assigned_priority: Optional[str] = None

class Observation(BaseModel):
    open_tickets: List[Ticket]
    resolved_tickets: List[Ticket]
    last_feedback: str = ""
    current_step: int = 0

class Action(BaseModel):
    ticket_id: str = Field(description="The ID of the ticket to triage.")
    category: str = Field(description="Must be one of: 'Billing', 'Tech Support', 'Refund', 'General'.")
    priority: str = Field(description="Must be one of: 'Low', 'Medium', 'High', 'Urgent'.")

class Reward(BaseModel):
    value: float = Field(description="Numeric reward value")
    feedback: str = Field(default="", description="Human-readable feedback")

class Info(BaseModel):
    error: Optional[str] = Field(default=None, description="Error message if any")
    reason: Optional[str] = Field(default=None, description="Reason for termination")

# --- ENVIRONMENT ---

class CustomerSupportEnv:
    VALID_CATEGORIES = {"Billing", "Tech Support", "Refund", "General"}
    VALID_PRIORITIES = {"Low", "Medium", "High", "Urgent"}
    MAX_STEPS = 10

    def __init__(self, initial_tickets: List[Dict[str, str]], ground_truth: Dict[str, Dict[str, str]]):
        """
        initial_tickets: List of dicts with 'id' and 'text'.
        ground_truth: Dict mapping ticket_id to expected category and priority for reward calculation.
        """
        self.initial_tickets = [Ticket(**t) for t in initial_tickets]
        self.ground_truth = ground_truth
        
        # Validate that all tickets have ground truth
        for ticket in self.initial_tickets:
            if ticket.id not in ground_truth:
                raise ValueError(f"Missing ground truth for ticket {ticket.id}")
        
        self._state: Observation = self._build_initial_state()
        self.total_reward = 0.0

    def _build_initial_state(self) -> Observation:
        return Observation(
            open_tickets=copy.deepcopy(self.initial_tickets),
            resolved_tickets=[],
            last_feedback="Environment initialized. Awaiting actions.",
            current_step=0
        )

    def reset(self) -> Observation:
        self._state = self._build_initial_state()
        self.total_reward = 0.0
        return self._state

    def state(self) -> Observation:
        return self._state

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Info]:
        self._state.current_step += 1
        reward_value = 0.0
        done = False
        info = Info()

        # 1. Validate Action limits & loops
        if self._state.current_step >= self.MAX_STEPS:
            self._state.last_feedback = "Max steps reached."
            reward = Reward(value=0.03, feedback="Max steps reached.")  # Avoid exact 0.0
            info.reason = "timeout"
            return self._state, reward, True, info

        # 2. Validate ticket exists and is open
        ticket_index = next((i for i, t in enumerate(self._state.open_tickets) if t.id == action.ticket_id), None)
        if ticket_index is None:
            reward_value = -0.1 # Penalty for hallucinating an ID or trying to resolve a closed ticket
            self._state.last_feedback = f"Error: Ticket {action.ticket_id} not found or already closed."
            reward = Reward(value=reward_value, feedback=self._state.last_feedback)
            info.error = self._state.last_feedback
            return self._state, reward, False, info

        # 3. Validate Enum fields
        if action.category not in self.VALID_CATEGORIES or action.priority not in self.VALID_PRIORITIES:
            reward_value = -0.1 # Penalty for invalid formatting
            self._state.last_feedback = "Error: Invalid category or priority applied."
            reward = Reward(value=reward_value, feedback=self._state.last_feedback)
            info.error = self._state.last_feedback
            return self._state, reward, False, info

        # 4. Process Action & Calculate Incremental Reward
        ticket = self._state.open_tickets.pop(ticket_index)
        ticket.assigned_category = action.category
        ticket.assigned_priority = action.priority
        ticket.status = "resolved"
        self._state.resolved_tickets.append(ticket)

        truth = self.ground_truth.get(ticket.id, {})
        
        # Incremental reward logic
        step_reward = 0.0
        if ticket.assigned_category == truth.get("category"):
            step_reward += 0.5
        if ticket.assigned_priority == truth.get("priority"):
            step_reward += 0.5
        
        # Penalty for completely wrong routing
        if step_reward == 0.0:
             step_reward = -0.2

        # Adjust reward to avoid exact 0.0, 0.5, 1.0 values  
        # Map rewards to valid range (0.01, 0.99)
        if step_reward == 0.0:
            step_reward = 0.01
        elif step_reward == 0.5:  
            step_reward = 0.51
        elif step_reward == 1.0:
            step_reward = 0.99
        elif step_reward == -0.2:
            step_reward = 0.02  # Small positive for wrong answer
        
        reward_value = step_reward
        self.total_reward += reward_value
        self._state.last_feedback = f"Successfully triaged ticket {ticket.id}. Reward: {reward_value}"

        # 5. Check Termination
        if not self._state.open_tickets:
            done = True
            self._state.last_feedback = "All tickets triaged."

        reward = Reward(value=reward_value, feedback=self._state.last_feedback)
        return self._state, reward, done, info
    