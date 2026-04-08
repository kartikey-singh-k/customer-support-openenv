import os
import json
import sys
from dotenv import load_dotenv
from openai import OpenAI
from src.env import CustomerSupportEnv, Action, Reward, Info
from src.tasks import get_task

# Load .env file for local testing (Hackathon container will inject these directly)
load_dotenv()

# --- Required Environment Variables per Guidelines ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# --- Initialize Client ---
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def run_inference(task_level: str):
    task_data = get_task(task_level)
    env = CustomerSupportEnv(initial_tickets=task_data["tickets"], ground_truth=task_data["ground_truth"])
    obs = env.reset()
    done = False
    
    # State tracking for required output
    steps = 0
    rewards = []
    success = False
    
    system_prompt = f"""You are an autonomous customer support triage agent. 
You will receive a queue of tickets. You must output a JSON action to triage ONE ticket at a time.
Action Schema: {Action.model_json_schema()}
Only output raw JSON. Do not include markdown code blocks.
"""

    # RULE: One [START] line at episode begin.
    print(f"[START] task={task_level} env=customer-support-triage model={MODEL_NAME}")

    try:
        while not done:
            steps += 1
            action_str = ""
            reward = 0.00
            error_msg = "null"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Current State: {obs.model_dump_json()}"}
            ]
            
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    response_format={"type": "json_object"}
                )
                
                # Parse action and format it without spaces for the STDOUT
                raw_content = response.choices[0].message.content
                action_dict = json.loads(raw_content)
                action = Action(**action_dict)
                action_str = json.dumps(action_dict, separators=(',', ':'))
                
                # Step the environment
                obs, reward_obj, done, info = env.step(action)
                
                # Extract reward value from Reward object
                reward = reward_obj.value if isinstance(reward_obj, Reward) else reward_obj
                
                if info.error:
                    error_msg = str(info.error).replace('\n', ' ')
                    
            except Exception as e:
                error_msg = str(e).replace('\n', ' ')
                action_str = "invalid_action"
                done = True # Abort on agent crash to prevent infinite loops
            
            rewards.append(reward)
            
            # Formatting rules: 2 decimal places, lowercase bools
            done_str = "true" if done else "false"
            reward_fmt = f"{reward:.2f}"
            
            # RULE: One [STEP] line per step, immediately after env.step() returns.
            print(f"[STEP] step={steps} action={action_str} reward={reward_fmt} done={done_str} error={error_msg}")

        # Define successful completion: queue empty and no critical errors
        if not env.state().open_tickets and error_msg == "null":
            success = True

    finally:
        # RULE: One [END] line after env.close(), always emitted (even on exception).
        success_str = "true" if success else "false"
        
        if not rewards:
            rewards_str = "0.00"
        else:
            rewards_str = ",".join([f"{r:.2f}" for r in rewards])
            
        print(f"[END] success={success_str} steps={steps} rewards={rewards_str}")

if __name__ == "__main__":
    # Execute the tasks sequentially
    # Do not add any extra print statements here!
    for level in ["easy", "medium", "hard"]:
        run_inference(level)