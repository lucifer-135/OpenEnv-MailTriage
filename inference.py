"""
MailTriageEnv — Baseline Inference Script
Reads: API_BASE_URL, MODEL_NAME, HF_TOKEN from environment
Runs all 3 tasks sequentially with seed=42
Emits [START], [STEP], [END] lines to stdout
"""
import os
import json
import requests
from typing import Optional, List
from openai import OpenAI

# --- Config ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK    = "mailtriage-env"

TASK_CONFIGS = [
    {"name": "classify_inbox",        "max_steps": 15, "success_threshold": 0.80},
    {"name": "triage_and_respond",    "max_steps": 30, "success_threshold": 0.70},
    {"name": "inbox_zero_with_policy","max_steps": 50, "success_threshold": 0.65},
]

SYSTEM_PROMPT = """
You are an expert email manager operating a business inbox.
You must process emails using ONLY these JSON actions:

{"action_type": "open_email", "email_id": "eml_001"}
{"action_type": "classify_email", "email_id": "eml_001", "category": "billing", "priority": "high"}
  Valid categories: billing, support, hr, security, vendor, spam, internal, other
  Valid priorities: urgent, high, normal, low
{"action_type": "draft_reply", "email_id": "eml_001", "reply_body": "Dear [Name], ..."}
{"action_type": "send_reply", "email_id": "eml_001"}
{"action_type": "route_email", "email_id": "eml_001", "department": "billing_team"}
  Valid departments: billing_team, support_tier2, legal, hr_dept, security_ops, manager
{"action_type": "escalate_email", "email_id": "eml_001", "reason": "VIP customer complaint"}
{"action_type": "archive_email", "email_id": "eml_001"}
{"action_type": "mark_spam", "email_id": "eml_001"}
{"action_type": "no_op"}

RULES:
- Always open_email before any other action on that email
- Security emails: route to security_ops immediately after opening
- VIP emails: always escalate, never archive
- Billing > $500: route to legal before sending any reply
- Angry customers: your reply MUST contain "apologize" or "sorry"
- Thread > 3 messages: do NOT archive

Respond with ONLY a valid JSON action object. No explanation. No markdown. Pure JSON.
"""

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} "
          f"done={str(done).lower()} error={err}", flush=True)

def log_end(success, steps, score, rewards):
    rstr = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.2f} rewards={rstr}", flush=True)

def env_reset(task_name: str, seed: int = 42) -> dict:
    r = requests.post(f"{ENV_URL}/reset",
                      json={"task_name": task_name, "seed": seed}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(action: dict) -> dict:
    r = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()

def env_grade() -> float:
    r = requests.post(f"{ENV_URL}/grade", timeout=30)
    r.raise_for_status()
    return r.json()["score"]

def env_close():
    requests.post(f"{ENV_URL}/close", timeout=10)

def get_action(client: OpenAI, obs: dict, history: List[str]) -> dict:
    inbox_summary = []
    for email in obs.get("inbox", []):
        inbox_summary.append(
            f"  {email['email_id']}: [{email['sender_name']}] "
            f"{email['subject']} (read={email['already_read']})"
        )
    
    current = obs.get("current_email")
    current_block = ""
    if current:
        current_block = (
            f"\nCURRENT OPEN EMAIL ({current['email_id']}):\n"
            f"From: {current['sender_name']} <{current['sender_email']}>\n"
            f"Subject: {current['subject']}\n"
            f"Body:\n{current['body']}\n"
            f"Thread length: {current['thread_length']}\n"
        )
    
    user_content = (
        f"TASK: {obs['task_objective']}\n\n"
        f"INBOX ({obs['inbox_size']} unread, step {obs['step_number']}, "
        f"{obs['steps_remaining']} steps remaining):\n"
        + "\n".join(inbox_summary)
        + current_block
        + f"\nLast result: {obs['last_action_result']}"
        + (f"\nLast error: {obs['last_action_error']}" if obs.get('last_action_error') else "")
        + f"\n\nPrevious actions:\n" + "\n".join(history[-6:])
        + "\n\nWhat is your next action? Reply with JSON only."
    )
    
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=200,
        )
        text = resp.choices[0].message.content.strip()
        text = text.strip("` \n")
        if text.startswith("json"):
            text = text[4:].strip()
        return json.loads(text)
    except Exception as e:
        return {"action_type": "no_op"}

def run_task(client: OpenAI, task_config: dict) -> tuple[float, List[float], int]:
    task_name = task_config["name"]
    max_steps = task_config["max_steps"]
    success_threshold = task_config["success_threshold"]
    
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    obs = env_reset(task_name=task_name, seed=42)
    
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    try:
        for step_n in range(1, max_steps + 1):
            if obs.get("done", False):
                break
            
            action = get_action(client, obs, history)
            action_str = json.dumps(action)
            
            result = env_step(action)
            obs = result["observation"]
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            error = obs.get("last_action_error")
            
            rewards.append(reward)
            steps_taken = step_n
            history.append(f"Step {step_n}: {action_str} → reward={reward:.2f}")
            
            log_step(step=step_n, action=action_str, reward=reward,
                     done=done, error=error)
            
            if done:
                break
        
        score = env_grade()
        success = score >= success_threshold
        
    except Exception as e:
        # Avoid crashing completely, log and continue to close/metrics
        print(f"[DEBUG] Task error: {e}", flush=True)
    finally:
        env_close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    
    return score, rewards, steps_taken

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    for task_config in TASK_CONFIGS:
        run_task(client, task_config)

if __name__ == "__main__":
    main()
