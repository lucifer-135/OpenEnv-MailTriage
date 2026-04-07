from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from env.env import MailTriageEnv
from env.models import EmailAction, EmailObservation, StepResponse, TaskInfo
from env.graders import get_grader
import threading

app = FastAPI(title="MailTriageEnv", version="1.0.0")

_env: Optional[MailTriageEnv] = None
_lock = threading.Lock()

class ResetRequest(BaseModel):
    task_name: str = "classify_inbox"
    seed: int = 42

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0", "env": "mailtriage-env"}

@app.post("/reset", response_model=EmailObservation)
def reset(request: ResetRequest):
    global _env
    with _lock:
        _env = MailTriageEnv()
        obs = _env.reset(task_name=request.task_name, seed=request.seed)
    return obs

@app.post("/step", response_model=StepResponse)
def step(action: EmailAction):
    global _env
    with _lock:
        if _env is None:
            raise HTTPException(status_code=400, detail="Call /reset first")
        resp = _env.step(action)
    return resp

@app.get("/state")
def state():
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return _env.state()

@app.post("/close")
def close():
    global _env
    with _lock:
        if _env:
            _env.close()
            _env = None
    return {"status": "closed"}

@app.get("/tasks", response_model=List[TaskInfo])
def list_tasks():
    return [
        TaskInfo(name="classify_inbox", difficulty="easy",
                 description="Classify 5 emails by category and priority",
                 max_steps=15, email_count=5),
        TaskInfo(name="triage_and_respond", difficulty="medium",
                 description="Classify 10 emails and draft replies",
                 max_steps=30, email_count=10),
        TaskInfo(name="inbox_zero_with_policy", difficulty="hard",
                 description="Process 20 emails under enterprise policy constraints",
                 max_steps=50, email_count=20),
    ]

@app.post("/grade")
def grade():
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    state_dict = _env.state()
    grader = get_grader(state_dict["task_name"])
    score = grader.grade(state_dict, _env._ground_truths)
    return {"score": score, "task_name": state_dict["task_name"]}

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
