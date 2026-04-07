# MailTriageEnv

## Overview
MailTriageEnv is a real-world OpenEnv environment where AI agents learn to triage, prioritize, and respond to business email inboxes. 

Email triage is one of the highest-volume cognitive tasks performed daily by knowledge workers. Training agents to triage intelligently — distinguishing urgent customer complaints from newsletters, routing escalations correctly, and drafting context-aware replies — has immense immediate value. 

## Environment Description
The agent acts as an inbox manager. In each episode, it must process a queue of emails within a fixed step budget. To succeed, the agent must open unread emails, classify them, route escalations appropriately, draft replies, and clear out spam or internal memos.

## Observation Space

| Field | Type | Description |
|---|---|---|
| `inbox` | `List[EmailItem]` | Snapshot of all emails not yet fully processed. Unread emails have `body=""`. |
| `current_email` | `Optional[EmailItem]` | The email currently being looked at (populated after `open_email`). |
| `thread_history` | `List[str]` | Prior messages in thread. |
| `inbox_size` | `int` | Total emails remaining to process. |
| `processed_count` | `int` | Number of emails fully processed. |
| `step_number` | `int` | Current step index in the episode. |
| `steps_remaining` | `int` | Step budget left. |
| `last_action_result` | `str` | Textual feedback from previous action. |
| `last_action_error` | `Optional[str]` | Error message if the last action was invalid. |
| `task_objective` | `str` | Natural language description of the current task goal. |

## Action Space

All actions are provided as JSON objects matching these properties:

| Action Type | Arguments | Description |
|---|---|---|
| `open_email` | `email_id: str` | Opens an email, revealing its body. |
| `classify_email` | `email_id: str`, `category: str`, `priority: str` | Classifies the email. Valid categories: `billing`, `support`, `hr`, `security`, `vendor`, `spam`, `internal`, `other`. Valid priorities: `urgent`, `high`, `normal`, `low`. |
| `draft_reply` | `email_id: str`, `reply_body: str` | Drafts a reply. Should be between 10-1500 chars. |
| `send_reply` | `email_id: str` | Sends the drafted reply, finishing processing for that email. |
| `route_email` | `email_id: str`, `department: str` | Routes to: `billing_team`, `support_tier2`, `legal`, `hr_dept`, `security_ops`, `manager`. |
| `escalate_email` | `email_id: str`, `reason: str` | Escalate for urgent action. |
| `archive_email` | `email_id: str` | Archive email. |
| `mark_spam` | `email_id: str` | Mark email as spam. |
| `no_op` | None | Does nothing. |

## Tasks

### Task 1: classify_inbox (Easy)
- **Objective:** Classify 5 emails by category and priority.
- **Budget:** 15 steps.
- **Success Threshold:** 0.80

### Task 2: triage_and_respond (Medium)
- **Objective:** Classify 10 emails and draft context-aware replies.
- **Budget:** 30 steps.
- **Success Threshold:** 0.70

### Task 3: inbox_zero_with_policy (Hard)
- **Objective:** Process 20 emails under strict enterprise policy rules.
- **Budget:** 50 steps.
- **Success Threshold:** 0.65

## Reward Function
- **Classify Correctly:** +0.15 for both category and priority; +0.07 for one; -0.05 for wrong.
- **Draft Reply:** Base +0.05, Quality Bonus +0.20 (max +0.25).
- **Send Reply:** +0.10 if high quality, -0.10 if reply sent when not needed.
- **Route / Escalate:** Correct route gives +0.15, correct escalate +0.20.
- **Spam:** +0.10 for correct spam marking, -0.15 for incorrect.

## Policy Rules (Task 3)
1. VIP customer emails MUST be escalated (never archived). Penalty: -0.15
2. Security alerts MUST be routed to `security_ops` within 3 steps of opening. Penalty: -0.15
3. Billing disputes > $500 MUST be routed to `legal` BEFORE sending any reply. Penalty: -0.15
4. Replies to angry customers MUST contain an apology ('apologize' or 'sorry'). Penalty: -0.15
5. Emails with `thread_length > 3` must NOT be archived. Penalty: -0.15

## Baseline Scores

| Task | Model | Score | Steps |
|------|-------|-------|-------|
| classify_inbox | Qwen2.5-72B-Instruct | TBD | TBD |
| triage_and_respond | Qwen2.5-72B-Instruct | TBD | TBD |
| inbox_zero_with_policy | Qwen2.5-72B-Instruct | TBD | TBD |

## Setup

### Local Development
### Local Development (using pip or uv)
```bash
# Using pip
pip install -e .
server

# Or using uv (recommended for OpenEnv validation)
uv pip install -e .
uv run server
```

### Docker
```bash
docker build -t mailtriage-env .
docker run -p 7860:7860 mailtriage-env
```

### Running Inference
```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ENV_URL=http://localhost:7860
python inference.py
```

### Running Tests
```bash
pytest tests/ -v
```

### OpenEnv Validation
Ensure you have `uv` installed, as OpenEnv uses it to check dependencies.
```bash
uv lock
openenv validate
```
