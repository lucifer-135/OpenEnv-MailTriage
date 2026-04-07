TASK_NAME = "triage_and_respond"
SEED = 42
MAX_STEPS = 30
OBJECTIVE = """You have 10 business emails. Your goal is to:
1. Open each email
2. Classify it (category + priority)
3. For non-spam, non-internal emails: draft a reply then send it
4. For spam: mark as spam
5. For internal memos: archive them
Write professional, context-aware replies. Address the sender by name."""
