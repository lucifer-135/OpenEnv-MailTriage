TASK_NAME = "classify_inbox"
SEED = 42
MAX_STEPS = 15
OBJECTIVE = """You have 5 unread business emails. Your goal is to:
1. Open each email using open_email(email_id)
2. Classify it using classify_email(email_id, category, priority)
Valid categories: billing, support, hr, security, vendor, spam, internal, other
Valid priorities: urgent, high, normal, low
Process all 5 emails as accurately as possible."""
