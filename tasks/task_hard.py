TASK_NAME = "inbox_zero_with_policy"
SEED = 42
MAX_STEPS = 50
OBJECTIVE = """You have 20 emails to process under strict enterprise policy:
POLICY RULES (violations are penalized):
1. VIP customer emails MUST be escalated (never archived)
2. Security alerts MUST be routed to security_ops within 3 steps of opening
3. Billing disputes > $500 MUST be routed to 'legal' BEFORE sending any reply
4. Replies to angry customers MUST contain an apology ('apologize' or 'sorry')
5. Emails with thread_length > 3 must NOT be archived
Process all emails. Apply the correct action sequence for each case."""
