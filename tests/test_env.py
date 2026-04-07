import pytest
import math
from env.env import MailTriageEnv
from env.models import (
    OpenEmailAction, ClassifyEmailAction, NoOpAction
)

@pytest.fixture
def env():
    e = MailTriageEnv()
    e.reset(task_name="classify_inbox", seed=42)
    return e

def test_reset_returns_observation(env):
    obs = env._build_observation()
    assert obs.step_number == 0
    assert len(obs.inbox) == 5
    assert obs.inbox[0].body == "" # Unread

def test_step_before_reset_raises():
    e = MailTriageEnv()
    with pytest.raises(ValueError):
        e.step(NoOpAction(action_type="no_op"))

def test_open_email_populates_body(env):
    email_id = env._inbox[0].email_id
    action = OpenEmailAction(action_type="open_email", email_id=email_id)
    resp = env.step(action)
    
    assert resp.observation.current_email is not None
    assert resp.observation.current_email.email_id == email_id
    assert resp.observation.current_email.body != ""
    assert resp.observation.last_action_error is None

def test_classify_wrong_reward(env):
    email_id = env._inbox[0].email_id
    env.step(OpenEmailAction(action_type="open_email", email_id=email_id))
    
    action = ClassifyEmailAction(
        action_type="classify_email",
        email_id=email_id,
        category="spam",  # likely wrong
        priority="low"
    )
    
    gt = env._ground_truths[email_id]
    action.category = "support" if gt.true_category != "support" else "spam"
    action.priority = "urgent" if gt.true_priority != "urgent" else "low"
    
    resp = env.step(action)
    assert resp.reward < 0

def test_classify_correct_reward(env):
    email_id = env._inbox[0].email_id
    env.step(OpenEmailAction(action_type="open_email", email_id=email_id))
    
    gt = env._ground_truths[email_id]
    action = ClassifyEmailAction(
        action_type="classify_email",
        email_id=email_id,
        category=gt.true_category,  # exact match
        priority=gt.true_priority
    )
    
    resp = env.step(action)
    assert resp.reward > 0

def test_no_op_penalized(env):
    action = NoOpAction(action_type="no_op")
    resp = env.step(action)
    assert math.isclose(resp.reward, -0.06)  # -0.05 (noop) - 0.01 (step penalty)

def test_episode_ends_on_budget(env):
    env._max_steps = 2
    env.step(NoOpAction(action_type="no_op"))
    resp = env.step(NoOpAction(action_type="no_op"))
    assert resp.done is True

def test_consecutive_errors_terminate(env):
    email_id = env._inbox[0].email_id
    # Classify without opening -> Error
    action = ClassifyEmailAction(action_type="classify_email", email_id=email_id, category="spam", priority="low")
    env.step(action)
    env.step(action)
    resp = env.step(action)
    assert resp.done is True
    assert "Terminated due to excessive errors" in resp.observation.last_action_result

def test_state_returns_dict_with_all_keys(env):
    state = env.state()
    keys = ["task_name", "seed", "step", "max_steps", "inbox_size", "processed", 
            "classifications", "drafts", "sent", "routed", "escalated", 
            "archived", "spam_marked", "total_reward", "step_rewards", 
            "policy_violations", "consecutive_errors"]
    for k in keys:
        assert k in state
