import pytest
from pydantic import ValidationError
from env.models import (
    EmailAction, OpenEmailAction, ClassifyEmailAction,
    EmailObservation, EmailGroundTruth
)
from env.env import MailTriageEnv

def test_action_discriminator():
    # Test valid models
    from pydantic import TypeAdapter
    action = TypeAdapter(EmailAction).validate_python({
        "action_type": "open_email", 
        "email_id": "eml_123"
    })
    assert isinstance(action, OpenEmailAction)
    assert action.email_id == "eml_123"

    action2 = TypeAdapter(EmailAction).validate_python({
        "action_type": "classify_email",
        "email_id": "eml_456",
        "category": "billing",
        "priority": "urgent"
    })
    assert isinstance(action2, ClassifyEmailAction)

    # Test invalid category
    with pytest.raises(ValidationError):
        TypeAdapter(EmailAction).validate_python({
            "action_type": "classify_email",
            "email_id": "eml_456",
            "category": "not_real",
            "priority": "urgent"
        })

def test_observation_fields():
    obs = EmailObservation(
        inbox=[],
        current_email=None,
        thread_history=[],
        inbox_size=0,
        processed_count=0,
        step_number=1,
        steps_remaining=5,
        last_action_result="test",
        last_action_error=None,
        task_objective="test obj"
    )
    assert obs.step_number == 1

def test_ground_truth_fields():
    gt = EmailGroundTruth(
        email_id="eml_1",
        true_category="spam",
        true_priority="low",
        requires_reply=False,
        requires_escalation=False,
        requires_routing=False,
        route_to=None,
        is_vip=False,
        dollar_amount=0,
        anger_level=0
    )
    assert gt.true_category == "spam"
