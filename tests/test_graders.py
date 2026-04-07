import pytest
import math
from env.graders import ClassifyInboxGrader, TriageAndRespondGrader, InboxZeroPolicyGrader
from env.models import EmailGroundTruth

@pytest.fixture
def gts():
    return {
        "eml_1": EmailGroundTruth(
            email_id="eml_1", true_category="billing", true_priority="high",
            requires_reply=True, requires_escalation=False, requires_routing=False,
            route_to=None, is_vip=False, dollar_amount=100.0, anger_level=0
        ),
        "eml_2": EmailGroundTruth(
            email_id="eml_2", true_category="spam", true_priority="low",
            requires_reply=False, requires_escalation=False, requires_routing=False,
            route_to=None, is_vip=False, dollar_amount=0, anger_level=0
        )
    }

def test_classify_grader_perfect(gts):
    grader = ClassifyInboxGrader()
    state = {
        "classifications": {
            "eml_1": {"category": "billing", "priority": "high"},
            "eml_2": {"category": "spam", "priority": "low"}
        }
    }
    score = grader.grade(state, gts)
    assert score == 1.0

def test_classify_grader_zero(gts):
    grader = ClassifyInboxGrader()
    state = {
        "classifications": {
            "eml_1": {"category": "spam", "priority": "low"},
            "eml_2": {"category": "billing", "priority": "high"}
        }
    }
    score = grader.grade(state, gts)
    assert score == 0.0

def test_classify_grader_partial(gts):
    grader = ClassifyInboxGrader()
    state = {
        "classifications": {
            "eml_1": {"category": "billing", "priority": "low"}, # cat correct, pri wrong -> 0.5
            "eml_2": {"category": "spam", "priority": "urgent"}  # cat correct, pri wrong -> 0.5
        }
    }
    score = grader.grade(state, gts)
    assert score == 0.5 # (0.5+0.5)/2

def test_triage_grader_full(gts):
    grader = TriageAndRespondGrader()
    state = {
        "classifications": {
            "eml_1": {"category": "billing", "priority": "high"},
            "eml_2": {"category": "spam", "priority": "low"}
        },
        "draft_qualities": {
            "eml_1": 0.8
        },
        "sent": ["eml_1"]
    }
    score = grader.grade(state, gts)
    assert score <= 1.0
    assert score > 0.8  # C_score=1.0 (*0.4), R_score=0.8 (*0.4), Cov=1.0 (*0.2) = 0.4+0.32+0.2=0.92
    assert math.isclose(score, 0.92)

def test_policy_grader_with_violations(gts):
    grader = InboxZeroPolicyGrader()
    state = {
        "classifications": {
            "eml_1": {"category": "billing", "priority": "high"},
            "eml_2": {"category": "spam", "priority": "low"}
        },
        "draft_qualities": {
            "eml_1": 0.8
        },
        "sent": ["eml_1"],
        "policy_violations": ["violation 1", "violation 2"],
        "processed": ["eml_1", "eml_2"],
        "step": 5,
        "max_steps": 10,
        "inbox_size": 0
    }
    score_with_v = grader.grade(state, gts)
    
    state_no_v = state.copy()
    state_no_v["policy_violations"] = []
    score_no_v = grader.grade(state_no_v, gts)
    
    assert score_with_v < score_no_v
