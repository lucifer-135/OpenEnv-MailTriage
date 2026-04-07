from typing import Optional, List, Annotated, Literal, Union
from pydantic import BaseModel, Field

class EmailItem(BaseModel):
    email_id: str
    sender_name: str
    sender_email: str
    subject: str
    body: str
    timestamp: str          # ISO-8601
    thread_id: str
    thread_length: int
    has_attachment: bool
    already_read: bool


class EmailGroundTruth(BaseModel):
    email_id: str
    true_category: str
    true_priority: str
    requires_reply: bool
    requires_escalation: bool
    requires_routing: bool
    route_to: Optional[str]
    is_vip: bool
    dollar_amount: float
    anger_level: int        # 0=normal, 1=frustrated, 2=angry


class EmailObservation(BaseModel):
    inbox: List[EmailItem]              # all unprocessed emails (body hidden until opened)
    current_email: Optional[EmailItem]  # email in focus after open_email action
    thread_history: List[str]           # prior thread messages if any
    inbox_size: int
    processed_count: int
    step_number: int
    steps_remaining: int
    last_action_result: str
    last_action_error: Optional[str]
    task_objective: str


class OpenEmailAction(BaseModel):
    action_type: Literal["open_email"]
    email_id: str

class ClassifyEmailAction(BaseModel):
    action_type: Literal["classify_email"]
    email_id: str
    category: Literal["billing","support","hr","security","vendor","spam","internal","other"]
    priority: Literal["urgent","high","normal","low"]

class DraftReplyAction(BaseModel):
    action_type: Literal["draft_reply"]
    email_id: str
    reply_body: str

class SendReplyAction(BaseModel):
    action_type: Literal["send_reply"]
    email_id: str

class RouteEmailAction(BaseModel):
    action_type: Literal["route_email"]
    email_id: str
    department: Literal["billing_team","support_tier2","legal","hr_dept","security_ops","manager"]

class EscalateEmailAction(BaseModel):
    action_type: Literal["escalate_email"]
    email_id: str
    reason: str

class ArchiveEmailAction(BaseModel):
    action_type: Literal["archive_email"]
    email_id: str

class MarkSpamAction(BaseModel):
    action_type: Literal["mark_spam"]
    email_id: str

class NoOpAction(BaseModel):
    action_type: Literal["no_op"]

EmailAction = Annotated[
    Union[OpenEmailAction, ClassifyEmailAction, DraftReplyAction,
          SendReplyAction, RouteEmailAction, EscalateEmailAction,
          ArchiveEmailAction, MarkSpamAction, NoOpAction],
    Field(discriminator="action_type")
]


class StepResponse(BaseModel):
    observation: EmailObservation
    reward: float
    done: bool
    info: dict

class TaskInfo(BaseModel):
    name: str
    difficulty: str
    description: str
    max_steps: int
    email_count: int
