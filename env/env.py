from typing import Optional, List, Dict, Set, Tuple
from env.models import (
    EmailAction, EmailObservation, StepResponse, EmailItem, EmailGroundTruth,
    OpenEmailAction, ClassifyEmailAction, DraftReplyAction, SendReplyAction,
    RouteEmailAction, EscalateEmailAction, ArchiveEmailAction, MarkSpamAction, NoOpAction
)
from env.email_generator import EmailGenerator
from env.rewards import StepRewardCalculator


class MailTriageEnv:
    TASK_CONFIGS = {
        "classify_inbox":        {"max_steps": 15, "email_count": 5},
        "triage_and_respond":    {"max_steps": 30, "email_count": 10},
        "inbox_zero_with_policy":{"max_steps": 50, "email_count": 20},
    }
    
    def __init__(self):
        self._task_name: Optional[str] = None
        self._seed: int = 42
        self._inbox: List[EmailItem] = []
        self._ground_truths: Dict[str, EmailGroundTruth] = {}
        self._processed: Set[str] = set()
        self._classifications: Dict[str, dict] = {}
        self._drafts: Dict[str, str] = {}
        self._draft_qualities: Dict[str, float] = {}
        self._sent: Set[str] = set()
        self._routed: Dict[str, str] = {}
        self._escalated: Dict[str, str] = {}
        self._archived: Set[str] = set()
        self._spam_marked: Set[str] = set()
        self._current_email_id: Optional[str] = None
        self._step: int = 0
        self._max_steps: int = 0
        self._consecutive_errors: int = 0
        self._total_reward: float = 0.0
        self._step_rewards: List[float] = []
        self._last_action_result: str = "Episode not started. Call reset() first."
        self._last_action_error: Optional[str] = None
        self._policy_violations: List[str] = []
        self._security_email_opened_at: Dict[str, int] = {}
        self._initialized: bool = False
        self._generator = EmailGenerator()
        self._reward_calc = StepRewardCalculator()

    def reset(self, task_name: str = "classify_inbox", seed: int = 42) -> EmailObservation:
        if task_name not in self.TASK_CONFIGS:
            task_name = "classify_inbox"
            
        self._task_name = task_name
        self._seed = seed
        self._max_steps = self.TASK_CONFIGS[task_name]["max_steps"]
        
        inbox_items, gt_items = self._generator.generate_inbox(task_name=task_name, seed=seed)
        self._inbox = inbox_items
        self._ground_truths = {gt.email_id: gt for gt in gt_items}
        
        self._processed = set()
        self._classifications = {}
        self._drafts = {}
        self._draft_qualities = {}
        self._sent = set()
        self._routed = {}
        self._escalated = {}
        self._archived = set()
        self._spam_marked = set()
        self._current_email_id = None
        self._step = 0
        self._consecutive_errors = 0
        self._total_reward = 0.0
        self._step_rewards = []
        self._last_action_result = "Welcome to MailTriageEnv. You have new emails."
        self._last_action_error = None
        self._policy_violations = []
        self._security_email_opened_at = {}
        self._initialized = True
        
        return self._build_observation()

    def step(self, action: EmailAction) -> StepResponse:
        if not self._initialized:
            raise ValueError("Environment not initialized. Call reset() first.")
            
        self._step += 1
        reward = self._reward_calc.step_penalty()
        self._last_action_error = None
        
        result_msg = ""
        action_reward = 0.0
        error = None
        
        try:
            if isinstance(action, OpenEmailAction):
                result_msg, action_reward, error = self._handle_open_email(action)
            elif isinstance(action, ClassifyEmailAction):
                result_msg, action_reward, error = self._handle_classify_email(action)
            elif isinstance(action, DraftReplyAction):
                result_msg, action_reward, error = self._handle_draft_reply(action)
            elif isinstance(action, SendReplyAction):
                result_msg, action_reward, error = self._handle_send_reply(action)
            elif isinstance(action, RouteEmailAction):
                result_msg, action_reward, error = self._handle_route_email(action)
            elif isinstance(action, EscalateEmailAction):
                result_msg, action_reward, error = self._handle_escalate_email(action)
            elif isinstance(action, ArchiveEmailAction):
                result_msg, action_reward, error = self._handle_archive_email(action)
            elif isinstance(action, MarkSpamAction):
                result_msg, action_reward, error = self._handle_mark_spam(action)
            elif isinstance(action, NoOpAction):
                result_msg, action_reward, error = self._handle_no_op(action)
            else:
                result_msg, action_reward, error = "Unknown action", -0.05, "Action type not supported"
        except Exception as e:
            result_msg, action_reward, error = f"Exception: {str(e)}", -0.05, str(e)
            
        if error:
            self._consecutive_errors += 1
            self._last_action_error = error
        else:
            self._consecutive_errors = 0
            
        reward += action_reward
        self._last_action_result = result_msg
        
        done = self._check_done()
        if done:
            all_processed = len(self._inbox) - len(self._processed) == 0
            bonus = self._reward_calc.episode_completion_bonus(all_processed, self._step, self._max_steps)
            reward += bonus
            if self._consecutive_errors >= 3:
                self._last_action_result += " | Terminated due to excessive errors."
            elif all_processed:
                self._last_action_result += " | Inbox cleared!"
            else:
                self._last_action_result += " | Step budget exhausted."

        self._step_rewards.append(reward)
        self._total_reward += reward
        
        return StepResponse(
            observation=self._build_observation(),
            reward=reward,
            done=done,
            info={"score": self._total_reward, "policy_violations": len(self._policy_violations)}
        )

    def state(self) -> dict:
        return {
            "task_name": self._task_name,
            "seed": self._seed,
            "step": self._step,
            "max_steps": self._max_steps,
            "inbox_size": len(self._inbox) - len(self._processed),
            "processed": list(self._processed),
            "classifications": self._classifications,
            "drafts": {k: v[:50] + "..." for k,v in self._drafts.items()},
            "draft_qualities": self._draft_qualities,
            "sent": list(self._sent),
            "routed": self._routed,
            "escalated": self._escalated,
            "archived": list(self._archived),
            "spam_marked": list(self._spam_marked),
            "total_reward": self._total_reward,
            "step_rewards": self._step_rewards,
            "policy_violations": self._policy_violations,
            "consecutive_errors": self._consecutive_errors,
        }

    def close(self):
        self._initialized = False

    def _get_email(self, email_id: str) -> Optional[EmailItem]:
        for e in self._inbox:
            if e.email_id == email_id:
                return e
        return None

    def _handle_open_email(self, action: OpenEmailAction) -> Tuple[str, float, Optional[str]]:
        email = self._get_email(action.email_id)
        if not email:
            return "Email not found", 0.0, "Invalid email_id"
        if action.email_id in self._processed:
            return "Email already processed", 0.0, "Cannot open processed email"
            
        email.already_read = True
        self._current_email_id = action.email_id
        
        gt = self._ground_truths[action.email_id]
        if gt.true_category == "security" and action.email_id not in self._security_email_opened_at:
            self._security_email_opened_at[action.email_id] = self._step
            
        return f"Opened email {action.email_id}", self._reward_calc.reward_open_email(), None

    def _handle_classify_email(self, action: ClassifyEmailAction) -> Tuple[str, float, Optional[str]]:
        if self._current_email_id != action.email_id:
            return "Email not currently open", -0.05, "Must open email before classifying"
        
        gt = self._ground_truths[action.email_id]
        self._classifications[action.email_id] = {"category": action.category, "priority": action.priority}
        
        reward = self._reward_calc.reward_classify(action, gt)
        return f"Classified {action.email_id} as {action.category}/{action.priority}", reward, None

    def _handle_draft_reply(self, action: DraftReplyAction) -> Tuple[str, float, Optional[str]]:
        if self._current_email_id != action.email_id:
            return "Email not currently open", -0.05, "Must open email before drafting"
        if len(action.reply_body) < 10 or len(action.reply_body) > 1500:
            return "Invalid draft length", -0.05, "Draft length must be 10-1500 chars"
            
        email = self._get_email(action.email_id)
        gt = self._ground_truths[action.email_id]
        
        self._drafts[action.email_id] = action.reply_body
        quality = self._reward_calc.reply_scorer.score(action.reply_body, email, gt)
        self._draft_qualities[action.email_id] = quality
        
        reward = self._reward_calc.reward_draft_reply(action, email, gt)
        return f"Drafted reply for {action.email_id} (quality: {quality:.2f})", reward, None

    def _handle_send_reply(self, action: SendReplyAction) -> Tuple[str, float, Optional[str]]:
        if self._current_email_id != action.email_id:
            return "Email not currently open", -0.05, "Must open email before sending"
            
        draft_quality = self._draft_qualities.get(action.email_id, -1.0)
        gt = self._ground_truths[action.email_id]
        
        # Policy Check: billing > $500 must be routed to legal before send
        if gt.true_category == "billing" and gt.dollar_amount > 500:
            if self._routed.get(action.email_id) != "legal":
                violation = f"Policy Violation: Email {action.email_id} billing > 500 not routed to legal before reply."
                self._policy_violations.append(violation)
                
        # Policy check: angry customer reply must contain apologize or sorry
        if gt.anger_level >= 2:
            draft_lower = self._drafts.get(action.email_id, "").lower()
            if "apologize" not in draft_lower and "sorry" not in draft_lower:
                violation = f"Policy Violation: Email {action.email_id} angry customer reply missing apology."
                self._policy_violations.append(violation)
        
        self._sent.add(action.email_id)
        self._processed.add(action.email_id)
        
        reward = self._reward_calc.reward_send_reply(draft_quality, gt)
        return f"Sent reply for {action.email_id}", reward, None

    def _handle_route_email(self, action: RouteEmailAction) -> Tuple[str, float, Optional[str]]:
        if self._current_email_id != action.email_id:
            return "Email not currently open", -0.05, "Must open email before routing"
            
        gt = self._ground_truths[action.email_id]
        self._routed[action.email_id] = action.department
        
        # Policy check: security email routed within 3 steps
        if gt.true_category == "security":
            open_step = self._security_email_opened_at.get(action.email_id, 0)
            if self._step - open_step > 3 or action.department != "security_ops":
                violation = f"Policy Violation: Email {action.email_id} (security) not routed to security_ops within 3 steps."
                self._policy_violations.append(violation)
        
        self._processed.add(action.email_id)
        reward = self._reward_calc.reward_route(action, gt)
        return f"Routed {action.email_id} to {action.department}", reward, None

    def _handle_escalate_email(self, action: EscalateEmailAction) -> Tuple[str, float, Optional[str]]:
        if self._current_email_id != action.email_id:
            return "Email not currently open", -0.05, "Must open email before escalating"
        if len(action.reason) < 10:
            return "Escalation reason too short", -0.05, "Reason length >= 10 chars required"
            
        gt = self._ground_truths[action.email_id]
        self._escalated[action.email_id] = action.reason
        self._processed.add(action.email_id)
        
        reward = self._reward_calc.reward_escalate(action, gt)
        return f"Escalated {action.email_id}", reward, None

    def _handle_archive_email(self, action: ArchiveEmailAction) -> Tuple[str, float, Optional[str]]:
        if self._current_email_id != action.email_id:
            return "Email not currently open", -0.05, "Must open email before archiving"
            
        email = self._get_email(action.email_id)
        gt = self._ground_truths[action.email_id]
        
        # VIP policy check
        if gt.is_vip:
            violation = f"Policy Violation: Email {action.email_id} is VIP but was archived."
            self._policy_violations.append(violation)
            
        # Thread length check
        if email.thread_length > 3:
            violation = f"Policy Violation: Email {action.email_id} thread_length > 3 but was archived."
            self._policy_violations.append(violation)
            
        self._archived.add(action.email_id)
        self._processed.add(action.email_id)
        
        reward = self._reward_calc.reward_archive(gt)
        return f"Archived {action.email_id}", reward, None

    def _handle_mark_spam(self, action: MarkSpamAction) -> Tuple[str, float, Optional[str]]:
        if self._current_email_id != action.email_id:
            return "Email not currently open", -0.05, "Must open email before marking spam"
            
        gt = self._ground_truths[action.email_id]
        self._spam_marked.add(action.email_id)
        self._processed.add(action.email_id)
        
        reward = self._reward_calc.reward_mark_spam(gt)
        return f"Marked {action.email_id} as spam", reward, None

    def _handle_no_op(self, action: NoOpAction) -> Tuple[str, float, Optional[str]]:
        return "No operation.", self._reward_calc.reward_no_op(), None

    def _build_observation(self) -> EmailObservation:
        obs_inbox = []
        for e in self._inbox:
            if e.email_id not in self._processed:
                # hide body if unread, else full
                item_copy = e.model_copy()
                if not item_copy.already_read:
                    item_copy.body = ""
                obs_inbox.append(item_copy)
                
        current_email = None
        if self._current_email_id and self._current_email_id not in self._processed:
            current_email = self._get_email(self._current_email_id)
            
        task_obj = ""
        if self._task_name == "classify_inbox":
            task_obj = "Classify emails by category and priority."
        elif self._task_name == "triage_and_respond":
            task_obj = "Classify emails, draft and send replies, archive internals, mark spam."
        else:
            task_obj = "Process emails under strict enterprise policy constraints (VIP, security, billing)."
            
        return EmailObservation(
            inbox=obs_inbox,
            current_email=current_email,
            thread_history=[],
            inbox_size=len(self._inbox) - len(self._processed),
            processed_count=len(self._processed),
            step_number=self._step,
            steps_remaining=self._max_steps - self._step,
            last_action_result=self._last_action_result,
            last_action_error=self._last_action_error,
            task_objective=task_obj
        )

    def _check_done(self) -> bool:
        unprocessed = len(self._inbox) - len(self._processed)
        return unprocessed == 0 or self._step >= self._max_steps or self._consecutive_errors >= 3
