import re
from env.models import (
    EmailItem, EmailGroundTruth, ClassifyEmailAction,
    DraftReplyAction, RouteEmailAction, EscalateEmailAction
)

class ReplyQualityScorer:
    """Rule-based, fully deterministic reply quality scorer. No LLM calls."""
    
    BOILERPLATE_PHRASES = [
        "per my last email", "as per", "please advise", "going forward",
        "touch base", "circle back", "synergy", "leverage", "dear customer",
        "to whom it may concern", "i hope this email finds you",
        "please let me know if you have any questions"
    ]
    
    CLOSING_PHRASES = ["regards", "sincerely", "thanks", "thank you", 
                        "best", "cheers", "warm regards"]
                        
    def score(self, reply_body: str, email: EmailItem, 
              ground_truth: EmailGroundTruth) -> float:
        score = 0.0
        reply_lower = reply_body.lower()
        
        # +0.20 if sender first name appears in reply
        sender_first_name = (" " + email.sender_name.split()[0].lower() + " ")
        # Using regex or simple string find
        # Adding spaces for rough word matching
        if email.sender_name.split()[0].lower() in reply_lower:
            score += 0.20
            
        # +0.20 if any word from subject (len > 4) appears in reply
        subject_words = [w.lower() for w in re.findall(r'\b\w+\b', email.subject) if len(w) > 4]
        for w in subject_words:
            if w in reply_lower:
                score += 0.20
                break
                
        # +0.20 if word count between 30 and 300
        word_count = len(re.findall(r'\b\w+\b', reply_body))
        if 30 <= word_count <= 300:
            score += 0.20
            
        # +0.20 if no boilerplate phrase found (case-insensitive)
        has_boilerplate = False
        for phrase in self.BOILERPLATE_PHRASES:
            if phrase.lower() in reply_lower:
                has_boilerplate = True
                break
        if not has_boilerplate:
            score += 0.20
            
        # +0.20 if reply ends with a closing phrase (last 20 words)
        last_20_words = " ".join(reply_lower.split()[-20:])
        has_closing = False
        for phrase in self.CLOSING_PHRASES:
            if phrase in last_20_words:
                has_closing = True
                break
        if has_closing:
            score += 0.20
            
        # Bonus +0.10 if anger_level >= 2 and reply contains 
        # "apologize" or "sorry" or "sincerely apologize"
        if ground_truth.anger_level >= 2:
            if "apologize" in reply_lower or "sorry" in reply_lower:
                score += 0.10
                
        return min(score, 1.0)


class StepRewardCalculator:
    def __init__(self):
        self.reply_scorer = ReplyQualityScorer()
    
    def reward_open_email(self) -> float:
        return 0.0  # neutral — opening is necessary but not rewarded
    
    def reward_classify(self, action: ClassifyEmailAction, 
                         gt: EmailGroundTruth) -> float:
        cat_correct = action.category == gt.true_category
        pri_correct = action.priority == gt.true_priority
        if cat_correct and pri_correct:
            return 0.15
        elif cat_correct:
            return 0.07
        elif pri_correct:
            return 0.05
        return -0.05
    
    def reward_draft_reply(self, action: DraftReplyAction,
                            email: EmailItem, gt: EmailGroundTruth) -> float:
        base = 0.05
        quality = self.reply_scorer.score(action.reply_body, email, gt)
        bonus = quality * 0.20
        return min(base + bonus, 0.25)
    
    def reward_send_reply(self, draft_quality: float, gt: EmailGroundTruth) -> float:
        if not gt.requires_reply:
            return -0.10
        if draft_quality < 0: # Sentinel for no draft
            return -0.05
        if draft_quality >= 0.7:
            return 0.05 + 0.10 # Wait, PRD: "+0.10 if >=0.5, +0.05 if >=0.7". Total 0.15?
            # actually logic says: +0.10 if >= 0.5, +0.05 if >= 0.7 (so total 0.15 for >= 0.7)
        if draft_quality >= 0.5:
            return 0.10
        return 0.0
    
    def reward_route(self, action: RouteEmailAction, gt: EmailGroundTruth) -> float:
        if not gt.requires_routing:
            return -0.05
        if action.department == gt.route_to:
            return 0.15
        return -0.10
    
    def reward_escalate(self, action: EscalateEmailAction, 
                         gt: EmailGroundTruth) -> float:
        if not gt.requires_escalation:
            return -0.05
        if len(action.reason) < 10:
            return -0.05
        return 0.20
    
    def reward_archive(self, gt: EmailGroundTruth) -> float:
        if gt.is_vip:
            return -0.25
        if gt.true_priority in ["urgent", "high"]:
            return -0.20
        # spam/internal mostly
        if gt.true_category in ["spam", "internal"]:
             return 0.05
        return 0.0
    
    def reward_mark_spam(self, gt: EmailGroundTruth) -> float:
        if gt.true_category == "spam":
            return 0.10
        return -0.15
    
    def reward_no_op(self) -> float:
        return -0.05
    
    def step_penalty(self) -> float:
        return -0.01  
    
    def episode_completion_bonus(self, all_processed: bool, 
                                  steps_used: int, max_steps: int) -> float:
        bonus = 0.0
        if all_processed:
            bonus += 0.30
            if max_steps > 0 and steps_used / max_steps < 0.80:
                bonus += 0.10
        return bonus
