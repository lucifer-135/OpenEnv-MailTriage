import json
import os
import random
import numpy as np
from faker import Faker
from typing import List, Tuple
from env.models import EmailItem, EmailGroundTruth

class EmailGenerator:
    def __init__(self, seed: int = 42):
        self._seed = seed
        self.fake = Faker()
        self.fake.seed_instance(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        # Load templates
        templates_path = os.path.join(os.path.dirname(__file__), "..", "data", "email_templates.json")
        with open(templates_path, "r", encoding="utf-8") as f:
            self.templates = json.load(f)

    def _fill_template(self, template: dict, email_id: str) -> dict:
        filled = template.copy()
        
        # We'll use Faker to populate specific values for filling
        sender_name = self.fake.name()
        recipient_name = self.fake.first_name()
        company = self.fake.company()
        date_str = self.fake.date_this_year().isoformat()
        product = self.fake.bs().title()
        
        # Override dollar_amount or properties if needed, or stick to template defaults
        amount = f"${template.get('dollar_amount', 0):.2f}" if template.get("dollar_amount") else f"${self.fake.random_int(min=50, max=5000)}.00"
        
        subject = template["subject"].format(
            company=company, amount=amount, product=product, 
            sender_name=sender_name, recipient_name=recipient_name, date=date_str
        )
        body = template["body"].format(
            company=company, amount=amount, product=product,
            sender_name=sender_name, recipient_name=recipient_name, date=date_str
        )
        
        filled["subject"] = subject
        filled["body"] = body
        filled["sender_name"] = sender_name
        filled["sender_email"] = f"{sender_name.replace(' ', '.').lower()}@{company.replace(' ', '').replace(',', '').lower()}.com"
        filled["timestamp"] = self.fake.date_time_this_month().isoformat()
        
        return filled

    def generate_inbox(self, task_name: str, seed: int = 42) -> Tuple[List[EmailItem], List[EmailGroundTruth]]:
        self.fake.seed_instance(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        inbox_items: List[EmailItem] = []
        ground_truths: List[EmailGroundTruth] = []
        
        # Based on task_name, pick composition
        composition = []
        if task_name == "classify_inbox":
            composition = [
                ("billing", lambda t: True),
                ("support", lambda t: True),
                ("hr", lambda t: True),
                ("spam", lambda t: True),
                ("internal", lambda t: True),
            ]
        elif task_name == "triage_and_respond":
            composition = [
                ("support", lambda t: True)] * 3 + \
                [("billing", lambda t: True)] * 2 + \
                [("internal", lambda t: True)] * 2 + \
                [("spam", lambda t: True)] * 2 + \
                [("security", lambda t: True)] * 1
        elif task_name == "inbox_zero_with_policy":
            composition = [
                ("billing", lambda t: t.get("is_vip") == True and t.get("dollar_amount", 0) >= 500), # 1 VIP billing ($750)
                ("security", lambda t: True), # 1 security
                ("billing", lambda t: not t.get("is_vip") and t.get("dollar_amount", 0) > 500), # >500 billing
                ("billing", lambda t: not t.get("is_vip") and t.get("dollar_amount", 0) > 500), # >500 billing
                ("billing", lambda t: not t.get("is_vip") and t.get("dollar_amount", 0) <= 500), # <=500
                ("billing", lambda t: not t.get("is_vip") and t.get("dollar_amount", 0) <= 500), # <=500
                ("hr", lambda t: True), ("hr", lambda t: True), ("hr", lambda t: True),
                ("vendor", lambda t: True), ("vendor", lambda t: True),
                ("support", lambda t: t.get("anger_level", 0) >= 2), # 1 angry support
                ("support", lambda t: t.get("anger_level", 0) < 2),
                ("support", lambda t: t.get("anger_level", 0) < 2),
                ("support", lambda t: t.get("anger_level", 0) < 2),
                ("internal", lambda t: True), ("internal", lambda t: True), ("internal", lambda t: True),
                ("spam", lambda t: True), ("spam", lambda t: True),
            ]
            
        else:
            # Fallback
            composition = [("support", lambda t: True)] * 5
            
        email_counter = 1
        for cat, condition in composition:
            available_templates = [t for t in self.templates.get(cat, []) if condition(t)]
            if not available_templates:
                # Fallback condition if none match perfectly
                available_templates = self.templates.get(cat, [])
                
            template = random.choice(available_templates)
            email_id = f"eml_{email_counter:03d}"
            
            filled = self._fill_template(template, email_id)
            
            # Additional logic for thread lengths etc.
            thread_id = f"thr_{email_counter:03d}"
            thread_length = random.randint(0, 5) if cat != "spam" and cat != "internal" else 0
            
            # Make email item
            item = self._make_email_item(filled, email_id, thread_id, thread_length)
            inbox_items.append(item)
            
            # Make ground truth
            gt = self._make_ground_truth(filled, email_id)
            ground_truths.append(gt)
            
            email_counter += 1
            
        # Shuffle inbox so they don't appear in categorized order
        combined = list(zip(inbox_items, ground_truths))
        random.shuffle(combined)
        inbox_items, ground_truths = zip(*combined)
            
        return list(inbox_items), list(ground_truths)

    def _make_email_item(self, filled: dict, email_id: str, 
                          thread_id: str, thread_length: int) -> EmailItem:
        return EmailItem(
            email_id=email_id,
            sender_name=filled["sender_name"],
            sender_email=filled["sender_email"],
            subject=filled["subject"],
            body=filled["body"],
            timestamp=filled["timestamp"],
            thread_id=thread_id,
            thread_length=thread_length,
            has_attachment=random.choice([True, False]),
            already_read=False
        )

    def _make_ground_truth(self, filled: dict, email_id: str) -> EmailGroundTruth:
        return EmailGroundTruth(
            email_id=email_id,
            true_category=filled["true_category"],
            true_priority=filled["true_priority"],
            requires_reply=filled.get("requires_reply", False),
            requires_escalation=filled.get("requires_escalation", False),
            requires_routing=filled.get("requires_routing", False),
            route_to=filled.get("route_to", None),
            is_vip=filled.get("is_vip", False),
            dollar_amount=float(filled.get("dollar_amount", 0)),
            anger_level=int(filled.get("anger_level", 0))
        )
