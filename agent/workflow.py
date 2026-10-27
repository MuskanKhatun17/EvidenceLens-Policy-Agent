"""Shared helpers (input validation, JSON parsing). Orchestration lives in orchestrator.py (Deep Agents)."""
import json, re

MAX_Q = 2000

def check_input(q):
    q = (q or "").strip()
    if not q:
        return None, "Please enter a policy question."
    if len(q) > MAX_Q:
        return None, f"Question too long ({len(q)} chars). Limit is {MAX_Q}."
    return q, None

def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    return json.loads(text)
