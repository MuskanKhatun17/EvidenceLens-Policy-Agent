"""Runs the real Deep Agents graph with a scripted fake model (no API). Checks delegation + tools + validation wiring."""
import json
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from agent import orchestrator
from tools import registry_tools

class FakeModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kw):
        return self

import itertools
_c = itertools.count()
def SID(): return f"t{next(_c)}"

def call(name, args, i):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{i}"}])

GOOD = {"summary": "s", "conflicts": [], "uncertainties": ["u"], "limitations": ["l"],
        "options": [{"name": "A", "description": "d", "benefits": "b", "tradeoffs": "t", "caveats": "c",
                     "evidence_ids": ["S1"], "scores": {"evidence_strength": 4, "cost": 3, "feasibility": 4, "fairness": 4, "risk": 4}}]}

def fake_factory(script):
    return lambda key: FakeModel(messages=iter(script))

def test_deepagent_delegates_and_submits(monkeypatch, tmp_path):
    monkeypatch.setattr(registry_tools, "_search", lambda q: [{"title": "Cochrane review on media literacy", "url": "https://x.org/r", "snippet": "systematic review", "date": "2024"}])
    script = [
        call("task", {"description": "Find evidence", "subagent_type": "evidence-researcher"}, 1),
        call("search_evidence", {"query": "media literacy systematic review"}, 2),   # runs inside researcher subagent
        AIMessage(content="Found S1."),                                              # researcher returns
        call("submit_final_report", {"report_json": json.dumps(GOOD)}, 3),
        AIMessage(content="Done."),
    ]
    res = orchestrator.run("How can government reduce misinformation?", keys=["k"], model_factory=fake_factory(script), tool_mode="local", state_dir=str(tmp_path), session_id=SID())
    assert res["status"] == "ok", res
    assert res["sources"][0]["id"] == "S1"
    assert any(x["tool"] == "task" and "researcher" in x["detail"] for x in res["trace"])

def test_unknown_citation_is_flagged(monkeypatch, tmp_path):
    monkeypatch.setattr(registry_tools, "_search", lambda q: [{"title": "t", "url": "https://x.org/r", "snippet": "", "date": ""}])
    bad = json.loads(json.dumps(GOOD)); bad["options"][0]["evidence_ids"] = ["S9"]
    script = [call("task", {"description": "go", "subagent_type": "evidence-researcher"}, 1),
              call("search_evidence", {"query": "q"}, 2), AIMessage(content="ok"),
              call("submit_final_report", {"report_json": json.dumps(bad)}, 3), AIMessage(content="x")]
    res = orchestrator.run("question?", keys=["k"], model_factory=fake_factory(script), tool_mode="local", state_dir=str(tmp_path), session_id=SID())
    assert res["status"] == "ok_with_issues" and any("S9" in i for i in res["validation_issues"])

def test_no_sources_means_no_recommendation(monkeypatch, tmp_path):
    monkeypatch.setattr(registry_tools, "_search", lambda q: [])
    script = [call("submit_final_report", {"report_json": json.dumps(GOOD)}, 1), AIMessage(content="Insufficient evidence.")]
    res = orchestrator.run("obscure claim?", keys=["k"], model_factory=fake_factory(script), tool_mode="local", state_dir=str(tmp_path), session_id=SID())
    assert res["status"] == "insufficient_evidence" and "report" not in res

def test_quota_error_rotates_key(tmp_path):
    seen = []
    def factory(key):
        seen.append(key)
        if key == "k1": raise Exception("429 RESOURCE_EXHAUSTED")
        return FakeModel(messages=iter([AIMessage(content="nothing")]))
    res = orchestrator.run("q?", keys=["k1", "k2"], model_factory=factory, tool_mode="local", state_dir=str(tmp_path), session_id=SID())
    assert seen == ["k1", "k2"] and res["status"] in ("insufficient_evidence", "no_report")

def test_all_keys_exhausted_message(tmp_path):
    def factory(key): raise Exception("429 quota")
    res = orchestrator.run("q?", keys=["SECRET1", "SECRET2"], model_factory=factory, tool_mode="local", state_dir=str(tmp_path), session_id=SID())
    assert res["status"] in ("insufficient_evidence", "partial_no_llm") and "SECRET" not in str(res)

def test_invalid_input():
    assert orchestrator.run("", keys=["k"])["status"] == "invalid_input"

def test_progress_events_are_emitted(tmp_path):
    events = []
    script = [AIMessage(content="nothing")]
    orchestrator.run("q?", keys=["k"], model_factory=fake_factory(script), tool_mode="local",
                     state_dir=str(tmp_path), session_id=SID(), on_event=events.append)
    assert any("LLM call #1 started" in e for e in events) and any("finished" in e for e in events)
