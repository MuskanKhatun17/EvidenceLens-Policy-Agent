"""Rate-limit handling: model chain, wait pass, quota-saver fallback, evidence-only last resort."""
import json
from agent import keys, orchestrator
from agent.store import SourceStore
from agent.keys import AllKeysFailed
from tests.test_orchestrator import GOOD

SRC = {"title": "Cochrane systematic review", "url": "https://x.org/r", "snippet": "systematic review", "date": "2024"}
def always_429(k): raise Exception("429 RESOURCE_EXHAUSTED")

def seed(tmp_path, sid):
    SourceStore(str(tmp_path / f"{sid}.json")).add_sources([dict(SRC)])

def test_generate_falls_back_to_next_model():
    calls = []
    def call_fn(key, model, *a):
        calls.append(model)
        if model == "flash": raise Exception("429 quota")
        return "ok"
    assert keys.generate("p", keys=["k1", "k2"], models=["flash", "lite"], call_fn=call_fn) == "ok"
    assert calls == ["flash", "flash", "lite"]

def test_generate_raises_when_every_model_limited():
    try: keys.generate("p", keys=["k"], models=["a", "b"], call_fn=lambda k, m, *a: (_ for _ in ()).throw(Exception("429")))
    except AllKeysFailed: return
    raise AssertionError("should raise")

def test_timeout_504_is_retryable():
    assert keys.is_retryable(Exception("504 DEADLINE_EXCEEDED"))

def test_agent_quota_exhausted_falls_back_to_quota_saver_report(tmp_path):
    seed(tmp_path, "q1")
    res = orchestrator.run("q?", keys=["k1", "k2"], model_factory=always_429, tool_mode="local",
                           state_dir=str(tmp_path), session_id="q1", generate_fn=lambda *a, **k: json.dumps(GOOD))
    assert res["status"] == "ok" and "Quota-saver" in res["fallback_note"] and res["report"]["options"]

def test_everything_rate_limited_gives_evidence_only_report(tmp_path):
    seed(tmp_path, "q2")
    def lite_fail(*a, **k): raise AllKeysFailed("limits")
    res = orchestrator.run("q?", keys=["SECRET"], model_factory=always_429, tool_mode="local",
                           state_dir=str(tmp_path), session_id="q2", generate_fn=lite_fail)
    assert res["status"] == "partial_no_llm" and res["sources"][0]["id"] == "S1" and "report" not in res
    assert "SECRET" not in str(res)

def test_lite_mode_uses_no_agent_calls(tmp_path):
    seed(tmp_path, "q3")
    res = orchestrator.run("q?", keys=["k"], model_factory=always_429, mode="lite", tool_mode="local",
                           state_dir=str(tmp_path), session_id="q3", generate_fn=lambda *a, **k: json.dumps(GOOD))
    assert res["status"] == "ok" and res["trace"] == []

def test_lite_flags_bad_citation(tmp_path):
    seed(tmp_path, "q4")
    bad = json.loads(json.dumps(GOOD)); bad["options"][0]["evidence_ids"] = ["S9"]
    res = orchestrator.run("q?", keys=["k"], model_factory=always_429, mode="lite", tool_mode="local",
                           state_dir=str(tmp_path), session_id="q4", generate_fn=lambda *a, **k: json.dumps(bad))
    assert res["status"] == "ok_with_issues"
