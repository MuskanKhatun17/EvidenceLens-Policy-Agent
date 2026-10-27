"""MCP: spawns the REAL local MCP server subprocess. Memory: short-term (session) and long-term (JSON)."""
import asyncio, json
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from agent import orchestrator
from agent.store import SourceStore
from agent.memory import LongTermMemory
from tests.test_orchestrator import FakeModel, call, GOOD

SRC = {"title": "Cochrane systematic review", "url": "https://x.org/r", "snippet": "systematic review", "date": "2024"}

def test_mcp_server_exposes_tools_and_shares_state(tmp_path):
    path = str(tmp_path / "s.json"); SourceStore(path).add_sources([dict(SRC)])
    async def go():
        client = MultiServerMCPClient(orchestrator.mcp_config(path))
        tools = {t.name: t for t in await client.get_tools()}
        a = await tools["assess_sources"].ainvoke({})
        v = await tools["validate_report"].ainvoke({"report_json": json.dumps(GOOD)})
        return sorted(tools), str(a), str(v)
    names, a, v = asyncio.run(go())
    assert names == ["assess_sources", "fetch_source", "score_policy_options", "search_evidence", "validate_report"]
    assert "Systematic review" in a                       # MCP server read the shared registry
    assert '\\"valid\\": true' in v or '"valid": true' in v
    assert SourceStore(path).by_id("S1")["quality"] == "High"   # ...and wrote back to it

def test_full_agent_run_over_mcp(tmp_path):
    SourceStore(str(tmp_path / "m1.json")).add_sources([dict(SRC)])
    script = [call("task", {"description": "assess", "subagent_type": "evidence-analyst"}, 1),
              call("assess_sources", {}, 2), AIMessage(content="assessed"),
              call("submit_final_report", {"report_json": json.dumps(GOOD)}, 3), AIMessage(content="done")]
    res = orchestrator.run("q?", keys=["k"], model_factory=lambda k: FakeModel(messages=iter(script)),
                           tool_mode="mcp", state_dir=str(tmp_path), session_id="m1")
    assert res["tool_mode_used"] == "mcp", res["tool_mode_used"]
    assert res["status"] == "ok", res
    assert SourceStore(str(tmp_path / "m1.json")).by_id("S1")["quality"] == "High"

def test_short_term_memory_sources_persist_across_followups(monkeypatch, tmp_path):
    from tools import registry_tools
    monkeypatch.setattr(registry_tools, "_search", lambda q: [dict(SRC)])
    s1 = [call("task", {"description": "go", "subagent_type": "evidence-researcher"}, 1),
          call("search_evidence", {"query": "q"}, 2), AIMessage(content="ok"), AIMessage(content="first done")]
    kw = dict(keys=["k"], tool_mode="local", state_dir=str(tmp_path), session_id="sess")
    r1 = orchestrator.run("first?", model_factory=lambda k: FakeModel(messages=iter(s1)), **kw)
    assert [s["id"] for s in r1["sources"]] == ["S1"]
    s2 = [call("submit_final_report", {"report_json": json.dumps(GOOD)}, 3), AIMessage(content="second done")]
    r2 = orchestrator.run("follow-up?", model_factory=lambda k: FakeModel(messages=iter(s2)), **kw)  # no new search
    assert r2["status"] == "ok" and r2["sources"][0]["id"] == "S1"

def test_long_term_memory_roundtrip_and_not_evidence(tmp_path):
    m = LongTermMemory(str(tmp_path / "lt.json"))
    m.save_preferences("low budget", {"cost": 5})
    assert m.save_approved_report({"question": "Q?", "report": GOOD, "sources": [dict(SRC)]})
    ctx = m.context_for_prompt()
    assert "low budget" in ctx and "NOT evidence" in ctx
    m.clear(); assert m.context_for_prompt() == ""

def test_memory_is_injected_into_task_prompt(tmp_path):
    m = LongTermMemory(str(tmp_path / "lt.json")); m.save_preferences("rural reach")
    seen = {}
    class Spy(FakeModel):
        def _generate(self, messages, *a, **k):
            seen["text"] = " ".join(str(x.content) for x in messages); return super()._generate(messages, *a, **k)
    orchestrator.run("q?", keys=["k"], model_factory=lambda k: Spy(messages=iter([AIMessage(content="x")])),
                     tool_mode="local", state_dir=str(tmp_path), session_id="mm", memory=m)
    assert "rural reach" in seen["text"]
