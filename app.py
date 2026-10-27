import json, uuid, streamlit as st
from agent.orchestrator import run
from agent.keys import LOG, load_keys
from agent.memory import LongTermMemory

MEM = LongTermMemory()
st.session_state.setdefault("sid", uuid.uuid4().hex[:8])

st.set_page_config(page_title="EvidenceLens AI", page_icon="🔍", layout="wide")
st.title("🔍 EvidenceLens AI")
st.caption("Policy recommendation agent for scientific misinformation. Recommendations need human review.")

with st.sidebar:
    st.header("Settings")
    check_urls = st.checkbox("Verify source URLs (slower)")
    st.subheader("Criteria weights")
    weights = {k: st.slider(k.replace("_", " ").title(), 0, 5, 1) for k in
               ["evidence_strength", "cost", "feasibility", "fairness", "risk"]}
    tool_mode = st.radio("Tools via", ["mcp", "local"], help="mcp = local MCP server (evidencelens-tools); local = in-process fallback")
    use_mem = st.checkbox("Use long-term memory", value=True)
    lite = st.checkbox("Quota-saver mode (1-2 Gemini calls)", help="Use when you hit rate limits. Tools run in Python; Gemini only writes the report.")
    st.write(f"Gemini keys configured: **{len(load_keys())}**/3")
    st.caption(f"Session: `{st.session_state['sid']}` (short-term memory)")
    if st.button("New session"):
        st.session_state["sid"] = uuid.uuid4().hex[:8]; st.session_state.pop("res", None); st.rerun()
    with st.expander("Long-term memory"):
        st.json(MEM.load())
        if st.button("Clear long-term memory"): MEM.clear(); st.rerun()

q = st.text_area("Policy question", height=100,
    placeholder="How can the government help citizens distinguish reliable scientific evidence from weak studies and misleading AI-generated claims?")
prio = st.text_input("Priorities / constraints (optional)", placeholder="low budget, rural reach, avoid censorship")

if st.button("Run analysis", type="primary"):
    LOG.clear()
    with st.status("Agent running - live steps below (also in your terminal and outputs/agent.log)", expanded=True) as status:
        if use_mem: MEM.save_preferences(prio, weights)
        res = run(q, prio, weights, check_urls, tool_mode, st.session_state["sid"], memory=MEM if use_mem else None,
                  on_event=lambda m: status.write(m), mode="lite" if lite else "agent")
        status.update(label="Done", state="complete")
    st.session_state["res"] = res

def show_sources(res):
    st.subheader("Sources")
    for x in res.get("sources", []):
        flag = " ⚠️ injection text removed" if x.get("injection_flagged") else ""
        ver = "" if "verified" not in x else (" ✅" if x["verified"] else " ❌ unverified")
        st.markdown(f"**{x['id']}** [{x['title']}]({x['url']}) - {x.get('quality', '?')} ({x.get('type', '?')}){ver}{flag}")

res = st.session_state.get("res")
if res:
    s = res["status"]
    if res.get("fallback_note"): st.info(res["fallback_note"])
    if s in ("invalid_input", "error", "insufficient_evidence", "no_report", "partial_no_llm"):
        st.warning(res["message"])
        if s == "partial_no_llm":
            st.markdown(f"**Evidence mix:** {res.get('evidence_mix', {}).get('types', [])}")
            show_sources(res)
    else:
        r = res["report"]
        st.subheader("Summary"); st.write(r.get("summary"))
        st.subheader("Policy options (ranked by weighted score)")
        for o in res["ranking"]["ranked"]:
            with st.expander(f"{o['name']}  -  score {o['weighted_score']}"):
                st.write(o.get("description")); st.markdown(f"**Benefits:** {o.get('benefits')}")
                st.markdown(f"**Trade-offs:** {o.get('tradeoffs')}"); st.markdown(f"**Caveats:** {o.get('caveats','')}")
                st.markdown(f"**Evidence:** {', '.join(o.get('evidence_ids', []))}"); st.json(o.get("scores", {}))
        if res["ranking"]["note"]: st.info(res["ranking"]["note"])
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Conflicts**"); [c1.write(f"- {x}") for x in r.get("conflicts", [])]
        c2.markdown("**Uncertainties**"); [c2.write(f"- {x}") for x in r.get("uncertainties", [])]
        c3.markdown("**Limitations**"); [c3.write(f"- {x}") for x in r.get("limitations", [])]
        show_sources(res)
        if res["validation_issues"]:
            st.error("Validation issues remain:"); [st.write(f"- {i}") for i in res["validation_issues"]]
        else:
            st.success("Validation passed")
        st.info(res["notice"])
        if st.button("Approve & save report metadata to memory"):
            st.success("Saved." if MEM.save_approved_report(res) else "Nothing to save.")
    st.caption(f"Tools used via: {res.get('tool_mode_used')}")
    st.download_button("Download JSON", json.dumps(res, indent=2, default=str), "report.json")
    if res.get("trace"):
        with st.expander("Deep Agents trace (plan, delegations, tool calls)"):
            for i, x in enumerate(res["trace"], 1): st.write(f"{i}. `{x['tool']}` {x['detail']}")
    with st.expander("Agent log (no secrets)"): st.json(LOG)
