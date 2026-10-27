"""Quota-saver pipeline: tools run in Python (no LLM), then 1 Gemini call (+1 optional fix). Used when the full
Deep Agents run is rate-limited, or when the user picks Quota-saver mode. If even that fails: evidence-only report."""
import json
from agent import prompts
from agent.keys import generate, AllKeysFailed, log
from agent.workflow import parse_json
from tools import registry_tools as rt
from tools.assess_evidence import compare_findings
from tools.compare_policies import compare_policies
from tools.validate_sources import validate_report

NOTICE = "Candidate recommendations for human review - not final policy decisions."

def _fmt(sources):
    return "\n".join(f"[{s['id']}] {s['title']} | {s['url']} | {s.get('date','')} | quality={s.get('quality','?')} ({s.get('type','?')})\n"
                     f"<untrusted_source>{s.get('passage') or s.get('snippet','')}</untrusted_source>" for s in sources)

def gather(store, q, fetch_n=2):
    if not store.all():
        for query in (q + " systematic review", q + " government policy evaluation"):
            rt.search(store, query)
    if store.all():
        rt.assess(store)
        for s in store.all()[:fetch_n]:
            if not s.get("passage"): rt.fetch(store, s["id"])
    return store.all()

def evidence_only(q, sources, reason):
    """No-LLM report: sources + quality labels only. Makes NO policy recommendation."""
    return {"status": "partial_no_llm", "question": q, "sources": sources, "evidence_mix": compare_findings(sources),
            "message": f"{reason} Showing the retrieved evidence only (no AI analysis, no recommendations). "
                       "Wait a few minutes and run again, or enable billing / add keys from different projects."}

def run_lite(q, priorities, weights, check_urls, store, keys=None, generate_fn=None):
    gen = generate_fn or generate
    sources = gather(store, q)
    if not sources:
        return {"status": "insufficient_evidence", "question": q, "sources": [],
                "message": "No sources were retrieved (search failed, no SERPER_API_KEY, or nothing found), so no recommendation is made."}
    srcs = _fmt(sources)
    try:
        draft = parse_json(gen(prompts.LITE_DRAFT.format(q=q, prio=priorities or "none stated", sources=srcs),
                               system=prompts.LITE_SYSTEM, json_mode=True, keys=keys))
    except AllKeysFailed:
        return evidence_only(q, sources, "Gemini quota/rate limits were hit on every key and model.")
    except ValueError:
        return evidence_only(q, sources, "The model returned unreadable output.")
    issues = validate_report(draft, sources, check_urls)
    if issues:
        try:
            draft = parse_json(gen(prompts.LITE_REVISE.format(issues="\n".join(issues), draft=json.dumps(draft), sources=srcs),
                                   system=prompts.LITE_SYSTEM, json_mode=True, keys=keys))
            issues = validate_report(draft, sources, check_urls)
        except Exception:
            pass   # keep first draft; issues stay visible
    return {"status": "ok" if not issues else "ok_with_issues", "question": q, "report": draft,
            "ranking": compare_policies(draft.get("options", []), weights), "evidence_mix": compare_findings(sources),
            "sources": sources, "validation_issues": issues, "notice": NOTICE}
