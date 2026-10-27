"""Tool logic shared by the MCP server and the local (non-MCP) fallback. All return JSON strings."""
import os, json, time, functools
from tools.logconfig import get_logger
from tools.search_evidence import search_evidence as _search
from tools.fetch_source import fetch_source as _fetch
from tools.assess_evidence import assess_evidence, compare_findings
from tools.compare_policies import compare_policies
from tools.validate_sources import validate_report as _validate

L = get_logger()

def _logged(fn):
    @functools.wraps(fn)
    def w(*a, **k):
        t = time.time(); out = fn(*a, **k)
        L.info("  [tool-impl] %s done in %.1fs (%d chars)", fn.__name__, time.time() - t, len(out)); return out
    return w

MAX_CALLS = int(os.getenv("MAX_TOOL_CALLS", "30"))
_OVER = json.dumps({"error": "Tool-call budget exhausted. Finish with what you have."})

@_logged
def search(store, query):
    if not store.bump_calls(MAX_CALLS): return _OVER
    res = store.add_sources(_search(query))
    if not res:
        return json.dumps({"results": [], "note": "No results (search failed, no SERPER_API_KEY, or nothing found)."})
    return json.dumps({"results": [{k: s.get(k, "") for k in ("id", "title", "url", "date", "snippet")} for s in res]})

@_logged
def fetch(store, source_id):
    if not store.bump_calls(MAX_CALLS): return _OVER
    s = store.by_id(source_id)
    if not s: return json.dumps({"error": f"Unknown source ID {source_id}"})
    f = _fetch(s["url"])
    kw = {"fetched": f["ok"], "injection_flagged": f["flagged"]}
    if f["ok"]: kw["passage"] = f["text"]
    else: kw["verified"] = False
    store.update_source(source_id, **kw)
    return json.dumps({"id": source_id, "ok": f["ok"], "error": f["error"], "untrusted_text": f["text"]})

@_logged
def assess(store):
    if not store.bump_calls(MAX_CALLS): return _OVER
    sources = store.all()
    for s in sources:
        a = assess_evidence(s); s.update(a); store.update_source(s["id"], **a)
    return json.dumps({"sources": [{k: s[k] for k in ("id", "title", "quality", "type")} for s in sources],
                       "mix": compare_findings(sources)})

@_logged
def score(store, options_json, weights=None):
    if not store.bump_calls(MAX_CALLS): return _OVER
    try: return json.dumps(compare_policies(json.loads(options_json), weights))
    except Exception as e: return json.dumps({"error": f"Bad JSON: {type(e).__name__}"})

@_logged
def validate(store, report_json, check_urls=False):
    if not store.bump_calls(MAX_CALLS): return _OVER
    try: issues = _validate(json.loads(report_json), store.all(), check_urls)
    except Exception as e: issues = [f"Report is not valid JSON: {type(e).__name__}"]
    return json.dumps({"issues": issues, "valid": not issues})
