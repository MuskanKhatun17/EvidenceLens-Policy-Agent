import requests

REQUIRED = ["summary", "options", "conflicts", "uncertainties", "limitations"]
OPT_REQUIRED = ["name", "description", "benefits", "tradeoffs", "evidence_ids"]

def check_required_fields(report):
    issues = [f"Missing field: {k}" for k in REQUIRED if k not in report]
    for i, o in enumerate(report.get("options", []), 1):
        issues += [f"Option {i} missing: {k}" for k in OPT_REQUIRED if not o.get(k)]
    return issues

def validate_sources(report, sources, check_urls=False):
    ids = {s["id"] for s in sources}
    issues = []
    for o in report.get("options", []):
        cited = o.get("evidence_ids") or []
        bad = [c for c in cited if c not in ids]
        if bad:
            issues.append(f"Option '{o.get('name')}' cites unknown source IDs: {bad}")
    if check_urls:
        for s in sources:
            try:
                r = requests.head(s["url"], timeout=6, allow_redirects=True)
                s["verified"] = r.status_code < 400
            except Exception:
                s["verified"] = False
            if not s["verified"]:
                issues.append(f"Source {s['id']} URL could not be verified (flagged, not removed).")
    return issues

def validate_report(report, sources, check_urls=False):
    return check_required_fields(report) + validate_sources(report, sources, check_urls)
