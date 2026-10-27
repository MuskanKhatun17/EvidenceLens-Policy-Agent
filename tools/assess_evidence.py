"""Heuristic evidence-quality labelling (transparent rules, not a substitute for expert review)."""
RULES = [
    (("systematic review", "meta-analysis", "cochrane", "campbell collaboration"), "High", "Systematic review / meta-analysis"),
    (("randomized", "randomised", "rct", "trial"), "Medium-High", "Experimental study"),
    ((".gov", "who.int", "oecd", "unesco", "europa.eu", "national academies", "report"), "Medium", "Official / institutional report"),
    (("pubmed", "doi.org", "journal", "study", "arxiv", "nature.com", "sciencedirect"), "Medium", "Individual study / journal"),
]

def assess_evidence(source):
    blob = f"{source.get('title','')} {source.get('url','')} {source.get('snippet','')}".lower()
    for keys, rating, kind in RULES:
        if any(k in blob for k in keys):
            return {"quality": rating, "type": kind}
    return {"quality": "Low/Unknown", "type": "News, blog or unclassified"}

def assess_all(sources):
    for s in sources:
        s.update(assess_evidence(s))
    return sources

def compare_findings(sources):
    """Flag when sources of different quality levels exist, so the analyst notes possible disagreement."""
    kinds = {s.get("type") for s in sources}
    return {"mixed_evidence_types": len(kinds) > 1, "types": sorted(k for k in kinds if k)}
