DEFAULT_WEIGHTS = {"evidence_strength": 1, "cost": 1, "feasibility": 1, "fairness": 1, "risk": 1}

def score_policy_options(options, weights=None):
    """Consistent weighted scoring in Python (scores 1-5 come from the model, maths is deterministic)."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    total_w = sum(w.values()) or 1
    ranked = []
    for o in options:
        sc = o.get("scores", {})
        val = sum(float(sc.get(k, 3)) * w[k] for k in w) / total_w
        ranked.append({**o, "weighted_score": round(val, 2)})
    return sorted(ranked, key=lambda x: x["weighted_score"], reverse=True)

def compare_policies(options, weights=None):
    ranked = score_policy_options(options, weights)
    close = len(ranked) > 1 and abs(ranked[0]["weighted_score"] - ranked[1]["weighted_score"]) < 0.3
    return {"ranked": ranked, "note": "Top options are close; no clear winner." if close else ""}
