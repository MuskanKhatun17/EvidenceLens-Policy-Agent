"""Long-term memory: user preferences + metadata of reports the user APPROVED. Plain JSON, human-readable.
Past conclusions are never treated as evidence - they are only hints; sources must be re-retrieved and validated."""
import json, os, datetime

class LongTermMemory:
    def __init__(self, path="outputs/memory/long_term.json"):
        self.path = path

    def load(self):
        try:
            with open(self.path, encoding="utf-8") as f: return json.load(f)
        except Exception:
            return {"preferences": {}, "approved_reports": []}

    def _save(self, d):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f: json.dump(d, f, indent=2)

    def save_preferences(self, priorities="", weights=None):
        d = self.load(); d["preferences"] = {"priorities": priorities, "weights": weights or {}}; self._save(d)

    def save_approved_report(self, res):
        """Store only metadata (no model-written conclusions as facts)."""
        if not res.get("report"): return False
        d = self.load()
        d["approved_reports"].append({
            "date": datetime.date.today().isoformat(), "question": res.get("question", ""),
            "option_names": [o.get("name") for o in res["report"].get("options", [])],
            "source_urls": [s["url"] for s in res.get("sources", [])][:10]})
        d["approved_reports"] = d["approved_reports"][-20:]; self._save(d); return True

    def clear(self):
        self._save({"preferences": {}, "approved_reports": []})

    def context_for_prompt(self):
        d = self.load(); parts = []
        p = d.get("preferences") or {}
        if p.get("priorities"): parts.append(f"Remembered user priorities: {p['priorities']}")
        past = d.get("approved_reports", [])[-5:]
        if past:
            parts.append("Previously approved reports (metadata only; NOT evidence - do not cite them, "
                         "re-retrieve and validate sources): " + "; ".join(
                             f"{r['date']} '{r['question'][:80]}' -> options {r['option_names']}" for r in past))
        return "\n".join(parts)
