import os, requests

def search_evidence(query, num=5):
    """Search the web via Serper. Returns list of source dicts (no IDs yet). [] if unavailable."""
    key = os.getenv("SERPER_API_KEY")
    if not key:
        return []
    try:
        r = requests.post("https://google.serper.dev/search", timeout=10,
                          headers={"X-API-KEY": key, "Content-Type": "application/json"},
                          json={"q": query, "num": num})
        r.raise_for_status()
        return [{"title": o.get("title", ""), "url": o.get("link", ""),
                 "snippet": o.get("snippet", ""), "date": o.get("date", "")}
                for o in r.json().get("organic", [])]
    except Exception:
        return []
