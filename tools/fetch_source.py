import re, requests
from bs4 import BeautifulSoup

INJECTION = re.compile(r"ignore (all |previous |the )?(instructions|rules)|you are now|system prompt|disregard", re.I)

def fetch_source(url, max_chars=2500):
    """Fetch a page's text (untrusted). Returns {ok, text, flagged, error}."""
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "EvidenceLens/1.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup(["script", "style", "nav", "footer"]):
            t.decompose()
        text = " ".join(soup.get_text(" ").split())[:max_chars]
        flagged = bool(INJECTION.search(text))
        if flagged:
            text = INJECTION.sub("[removed]", text)
        return {"ok": True, "text": text, "flagged": flagged, "error": None}
    except Exception as e:
        return {"ok": False, "text": "", "flagged": False, "error": type(e).__name__}
