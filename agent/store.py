"""File-backed source registry shared by the agent process and the MCP server subprocess.
(MCP stdio tools run in a separate process, so shared state must live on disk.)"""
import json, os, time, contextlib

class SourceStore:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    @contextlib.contextmanager
    def _lock(self, timeout=5.0):
        lock, start = self.path + ".lock", time.time()
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL); break
            except FileExistsError:
                if time.time() - start > timeout:      # stale lock
                    try: os.remove(lock)
                    except OSError: pass
                    start = time.time()
                time.sleep(0.02)
        try:
            yield
        finally:
            os.close(fd)
            try: os.remove(lock)
            except OSError: pass

    def _read(self):
        try:
            with open(self.path, encoding="utf-8") as f: return json.load(f)
        except Exception:
            return {"sources": [], "calls": 0}

    def _mutate(self, fn):
        with self._lock():
            d = self._read(); out = fn(d)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f: json.dump(d, f)
            os.replace(tmp, self.path)
            return out

    def all(self):
        with self._lock(): return self._read()["sources"]

    def by_id(self, sid):
        return next((s for s in self.all() if s["id"] == sid), None)

    def add_sources(self, items):
        def fn(d):
            known = {s["url"]: s for s in d["sources"]}; out = []
            for it in items:
                if not it.get("url"): continue
                if it["url"] not in known:
                    it = {**it, "id": f"S{len(d['sources']) + 1}"}
                    d["sources"].append(it); known[it["url"]] = it
                out.append(known[it["url"]])
            return out
        return self._mutate(fn)

    def update_source(self, sid, **kw):
        def fn(d):
            for s in d["sources"]:
                if s["id"] == sid: s.update(kw)
        self._mutate(fn)

    def bump_calls(self, max_calls):
        def fn(d):
            d["calls"] = d.get("calls", 0) + 1
            return d["calls"] <= max_calls
        return self._mutate(fn)

    def reset_calls(self):
        def fn(d): d["calls"] = 0
        self._mutate(fn)
