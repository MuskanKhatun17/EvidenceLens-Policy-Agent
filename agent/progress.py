"""Live progress callback: logs every LLM call and tool call with timings (terminal, outputs/agent.log, optional UI)."""
import time
from langchain_core.callbacks import AsyncCallbackHandler
from tools.logconfig import get_logger

L = get_logger()

class ProgressCallback(AsyncCallbackHandler):
    def __init__(self, on_event=None):
        self.t0, self.starts, self.n_llm, self.on_event = time.time(), {}, 0, on_event

    def _emit(self, msg):
        msg = f"[+{time.time() - self.t0:5.1f}s] {msg}"
        L.info(msg)
        if self.on_event:
            try: self.on_event(msg)
            except Exception: pass

    async def on_chat_model_start(self, serialized, messages, *, run_id, **kw):
        self.n_llm += 1; self.starts[run_id] = time.time()
        self._emit(f"LLM call #{self.n_llm} started")

    async def on_llm_end(self, response, *, run_id, **kw):
        self._emit(f"LLM call finished in {time.time() - self.starts.pop(run_id, time.time()):.1f}s")

    async def on_llm_error(self, error, *, run_id, **kw):
        self._emit(f"LLM ERROR after {time.time() - self.starts.pop(run_id, time.time()):.1f}s: {type(error).__name__}: {str(error)[:120]}")

    async def on_tool_start(self, serialized, input_str, *, run_id, **kw):
        name = kw.get("name") or (serialized or {}).get("name", "?")
        self.starts[run_id] = time.time()
        self._emit(f"TOOL {name} started: {str(input_str)[:80]}")

    async def on_tool_end(self, output, *, run_id, **kw):
        self._emit(f"TOOL finished in {time.time() - self.starts.pop(run_id, time.time()):.1f}s")

    async def on_tool_error(self, error, *, run_id, **kw):
        self._emit(f"TOOL ERROR: {type(error).__name__}: {str(error)[:120]}")
