"""Gemini calls with 3-key fallback. Rotates ONLY on quota/rate-limit/overload errors."""
import os, time
from dotenv import load_dotenv
load_dotenv()

RETRYABLE = ("429", "resource_exhausted", "quota", "rate limit", "503", "unavailable",
             "504", "deadline", "timeout", "timed out")
from tools.logconfig import get_logger
_L = get_logger()
LOG = []  # observability: tool names, timing, errors (never keys)

class AllKeysFailed(Exception):
    pass

def log(event, **kw):
    LOG.append({"t": round(time.time(), 2), "event": event, **kw})
    _L.info("EVENT %s %s", event, kw)

def load_keys():
    keys = [os.getenv(f"GEMINI_API_KEY_{i}") for i in (1, 2, 3)]
    try:  # Streamlit secrets, if present
        import streamlit as st
        keys = [k or st.secrets.get(f"GEMINI_API_KEY_{i+1}") for i, k in enumerate(keys)]
    except Exception:
        pass
    return [k for k in keys if k]

def is_retryable(err):
    s = str(err).lower()
    return any(t in s for t in RETRYABLE)

def with_key_fallback(fn, keys=None, attempts_per_key=2, backoff=1.5):
    """Call fn(api_key). Retry with backoff, then move to next key on retryable errors only."""
    keys = keys if keys is not None else load_keys()
    if not keys:
        raise AllKeysFailed("No Gemini API keys configured. Set GEMINI_API_KEY_1..3.")
    for i, key in enumerate(keys, 1):
        for attempt in range(attempts_per_key):
            try:
                out = fn(key)
                log("gemini_ok", key_slot=i)
                return out
            except Exception as e:  # noqa
                if not is_retryable(e):
                    log("gemini_nonretryable_error", key_slot=i, error=type(e).__name__)
                    raise  # auth/invalid request: do not rotate
                log("gemini_retryable_error", key_slot=i, attempt=attempt + 1)
                time.sleep(backoff * (attempt + 1))
        log("rotating_key", from_slot=i)
    raise AllKeysFailed("All Gemini keys hit quota/rate limits. Please try again later.")

def models_list():
    """Quota is tracked PER MODEL, so falling back to another model often works when one is exhausted."""
    first = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    raw = os.getenv("GEMINI_MODELS", f"{first},gemini-2.5-flash-lite")
    out = []
    for m in (x.strip() for x in raw.split(",")):
        if m and m not in out: out.append(m)
    return out

def _gemini_call(key, model, prompt, system, json_mode):
    from google import genai
    from google.genai import types
    cfg = types.GenerateContentConfig(system_instruction=system, temperature=0.2,
                                      response_mime_type="application/json" if json_mode else None)
    return genai.Client(api_key=key).models.generate_content(model=model, contents=prompt, config=cfg).text

def generate(prompt, system=None, json_mode=False, keys=None, models=None, call_fn=None):
    """One Gemini call. Tries every key on each model; moves to the next model when all keys are limited."""
    call_fn, models = call_fn or _gemini_call, models or models_list()
    last = None
    for model in models:
        try:
            out = with_key_fallback(lambda k: call_fn(k, model, prompt, system, json_mode), keys, attempts_per_key=1)
            log("generate_ok", model=model); return out
        except AllKeysFailed as e:
            last = e; log("model_exhausted", model=model)
    raise last or AllKeysFailed("No models configured.")
