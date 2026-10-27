"""Terminal + file logging. Terminal (stderr) shows live progress; full log is in outputs/agent.log. Never logs API keys."""
import logging, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_logger():
    lg = logging.getLogger("evidencelens")
    if not lg.handlers:
        lg.setLevel(logging.INFO); lg.propagate = False
        fmt = logging.Formatter("%(asctime)s [%(process)d] %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler(sys.stderr); sh.setFormatter(fmt); lg.addHandler(sh)
        try:
            os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
            fh = logging.FileHandler(os.path.join(ROOT, "outputs", "agent.log"), encoding="utf-8")
            fh.setFormatter(fmt); lg.addHandler(fh)
        except OSError:
            pass
    return lg
