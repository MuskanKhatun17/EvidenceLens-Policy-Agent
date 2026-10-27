from dataclasses import dataclass, field

@dataclass
class RunState:
    """Per-run state for the harness-side submit tool (sources live in the shared SourceStore)."""
    max_submits: int = 3
    submits: int = 0
    report: dict | None = None
    issues: list = field(default_factory=list)
    accepted: bool = False
