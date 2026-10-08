"""Section 2.4 - the trace file and the failure log.

Every step the graph takes (a model call, a tool call, the remember step)
appends one JSON line to traces/<session_id>.jsonl. That line has what the
assignment asks a trace to reconstruct a run from: session id, step name,
tool name, arguments, duration, error if any, and the final answer.

A failing tool call is also logged through the standard `logging` module with
the session id, the tool name, and the error - a second, independent record of
the same failure, separate from the trace file.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("satellite_image_assistant")

TRACE_DIR = Path(__file__).resolve().parent.parent.parent / "traces"


def now_ms() -> float:
    """A monotonic timer start, paired with elapsed_ms() around a step."""
    return time.monotonic()


def elapsed_ms(start: float) -> float:
    return round((time.monotonic() - start) * 1000, 1)


def write_trace(
    *,
    session_id: str,
    step: str,
    tool_name: str | None = None,
    arguments: dict | None = None,
    duration_ms: float | None = None,
    error: str | None = None,
    final_answer: str | None = None,
    skill: str | None = None,
) -> None:
    """Append one reconstructable step to this session's trace file."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "step": step,
        "tool_name": tool_name,
        "arguments": arguments,
        "duration_ms": duration_ms,
        "error": error,
        "final_answer": final_answer,
        "skill": skill,
    }
    TRACE_DIR.mkdir(exist_ok=True)
    with open(TRACE_DIR / f"{session_id}.jsonl", "a") as trace_file:
        trace_file.write(json.dumps(record) + "\n")


def log_tool_failure(*, session_id: str, tool_name: str, error: str) -> None:
    """Record a failing tool call. The graph keeps running; this is just the log."""
    logger.error(
        "tool call failed session_id=%s tool=%s error=%s",
        session_id,
        tool_name,
        error,
    )
