"""Small offline checks of the context policy, skill routing, and tracing."""

import json
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent import graph, tracing  # noqa: E402
from agent.skills import load_skill_index, match_skill  # noqa: E402


def test_window_is_bounded_and_starts_on_a_non_tool_message():
    messages = [HumanMessage(str(i)) for i in range(30)]
    messages[-graph.MAX_MESSAGES] = ToolMessage("orphan", tool_call_id="x")
    window = graph.recent_messages(messages)
    assert len(window) <= graph.MAX_MESSAGES
    assert not isinstance(window[0], ToolMessage)


def test_long_tool_output_is_truncated():
    long = ToolMessage("a" * 5000, tool_call_id="x")
    (short,) = graph.recent_messages([HumanMessage("q"), long])[1:]
    assert len(short.text) <= graph.MAX_TOOL_CHARS


def test_session_facts_numbers_the_shortlist():
    facts = graph.session_facts(
        {"messages": [], "shortlist": [
            {"id": "A", "datetime": "d1", "cloud_cover": 1},
            {"id": "B", "datetime": "d2", "cloud_cover": 2},
        ]}
    )
    assert "2. B" in facts
    assert graph.session_facts({"messages": []}) == ""


def test_skill_loads_only_for_matching_request():
    index = load_skill_index()
    assert match_skill("Which bands should I use for NDVI?", index)
    assert match_skill("What was the cloud cover?", index) is None
    # Regression: a plain asset lookup must not pull in the NDVI skill.
    assert match_skill(
        "Look up the second scene from that shortlist and list its available assets.",
        index,
    ) is None


def test_trace_record_has_required_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(tracing, "TRACE_DIR", tmp_path)
    tracing.write_trace(session_id="s", step="tools", tool_name="t",
                        arguments={"a": 1}, error="boom")
    record = json.loads((tmp_path / "s.jsonl").read_text())
    for field in ("session_id", "step", "tool_name", "arguments", "error", "final_answer"):
        assert field in record
