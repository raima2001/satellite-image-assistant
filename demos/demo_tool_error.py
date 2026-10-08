"""Section 2.4 demo turn - force a tool error and show the agent survives it.

Asks the agent to look up a scene id that does not exist. The MCP server
raises a clear error, the server stays up, the agent's reply explains the
failure instead of guessing, and the trace file for this session records the
error
"""

import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_core.messages import HumanMessage

from agent.a2a_client import resolve_place_to_bbox
from agent.graph import build_graph
from agent.mcp_tools import load_mcp_tools
from agent.tracing import TRACE_DIR


async def main() -> None:
    session_id = f"demo-error-{uuid4().hex[:8]}"
    tools = await load_mcp_tools()
    tools.append(resolve_place_to_bbox)
    graph = build_graph(tools)
    config = {"configurable": {"thread_id": session_id}}

    question = "Look up scene ID deliberately-invalid-scene."
    print(f"Session: {session_id}")
    print(f"User: {question}")

    result = await graph.ainvoke({"messages": [HumanMessage(question)]}, config=config)
    print(f"Agent: {result['messages'][-1].content}")

    trace_path = TRACE_DIR / f"{session_id}.jsonl"
    print(f"\nTrace file: {trace_path}")
    for line in trace_path.read_text().splitlines():
        record = json.loads(line)
        print(f"  {record['step']:9s} tool={record['tool_name']} "
              f"error={record['error']!r} duration_ms={record['duration_ms']}")


if __name__ == "__main__":
    asyncio.run(main())
