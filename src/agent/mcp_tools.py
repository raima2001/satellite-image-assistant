"""Load the MCP tools for the agent graph (section 2.3).

The agent calls its tools through MCP: this spawns the MCP server as a
subprocess over stdio and loads its tools as LangChain tools backed by real
MCP tool calls. Importing the tool functions from src/mcp_server directly
would not go through MCP, so the graph never does that.
"""

from pathlib import Path

from langchain_core.tools import BaseTool
from langchain.mcp import MCPAdapter

SERVER_SCRIPT = Path(__file__).resolve().parent.parent / "mcp_server" / "server.py"


async def load_mcp_tools() -> list[BaseTool]:
    """Start the MCP server and return its tools as LangChain tools.

    A `Path` target is launched over stdio with the current interpreter. The
    returned tools keep the adapter's client, so they stay callable after the
    context exits.
    """
    async with MCPAdapter(SERVER_SCRIPT) as adapter:
        return await adapter.list_tools()
