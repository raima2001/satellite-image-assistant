"""Load the MCP tools for the agent graph (section 2.3).

The agent calls its tools through MCP: this spawns the MCP server as a
subprocess over stdio and loads its tools as LangChain tools backed by real
MCP tool calls. Importing the tool functions from src/mcp_server directly
would not go through MCP, so the graph never does that.
"""

import sys
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

SERVER_SCRIPT = Path(__file__).resolve().parent.parent / "mcp_server" / "server.py"


async def load_mcp_tools() -> list[BaseTool]:
    """Start the MCP server and return its tools as LangChain tools."""
    client = MultiServerMCPClient(
        {
            "sentinel-catalogue": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(SERVER_SCRIPT)],
            }
        }
    )
    return await client.get_tools()
