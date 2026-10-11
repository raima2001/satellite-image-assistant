"""Section 2.5 - the HTTP API.

Exposes the agent (2.1-2.4) over HTTP. POST /chat takes a message and a
session id and returns the agent's reply for that session. Two read-only debug
endpoints show a session's saved state and its checkpoint history. The MCP
server (2.3) is started once, at startup, as the agent's tool connection.

Run with: python src/api/app.py
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

# Read GROQ_API_KEY and MODEL_NAME from the repo-root .env before the agent
# modules read them. Variables already exported in the shell take precedence.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agent.a2a_client import resolve_place_to_bbox
from agent.graph import build_graph
from agent.mcp_tools import load_mcp_tools
from agent.tracing import logger, write_trace


@asynccontextmanager
async def lifespan(app: FastAPI):
    tools = await load_mcp_tools()
    tools.append(resolve_place_to_bbox)
    app.state.graph = build_graph(tools)
    yield


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str


@app.get("/")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or uuid4().hex
    config = {"configurable": {"thread_id": session_id}}
    try:
        result = await app.state.graph.ainvoke(
            {"messages": [HumanMessage(request.message)]}, config=config
        )
    except Exception as error:
        # Last line of defence: tool and model errors are handled inside the
        # graph, so reaching here means something unexpected. Log, trace, and
        # reply instead of returning a bare 500.
        logger.exception("chat request failed session_id=%s", session_id)
        write_trace(
            session_id=session_id,
            step="api",
            error=f"{type(error).__name__}: {error}",
        )
        return JSONResponse(
            status_code=500,
            content={
                "session_id": session_id,
                "reply": "Something went wrong while handling your request, "
                "so I could not answer. Please try again.",
            },
        )
    return ChatResponse(session_id=session_id, reply=result["messages"][-1].content)


def _thread_config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


@app.get("/sessions/{session_id}/state")
async def session_state(session_id: str) -> dict:
    """The latest checkpoint for a session: its pinned facts and message count."""
    snapshot = await app.state.graph.aget_state(_thread_config(session_id))
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="No state for that session id.")
    return {
        "session_id": session_id,
        "checkpoint_id": snapshot.config["configurable"].get("checkpoint_id"),
        "created_at": snapshot.created_at,
        "criteria": snapshot.values.get("criteria"),
        "shortlist": snapshot.values.get("shortlist"),
        "selected": snapshot.values.get("selected"),
        "message_count": len(snapshot.values.get("messages", [])),
    }


@app.get("/sessions/{session_id}/history")
async def session_history(session_id: str) -> dict:
    """Every checkpoint for a session, newest first.

    Lists each checkpoint's step and which node runs next, not the stored
    values: older checkpoints can still hold full, untruncated tool results.
    """
    checkpoints = [
        {
            "checkpoint_id": snapshot.config["configurable"].get("checkpoint_id"),
            "step": snapshot.metadata.get("step"),
            "source": snapshot.metadata.get("source"),
            "next": list(snapshot.next),
            "created_at": snapshot.created_at,
        }
        async for snapshot in app.state.graph.aget_state_history(
            _thread_config(session_id)
        )
    ]
    if not checkpoints:
        raise HTTPException(status_code=404, detail="No state for that session id.")
    return {"session_id": session_id, "checkpoints": checkpoints}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
