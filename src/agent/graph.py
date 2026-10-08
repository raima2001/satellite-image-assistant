"""Sections 2.1, 2.2, and 2.4 - the agent, its context policy, and its traces.

One agent that helps an environmental analyst choose Sentinel-2 imagery for
vegetation monitoring. Built as a graph with a tool-calling loop and a
checkpointer that keys conversation state by the API session id.
"""

import json
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import Annotated, NotRequired, TypedDict

from .skills import load_skill_index, match_skill
from .tracing import elapsed_ms, log_tool_failure, now_ms, write_trace

MODEL_ID = os.environ.get("MODEL_NAME", "openai/gpt-oss-120b")

# Context bounds (2.2). The window is a number of recent messages; tool outputs
# are truncated so raw catalogue payloads never accumulate in the history.
MAX_MESSAGES = 12
MAX_TOOL_CHARS = 800

SYSTEM_PROMPT = """You help an environmental analyst choose Sentinel-2 imagery \
for vegetation monitoring. You search the Earth Search catalogue and explain \
scene metadata. You do not assess vegetation health.

Use the tools to search for scenes by area, date range, and maximum cloud \
cover, and to look up the details and available bands of a single scene by id. \
If the analyst names a place instead of giving coordinates, resolve it to a \
bounding box first.

Ground every answer in tool output. Quote the scene ids, dates, and cloud \
percentages the tools return. If the tools do not give you enough information, \
say so instead of guessing.

When you present a shortlist, number it, and keep that numbering for the rest \
of the conversation so the analyst can refer to "the second one". If the \
session facts below list a shortlist, a selected scene, or search criteria, \
resolve references like "the second one", "that site", or "the same date" \
against them rather than asking the analyst to repeat the details.

Be explicit about what the catalogue can and cannot tell you. A suitable scene \
is a candidate based on location, date, and the cloud cover reported for the \
whole scene. That does not guarantee the analyst's own site is cloud-free, and \
it says nothing about vegetation health.

If a tool call fails, tell the analyst plainly what you tried to do and that \
it failed, using the error you were given. Do not guess a scene's details to \
cover for a failed lookup.

The get_scene_details tool returns only scene ID, datetime, cloud cover,\
and asset names. Report asset names exactly as returned. Do not add\
wavelengths, spatial resolutions, URLs, or other scene-specific properties\
unless they are explicitly present in the tool output.\

If asked for missing properties, say: "The tool output does not include\
that information." Clearly distinguish general background explanations\
from facts retrieved about a particular scene.

Describe search results as a returned shortlist. Never claim they are\
all matching scenes or the only available scenes, because the search\
tool does not retrieve every page of catalogue results.


"""

MODEL_ERROR_REPLY = (
    "I could not get a response from the language model (the request failed), "
    "so I was unable to answer. Please try again shortly."
)

SCENE_FIELDS = ("id", "datetime", "cloud_cover")

# Bonus 3.B. Only frontmatter (name, description) is read here, at import
# time - cheap, and resident for every turn. A skill's body is read and
# added to the prompt only on a turn that matches it (see call_model).
SKILL_INDEX = load_skill_index()


class AgentState(TypedDict):
    """Graph state.

    `messages` is the conversation; `add_messages` appends each turn instead of
    overwriting. The other three are the pinned facts of 2.2: they are small,
    they are updated from tool results, and they are sent on every turn, so a
    reference still resolves after the message window has dropped the turn that
    introduced it.
    """

    messages: Annotated[list, add_messages]
    criteria: NotRequired[dict]
    shortlist: NotRequired[list]
    selected: NotRequired[dict]


def build_model(tools: list[BaseTool]):
    """Call GPT-OSS through Groq."""
    token = os.getenv("GROQ_API_KEY")
    if not token:
        raise RuntimeError("Set GROQ_API_KEY before starting the app.")

    chat_model = ChatOpenAI(
        model=MODEL_ID,
        base_url="https://api.groq.com/openai/v1",
        api_key=token,
        max_retries=2,
        timeout=120,
    )

    return chat_model.bind_tools(tools)

def session_facts(state: AgentState) -> str:
    """The pinned facts, rendered for the prompt. Empty before the first search."""
    lines = []
    if state.get("criteria"):
        lines.append(f"Search criteria: {json.dumps(state['criteria'])}")
    if state.get("shortlist"):
        lines.append("Shortlist, numbered as it was shown to the analyst:")
        for position, scene in enumerate(state["shortlist"], start=1):
            lines.append(
                f"  {position}. {scene.get('id')} - {scene.get('datetime')}"
                f" - cloud cover {scene.get('cloud_cover')}%"
            )
    if state.get("selected"):
        lines.append(f"Scene last looked up in detail: {state['selected'].get('id')}")
    if not lines:
        return ""
    return "Session facts, kept outside the message window:\n" + "\n".join(lines)


def recent_messages(messages: list) -> list:
    """The last MAX_MESSAGES messages, with long tool outputs truncated.

    A window can start part-way through a batch of tool results, which the model
    API rejects because the matching tool call is no longer there, so any leading
    tool messages are dropped.
    """
    window = messages[-MAX_MESSAGES:]
    while window and isinstance(window[0], ToolMessage):
        window = window[1:]
    return [_truncate(message) for message in window]


def _truncate(message):
    # `.text` normalizes content whether the tool returned a plain string or
    # the list of content blocks MCP tool results come back as.
    if not isinstance(message, ToolMessage):
        return message
    content = message.text
    if len(content) <= MAX_TOOL_CHARS:
        return message
    suffix = "\n... [truncated]"
    short = content[:MAX_TOOL_CHARS - len(suffix)] + suffix
    return message.model_copy(update={"content": short})


def _session_id(config: RunnableConfig | None) -> str:
    return (config or {}).get("configurable", {}).get("thread_id", "unknown")


def build_graph(tools: list[BaseTool]):
    """Compile the agent graph.

    Nodes: `model` calls the LLM, `tools` runs whatever tool calls it asked for
    (the search and scene-detail tools from the MCP server), and `remember`
    pins the facts of 2.2 from those results. The loop is
    model -> tools -> remember -> model. It stops when the model replies with no
    tool calls, which routes to END.

    Each node also writes a trace record for 2.4: session id, step name, tool
    name, arguments, duration, error if any, and the final answer. A failing
    tool never raises past this graph - the MCP server returns a clear error
    instead of crashing, `run_tools` logs it, and `remember` leaves the pinned
    facts untouched so the conversation keeps going.
    """
    model = build_model(tools)
    tool_node = ToolNode(tools)

    def call_model(state: AgentState, config: RunnableConfig) -> dict:
        session_id = _session_id(config)
        prompt = [SystemMessage(SYSTEM_PROMPT)]
        facts = session_facts(state)
        if facts:
            prompt.append(SystemMessage(facts))

        # Bonus 3.B: load a skill's body into the prompt only for a turn
        # whose latest human message matches it. Not stored in state, so it
        # never lingers into a later, unrelated turn.
        latest_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        matched_skill = (
            match_skill(latest_human.content, SKILL_INDEX) if latest_human else None
        )
        if matched_skill:
            prompt.append(SystemMessage(matched_skill.body()))

        prompt += recent_messages(state["messages"])

        start = now_ms()
        try:
            response = model.invoke(prompt)
        except Exception as error:  # provider quota, auth, or availability
            message = f"Model call failed: {type(error).__name__}: {error}"
            write_trace(
                session_id=session_id,
                step="model",
                duration_ms=elapsed_ms(start),
                error=message,
                final_answer=MODEL_ERROR_REPLY,
            )
            return {"messages": [AIMessage(MODEL_ERROR_REPLY)]}
        write_trace(
            session_id=session_id,
            step="model",
            duration_ms=elapsed_ms(start),
            final_answer=response.content if not response.tool_calls else None,
            skill=matched_skill.name if matched_skill else None,
        )
        return {"messages": [response]}

    async def run_tools(state: AgentState, config: RunnableConfig) -> dict:
        """Run the pending tool calls and trace/log each result.

        A failing tool does not raise here: the MCP tool raises a clear error
        on its side, and the MCP client turns that into a `ToolMessage` with
        `status="error"` instead of an exception, so this stays a normal step.
        """
        session_id = _session_id(config)
        tool_calls = {call["id"]: call for call in state["messages"][-1].tool_calls}

        start = now_ms()
        result = await tool_node.ainvoke(state, config)
        duration_ms = elapsed_ms(start)

        for message in result["messages"]:
            call = tool_calls.get(message.tool_call_id, {})
            failed = getattr(message, "status", "success") == "error"
            error = message.text if failed else None
            if failed:
                log_tool_failure(
                    session_id=session_id,
                    tool_name=call.get("name", "unknown"),
                    error=error,
                )
            write_trace(
                session_id=session_id,
                step="tools",
                tool_name=call.get("name"),
                arguments=call.get("args"),
                duration_ms=duration_ms,
                error=error,
            )
        return result

    def remember(state: AgentState, config: RunnableConfig) -> dict:
        """Update the pinned facts from the tool results just produced.

        A search result carries `criteria` and `scenes`; a scene lookup carries a
        single scene with an `id`. Anything else, including a tool error, leaves
        the pinned facts as they were.
        """
        start = now_ms()
        updates: dict = {}
        for message in reversed(state["messages"]):
            if not isinstance(message, ToolMessage):
                break
            try:
                payload = json.loads(message.text)
            except (TypeError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            if "scenes" in payload:
                updates["criteria"] = payload.get("criteria", {})
                updates["shortlist"] = [
                    {field: scene.get(field) for field in SCENE_FIELDS}
                    for scene in payload["scenes"]
                ]
            elif "id" in payload:
                updates["selected"] = {
                    field: payload.get(field) for field in SCENE_FIELDS
                }
                # Extract pinned facts above, then replace stored tool
        # messages with shortened copies that retain their IDs.
        shortened = []
        for message in reversed(state["messages"]):
            if not isinstance(message, ToolMessage):
                break
            shortened.append(_truncate(message))

        updates["messages"] = list(reversed(shortened))
        write_trace(
            session_id=_session_id(config),
            step="remember",
            duration_ms=elapsed_ms(start),
        )
        return updates

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        return "tools" if last_message.tool_calls else END

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", run_tools)
    graph.add_node("remember", remember)

    graph.set_entry_point("model")
    graph.add_conditional_edges("model", should_continue, ["tools", END])
    graph.add_edge("tools", "remember")
    graph.add_edge("remember", "model")

    # In-memory checkpointer: conversation state is persisted per thread, and the
    # API passes its session id as the thread id. It is lost on restart; a
    # LangGraph SQLite or Postgres checkpointer would survive one.
    return graph.compile(checkpointer=InMemorySaver())
