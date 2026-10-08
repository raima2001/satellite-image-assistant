# Satellite Image Assistant

A single LangGraph agent that helps an environmental analyst choose Sentinel-2 imagery for vegetation monitoring. It searches the Earth Search catalogue, remembers a numbered shortlist across turns, and retrieves scene metadata through a custom MCP server. It selects candidate imagery; it does not download or analyse pixels or assess vegetation health.

## Quick start

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), an internet connection, and a [Groq API key](https://console.groq.com/keys). The live workflow has been exercised on Python 3.14. Other Python versions are untested.

Run all commands from the repository root. Install dependencies from `pyproject.toml` and `uv.lock`:

```bash
uv sync --locked
```

Configure the hosted model by copying the example env file and adding your key:

```bash
cp .env.example .env
# then edit .env and set GROQ_API_KEY=your_groq_api_key
```

The API loads `.env` at startup. `.env` is git-ignored. Exporting `GROQ_API_KEY` and `MODEL_NAME` in the shell also works and takes precedence.

Start the API:

```bash
uv run python src/api/app.py
```

Wait for `Application startup complete`. The API launches the MCP server automatically as a subprocess over stdio; no second server terminal is required. Leave this terminal running. The geo agent for place-name queries (Bonus 3.A) is optional and started separately; see that section.

- Health check: <http://localhost:8000/>
- Interactive API documentation: <http://localhost:8000/docs>

The model runs on Groq through its OpenAI-compatible endpoint, `https://api.groq.com/openai/v1`, using LangChain's `ChatOpenAI`. No local model download or GPU is required. Groq offers a Free plan with request and token limits; availability and account quotas can change. An agent turn can require several model requests. Keep API keys out of source control.

If dependencies are changed during development, run `uv lock` and `uv sync`, then commit both `pyproject.toml` and `uv.lock`.

## Chat from the terminal

With the API running, open a second terminal for an interactive chat that renders replies as formatted tables and text instead of raw JSON:

```bash
uv run python demos/chat.py --session demo-3
```

All messages in one run share the session ID. Type `/trace` to print a one-line-per-step summary of the session's trace, and `/exit` to quit. The three demo messages below can be typed here directly.

## Three-turn HTTP demo

In another terminal, run these requests in order. They use the same session ID. Choose a new ID if repeating the demo against the same running process, and use it consistently in all commands and the trace path.

### 1. Search for imagery

```bash
curl -sS http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "demo-1",
    "message": "Find Sentinel-2 imagery within bbox [-122.5, 37.5, -122.4, 37.6] for June 2025 with less than 20% cloud cover. Show a numbered shortlist."
  }'
```

Expected: the agent calls `search_scenes` and reports a numbered shortlist with scene IDs, dates, and cloud cover. Results depend on the live catalogue.

### 2. Follow up using conversation context

```bash
curl -sS http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "demo-1",
    "message": "Look up the second scene from that shortlist and list its available assets."
  }'
```

Expected: the agent resolves the second scene from session state and calls `get_scene_details` without requiring the user to repeat its ID. This demo assumes the search returns at least two scenes, as it did in the recorded run.

### 3. Force a tool error

```bash
curl -sS http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "demo-1",
    "message": "Look up the exact scene ID INVALID_SCENE_TEST_404. Do not substitute another scene or search for alternatives. Explain if the lookup fails."
  }'
```

Expected: the catalogue returns a missing-scene error, the agent explains what failed, and the API returns a normal chat reply rather than terminating the process. The failure is logged with the session ID, tool name, and error.

Confirm the server is still responding, then inspect the demo trace:

```bash
curl -sS http://localhost:8000/
cat traces/demo-1.jsonl
```

Match each reply to a `model` record with a non-null `final_answer`. The invalid-ID lookup has a `tools` record with `tool_name`, the attempted `scene_id` in `arguments`, and a non-null `error`.

## Graph and session state (2.1)

`src/agent/graph.py` builds one agent with three nodes:

- `model`: sends the system prompt, pinned session facts, and recent messages to the model.
- `tools`: executes the model's requested tools through MCP and logs their outcomes.
- `remember`: extracts compact facts from tool results, then shortens the latest tool messages in the current state.

Execution starts at `model`. If its response contains tool calls, execution follows `model → tools → remember → model`. The loop ends at `END` when the model returns a response with no tool calls.

The graph uses `InMemorySaver`. The API maps its `session_id` to LangGraph's `thread_id`, allowing successive requests to resume the same conversation. Restarting the API clears this in-memory state; existing trace files do not restore it. For restart durability, use a SQLite checkpointer for a small single-process deployment or a PostgreSQL checkpointer for a shared deployment, with the corresponding database setup and lifecycle management.

## Context policy (2.2)

### What is stored

Each session stores conversation messages and three structured facts extracted from tool results:

| Fact | Contents |
| --- | --- |
| `criteria` | Latest search bounding box, date range, and cloud-cover limit |
| `shortlist` | Latest search results in tool-returned order, reduced to ID, datetime, and cloud cover |
| `selected` | Last successfully retrieved scene, reduced to ID, datetime, and cloud cover |

The prompt instructs the model to preserve shortlist numbering. A failed lookup leaves the pinned facts unchanged. These facts let references such as “the second one” or “that site” resolve after the original exchange leaves the recent-message window.

### What is sent to the model

Every model call receives:

1. The system prompt.
2. Pinned session facts, if present: search criteria, the numbered shortlist,
   and the selected scene’s ID.
3. Matching skill instructions, when applicable.
4. Up to the latest 12 conversation messages, counting user, assistant, and tool messages.

Leading tool messages are removed if the window no longer contains their matching assistant tool call. Tool-message text is limited to 800 characters, including the truncation marker. `remember` extracts facts before replacing the latest tool messages with shortened copies retaining their message IDs.

This bounds the number of messages and size of individual tool outputs sent to the model. It is not a total token budget: user and assistant messages can still be long. The current message history and older in-memory checkpoint snapshots can continue to grow; production use would require retention or pruning. Older snapshots may retain pre-truncation tool results.

### Long-conversation demonstration

The live `groq-test-1` run completed 12 user turns. Use one fresh session ID for the following sequence to reproduce it:

| Turn | Message |
| --- | --- |
| 1 | Find Sentinel-2 imagery within bbox [-122.5, 37.5, -122.4, 37.6] for June 2025 with less than 20% cloud cover. Show a numbered shortlist. |
| 2 | For the next few turns, keep replies to one sentence and do not repeat the scene IDs or search criteria. What does scene-wide cloud cover mean? |
| 3 | Does low scene-wide cloud cover guarantee my specific site is clear? |
| 4 | Can catalogue metadata alone tell me whether vegetation is healthy? |
| 5 | What is the difference between choosing an image and analysing it? |
| 6 | Why might clouds interfere with vegetation monitoring? |
| 7 | Why should I check cloud shadows as well? |
| 8 | What should I verify before comparing vegetation across two images? |
| 9 | Look up the second scene from our original shortlist and report its ID and available bands. |
| 10 | What exact bounding box, date range, and cloud-cover limit did we use originally? Do not run another search. |
| 11 | Look up the exact scene ID INVALID_SCENE_TEST_404. Do not substitute another scene. Explain if the lookup fails. |
| 12 | Which valid scene did we last successfully look up before that failed request? Give its ID. |

By turn 9, the original search exchange falls outside the 12-message window. In the recorded run, turn 9 called `get_scene_details` for `S2B_T10SEG_20250622T190344_L2A`; turn 10 recalled the exact original criteria without searching again; turn 11 logged the deliberate error; and turn 12 identified the same valid scene. The trace is `traces/groq-test-1.jsonl`.

This run demonstrated context continuity and exposed unsupported metadata additions in a reply. The grounding prompt was subsequently tightened and tested separately as described below. The current trace does not record the actual prompt window or its message count; the window behavior is inferred from the reviewed code and the conversation sequence.

## MCP server and grounding (2.3)

`src/mcp_server/server.py` defines a custom `FastMCP` server. Both tools call the external Earth Search STAC API at <https://earth-search.aws.element84.com/v1>, using collection `sentinel-2-c1-l2a`.

| Tool | External operation | Returned data |
| --- | --- | --- |
| `search_scenes` | `POST /search` | Search criteria and a shortlist of scene IDs, datetimes, and cloud cover, sorted by cloud cover |
| `get_scene_details` | `GET /collections/{collection}/items/{scene_id}` | Scene ID, datetime, cloud cover, and sorted asset names |

`src/agent/mcp_tools.py` starts the server over stdio and uses `langchain-mcp-adapters` to expose MCP-backed tools to the graph. The graph receives these tools rather than importing the server's Python tool functions. The tools node runs asynchronously, and the API drives the graph with `ainvoke`.

The API starts the MCP process automatically. Its standalone stdio entry point is:

```bash
uv run python src/mcp_server/server.py
```

This standalone command waits for an MCP client; it is not an HTTP server and is not needed in addition to the normal API startup.

The system prompt requires scene-specific claims to come from tool outputs. Asset names do not establish wavelengths, spatial resolutions, download URLs, or vegetation condition. If requested fields are absent, the agent must say so and distinguish general background knowledge from retrieved facts. Scene-wide cloud cover does not guarantee that the user's smaller area is cloud-free.

A separate live grounding check used session `grounding-test-1`:

```json
{
  "session_id": "grounding-test-1",
  "message": "Look up S2B_T10SEG_20250622T190344_L2A. List its available assets and tell me their wavelengths and spatial resolutions."
}
```

The reply listed asset names and explicitly stated that wavelengths and resolutions were absent from the tool output. This is one successful behavioral check, not a guarantee against all unsupported claims.

### Catalogue limitations

The current search fetches the first page of up to 100 items, filters cloud cover locally, sorts that page, and returns up to the requested shortlist limit. It does not follow pagination. Results are therefore a shortlist from the fetched page, not necessarily an exhaustive search or the globally least-cloudy scenes. The cloud filter currently uses `<=`, so "less than 20%" in the demo prompts is applied as "at most 20%".

## Errors and traces (2.4)

The tools translate HTTP and network failures into errors describing the attempted operation. In the tested MCP integration, those failures become error-status tool messages, allowing the graph to log the error, preserve pinned facts, and ask the model for a clear user-facing explanation.

`src/agent/tracing.py` logs tool failures with session ID, tool name, and error. It also appends JSON Lines records to `traces/<session_id>.jsonl`:

| Field | Meaning |
| --- | --- |
| `timestamp` | Time the trace record was written |
| `session_id` | Conversation identifier |
| `step` | `model`, `tools`, `remember`, or `api` (an unexpected request failure) |
| `tool_name` | Tool name for a tool step; otherwise null |
| `arguments` | Tool arguments for a tool step; otherwise null |
| `duration_ms` | Measured step duration |
| `error` | Tool, model, or API error when present |
| `final_answer` | Final model reply when the turn ends |
| `skill` | Agent Skill loaded for a model step (Bonus 3.B); otherwise null |

For multiple tool calls in one tools-node execution, the current implementation records the same overall batch duration for each result; it does not measure each tool independently. Traces currently omit user input, request IDs, tool-result payloads, and prompt-window counts. These would improve reconstruction and grounding audits.

The invalid-ID test verified a logged tool failure, a clear reply, and successful continued conversation. Hosted-model failures (provider quota, authentication, or availability errors) are also caught in `call_model`: the failure is written to the trace in the `error` field and the user gets a plain reply saying the model request failed. This path is covered by code review only; it has not been exercised against a live provider outage.

## HTTP API (2.5)

| Route | Purpose |
| --- | --- |
| `GET /` | Health response: `{"status": "ok"}` |
| `POST /chat` | Submit `message` and `session_id`; receive `session_id` and `reply` |
| `GET /docs` | Interactive request UI |

Use a stable session ID for related turns and a fresh ID for a separate conversation. The API retains the graph and MCP client setup on application state across requests. Stop the API with `Ctrl+C` when finished.

## Bonus 3.A: Multi-agent delegation with A2A

A separate geocoding agent resolves place names to bounding boxes using
OpenStreetMap Nominatim. It runs as its own FastAPI service on port 8001,
separate from the primary LangGraph agent and API on port 8000.

### How delegation works

1. When a user names a place instead of providing coordinates, the primary
   model can call the `resolve_place_to_bbox` tool.
2. The A2A client fetches the geo agent’s Agent Card from
   `GET /.well-known/agent-card.json`.
3. It checks that the card advertises A2A `0.3.0`, JSON-RPC transport, and
   the `resolve-place-to-bbox` capability.
4. It sends a JSON-RPC `message/send` request to the endpoint advertised
   in the card’s `url` field. The request includes a request ID and a
   separate message ID.
5. The geo agent queries Nominatim and returns a matched place name and
   bounding box.
6. The client validates the response ID, message structure, and coordinate
   bounds before returning the result to the primary agent.
7. The primary agent uses that bounding box when calling the
   MCP-backed `search_scenes` tool and produces a reply from the results.

The LangChain tool is a wrapper around an HTTP delegation to another
process. The geo agent is not a second node inside the primary graph.
The Agent Card’s advertised capability is separate from the Agent Skill
described in section 3.B.

### Start both services

Run commands from the repository root after installing dependencies.

Terminal 1 — primary API, which also starts the MCP server (reads the key from `.env`):

```bash
uv run python src/api/app.py
```

Terminal 2 — geo agent:

```bash
uv run python src/geo_agent/service.py
```

The geo agent does not require a model API key. Its geocoding operation
is deterministic and uses the external Nominatim API.

### Inspect the Agent Card and run the standalone demo

In another terminal:

```bash
curl -sS http://localhost:8001/.well-known/agent-card.json
uv run python demos/demo_a2a_delegation.py
```

The card includes:

```json
{
  "protocolVersion": "0.3.0",
  "name": "place-geocoder",
  "url": "http://localhost:8001/",
  "preferredTransport": "JSONRPC",
  "version": "1.0.0",
  "capabilities": {
    "streaming": false,
    "pushNotifications": false
  },
  "defaultInputModes": ["text"],
  "defaultOutputModes": ["text"],
  "skills": [
    {
      "id": "resolve-place-to-bbox",
      "name": "Resolve place to bounding box",
      "description": "Given a place name, returns its bounding box as JSON.",
      "tags": ["geocoding"],
      "examples": ["Rotterdam, Netherlands", "Amazon basin"]
    }
  ]
}
```

Short transcript from the successful standalone demo:

```text
Delegating one task to the geo agent: resolve 'Rotterdam, Netherlands'

Geo agent reply:
{
  "place": "Rotterdam, Zuid-Holland, Nederland",
  "bbox": [4.3793095, 51.8616672, 4.6018092, 51.9942735]
}
```

Coordinates are ordered as `[min_lon, min_lat, max_lon, max_lat]`.
Live geocoding results may change.

### Demonstrate delegation through the primary agent

With both services running:

```bash
curl -sS http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "a2a-demo",
    "message": "Find Sentinel-2 imagery over Rotterdam, Netherlands in June 2025 with less than 20% cloud cover."
  }'
```

Inspect the resulting trace:

```bash
cat traces/a2a-demo.jsonl
```

Expected sequence:

1. `resolve_place_to_bbox` receives `"Rotterdam, Netherlands"`.
2. `search_scenes` uses the returned bounding box.
3. A final `model` record contains the scene shortlist.

The recorded integration run is available in
[traces/a2a-final.jsonl](traces/a2a-final.jsonl). It shows a successful
geocoding call, a catalogue search using the same bounding box, and a final
reply listing two candidate scenes, with no recorded errors.

### Error handling and scope

The client raises a descriptive `ValueError` if discovery fails, the card
is incompatible, the remote service returns an error, or the result is
invalid. The error names the attempted place lookup. Turning that exception
into a user-facing tool result is the responsibility of the primary graph’s
tool-error handling.

This is a hand-written A2A v0.3.0 JSON-RPC subset supporting immediate
`message/send` replies. It does not use `a2a-sdk` or implement streaming,
push notifications, persistent tasks, or multi-turn conversations in the
geo service. Successful demonstrations verify the implemented path;
they are not a full protocol-conformance test.

## Bonus 3.B: Agent Skill

The skill at `skills/ndvi-band-guidance/SKILL.md` provides guidance for
choosing NDVI inputs and explaining the formula. Its frontmatter contains
`name` and `description`; its Markdown body contains the instructions.

This Agent Skill supplies instructions to the primary model. It is separate
from the capabilities advertised in the geo agent’s A2A Agent Card.

### How the skill is selected and loaded

At startup, `src/agent/skills.py` scans `skills/*/SKILL.md`. The loader reads
each file and retains its name, description, and path in an index. The skill
body is not automatically included in model prompts.

For each model call, `call_model()` checks the latest human message:

1. Convert each skill description to lowercase, remove punctuation from
   its words, and exclude stopwords, words shorter than four characters,
   and words generic to this assistant (such as "asset", "band", and
   "sentinel2"). For the NDVI skill this leaves `ndvi`, `vegetation`,
   `index`, and `nearinfrared`.
2. Normalize the user message the same way and check whether it shares
   a whole word with those keywords.
3. Select the first matching skill.
4. Read its body and add it as an extra `SystemMessage` for that model call.

This selection uses Python string matching, with no additional model call
or embedding search.

During a tool-calling loop, the latest human message stays the same, so the
skill can be included in multiple model calls within that user turn. On an
unrelated subsequent turn, the skill body is omitted if nothing matches.

The skill instructions are not stored in conversation state. However,
earlier assistant replies influenced by the skill can remain in the
recent-message window.

### How it changes behavior

For an NDVI question, the skill instructs the agent to:

- Check the scene’s available assets using `get_scene_details`.
- Recommend `red` and `nir` only when those asset names are actually present.
- Explain the formula `(NIR - Red) / (NIR + Red)`.
- Avoid claiming to calculate NDVI pixels.
- Avoid adding wavelengths, resolutions, or other scene metadata absent
  from the tool output.

The formula and band guidance come from the skill; claims about the
particular scene’s available assets must come from the tool result.

### Run the selection demo

From the repository root:

```bash
uv run python demos/demo_skill_ndvi.py
```

This demo requires no model call or network access. It shows:

- A vegetation-index question loading `ndvi-band-guidance`.
- A cloud-cover question matching no skill.

### Test it in a live conversation

With the main API running and its model key configured:

```bash
curl -sS http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "skill-demo",
    "message": "Look up scene S2B_T31UET_20250618T104619_L2A. Which available assets should I use for NDVI, and what is the formula?"
  }'
```

Expected: the agent retrieves the scene’s details, identifies the available
NDVI assets, and explains the formula. The model trace records:

```json
{"step": "model", "skill": "ndvi-band-guidance"}
```

Then ask an unrelated follow-up in the same session:

```bash
curl -sS http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "skill-demo",
    "message": "What was the cloud cover of that scene?"
  }'
```

Expected: the agent retains the scene context, but the new model record has:

```json
{"step": "model", "skill": null}
```

Inspect the trace:

```bash
cat traces/skill-demo.jsonl
```

Use a fresh session ID when repeating the demonstration.

### Recorded evidence

[traces/skill-check-2.jsonl](traces/skill-check-2.jsonl) contains a successful
live run:

- The NDVI turn loaded the skill and called `get_scene_details` for the
  requested scene.
- The reply identified `red` and `nir`, provided the NDVI formula, and stated
  that wavelengths, spatial resolutions, and URLs were not supplied.
- The subsequent cloud-cover question answered from the conversation
  context with `"skill": null`.

This demonstrates selective loading and behavior consistent with the skill.
It is not a controlled comparison against the same request without the skill.

### Limitations

Matching is intentionally simple: one shared keyword is sufficient, which
can cause false positives (for example, any message mentioning
"vegetation") or miss differently phrased requests. An earlier version
matched substrings, so "list its available assets" wrongly loaded the NDVI
skill; whole-word matching and a regression test now prevent that.
If several skills match, the first one in sorted directory order wins.

The frontmatter parser supports the simple single-line `name: value` and
`description: value` format used here; it is not a general YAML parser.
Future improvements would include YAML validation and more precise matching.

## Example traces

| Demonstration | Recorded trace |
| --- | --- |
| The three-turn demo above: search, second-scene follow-up, forced tool error | [demo-2.jsonl](traces/demo-2.jsonl) |
| Earlier core run (first-scene follow-up and tool error) | [test-1.jsonl](traces/test-1.jsonl) |
| Long conversation and context retention | [groq-test-1.jsonl](traces/groq-test-1.jsonl) |
| Missing-metadata grounding check | [grounding-test-1.jsonl](traces/grounding-test-1.jsonl) |
| A2A delegation and subsequent scene search | [a2a-final.jsonl](traces/a2a-final.jsonl) |
| Conditional NDVI skill loading | [skill-check-2.jsonl](traces/skill-check-2.jsonl) |

Running the three-turn demo writes `traces/demo-1.jsonl`; `demo-2.jsonl` is the recorded run of the same three messages, after the final skill-matching fix.

The long-conversation trace predates the final grounding-prompt changes.

## Tests

Offline checks of the message window, tool-output truncation, pinned facts, skill routing, and trace fields (no network or API key needed):

```bash
uv run pytest
```

## Repository guide

| Path | Purpose |
| --- | --- |
| `src/api/app.py` | FastAPI endpoints and startup lifecycle |
| `src/agent/graph.py` | Model configuration, graph, and context policy |
| `src/agent/mcp_tools.py` | MCP connection and tool loading |
| `src/agent/tracing.py` | Structured traces and failure logging |
| `src/mcp_server/server.py` | Custom MCP server and external catalogue calls |
| `src/geo_agent/service.py` | Bonus 3.A: second agent service, Agent Card, A2A endpoint |
| `src/agent/a2a_client.py` | Bonus 3.A: A2A client and delegation tool |
| `demos/chat.py` | Interactive terminal chat client for the API |
| `demos/demo_a2a_delegation.py` | Bonus 3.A: Agent Card and delegation transcript demo |
| `src/agent/skills.py` | Bonus 3.B: Agent Skill index, loading, and matching |
| `skills/ndvi-band-guidance/SKILL.md` | Bonus 3.B: the one Agent Skill |
| `demos/demo_skill_ndvi.py` | Bonus 3.B: skill-loading decision demo |
| `tests/test_core.py` | Offline tests for the context policy, skills, and traces |
| `traces/` | Per-session JSONL traces |
| `pyproject.toml`, `uv.lock` | Dependencies and locked environment |

## What I left out and would do next

Left out, and why: the project selects candidate scenes; it does not download imagery, calculate NDVI, or assess vegetation health. That keeps the scope small enough to finish and verify end to end. Pagination and durable checkpoints were also cut for time.

Next, I would strengthen the agent’s execution and evaluation harness. I would add repeatable tests for tool selection, multi-turn reference resolution, grounding, and error recovery; enforce tool-call and time limits; and handle model-provider failures with selective retries and clear responses. I would also add token-budgeted context management, durable checkpoints with retention limits, catalogue pagination, and request-linked traces containing bounded tool results.
