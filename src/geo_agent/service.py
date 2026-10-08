"""Bonus 3.A: a separate geocoding agent using A2A v0.3.0.

Publishes an Agent Card and supports JSON-RPC message/send.
Run: uv run python src/geo_agent/service.py
"""

import json
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "satellite-image-assistant-geo-agent/1.0"

AGENT_CARD = {
    "protocolVersion": "0.3.0",
    "name": "place-geocoder",
    "description": (
        "Resolves a place name to a bounding box "
        "(min_lon, min_lat, max_lon, max_lat) "
        "using OpenStreetMap Nominatim."
    ),
    "url": "http://localhost:8001/",
    "preferredTransport": "JSONRPC",
    "version": "1.0.0",
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
    },
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "skills": [
        {
            "id": "resolve-place-to-bbox",
            "name": "Resolve place to bounding box",
            "description": (
                "Given a place name, returns its bounding box as JSON."
            ),
            "tags": ["geocoding"],
            "examples": ["Rotterdam, Netherlands", "Amazon basin"],
        }
    ],
}

app = FastAPI(title="Place Geocoder A2A Agent")


@app.get("/.well-known/agent-card.json")
def agent_card() -> dict:
    return AGENT_CARD


def rpc_error(request_id, code: int, message: str) -> JSONResponse:
    return JSONResponse(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
    )


async def geocode(place_name: str) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            NOMINATIM_URL,
            params={
                "q": place_name,
                "format": "json",
                "limit": 1,
            },
            headers={"User-Agent": USER_AGENT},
        )

    response.raise_for_status()
    results = response.json()

    if not results:
        raise ValueError(f"No place found matching '{place_name}'.")

    south, north, west, east = (
        float(value) for value in results[0]["boundingbox"]
    )

    return {
        "place": results[0]["display_name"],
        "bbox": [west, south, east, north],
    }


@app.post("/")
async def rpc(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except (ValueError, UnicodeDecodeError):
        return rpc_error(None, -32700, "Parse error: invalid JSON.")

    if not isinstance(body, dict):
        return rpc_error(None, -32600, "Invalid request: expected an object.")

    request_id = body.get("id")
    if (
        body.get("jsonrpc") != "2.0"
        or not isinstance(body.get("method"), str)
        or not isinstance(request_id, (str, int))
        or isinstance(request_id, bool)
    ):
        return rpc_error(
            None,
            -32600,
            "Invalid request: expected jsonrpc '2.0', method, and an id.",
        )

    if body["method"] != "message/send":
        return rpc_error(request_id, -32601, "Method not found.")

    params = body.get("params")
    if not isinstance(params, dict):
        return rpc_error(request_id, -32602, "params must be an object.")

    message = params.get("message")
    if not isinstance(message, dict):
        return rpc_error(request_id, -32602, "message must be an object.")

    if message.get("kind") != "message" or message.get("role") != "user":
        return rpc_error(
            request_id,
            -32602,
            "Expected message kind 'message' and role 'user'.",
        )

    message_id = message.get("messageId")
    if not isinstance(message_id, str) or not message_id.strip():
        return rpc_error(
            request_id, -32602, "A non-empty messageId is required."
        )

    context_id = message.get("contextId")
    if context_id is not None and (
        not isinstance(context_id, str) or not context_id.strip()
    ):
        return rpc_error(request_id, -32602, "Invalid contextId.")

    # This service returns immediate messages and does not retain tasks.
    if message.get("taskId") is not None:
        return rpc_error(request_id, -32001, "Task not found.")

    parts = message.get("parts")
    if not isinstance(parts, list) or not parts:
        return rpc_error(
            request_id, -32602, "message.parts must be a non-empty list."
        )

    text_parts = []
    for part in parts:
        if not isinstance(part, dict):
            return rpc_error(request_id, -32602, "Invalid message part.")

        if part.get("kind") != "text":
            return rpc_error(
                request_id, -32005, "Only text parts are supported."
            )

        if not isinstance(part.get("text"), str):
            return rpc_error(request_id, -32602, "Text must be a string.")

        text_parts.append(part["text"])

    place_name = " ".join(text_parts).strip()
    if not place_name:
        return rpc_error(request_id, -32602, "A place name is required.")

    try:
        result = await geocode(place_name)
    except httpx.HTTPError as error:
        return rpc_error(
            request_id,
            -32000,
            f"Geocoding failed for '{place_name}': {error}",
        )
    except (ValueError, KeyError, TypeError, IndexError) as error:
        return rpc_error(
            request_id,
            -32000,
            f"Could not resolve '{place_name}': {error}",
        )

    return JSONResponse(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "messageId": str(uuid4()),
                "contextId": context_id or str(uuid4()),
                "role": "agent",
                "kind": "message",
                "parts": [
                    {
                        "kind": "text",
                        "text": json.dumps(result),
                    }
                ],
            },
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)