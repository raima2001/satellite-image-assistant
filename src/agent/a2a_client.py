"""Bonus 3.A: the primary agent's A2A v0.3.0 client.

Discovers the geo agent's Agent Card and sends a message/send request
to its advertised endpoint. The geo service must be running separately.
"""

import json
import os
from uuid import uuid4

import httpx
from langchain_core.tools import tool

GEO_AGENT_URL = os.environ.get(
    "GEO_AGENT_URL", "http://localhost:8001"
).rstrip("/")


def fetch_agent_card(base_url: str = GEO_AGENT_URL) -> dict:
    response = httpx.get(
        f"{base_url.rstrip('/')}/.well-known/agent-card.json",
        timeout=10.0,
    )
    response.raise_for_status()
    card = response.json()

    if not isinstance(card, dict):
        raise ValueError("The Agent Card must be a JSON object.")

    return card


def delegate_place_lookup(
    place_name: str,
    base_url: str = GEO_AGENT_URL,
) -> dict:
    """Discover the geo agent and request a place's bounding box."""
    attempted = (
        f"ask the geo agent to resolve '{place_name}' to a bounding box"
    )

    try:
        card = fetch_agent_card(base_url)

        if card.get("protocolVersion") != "0.3.0":
            raise ValueError("Expected an A2A v0.3.0 Agent Card.")

        if card.get("preferredTransport", "JSONRPC") != "JSONRPC":
            raise ValueError("This client requires JSONRPC transport.")

        endpoint = card.get("url")
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError("The Agent Card has no valid endpoint URL.")

        skills = card.get("skills", [])
        if not any(
            isinstance(skill, dict)
            and skill.get("id") == "resolve-place-to-bbox"
            for skill in skills
        ):
            raise ValueError(
                "The agent does not advertise resolve-place-to-bbox."
            )

        request_id = str(uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "message/send",
            "params": {
                "message": {
                    "kind": "message",
                    "role": "user",
                    "messageId": str(uuid4()),
                    "parts": [
                        {"kind": "text", "text": place_name}
                    ],
                }
            },
        }

        response = httpx.post(
            endpoint,
            json=payload,
            timeout=45.0,
        )
        response.raise_for_status()
        body = response.json()

        if not isinstance(body, dict):
            raise ValueError("The geo agent returned an invalid response.")

        if (
            body.get("jsonrpc") != "2.0"
            or body.get("id") != request_id
        ):
            raise ValueError("The JSON-RPC response does not match the request.")

        if "error" in body:
            error = body["error"]
            detail = (
                error.get("message", str(error))
                if isinstance(error, dict)
                else str(error)
            )
            raise ValueError(detail)

        message = body.get("result")
        if not isinstance(message, dict):
            raise ValueError("The response is missing its result.")

        # This client supports the geo service's immediate Message replies.
        if (
            message.get("kind") != "message"
            or message.get("role") != "agent"
            or not isinstance(message.get("messageId"), str)
            or not message["messageId"]
        ):
            raise ValueError("Expected an agent Message with a messageId.")

        parts = message.get("parts")
        if not isinstance(parts, list):
            raise ValueError("The reply has no valid message parts.")

        text_parts = [
            part["text"]
            for part in parts
            if isinstance(part, dict)
            and part.get("kind") == "text"
            and isinstance(part.get("text"), str)
        ]
        if not text_parts:
            raise ValueError("The geo agent returned no text result.")

        result = json.loads("\n".join(text_parts))
        if not isinstance(result, dict):
            raise ValueError("The geocoding result must be a JSON object.")

        bbox = result.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                for value in bbox
            )
        ):
            raise ValueError("The geo agent returned an invalid bounding box.")

        west, south, east, north = bbox
        if not (
            -180 <= west <= east <= 180
            and -90 <= south <= north <= 90
        ):
            raise ValueError("Bounding-box coordinates are out of range.")

        if not isinstance(result.get("place"), str):
            raise ValueError("The geocoding result is missing the place name.")

        return result

    except (httpx.HTTPError, ValueError, TypeError) as error:
        raise ValueError(f"Could not {attempted}: {error}") from error


@tool
def resolve_place_to_bbox(place_name: str) -> str:
    """Resolve a place name to a bounding box using a separate A2A agent.

    Returns JSON with the matched place and bbox in the order
    (min_lon, min_lat, max_lon, max_lat). Use this before searching
    when the analyst names a place instead of giving coordinates.
    """
    return json.dumps(delegate_place_lookup(place_name))