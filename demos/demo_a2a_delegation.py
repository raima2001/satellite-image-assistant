"""Bonus 3.A demo - show the geo agent's Agent Card and one delegated task.

Needs the geo agent running first: python src/geo_agent/service.py

Fetches the Agent Card from its well-known path, then sends it one A2A
message/send task through the same client code the main agent uses
(agent.a2a_client), and prints the request and response.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent.a2a_client import GEO_AGENT_URL, delegate_place_lookup, fetch_agent_card


def main() -> None:
    print(f"Agent Card from {GEO_AGENT_URL}/.well-known/agent-card.json:")
    print(json.dumps(fetch_agent_card(), indent=2))

    place_name = "Rotterdam, Netherlands"
    print(f"\nDelegating one task to the geo agent: resolve '{place_name}'")
    result = delegate_place_lookup(place_name)
    print(f"Geo agent reply: {json.dumps(result)}")


if __name__ == "__main__":
    main()
