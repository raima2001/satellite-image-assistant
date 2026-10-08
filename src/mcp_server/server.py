"""Section 2.3 - the MCP server.

Two tools, both calling the Earth Search STAC API for the sentinel-2-c1-l2a
collection: search for scenes, and get one scene's details and assets by id.

Run directly as a subprocess over stdio (the agent spawns it; see
src/agent/mcp_tools.py). Not meant to be imported as a library.
"""

import json

import httpx
from mcp.server.fastmcp import FastMCP

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-c1-l2a"

mcp = FastMCP("sentinel-catalogue")


@mcp.tool()
def search_scenes(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 20.0,
    limit: int = 5,
) -> str:
    """Search Sentinel-2 scenes by bounding box, date range, and max cloud cover.

    Dates are "YYYY-MM-DD". Returns JSON with the search criteria and a
    shortlist of scenes (id, datetime, cloud_cover), sorted by cloud cover.

    Raises an error (2.4) if the search itself fails - for example a
    malformed date - naming the search that was attempted and why it failed.
    """
    bbox = [min_lon, min_lat, max_lon, max_lat]
    attempted = f"search for {COLLECTION} scenes in bbox {bbox} between {start_date} and {end_date}"
    try:
        response = httpx.post(
            f"{EARTH_SEARCH_URL}/search",
            json={
                "collections": [COLLECTION],
                "bbox": bbox,
                "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
                "limit": 100,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        features = response.json()["features"]
    except httpx.HTTPStatusError as error:
        raise ValueError(
            f"Could not {attempted}: the catalogue rejected the request "
            f"({error.response.status_code}). Check the dates and bbox."
        ) from error
    except httpx.RequestError as error:
        raise ValueError(f"Could not {attempted}: {error}") from error

    scenes = [
        {
            "id": feature["id"],
            "datetime": feature["properties"].get("datetime"),
            "cloud_cover": feature["properties"].get("eo:cloud_cover"),
        }
        for feature in features
        if feature["properties"].get("eo:cloud_cover", 100) <= max_cloud_cover
    ]
    scenes.sort(key=lambda scene: scene["cloud_cover"] or 0.0)

    return json.dumps(
        {
            "criteria": {
                "bbox": bbox,
                "start_date": start_date,
                "end_date": end_date,
                "max_cloud_cover": max_cloud_cover,
            },
            "scenes": scenes[:limit],
        }
    )


@mcp.tool()
def get_scene_details(scene_id: str) -> str:
    """Get one scene's metadata and available image assets by its id.

    Raises an error (2.4) if the scene id does not exist, or if the lookup
    itself fails, naming the scene id that was attempted and why it failed.
    """
    attempted = f"look up scene '{scene_id}' in the {COLLECTION} collection"
    try:
        response = httpx.get(
            f"{EARTH_SEARCH_URL}/collections/{COLLECTION}/items/{scene_id}",
            timeout=30.0,
        )
        if response.status_code == 404:
            raise ValueError(f"Could not {attempted}: no scene with that id exists.")
        response.raise_for_status()
        feature = response.json()
    except httpx.HTTPStatusError as error:
        raise ValueError(
            f"Could not {attempted}: the catalogue returned "
            f"{error.response.status_code}."
        ) from error
    except httpx.RequestError as error:
        raise ValueError(f"Could not {attempted}: {error}") from error

    return json.dumps(
        {
            "id": feature["id"],
            "datetime": feature["properties"].get("datetime"),
            "cloud_cover": feature["properties"].get("eo:cloud_cover"),
            "assets": sorted(feature["assets"].keys()),
        }
    )


if __name__ == "__main__":
    mcp.run()
