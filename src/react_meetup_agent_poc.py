from __future__ import annotations
import time
from functools import wraps
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from google_api_based_gis_tools import (
    getCommon,
    getIsochrone,
    getSuggestions,
    parse_travel_time,
    queryAI,
    sortPlaces,
)


def timing_measurement(label: str, name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                print(f"{label} {name} took {elapsed_ms:.2f} ms")

        return wrapper

    return decorator


# -----------------------------------------------------------------------------
# Local GPU handshake swap:
# Change ONLY the BASE_URL assignment below to:
#   http://localhost:8001/v1
# once your teammate's local AMD GPU vLLM server is ready.
# Keep the API_KEY value pointed at that server's expected auth token.
# -----------------------------------------------------------------------------
BASE_URL = "http://127.0.0.1:8001/v1"  # Note: ensure VSCode port forwarding is active if running locally
API_KEY = "EMPTY"  # vLLM local endpoints usually accept any string here unless auth is explicitly configured
MODEL_NAME = "gatherpoint-local"

def _normalize_coordinate_entry(entry: dict, index: int) -> dict:
    latitude = entry.get("latitude") if entry.get("latitude") is not None else entry.get("lat")
    longitude = entry.get("longitude") if entry.get("longitude") is not None else entry.get("lng")

    if latitude is None or longitude is None:
        raise ValueError(
            f"Coordinate #{index + 1} must include latitude and longitude values."
        )

    return {"lat": float(latitude), "lng": float(longitude)}


@tool
def find_optimal_meeting_spots(
    user_coordinates: list[dict[str, float]],
    transport_modes: list[str],
    place_type: str,
    travel_time: str = "30m",
) -> str:
    """Find the best meetup recommendations from raw coordinate dictionaries.

    Input format:
    - user_coordinates must be a list of dictionaries that already contain coordinates.
    - Each coordinate dictionary must include latitude and longitude values, or lat and lng aliases.
    - transport_modes must align with user_coordinates one-for-one.
    - Example: {"user_coordinates": [{"latitude": 43.6426, "longitude": -79.3871}, {"latitude": 43.669, "longitude": -79.3832}], "transport_modes": ["WALK", "DRIVE"], "place_type": "cafe", "travel_time": "30m"}
    - Do not pass addresses, city names, or place names here; geocoding must happen upstream on the local machine.
    """
    try:
        if len(user_coordinates) != len(transport_modes):
            return (
                "Observation: Error executing GIS tool: user_coordinates and transport_modes must "
                "have the same length."
            )

        normalized_points = [
            _normalize_coordinate_entry(point, index)
            for index, point in enumerate(user_coordinates)
        ]
        travel_time_seconds = parse_travel_time(travel_time)

        isochrones = []
        for point, mode in zip(normalized_points, transport_modes):
            iso = getIsochrone(point, mode, travel_time_seconds)
            isochrones.append(iso)

        common = getCommon(isochrones)
        if not common.get("intersection_valid"):
            return (
                "Observation: The isochrones of all participants have no overlap; consider "
                "increasing the travel_time."
            )

        suggestions = getSuggestions(isochrones, place_type)
        sorted_places = sortPlaces(suggestions)[:5]
        final_text = queryAI(sorted_places)
        if "Based on the above information" in final_text:
            final_text = final_text.split("Based on the above information")[0].rstrip()

        return final_text

    except Exception as e:
        return f"Observation: Error executing GIS tool: {str(e)}. Please check your inputs and try again."


PROMPT = PromptTemplate.from_template(
    """You are a helpful meetup planning assistant. 
    You have access to the following tools:
    {tools}
    
    Rules:
    - Extract raw coordinate dictionaries from the upstream request.
    - Extract transport modes. Default to DRIVE if missing.
    - Extract the venue type (place_type).
    - Default travel_time is "30m".
    - When calling find_optimal_meeting_spots, Action Input must be a valid JSON object containing user_coordinates, transport_modes, place_type, and travel_time. Example: {{"user_coordinates": [{{"latitude": 43.6426, "longitude": -79.3871}}, {{"latitude": 43.6690, "longitude": -79.3832}}], "transport_modes": ["WALK", "DRIVE"], "place_type": "cafe", "travel_time": "30m"}}
    - Never send addresses, city names, or place names to the tool.
    
    EDGE CASE HANDLING:
    - If you receive an Observation stating "no overlap" or that participants are too far apart, you MUST call the tool again and increase the "travel_time" (e.g., to "45m" or "1h").
    - MAXIMUM RETRY RULE: If you have increased the travel_time up to "2h" and still receive a "no overlap" error, DO NOT use the tool again. Immediately output "Final Answer: " explaining to the user that they are physically too far apart for a practical meetup.

    CRITICAL STOPPING RULE: 
    - Once you receive the list of recommended places from find_optimal_meeting_spots, DO NOT use any more tools. 
    - You must immediately output "Final Answer: " followed by a friendly summary of the top 2 recommended spots for the user.

    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Question: {input}
    Thought:{agent_scratchpad}"""
)


def build_agent() -> AgentExecutor:
    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=0,
    )

    tools = [find_optimal_meeting_spots]
    agent = create_react_agent(llm=llm, tools=tools, prompt=PROMPT)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,
    )


def main() -> None:
    executor = build_agent()

    prompt = (
        'Alice is at {"latitude": 49.2606, "longitude": -123.2460} and will DRIVE. '
        'Bob is at {"latitude": 43.6690, "longitude": -79.3832} and will DRIVE. '
        'Find a cafe for them to meet at.'
    )

    @timing_measurement("[LLM LATENCY]", "agent_reasoning")
    def run_agent():
        return executor.invoke({"input": prompt})

    result = run_agent()

    print("\n=== Final Result ===")
    print(result["output"])


if __name__ == "__main__":
    main()
