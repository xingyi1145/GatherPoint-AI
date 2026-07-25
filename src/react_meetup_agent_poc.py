from __future__ import annotations

import ast
import json
from typing import Any

from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


# -----------------------------------------------------------------------------
# Local GPU handshake swap:
# Change ONLY the BASE_URL assignment below to:
#   http://localhost:8001/v1
# once your teammate's local AMD GPU vLLM server is ready.
# Keep the API_KEY value pointed at that server's expected auth token.
# -----------------------------------------------------------------------------
BASE_URL = "https://your-openai-compatible-endpoint.example/v1"
API_KEY = "YOUR_API_KEY_HERE"
MODEL_NAME = "qwen-plus"


@tool
def calculate_midpoint(locations: list[str]) -> str:
    """Calculate a meetup midpoint for a list of locations.

    Use this tool when the user provides two or more place names, neighborhoods,
    or rough location descriptions and wants a single midpoint coordinate to
    anchor the planning search.

    Input format:
    - Pass a JSON array or Python-style list of location strings.
    - Example: ["Alice is downtown", "Bob is in the suburbs"]

    Output:
    - Always returns the mock coordinate string "43.65°N, 79.38°W".
    - This is a proof-of-concept stub, not a real geocoder.
    """

    normalized_locations: list[str] = []
    for location in locations:
        normalized_locations.append(str(location).strip())

    _ = normalized_locations
    return "43.65°N, 79.38°W"


@tool
def search_nearby_places(center_coordinate: str, query: str) -> str:
    """Search for nearby places around a center coordinate.

    Use this tool after a midpoint has been found and the agent needs candidate
    meetup spots near that coordinate.

    Input format:
    - center_coordinate: a coordinate string such as "43.65°N, 79.38°W"
    - query: the type of place to search for, such as "vegan restaurant"

    Output:
    - Always returns a hardcoded JSON string with two mock restaurants.
    - This is a proof-of-concept stub, not a real place search.
    """

    _ = center_coordinate
    _ = query

    results = {
        "center_coordinate": center_coordinate,
        "query": query,
        "results": [
            {
                "name": "Green Leaf Vegan",
                "address": "123 Queen St W, Toronto, ON",
                "rating": 4.8,
            },
            {
                "name": "Plant Power Bites",
                "address": "456 King St W, Toronto, ON",
                "rating": 4.6,
            },
        ],
    }
    return json.dumps(results, ensure_ascii=False)


PROMPT = PromptTemplate.from_template(
    """You are a practical meetup-planning assistant.

You have access to the following tools:

{tools}

Use this exact format for every step:

Question: the user's request
Thought: think about the best next step
Action: one of [{tool_names}]
Action Input: the tool input
Observation: the tool result
... repeat Thought / Action / Action Input / Observation as needed ...
Thought: I now know the final answer
Final Answer: the answer to the user

Important rules:
- If the user mentions multiple people or locations, first use calculate_midpoint.
- After you have a midpoint, use search_nearby_places for a relevant venue type.
- When a tool takes multiple fields, write Action Input as JSON.
- Keep the reasoning concise and focused on the next decision.

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

    tools = [calculate_midpoint, search_nearby_places]
    agent = create_react_agent(llm=llm, tools=tools, prompt=PROMPT)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=4,
    )


def main() -> None:
    executor = build_agent()

    prompt = "Alice is downtown, Bob is in the suburbs. Find a vegan restaurant midway between them."
    result = executor.invoke({"input": prompt})

    print("\n=== Final Result ===")
    print(result["output"])


if __name__ == "__main__":
    main()
