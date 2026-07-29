from __future__ import annotations

from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from google_api_based_gis_tools import full_pipeline


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

@tool
def find_optimal_meeting_spots(
    user_addresses: list[str],
    transport_modes: list[str],
    place_type: str,
) -> str:
    """Find the best meetup recommendations for multiple participants.

    Use this tool when the user has provided the full set of participant
    locations and wants a practical meeting suggestion. Pass one list item per
    participant in `user_addresses`, and keep `transport_modes` aligned by
    index with the same participant order.

    Arguments:
    - user_addresses: a list of raw address strings or place descriptions.
      Example: ["Union Station, Toronto", "Yonge and Bloor, Toronto"]
    - transport_modes: a list of travel modes for each address.
      Supported values are DRIVE, WALK, BICYCLE, and TRANSIT.
      Example: ["WALK", "BICYCLE"]. If the user does not specify a mode,
      the caller should supply "DRIVE" for that participant.
    - place_type: the venue category to recommend, such as "cafe",
      "restaurant", "co-working space", or "park".

    Returns:
    - On success, returns the `ai_text` field from the GIS pipeline.
    - If the pipeline reports an error, returns that exact error string.
    - If an exception is raised, returns a readable error message for the LLM.
    """

    try:
        result = full_pipeline(
            user_addresses=user_addresses,
            transport_modes=transport_modes,
            place_type=place_type,
        )
        if isinstance(result, dict) and "error" in result:
            return str(result["error"])
        if isinstance(result, dict) and "ai_text" in result:
            return str(result["ai_text"])
        return "Error executing GIS tool: Pipeline returned an unexpected response. Please check your inputs and try again."
    except Exception as e:
        return f"Error executing GIS tool: {str(e)}. Please check your inputs and try again."


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
- Extract all user addresses and their respective transport modes from the user's request.
- Accept transport modes such as DRIVE, WALK, BICYCLE, or TRANSIT.
- If a transport mode is not specified for a person, default that person's mode to DRIVE.
- Extract the venue type the user wants, such as cafe or restaurant.
- When calling find_optimal_meeting_spots, Action Input must be a valid JSON object with exactly these keys: user_addresses, transport_modes, and place_type.
- Use JSON arrays for user_addresses and transport_modes.
- Do not write key=value pairs, YAML, bullets, or prose in Action Input.
- Example Action Input: {{"user_addresses": ["Union Station, Toronto", "Yonge and Bloor, Toronto"], "transport_modes": ["WALK", "BICYCLE"], "place_type": "cafe"}}
- Use find_optimal_meeting_spots to get the recommendation.
- Return the tool result directly in the Final Answer.

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
        max_iterations=4,
    )


def main() -> None:
    executor = build_agent()

    prompt = "Alice is at Union Station, Toronto and will WALK. Bob is at Yonge and Bloor, Toronto and will BICYCLE. Find a cafe for them to meet at."
    result = executor.invoke({"input": prompt})

    print("\n=== Final Result ===")
    print(result["output"])


if __name__ == "__main__":
    main()
