from __future__ import annotations
import json
import time
from functools import wraps
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from google_api_based_gis_tools import full_pipeline


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

@tool
def find_optimal_meeting_spots(input_json_str: str) -> str:
    """Find the best meetup recommendations for multiple participants.

    Input format:
    - A single JSON string with exactly these keys: user_addresses, transport_modes, place_type, and travel_time.
    - Example: '{"user_addresses": ["Union Station", "Yonge and Bloor"], "transport_modes": ["WALK", "BICYCLE"], "place_type": "cafe", "travel_time": "30m"}'
    """
    try:
        clean_str = input_json_str.strip().strip("'").strip('"')
        args = json.loads(clean_str)
        
        user_addresses = args.get("user_addresses", [])
        transport_modes = args.get("transport_modes", [])
        place_type = args.get("place_type", "cafe")
        # Extract travel_time, default to 30m
        travel_time = args.get("travel_time", "30m")

        # Pass travel_time to the pipeline
        result = full_pipeline(user_addresses, transport_modes, place_type, travel_time)

        if isinstance(result, dict) and "error" in result:
             return f"Observation: API Error - {result['error']}. Please try adjusting your parameters, such as increasing the travel_time."
        
        final_text = result.get("ai_text", str(result))
        if "Based on the above information" in final_text:
            final_text = final_text.split("Based on the above information")[0]
        
        return final_text

    except json.JSONDecodeError:
        return "Observation: Error parsing input. You must provide a valid JSON string with 'user_addresses', 'transport_modes', and 'place_type'."
    except Exception as e:
        return f"Observation: Error executing GIS tool: {str(e)}. Please check your inputs and try again."


PROMPT = PromptTemplate.from_template(
    """You are a helpful meetup planning assistant. 
    You have access to the following tools:
    {tools}
    
    Rules:
    - Extract all user addresses.
    - Extract transport modes. Default to DRIVE if missing.
    - Extract the venue type (place_type).
    - Default travel_time is "30m".
    - When calling find_optimal_meeting_spots, Action Input must be a valid JSON object. Example: {{"user_addresses": ["Union Station, Toronto"], "transport_modes": ["WALK"], "place_type": "cafe", "travel_time": "30m"}}
    
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

    prompt = "Alice is at University of British Columbia, Vancouver and will DRIVE. Bob is at Yonge and Bloor, Toronto and will DRIVE. Find a cafe for them to meet at."

    @timing_measurement("[LLM LATENCY]", "agent_reasoning")
    def run_agent():
        return executor.invoke({"input": prompt})

    result = run_agent()

    print("\n=== Final Result ===")
    print(result["output"])


if __name__ == "__main__":
    main()
