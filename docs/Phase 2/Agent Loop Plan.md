# Phase 2: Agent Loop Integration & Robust Error Handling

**Primary Goal:** Merge the real geographic Python scripts into the ReAct loop and bulletproof the agent against LLM hallucinations and bad data inputs.

## Step 1: Merge Real GIS Tools
* **Action:** Coordinate with Person 2 (GIS Tools) to pull their completed `calculate_midpoint` and `search_nearby_places` scripts into your working directory.
* **Action:** Swap out the dummy JSON payloads in your `@tool` functions for real API calls and geographic math.
* **Why:** The agent needs to transition from proving the reasoning loop to actually solving the Multi-Agent Meeting problem.

## Step 2: Implement Self-Correcting Error Loops
* **Action:** Expand the `try/except` string parsing we built in Phase 1.
* **Action:** If a tool fails (e.g., the map API returns a 404, or the LLM passes an invalid coordinate format), catch the error in Python.
* **Action:** Return the explicit error string back to the LLM as an `Observation` instead of letting the script crash.
* **Why:** LLMs hallucinate. A robust agent must be forced to read its own errors and try a different tool or parameter format automatically.

## Step 3: Edge Case Prompt Engineering
* **Action:** Update the LangChain System Prompt to explicitly instruct the agent on how to handle geographic anomalies.
* **Action:** Feed the agent stress-test queries. Example: "What if the calculated midpoint lands in the middle of Lake Ontario?"
* **Why:** The LLM needs explicit instructions to verify coordinates and expand its search radius if the initial location is unviable.

## Step 4: Hardware Latency Profiling
* **Action:** Run the fully integrated loop (Real Tools + LLM) locally through the SSH tunnel to the AMD Radeon GPU.
* **Action:** Monitor the time it takes for the model to process the much larger JSON payloads returned by real Map APIs.
* **Why:** You need to ensure the `gatherpoint-local` model does not hit token limits or timeout when dealing with real-world geographic data.