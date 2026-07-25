# Phase 1: Agent Loop Baseline & Hardware Handshake

**Goal:** Prove the core reasoning logic of the agent works using a placeholder model before the actual local deployment and complex geographic tools are finished.

## Step 1: Initialize the Framework
* **Action:** Set up the Python virtual environment and install LangChain along with its OpenAI integrations (since we will be using an OpenAI-compatible endpoint).
* **Why:** LangChain will handle the ReAct (Reason + Act) orchestration loop, tool binding, and prompt execution.

## Step 2: Connect to a Placeholder LLM Endpoint
* **Action:** Since the local AMD ROCm environment is still being set up, point the LangChain model to the Radeon Cloud Token Factory's Public Free Model API (e.g., Qwen or DeepSeek).
* **Why:** This provides a temporary, free "brain" to test the logic loops without needing a GPU instance immediately.

## Step 3: Create "Dummy" Tools
* **Action:** Write mock Python functions for the GIS tools that immediately return hardcoded strings or JSON data. 
    * `calculate_midpoint(locations: list)` -> Returns "43.65°N, 79.38°W".
    * `search_nearby_places(center: str, query: str)` -> Returns a minimal JSON string with two fake restaurants.
* **Why:** You need to simulate these tools immediately to verify if the LLM knows *how* to call them and parse their outputs.

## Step 4: Construct the ReAct Prompt and Execution Loop
* **Action:** Inject the dummy tools into LangChain and engineer the core ReAct system prompt (forcing the model to evaluate the goal, choose a tool, observe the result, and act). 
* **Action:** Run the test prompt: *"Alice is downtown, Bob is in the suburbs. Find a vegan restaurant midway between them."* 
* **Success Metric:** The terminal should show the LLM reasoning to find the midpoint, calling `calculate_midpoint`, taking the result to call `search_nearby_places`, and outputting a final answer.

## Step 5: The Handshake (Cross-Learning Sync)
* **Action:** Once the local `vllm/vllm-openai-rocm:latest` server is running on the AMD GPU, swap the placeholder Base URL for the local endpoint (e.g., `http://localhost:8001/v1`) and run the test again[.
* **Success Metric:** If the script runs successfully, the team has proven core inference runs entirely locally on an AMD Radeon GPU.