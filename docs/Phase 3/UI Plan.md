# Phase 3 Action Plan: Agent Loop - Wiring Placeholders

**Role:** Person 3 (Agent Loop Developer)
**Goal:** Connect the Streamlit UI and context management system to the real ReAct agent and local ChromaDB backend.

---

## Step 1: Connect the Local ReAct Agent
Currently, the Streamlit UI throws a placeholder error when trying to generate a response. We need to wire it to the real agent built in Phase 2.

* **File to Edit:** `src/agent_service.py`
* **Action Items:**
    1. Import the `build_agent` function from your existing script:
       `from react_meetup_agent_poc import build_agent`
    2. Locate the `_invoke_local_planning_agent(prompt: str)` function.
    3. Delete the `raise RuntimeError(...)` placeholder.
    4. Initialize the agent executor and pass the bounded `prompt` to it.
    5. Return the agent's final text output.

## Step 2: Connect the ChromaDB (RAG) Retrieval
Currently, the system retrieves no memories. We need to wire it to Person 1's newly deployed local database.

* **File to Edit:** `src/agent_service.py`
* **Action Items:**
    1. Coordinate with Person 1 to import their database retrieval function (e.g., `retrieve_user_context` from their deployment script).
    2. Locate the `retrieve_relevant_memories(group_id, user_message)` function.
    3. Replace the `return []` placeholder with a call to the RAG retrieval function, passing the `group_id` and `user_message`.
    4. Ensure the returned data is formatted as a list of strings so the UI can display it properly in the "Live Plan" panel.

## Step 3: End-to-End Testing
Verify that the entire application stack runs locally on the AMD Radeon GPU.

* **Action Items:**
    1. Start the Streamlit application: `streamlit run src/app.py`
    2. Enter a natural language prompt in the UI (e.g., "Alice is at Union, Bob is at Bloor, find a cafe").
    3. Watch the UI closely to ensure:
        * The "Live Plan" panel correctly populates the Active Constraints and Memory Retrieval sections.
        * The agent successfully reaches out to the `vLLM` server and Google APIs.
        * The final recommendation is seamlessly rendered into the chat interface.