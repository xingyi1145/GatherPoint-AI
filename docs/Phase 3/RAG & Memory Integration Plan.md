# Integration Plan: Restoring RAG and Memory

## 1. Reflection on Previous Roadblocks
Before bypassing the system, we encountered two major roadblocks:
* **The Hardcoded Guardrail:** The `run_agent_turn` function had a hardcoded `if not profiles:` block that manually returned a failure string before the LLM was even called. We have successfully deleted this.
* **Instruction Override (Attention Hijacking):** When RAG or Memory returned empty results, the context formatting functions injected fallback instructions (e.g., asking for participant locations). Local open-source models like Llama 3 are highly susceptible to instruction override. The model read the fallback sentence, assumed its only job was to ask that question, and completely ignored the user's actual prompt at the bottom.

## 2. Core Strategy: "Silent Context"
To successfully integrate RAG and Memory, we must adopt a "Silent Context" strategy. If a database query returns no profiles or no memories, we must feed the LLM nothing regarding those categories. We will only inject context blocks if real data exists.

## 3. Step-by-Step Implementation

### Step 1: Update Context Formatters
* Update `_format_memory_context(memories)` to adhere to the Silent Context strategy, returning empty strings if no data is present.

### Step 2: Rebuild `_build_planning_prompt`
* Remove the `**kwargs` direct-bypass hack and restore the actual prompt assembly.
* **Critical Rule:** The `user_message` must be placed at the very bottom of the prompt to mitigate recency bias, ensuring the LLM pays the most attention to the user's immediate request.
* Update this function in `src/agent_service.py`.

### Step 3: Validate the Entry Point
* Verify that `run_agent_turn` correctly loads profiles, retrieves memories, passes them all into the newly restored `_build_planning_prompt`, and then feeds that complete string into `_invoke_local_planning_agent`.
* Ensure the hardcoded `if not profiles:` guardrail remains completely deleted.

### Step 4: Test and Clear Session State
1. Save `src/agent_service.py`.
2. Go to the Streamlit UI.
3. **CRITICAL:** Click the "↻ New plan" button or hard refresh the browser to wipe Streamlit's `st.session_state`. If you skip this, the UI will feed your old bypass state back into the new code.
4. Type your test prompt (e.g., "Alice is at Union Station, Toronto and will DRIVE...").