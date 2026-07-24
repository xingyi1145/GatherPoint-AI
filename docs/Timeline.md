# Hackathon Roadmap: GatherPoint AI (Agentic Track)

**Project Deadline:** August 6, 2026
**Core Objective:** Deliver a locally deployed, privacy-preserving multi-person meetup planning agent running on AMD Radeon GPU and ROCm.
**Team Goal:** Deliver a functional MVP while ensuring all three members gain hands-on experience across the entire AI engineering stack (Hardware Deployment, Python Tooling, and LLM Orchestration).

---

## Phase 1: Environment & Baseline (July 24 – July 26)
**Goal:** Prove the hardware works and the basic Python logic is sound.

*   **Person 1 (Deployment):** Set up the AMD ROCm environment on the Radeon cloud instance. Pull the `vllm/vllm-openai-rocm:latest` Docker image and serve a quantized base model (e.g., Llama-3-8B-Instruct) on a local endpoint (`localhost:8001/v1`).
*   **Person 2 (GIS Tools):** Write the deterministic Python scripts for `calculate_midpoint()` and `search_nearby_places()` using free APIs (e.g., OpenStreetMap). *Note: Do not integrate AI yet; ensure raw data is formatted cleanly.*
*   **Person 3 (Agent Loop):** Set up the baseline agent framework (e.g., `smolagents` or `LangChain`). Connect it to a placeholder API endpoint to verify the ReAct (Reason + Act) loop functions properly in Python.
*   **Cross-Learning Sync:** Person 1 teaches the team how to query the local vLLM endpoint. Person 3 connects Person 1's vLLM endpoint into the agent framework, ensuring local inference is active.

## Phase 2: Tooling Integration & Error Handling (July 27 – July 30)
**Goal:** Connect the LLM to the tools, fulfill the Tool Invocation requirement, and fix context/hallucination errors.

*   **Person 1 (Deployment):** Monitor the vLLM server logs. Adjust maximum token lengths and optimize the KV-cache to prevent Out-Of-Memory (OOM) errors during heavy tool use.
*   **Person 2 (GIS Tools):** Write a "middleman" parser that compresses verbose POI JSON data (Name, Address, Rating) into tight, LLM-friendly strings to preserve the context window.
*   **Person 3 (Agent Loop):** Inject Person 2's tools into the agent prompt. Write robust Try/Catch error-handling loops so that if the LLM hallucinates a parameter, the Python script forces the LLM to correct itself instead of crashing.
*   **Cross-Learning Sync:** The team conducts a "Prompt Engineering" session. Everyone tests edge-case inputs (e.g., conflicting geographic constraints) to break the agent and iteratively improve the system prompt.

## Phase 3: Memory, RAG, & UI (July 31 – August 2)
**Goal:** Implement the Local Multi-turn Memory requirement and wrap the agent in a functional user interface.

*   **Person 1 (Deployment):** Assist Person 3 with setting up a local vector database (e.g., ChromaDB) for the Local Knowledge Retrieval (RAG) component, ensuring the embedding model runs smoothly on the ROCm stack.
*   **Person 2 (GIS Tools):** Build a local SQLite database or JSON file system to save and retrieve "Friend Profiles" (e.g., dietary restrictions, preferred transit modes).
*   **Person 3 (Agent Loop):** Build a Streamlit or Chainlit frontend. Integrate the Friend Profiles so the agent remembers context across multi-turn interactions.
*   **Cross-Learning Sync:** Person 2 and 3 wire the backend logic into the Streamlit UI. Person 1 profiles the overall latency (tokens-per-second) to ensure the local runtime performance is smooth for the final demo.

## Phase 4: Buffer, Optimization, & Submission (August 3 – August 5)
**Goal:** Freeze codebase, finalize documentation, and prepare deliverables for the August 6 deadline.

*   **August 3 (Code Freeze):** No new features. The team focuses entirely on fixing bugs and stabilizing the multi-turn interaction experience.
*   **August 4 (Documentation & Media):** 
    *   Person 1 drafts the "Optimization description for inference speed on AMD Radeon GPU" for the Project Specification Document.
    *   Person 2 and 3 record the 3–5 minute Demo Video, ensuring it clearly shows actual execution performance on the AMD Radeon GPU (from command line/GUI to the final result).
*   **August 5 (Final Review):** 
    *   Finalize the GitHub repository structure and the README file (must include environment configuration, startup guide, and dependency list).
    *   Review the supplementary PPT/Poster.
    *   Submit all materials via the official Pull Request process.