# Project Scope Statement: GatherPoint AI

## Executive Summary
**Problem/Opportunity:** Planning group activities involves conflicting preferences (dietary, budget, activity) combined with complex geographic constraints. Standard map tools only calculate geographic midpoints, ignoring human factors like transit modes or semantic requirements (e.g., "vegan food near a basketball court"). 
**Strategic Alignment:** This project is being executed for Track 2 (Development & Local Deployment of Private AI Agents) of the AMD AI DevMaster Hackathon. It addresses the "Life management AI Agent" scenario by delivering a privacy-preserving, locally hosted intelligent assistant that autonomously orchestrates group meetup planning.

---

## Project Objectives & Success Criteria (SMART)
*   **Specific:** Build a fully locally deployed AI Agent system on the AMD Radeon GPU and ROCm platform that utilizes toolchain invocation to recommend optimal group meetup locations.
*   **Measurable:** The agent must successfully implement at least two core capabilities from the hackathon rubric: Tool Invocation and Local Multi-turn Memory. 
*   **Achievable:** The scope is restricted to three specific Python tools (Geocoding, Midpoint Calculation, POI Search) to ensure completion within the hackathon timeframe.
*   **Relevant:** The project directly aligns with the Track 2 requirement to break the limitations of general AI tools by creating an agent with scenario-based service capabilities.
*   **Time-bound:** The project, including all code and supplementary materials, must be submitted before the final deadline of August 6, 2026.

---

## Detailed In-Scope Deliverables & Activities
*   **Core Software Architecture:**
    *   A local Large Language Model (LLM) deployed via vLLM or llama.cpp adapted for the AMD Radeon GPU.
    *   An agent orchestration loop built using a neutral framework (e.g., LangChain or smolagents) to handle multi-step task planning.
    *   A local memory management system (SQLite/JSON) to store user preferences across interactions.
*   **Custom Toolchain (Python):**
    *   A geocoding and travel-time calculation tool.
    *   A semantic Point of Interest (POI) search tool using open-source map data (e.g., OpenStreetMap).
*   **User Interface:**
    *   A locally hosted web UI (e.g., Streamlit or Chainlit) for users to input locations and view outputs.
*   **Hackathon Submission Materials:**
    *   A Project Specification Document detailing the architecture, capabilities, and GPU optimization.
    *   A complete source code repository with a detailed README (environment configuration, startup guide, dependencies).
    *   A 3–5 minute Demo Video demonstrating actual execution performance on an AMD Radeon GPU.
    *   A supplementary PPT or Poster highlighting the creative scenario and practical value.

---

## Detailed Out-of-Scope Items
*   Integration with closed-source, remote LLM APIs (e.g., OpenAI API, Anthropic API) for core inference processes, as this is strictly prohibited by hackathon rules.
*   Deployment on general cloud providers; the solution must run exclusively on the Radeon cloud + ROCm software stack.
*   Direct reservation or booking actions (e.g., calling a restaurant API to book a table). The agent will only provide recommendations.
*   A mobile application or complex frontend redesign. The interface will be a functional, standard local web GUI.

---

## Key Stakeholders & Roles
*   **Deployment Engineer (Team Member 1):** Responsible for the AMD ROCm environment setup, local model hosting (vLLM/llama.cpp), and targeted optimization for inference speed.
*   **GIS Tool Developer (Team Member 2):** Responsible for writing the deterministic Python functions for routing, midpoint calculation, and POI filtering.
*   **Agent Logic & UI Developer (Team Member 3):** Responsible for the ReAct orchestration loop, prompt engineering, and building the Streamlit/Chainlit frontend.
*   **Hackathon Evaluators (Judges):** Key stakeholders assessing the project based on functional completeness (60 points) and AMD GPU/ROCm adaptation (40 points).

---

## Constraints, Assumptions, & Dependencies
*   **Hardware Constraint:** Core inference must be executed locally on an AMD Radeon GPU; remote APIs are not allowed for core functions.
*   **Timeline Constraint:** The project must be fully complete by August 6, 2026.
*   **Dependency:** The project relies on the availability and stability of the provided Radeon cloud instances.
*   **Assumption:** Free tier mapping/POI APIs (like OpenStreetMap's Overpass API) will provide sufficient rate limits and accuracy for the agent's tool calls.

---

## High-Level Milestones & Timeline
*   **July 25, 2026:** Local ROCm environment configured and base LLM successfully running on Radeon cloud GPU.
*   **July 28, 2026:** Python geographic and POI search tools built and unit-tested.
*   **August 1, 2026:** Agent orchestration loop complete; LLM successfully invoking tools and handling data constraints.
*   **August 3, 2026:** Local memory implementation complete and UI integrated. 
*   **August 5, 2026:** 3–5 minute demo video recorded showing actual execution on the AMD GPU.
*   **August 6, 2026:** Final submission of codebase, README, video, and PPT/Poster before the deadline.

---

## Risks & Mitigation Strategies
1.  **Risk: GPU Out-of-Memory (OOM) Errors.**
    *   *Mitigation:* Utilize model quantization (4-bit or 8-bit) and vLLM optimization techniques to ensure the LLM fits within the provided Radeon GPU VRAM.
2.  **Risk: Token Exhaustion from API Payloads.**
    *   *Mitigation:* Build a strict parser within the Python POI tool to strip unnecessary metadata from map API responses before returning the context to the LLM.
3.  **Risk: Agent Infinite Action Loops.**
    *   *Mitigation:* Implement a hard limit on maximum iterations within the agent framework and provide fallback prompt instructions if a search yields zero results.

---

## Acceptance Criteria
*   The agent successfully processes at least three distinct user locations and semantic preferences to output a logical meetup venue.
*   The core inference processes execute entirely locally on the AMD Radeon GPU without any remote LLM API calls.
*   The system successfully demonstrates tool invocation and local multi-turn memory.
*   The final submission package (Source Code, README, Project Spec, Demo Video, and PPT/Poster) is verified and uploaded prior to the August 6, 2026 deadline.