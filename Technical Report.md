# GatherPoint AI — Technical Report

**Track 2: Development & Local Deployment of Private AI Agents**
**AMD AI DevMaster Hackathon (AMD Radeon GPU + ROCm)**

> A privacy-preserving, fully local AI Agent that plans the *geographically fairest and semantically most suitable* group meetup spot by orchestrating a ReAct reasoning loop, an isochrone-intersection GIS toolchain, on-device spatial kernels, and a local RAG memory store.

---

## 1. Overview

Group meetup planning is notoriously hard: participants have conflicting dietary, budget, mobility, and scheduling constraints, combined with strict geographic reality. Generic map tools only compute a *geometric* midpoint — they ignore whether a spot is actually reachable by everyone within a reasonable travel time, and they cannot reason about *semantic* requirements such as "vegan food near a basketball court".

**GatherPoint AI** solves this by combining a locally deployed LLM with a deterministic spatial planning engine:

1. A **ReAct agent** (LangChain) running on a **local vLLM endpoint** on an AMD Radeon GPU decomposes a natural-language request into structured tool calls.
2. A **GIS toolchain** fetches per-person travel isochrones, computes their *intersection* (the region every participant can reach), and filters Points of Interest (POIs) inside it.
3. A **custom HIP C++ kernel** accelerates the bottleneck — massively parallel spatial intersection over map grids — directly on the Radeon GPU, bypassing Python's GIL.
4. A **local ChromaDB + ROCm-native embedding** store provides RAG: retrieving and (planned) *recording* user preferences and group memory entirely on-device.

The whole system runs as a split deployment (local frontend + Radeon Cloud GPU backend bridged over SSH) with **core inference executed locally on the AMD Radeon GPU** — no remote LLM APIs are used for core functions.

---

## 2. Application Scenarios

GatherPoint targets the **"Life management AI Agent"** scenario defined by the competition. Concrete use cases:

| Scenario | Example interaction |
|---|---|
| **Group dining** | *"Alice is at Union Station (walking), Bob is at Yonge & Bloor (biking). Find a good vegan restaurant both can reach in 30 minutes."* |
| **Friends with conflicting mobility** | One member drives, another relies on transit — the agent reasons over per-person travel modes instead of assuming one mode for everyone. |
| **Fairness-constrained planning** | The agent prefers venues that minimize the *worst-case* individual travel time rather than optimizing for one person. |
| **Impossible-request detection** | When isochrones do not overlap (participants too far apart), the agent gracefully explains and proposes relaxations (longer travel time, different venue type). |
| **Persistent preference handling** *(planned, see §7)* | "I'm vegan", "I avoid spicy food", "I only have evenings free" are recorded into the local vector store and re-injected into future planning turns across sessions. |

The key differentiator vs. generic assistants: GatherPoint does not stop at chat — it **executes a real spatial computation pipeline on the GPU** and returns verifiable, constraint-satisfying venue options, not hallucinated suggestions.

---

## 3. System Architecture

```
┌─────────────────────────────── LOCAL (Laptop) ───────────────────────────────┐
│                                                                              │
│  Streamlit UI (src/app.py)                                                   │
│    • Chat interface, sidebar (Active group / Saved Profiles / System status) │
│    • Live agent status panel (st.status)                                     │
│    • Conversation state & session summary (src/memory_service.py)            │
│                        │                                                     │
│  Agent Orchestrator (src/agent_service.py)                                   │
│    • run_agent_turn(): load profiles → retrieve RAG memory → build prompt     │
│    • Bounded context blocks (profiles / memory / conversation)               │
│                        │                                                     │
│  ReAct Agent (src/react_meetup_agent_poc.py)                                 │
│    • LangChain create_react_agent  →  local vLLM (OpenAI-compatible)         │
│    • Tool: find_optimal_meeting_spots (JSON schema enforcement)              │
│    • Self-correcting loop: max_iterations=6, error Observations fed back     │
│                        │                                                     │
│  GIS Toolchain (src/google_api_based_gis_tools.py, src/gistool.py)           │
│    • preProcess → getIsochrone → getCommon → getSuggestions → sortPlaces     │
│    • External map APIs used ONLY for tool data (not core inference)          │
│                                                                              │
└───────────────┬──────────────────────────────────────────────────────────────┘
                │ SSH port-forward (8001)
┌───────────────▼────────────────────────── RADEON CLOUD (AMD GPU) ────────────┐
│                                                                              │
│  vLLM instance  (port 8001, model: gatherpoint-local)                        │
│    • Quantized LLM served via vllm/vllm-openai-rocm                          │
│    • --gpu-memory-utilization tuned to reserve VRAM for embeddings           │
│                                                                              │
│  ROCm Spatial Engine                                                        │
│    • src/intersect.hip  (custom HIP kernel, one GPU thread per grid point)   │
│    • src/hip_wrapper.py (ctypes bridge, zero-copy NumPy pointers)            │
│    • src/build.sh      (hipcc -O3 -fPIC -shared → libintersect.so)          │
│                                                                              │
│  Local RAG Store (ChromaDB, ./gatherpoint_db)                               │
│    • Collection "friend_profiles" — hybrid schema:                           │
│        document: semantic text for the LLM                                   │
│        metadata:  geospatial cache for the GIS tools                         │
│    • Embedding: all-MiniLM-L6-v2 executed on ROCm (no CPU fallback)          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Data flow of one planning turn:**

1. User types a natural-language request in the Streamlit UI.
2. `agent_service.run_agent_turn()` loads group profiles, queries ChromaDB for relevant memories, and builds a bounded prompt.
3. The ReAct agent calls the local vLLM endpoint on the Radeon GPU; the LLM emits a structured `Action Input` for the GIS tool.
4. The GIS toolchain geocodes addresses, fetches per-person isochrones, computes the intersection (GPU-accelerated by the HIP kernel where enabled), and filters/sorts candidate POIs.
5. The tool result returns as an `Observation`; the agent produces a `Final Answer` with the top venue choices and per-person travel times.

---

## 4. Core Capabilities

### 4.1 ReAct Agent with Multi-Step Task Planning
`src/react_meetup_agent_poc.py` builds a LangChain ReAct agent (`create_react_agent`) backed by a **local vLLM endpoint** (`http://127.0.0.1:8001/v1`, model `gatherpoint-local`). The agent:

- **Decomposes** free-form requests into a JSON tool call with `user_addresses`, `transport_modes`, `place_type`, `travel_time`.
- **Plans iteratively**: the system prompt encodes edge-case behavior — if an `Observation` reports no isochrone overlap, the agent must retry with a larger `travel_time` (up to `2h`), then stop and explain that participants are physically too far apart.
- **Stops deterministically**: a hard stopping rule prevents infinite tool loops; `max_iterations=6` and `handle_parsing_errors=True` add belt-and-braces protection against LLM hallucinations.

### 4.2 Tool Invocation & Workflow Orchestration
`src/google_api_based_gis_tools.py` / `src/gistool.py` implement a complete, deterministic pipeline exposed to the agent as a single `@tool` (`find_optimal_meeting_spots`):

- `preProcess` — address → coordinates (`places_autocomplete` + `place` details).
- `getIsochrone` — travel-time isochrone polygons per person, per transport mode (`DRIVE / TRANSIT / WALK / BICYCLE / TWO_WHEELER`).
- `getCommon` — grid-sampled **intersection** of all isochrones (the region everyone can reach), returning centroid + coverage radius.
- `getSuggestions` — POI search inside the intersection, **precise ray-casting polygon filter**, per-person travel times via Distance Matrix.
- `sortPlaces` — ranking by (avg travel time ↑, rating ↓, review count ↓).
- `queryAI` — compresses candidates into a tight, LLM-friendly context block to protect the local context window.

Every stage is wrapped in a `baseline_measurement` timing decorator, and errors are returned to the LLM as `Observation` strings instead of crashing the process.

### 4.3 On-Device Spatial Computation (Custom HIP Kernel)
`src/intersect.hip` implements a **bare-metal GPU kernel** for the spatial bottleneck: it assigns **one GPU thread per map grid point** and computes Haversine distance against every participant to flag valid meeting zones. `src/hip_wrapper.py` loads `libintersect.so` via `ctypes` and passes contiguous `float32` NumPy arrays by memory pointer (**zero-copy**). In internal benchmarking, map-intersection calculations dropped from tens of seconds (Python loop) to **~110 ms for 1,000,000 locations** (see README). This is the project's flagship ROCm optimization: the Radeon GPU is used as a true compute engine, not a black box.

### 4.4 Local RAG & Multi-Turn Memory
- **Local vector store**: `src/init_db.py` provisions a persistent ChromaDB (`./gatherpoint_db`, collection `friend_profiles`) with a **hybrid schema** — semantic `documents` for the LLM plus structured `metadatas` (home coordinates, default transit mode) for the GIS tools.
- **ROCm-native embeddings**: `all-MiniLM-L6-v2` is loaded through `sentence-transformers` with the PyTorch backend mapped to ROCm, so retrieval executes on the GPU without CPU fallback.
- **Semantic retrieval**: `src/test_retrieval.py` exposes `retrieve_user_context(query)`; `agent_service.retrieve_relevant_memories()` delegates to it, ranking the top-2 relevant snippets for injection into the planning prompt.
- **Session memory**: `src/memory_service.py` keeps a bounded recent-message window and compresses older turns into a `summary`, preventing unbounded local-inference prompt growth while preserving multi-turn continuity.

### 4.5 Local Deployment & Privacy
All core inference runs on the AMD Radeon GPU via vLLM; the GIS API calls are data-fetching *tools*, not inference. User profiles and memories persist only in the local ChromaDB/SQLite layer — no personal data leaves the deployment. The UI (Streamlit) exposes a clean chat product with a fixed sidebar showing the active group, saved profiles, memory status, and live vLLM status.

---

## 5. Model Introduction & Local Deployment Plan

### 5.1 Model Selection
| Component | Model | Rationale |
|---|---|---|
| **Reasoning LLM** | Quantized Llama-3-8B-Instruct (served as `gatherpoint-local` via vLLM) | Open-source, tool-calling capable, fits Radeon GPU VRAM when quantized |
| **Embedding model** | `all-MiniLM-L6-v2` (sentence-transformers) | Lightweight, fast, small VRAM footprint; ideal for short preference/memory search |

### 5.2 Deployment Topology (Radeon Cloud + ROCm)
1. **GPU node (Radeon Cloud)**: pull `vllm/vllm-openai-rocm:latest`, serve the quantized LLM on `localhost:8001/v1` with an OpenAI-compatible API. The HIP kernel is compiled on-node with `hipcc -O3 -fPIC -shared intersect.hip -o libintersect.so` (`src/build.sh`). ChromaDB persists to disk on the node.
2. **Local node**: Streamlit frontend + agent orchestrator + GIS toolchain (outbound internet for map APIs).
3. **Bridge**: SSH port-forward `8001` connects the local orchestrator to the GPU LLM, bypassing cloud firewall restrictions (`ssh -N -L 8001:127.0.0.1:8001 ...`).

### 5.3 VRAM Partitioning Strategy
vLLM's `--gpu-memory-utilization` is tuned to **0.80–0.85** instead of the default 0.9, explicitly reserving ~15–20% of Radeon VRAM for the concurrent `sentence-transformers` embedding model (RAG) and auxiliary tensors. This prevents OOM crashes during simultaneous LLM inference + retrieval, verified via `rocm-smi`.

---

## 6. AMD Radeon GPU / ROCm Optimization Description

| # | Optimization | Where | Impact |
|---|---|---|---|
| 1 | **Custom HIP kernel for spatial intersection** | `intersect.hip`, `hip_wrapper.py` | ~1M grid points in ~110 ms vs. tens of seconds in Python; one thread per grid point; zero-copy `ctypes` memory sharing |
| 2 | **ROCm-native embedding inference** | `test_embedder.py`, `init_db.py` | RAG retrieval executes on GPU (PyTorch ROCm backend), no CPU fallback, no cloud embedding latency |
| 3 | **Local LLM serving via vLLM on ROCm** | `react_meetup_agent_poc.py` | Core inference 100% on-device on the Radeon GPU; OpenAI-compatible endpoint reused by LangChain with zero extra wiring |
| 4 | **VRAM memory partitioning** | vLLM startup flags | Safe co-residency of LLM KV-cache + embedding model; prevents OOM |
| 5 | **Context-window hygiene** | `memory_service.py`, `queryAI()` | Bounded prompt + compressed POI payloads → fewer tokens per turn → lower latency per inference |
| 6 | **Deterministic stopping rules** | ReAct prompt | `max_iterations=6`, retry-with-larger-travel_time, hard "too far apart" bailout → bounded worst-case latency, no runaway loops |
| 7 | **Performance instrumentation** | `baseline_measurement`, `timing_measurement` | Every API call and GIS stage is timed, enabling quantified demo evidence of GPU speedup |

These optimizations directly address the 40-point "Adaptation & Optimization for AMD Radeon GPU / ROCm" criterion: **core inference on the Radeon GPU (20 pts)** and **targeted inference-speed optimization (20 pts)**.

---

## 7. RAG-Based User Preference Recording — Status & Plan

> **Status:** the *retrieval* half of RAG is operational (ChromaDB + ROCm embeddings + `retrieve_user_context`). The *recording* half — automatically persisting user preferences from conversation and re-injecting them — is **planned** and is a core part of the roadmap below.

### 7.1 What Exists Today
- Persistent ChromaDB (`friend_profiles`) with hybrid schema and ROCm-generated embeddings (`src/init_db.py`).
- Semantic retrieval helper `retrieve_user_context()` (`src/test_retrieval.py`) and its integration point `agent_service.retrieve_relevant_memories()`.
- Profile normalization and conversation summarization (`src/memory_service.py`), plus design-intent prompt scaffolding (profiles + memory + conversation blocks) in `agent_service.py`.
- GIS-side caching contract: `get_cached_metadata(user_id)` lets the toolchain skip redundant map API calls when a profile's isochrone is cached.

### 7.2 Planned: Preference Recording Pipeline (RAG Write Path)
1. **Preference extraction**: after each turn, a lightweight local pass (rule-based + LLM-assisted) extracts structured preference facts — dietary restrictions, transit mode, budget, availability, *venue rejections* — from the conversation.
2. **Upsert into ChromaDB**: facts are written as new `documents` in the group's collection with metadata (`group_id`, `source_turn`, `timestamp`, `confidence`), embedded on the ROCm GPU.
3. **Profile persistence**: replace the placeholder `memory_service.load_group_profiles` with the SQLite-backed `group_members` schema already designed in the codebase; wire `save_group_memory` / `search_group_memory` to the vector store.
4. **Prompt re-injection**: switch `_build_planning_prompt` from the current direct-route bypass to the full bounded-context builder, injecting retrieved preferences + memory + compressed conversation on every turn — giving the agent *across-session* personalization.
5. **Conflict handling**: if a new preference conflicts with an existing record (e.g., "now eats fish"), the newer fact wins and the older one is marked superseded — avoiding contradictory context.
6. **Evaluation**: measure retrieval hit-rate and answer-quality delta with vs. without preference memory on a small group scenario suite.

This completes the **"Local knowledge retrieval (RAG)" + "Local multi-turn memory"** requirement pair with a genuine *write + retrieve* loop, and strengthens the "Complete core capabilities (task decomposition, tool invocation, RAG, memory management)" scoring item.

---

## 8. Compliance with Track 2 Evaluation Criteria

### Minimum functional requirements (≥2 of 5 required)
| Requirement | GatherPoint status |
|---|---|
| Local knowledge retrieval (RAG) | ✅ Operational (ChromaDB + ROCm embeddings); preference-recording write path planned (§7) |
| Tool invocation | ✅ `find_optimal_meeting_spots` ReAct tool + full GIS pipeline |
| Multi-step task planning | ✅ ReAct loop with retry/edge-case planning |
| Local multi-turn memory | ✅ Session window + summary compression; persistent memory extension planned |
| Permission control & privacy | ✅ Fully local core inference, local-only data persistence, no remote LLM APIs |

### Scoring rubric mapping (120 pts)
| Criterion | Points | How GatherPoint addresses it |
|---|---|---|
| Clear task positioning & creative scenario | 20 | "Life management AI Agent" with fairness-aware spatial planning — a concrete, novel scenario (§2) |
| Complete core capabilities (decomposition, tools, RAG, memory) | 20 | ReAct decomposition, GIS toolchain, RAG store, session memory (§4) |
| Smooth multi-turn interaction | 20 | Streamlit chat UX, live status panel, bounded context, error recovery (§4.5) |
| Core inference on AMD Radeon GPU | 20 | vLLM on ROCm; all reasoning local (§5) |
| Targeted inference-speed optimization | 20 | HIP kernel, ROCm embeddings, VRAM partitioning, context hygiene (§6) |
| Optional bonus: Radeon Cloud Model API + quantization/distillation | 20 | Quantized Llama-3-8B-Instruct on Radeon Cloud; quantization pipeline applicable (§5) |

---

## 9. Roadmap & Future Work

- **Now → deadline**: complete the RAG preference-recording write path (§7.2), enable the full bounded prompt builder, and record the 3–5 min demo video showing GPU execution performance.
- **Post-hackathon**: multi-agent planning (one planner agent + one GIS agent), support for more map providers (OSM/Overpass) to reduce external API dependence, and a lighter LLM (distilled) for faster time-to-first-token.

---

## Appendix A — Repository Layout

```
GatherPoint-AI/
├── README.md                     # Setup, SSH tunnel, run guide
├── requirements.txt              # Python dependencies
├── .env.example                  # GOOGLE_API_KEY
├── docs/
│   ├── Project Scope.md          # Scope, SMART objectives, risks
│   ├── Timeline.md               # 4-phase roadmap
│   └── Phase 1..3/               # Agent loop, RAG, deployment plans
├── src/
│   ├── app.py                    # Streamlit UI
│   ├── agent_service.py          # Agent turn orchestration, prompt building
│   ├── memory_service.py         # Conversation state, profiles, memory API
│   ├── react_meetup_agent_poc.py # ReAct agent + tool binding
│   ├── google_api_based_gis_tools.py / gistool.py   # GIS pipeline
│   ├── intersect.hip             # Custom HIP kernel (GPU spatial math)
│   ├── hip_wrapper.py            # ctypes bridge to libintersect.so
│   ├── build.sh                  # hipcc compile script
│   ├── init_db.py                # ChromaDB + embedding init
│   ├── test_retrieval.py         # RAG retrieval helpers
│   └── test_embedder.py          # ROCm embedding verification
└── gatherpoint_db/               # Persistent ChromaDB store
```

## Appendix B — Quick Start

```bash
# GPU node (Radeon Cloud)
./build.sh                                   # compile libintersect.so with hipcc
python src/init_db.py                        # init ChromaDB + embeddings on GPU
# start vLLM serving the quantized model on :8001

# Local node
pip install -r requirements.txt
ssh -N -L 8001:127.0.0.1:8001 root@<instance> -p <port>   # tunnel to GPU
streamlit run src/app.py                     # launch UI → http://localhost:8501
```
