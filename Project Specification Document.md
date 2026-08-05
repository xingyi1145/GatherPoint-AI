# GatherPoint AI — Project Specification Document

**Track 2: Development & Local Deployment of Private AI Agents**
**AMD AI DevMaster Hackathon (AMD Radeon GPU + ROCm)**

> A privacy-preserving, fully local AI Agent that plans the *geographically fairest and semantically most suitable* group meetup spot by orchestrating a ReAct reasoning loop, an isochrone-intersection GIS toolchain, on-device spatial kernels, and a local RAG memory store.

---

## 1. Overview

Group meetup planning is notoriously hard: participants have conflicting dietary, budget, mobility, and scheduling constraints, combined with strict geographic reality. Generic map tools only compute a *geometric* midpoint — they ignore whether a spot is actually reachable by everyone within a reasonable travel time, and they cannot reason about *semantic* requirements such as "vegan food near a basketball court".

**GatherPoint AI** solves this by combining a locally deployed LLM with a deterministic spatial planning engine:

1. A **ReAct agent** (LangChain) running on a **local vLLM endpoint** on an AMD Radeon GPU decomposes a natural-language request into structured tool calls.
2. A **GIS toolchain** fetches per-person travel isochrones, computes their *intersection* (the region every participant can reach), and filters Points of Interest (POIs) inside it.
3. A **custom HIP C++ kernel** accelerates the bottleneck — massively parallel spatial reduction over map coordinates — directly on the Radeon GPU, bypassing Python's GIL.
4. A **FastAPI microservice (`src/rag_server.py`)** on the Radeon node serves both RAG retrieval and GPU GIS center calculation endpoints, backed by local ChromaDB + ROCm-native embeddings, including preference write + retrieve memory loops.

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
| **Persistent preference handling** *(implemented, see §7)* | "I'm vegan", "I avoid spicy food", "I only have evenings free" are recorded into the local vector store and re-injected into future planning turns across sessions. |

The key differentiator vs. generic assistants: GatherPoint does not stop at chat — it **executes a real spatial computation pipeline on the GPU** and returns verifiable, constraint-satisfying venue options, not hallucinated suggestions.

---

## 3. System Architecture

<img src="Agent%20architecture%20diagram.png" alt="GatherPoint AI system architecture" width="100%">

**Data flow of one planning turn:**

1. User types a natural-language request in the Streamlit UI.
2. `agent_service.run_agent_turn()` loads group profiles, queries ChromaDB for relevant memories, and builds a bounded prompt.
3. The ReAct agent calls the local vLLM endpoint on the Radeon GPU; the LLM emits a structured `Action Input` for the GIS tool.
4. The GIS toolchain geocodes addresses and fetches per-person isochrones locally, then calls the GPU microservice (`POST /calculate_intersection`) to compute the optimal center.
5. Local Google Places queries use the returned center coordinate, then candidate POIs are filtered/sorted and returned to the agent as `Observation`.
6. The agent produces a `Final Answer` with top venue choices and per-person travel-time rationale.

---

## 4. Core Capabilities

### 4.1 ReAct Agent with Multi-Step Task Planning
`src/react_meetup_agent_poc.py` builds a LangChain ReAct agent (`create_react_agent`) backed by a **local vLLM endpoint** (`http://127.0.0.1:8001/v1`, model `gatherpoint-local`). The agent:

- **Decomposes** free-form requests into a JSON tool call with `user_addresses`, `transport_modes`, `place_type`, `travel_time`.
- **Plans iteratively**: the system prompt encodes edge-case behavior — if an `Observation` reports no isochrone overlap, the agent must retry with a larger `travel_time` (up to `2h`), then stop and explain that participants are physically too far apart.
- **Stops deterministically**: a hard stopping rule prevents infinite tool loops; `max_iterations=6` and `handle_parsing_errors=True` add belt-and-braces protection against LLM hallucinations.

### 4.2 Tool Invocation & Workflow Orchestration
`src/google_api_based_gis_tools.py` implements a complete, deterministic pipeline exposed to the agent as a single `@tool` (`find_optimal_meeting_spots`):

- `preProcess` — address → coordinates (`places_autocomplete` + `place` details).
- `getIsochrone` — travel-time isochrone polygons per person, per transport mode (`DRIVE / TRANSIT / WALK / BICYCLE / TWO_WHEELER`).
- `getCommon` — sends flat latitude/longitude arrays to the cloud endpoint (`POST /calculate_intersection`) to offload heavy center math to ROCm.
- `getSuggestions` — local POI search around the GPU-computed center, then per-person travel times via Distance Matrix.
- `sortPlaces` — ranking by (avg travel time ↑, rating ↓, review count ↓).
- `queryAI` — compresses candidates into a tight, LLM-friendly context block to protect the local context window.

Every stage is wrapped in a `baseline_measurement` timing decorator, and errors are returned to the LLM as `Observation` strings instead of crashing the process.

### 4.3 On-Device Spatial Computation (Custom HIP Kernel)
`src/intersect.hip` implements the GPU center-reduction kernel used by the FastAPI microservice. The current implementation uses block-level shared-memory reduction plus one global atomic update per block, then computes the final averaged center (`sum / count`) with a zero-count safety guard. `src/rag_server.py` bridges Python-to-C++ via `ctypes`, calling `calculate_center` from `libintersect.so` and returning JSON coordinates. This is the project's flagship ROCm optimization: the Radeon GPU is used as a true compute engine, not a black box.

### 4.4 Local RAG & Multi-Turn Memory
- **Local vector store**: `src/init_db.py` provisions a persistent ChromaDB (`./gatherpoint_db`, collection `friend_profiles`) with a **hybrid schema** — semantic `documents` for the LLM plus structured `metadatas` (home coordinates, default transit mode) for the GIS tools.
- **ROCm-native embeddings**: `all-MiniLM-L6-v2` is loaded through `sentence-transformers` with the PyTorch backend mapped to ROCm, so retrieval executes on the GPU without CPU fallback.
- **Semantic retrieval**: `src/agent_service.py` now calls the remote RAG endpoint (`POST /retrieve_profiles`) and injects returned documents into the planning prompt.
- **Preference recording (write path)**: conversational preference facts are persisted into local memory storage and upserted into the vector context so future turns can retrieve and reuse them.
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
1. **GPU node (Radeon Cloud)**: pull `vllm/vllm-openai-rocm:latest`, serve the quantized LLM on `localhost:8001/v1`, and run `src/rag_server.py` on `localhost:8000`. The HIP kernel is compiled on-node with `hipcc -O3 -fPIC -shared intersect.hip -o libintersect.so` (`src/build.sh`). ChromaDB persists to disk on the node.
2. **Local node**: Streamlit frontend + agent orchestrator + GIS toolchain (outbound internet for map APIs).
3. **Bridge**: SSH port-forward `8000` and `8001` connect the local orchestrator to cloud FastAPI + vLLM, bypassing cloud firewall restrictions (`ssh -N -L 8000:127.0.0.1:8000 -L 8001:127.0.0.1:8001 ...`).

### 5.3 VRAM Partitioning Strategy
vLLM's `--gpu-memory-utilization` is tuned to **0.80–0.85** instead of the default 0.9, explicitly reserving ~15–20% of Radeon VRAM for the concurrent `sentence-transformers` embedding model (RAG) and auxiliary tensors. This prevents OOM crashes during simultaneous LLM inference + retrieval, verified via `rocm-smi`.

---

## 6. AMD Radeon GPU / ROCm Optimization Description

| # | Optimization | Where | Impact |
|---|---|---|---|
| 1 | **Custom HIP kernel for spatial center reduction** | `intersect.hip`, `rag_server.py` | Shared-memory block reduction + reduced global atomics to accelerate GPU center computation |
| 2 | **ROCm-native embedding inference** | `test_embedder.py`, `init_db.py` | RAG retrieval executes on GPU (PyTorch ROCm backend), no CPU fallback, no cloud embedding latency |
| 3 | **Local LLM serving via vLLM on ROCm** | `react_meetup_agent_poc.py` | Core inference 100% on-device on the Radeon GPU; OpenAI-compatible endpoint reused by LangChain with zero extra wiring |
| 4 | **VRAM memory partitioning** | vLLM startup flags | Safe co-residency of LLM KV-cache + embedding model; prevents OOM |
| 5 | **Context-window hygiene** | `memory_service.py`, `queryAI()` | Bounded prompt + compressed POI payloads → fewer tokens per turn → lower latency per inference |
| 6 | **Deterministic stopping rules** | ReAct prompt | `max_iterations=6`, retry-with-larger-travel_time, hard "too far apart" bailout → bounded worst-case latency, no runaway loops |
| 7 | **Performance instrumentation** | `baseline_measurement`, `timing_measurement` | Every API call and GIS stage is timed, enabling quantified demo evidence of GPU speedup |

These optimizations directly address the 40-point "Adaptation & Optimization for AMD Radeon GPU / ROCm" criterion: **core inference on the Radeon GPU (20 pts)** and **targeted inference-speed optimization (20 pts)**.

---

## 7. RAG-Based User Preference Recording — Implemented

> **Status:** both halves of RAG are operational: preference facts are recorded (write path) and later retrieved (read path) for prompt personalization across turns.

### 7.1 What Exists Today
- Persistent ChromaDB (`friend_profiles`) with hybrid schema and ROCm-generated embeddings (`src/init_db.py`).
- Semantic retrieval served through FastAPI (`src/rag_server.py` → `POST /retrieve_profiles`) and consumed by `agent_service.retrieve_relevant_memories()`.
- Profile normalization, conversation summarization, and durable memory storage hooks (`src/memory_service.py`).
- GIS-side caching contract: `get_cached_metadata(user_id)` lets the toolchain skip redundant map API calls when a profile's isochrone is cached.

### 7.2 Implemented: Preference Recording Pipeline (RAG Write Path)
1. **Preference extraction**: each turn can produce structured preference facts (dietary restrictions, transit mode, budget, availability, venue rejections) from conversation context.
2. **Preference persistence**: memory records are stored with metadata (`group_id`, source context, timestamp-like fields) for durable local reuse.
3. **Vector upsert**: persisted facts are added to vector-searchable context so semantic retrieval can surface them in future turns.
4. **Prompt re-injection**: retrieved memories are injected into the bounded planning prompt on every turn, enabling across-session personalization.
5. **Conflict strategy**: new facts can supersede stale records to avoid contradictory context.
6. **Evaluation workflow**: retrieval quality and answer-quality deltas are measurable by toggling memory-enabled vs memory-disabled runs.

This satisfies the **"Local knowledge retrieval (RAG)" + "Local multi-turn memory"** requirement pair with a full *write + retrieve* loop.

---

## 8. Compliance with Track 2 Evaluation Criteria

### Minimum functional requirements (≥2 of 5 required)
| Requirement | GatherPoint status |
|---|---|
| Local knowledge retrieval (RAG) | ✅ Operational (ChromaDB + ROCm embeddings + preference write/retrieve loop in §7) |
| Tool invocation | ✅ `find_optimal_meeting_spots` ReAct tool + full GIS pipeline |
| Multi-step task planning | ✅ ReAct loop with retry/edge-case planning |
| Local multi-turn memory | ✅ Session window + summary compression + persistent preference memory |
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

- **Now → deadline**: harden preference-memory quality checks, expand benchmark coverage, and record the 3–5 min demo video showing GPU execution performance.
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
│   └── Phase 1..4/               # Agent loop, RAG, deployment, GPU microservice plans
├── src/
│   ├── app.py                    # Streamlit UI
│   ├── agent_service.py          # Agent turn orchestration, prompt building
│   ├── memory_service.py         # Conversation state, profiles, memory API
│   ├── react_meetup_agent_poc.py # ReAct agent + tool binding
│   ├── google_api_based_gis_tools.py   # GIS pipeline + microservice bridge
│   ├── rag_server.py             # FastAPI RAG + GIS service (cloud)
│   ├── intersect.hip             # Custom HIP kernel (GPU center math)
│   ├── hip_wrapper.py            # Standalone ROCm wrapper benchmark helper
│   ├── build.sh                  # hipcc compile script
│   ├── init_db.py                # ChromaDB + embedding init
│   └── tests/                    # Tests and experiments
└── gatherpoint_db/               # Persistent ChromaDB store
```

## Appendix B — Quick Start

```bash
# GPU node (Radeon Cloud)
./build.sh                                   # compile libintersect.so with hipcc
python src/init_db.py                        # init ChromaDB + embeddings on GPU
python src/rag_server.py                     # start FastAPI microservice on :8000
# start vLLM serving the quantized model on :8001

# Local node
pip install -r requirements.txt
ssh -N -L 8000:127.0.0.1:8000 -L 8001:127.0.0.1:8001 root@<instance> -p <port>   # tunnel to GPU
streamlit run src/app.py                     # launch UI → http://localhost:8501
```
