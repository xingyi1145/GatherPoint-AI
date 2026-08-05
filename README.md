# GatherPoint AI

GatherPoint AI is a local-first meetup planning assistant with an AMD ROCm cloud backend for heavy compute.
It combines:

1. Profile and conversation-aware planning.
2. RAG retrieval with ChromaDB + sentence-transformers.
3. GPU-accelerated GIS center calculation via a HIP shared library.

## Architecture

The system is split across two runtimes.

1. Local Orchestrator
   - UI: src/app.py (Streamlit)
   - Agent loop: src/agent_service.py
   - GIS data gathering: src/google_api_based_gis_tools.py (Google APIs)

2. AMD Cloud Compute Node
   - LLM serving: vLLM endpoint on port 8001
   - RAG + GIS microservice: src/rag_server.py on port 8000
   - HIP kernel shared object: src/libintersect.so compiled from src/intersect.hip
<img src="Agent%20architecture%20diagram.png" alt="GatherPoint AI system architecture" width="100%">

## Key Endpoints

The FastAPI service in src/rag_server.py exposes:

1. POST /retrieve_profiles
   - Input: {"query": "..."}
   - Output: top matching profile/memory documents from ChromaDB.

2. POST /calculate_intersection
   - Input: {"latitudes": [...], "longitudes": [...]} (flat float lists)
   - Output: {"latitude": float, "longitude": float}
   - Uses ctypes to call calculate_center inside libintersect.so.

## Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Current required packages are listed in requirements.txt.

## Environment Variables

You can override defaults with these variables.

1. GATHERPOINT_CHROMA_PATH
   - Default: repo/gatherpoint_db
2. GATHERPOINT_CHROMA_COLLECTION
   - Default: friend_profiles
3. GATHERPOINT_LIBINTERSECT_PATH
   - Default: src/libintersect.so
4. GATHERPOINT_EMBEDDER_PATH
   - Optional local path to all-MiniLM-L6-v2 model folder
5. GATHERPOINT_INTERSECTION_API_URL
   - Default in GIS tools: http://127.0.0.1:8000/calculate_intersection
6. GOOGLE_API_KEY
   - Required by src/google_api_based_gis_tools.py

## Setup

### 1) Cloud Node Setup (AMD ROCm)

Run on the remote GPU node inside src:

```bash
chmod +x build.sh
./build.sh
```

build.sh now fails fast if libintersect.so is not created or if calculate_center symbol is missing.

If you run in an outbound-restricted environment, ensure a local embedding model folder exists on disk:

1. repo/all-MiniLM-L6-v2, or
2. src/all-MiniLM-L6-v2, or
3. parent-of-repo/all-MiniLM-L6-v2, or
4. set GATHERPOINT_EMBEDDER_PATH explicitly.

Then start the FastAPI server:

```bash
python src/rag_server.py
```

Optional one-time Chroma seed:

```bash
python src/init_db.py
```

### 2) Local Machine Setup

Install dependencies and run Streamlit:

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

## Port Forwarding

To access cloud services from local tooling:

```bash
ssh -N -L 8000:127.0.0.1:8000 -L 8001:127.0.0.1:8001 root@<CLOUD_IP> -p <SSH_PORT>
```

Keep that tunnel terminal open.

## Quick Health Checks

From the host where rag_server.py runs:

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/retrieve_profiles -H "Content-Type: application/json" -d '{"query":"Alice and Bob"}'
curl -s -X POST http://127.0.0.1:8000/calculate_intersection -H "Content-Type: application/json" -d '{"latitudes":[43.64,43.67],"longitudes":[-79.38,-79.39]}'
```

## Repository Layout

1. src/app.py: Streamlit frontend
2. src/agent_service.py: prompt assembly and agent turn execution
3. src/google_api_based_gis_tools.py: Google API collection and microservice bridge
4. src/rag_server.py: FastAPI service for RAG retrieval and HIP GIS endpoint
5. src/intersect.hip: HIP kernel and C ABI wrapper
6. src/build.sh: HIP shared library build/validation script
7. gatherpoint_db/: ChromaDB persistent storage
