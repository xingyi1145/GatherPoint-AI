# Phase 3 Deployment Plan: ROCm-Optimized Hybrid RAG System

**Owner:** Team Member 1 (Deployment Engineer)
**Goal:** Implement the "Local Multi-turn Memory" requirement by deploying a local, persistent database that stores both semantic user preferences (for the LLM) and pre-calculated geospatial data (for the GIS tools) without exceeding the AMD Radeon GPU's VRAM limits.

## 1. VRAM Partitioning & Safety Strategy
Running a main LLM (via vLLM) and an embedding model simultaneously on the same GPU risks an Out-Of-Memory (OOM) crash.
* **Action:** Modify the vLLM server startup script.
* **Parameter Adjustment:** Change `--gpu-memory-utilization` from the default `0.9` down to `0.80` or `0.85`. This reserves 15-20% of the VRAM explicitly for the `sentence-transformers` embedding model.
* **Validation:** Monitor `rocm-smi` during a concurrent RAG + LLM inference test to ensure memory stays within limits.

## 2. ROCm-Native Embedding Model Configuration
The embedding model must execute on the AMD GPU to maintain fast retrieval speeds, rather than falling back to the CPU.
* **Model Selection:** Use `sentence-transformers/all-MiniLM-L6-v2`. It is extremely lightweight (fast inference, small VRAM footprint) but highly effective for short semantic searches like user preferences.
* **Backend Verification:** In the Python startup script, explicitly map the model to the ROCm device:
    ```python
    import torch
    from sentence_transformers import SentenceTransformer
    
    # Verify PyTorch recognizes the ROCm GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model on: {device}")
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    ```

## 3. Persistent Local ChromaDB Setup
To fulfill the hackathon's "Local Multi-turn Memory" requirement, the database must save to the local disk of the Radeon Cloud instance.
* **Action:** Initialize ChromaDB in Persistent Client mode.
* **Configuration:**
    ```python
    import chromadb
    client = chromadb.PersistentClient(path="./gatherpoint_db")
    collection = client.get_or_create_collection(name="friend_profiles")
    ```

## 4. Hybrid Data Schema Design
The database must store text for the LLM to read, and JSON/Metadata for the GIS tools to process. 
* **Schema Structure for insertion:**
    * `id`: `["alice_profile", "bob_profile"]`
    * `document`: (Semantic Text for Agent) `"Alice is vegan, hates driving, takes the subway, and lives at Union Station."`
    * `metadata`: (Geospatial Cache for Python Tools) 
        ```json
        {
          "home_lat": 43.6452,
          "home_lng": -79.3806,
          "default_mode": "TRANSIT",
          "isochrone_1h": "[[43.65,-79.39], [43.66,-79.38], ...]" 
        }
        ```

## 5. Handoff & Integration Steps
Once the database is running, coordinate with Person 2 (GIS) and Person 3 (Agent Loop):
* **To Person 2 (GIS):** Provide a Python helper function `get_cached_isochrone(user_id)`. Instruct them to update `full_pipeline()` to check this cache *before* calling the Google Routes API. If it exists, they skip the API call and immediately run `getCommon()`.
* **To Person 3 (Agent):** Provide a Python helper function `get_user_context(names_list)`. When the user types "Plan a meetup for Alice and Bob", the agent framework should fetch the text `documents` from ChromaDB and inject them into the System Prompt before making decisions.