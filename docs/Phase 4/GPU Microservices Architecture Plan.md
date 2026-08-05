# GPU Microservices Architecture Plan (GIS & RAG)

## Overview
Due to the outbound network firewall on the AMD Radeon Cloud instance, direct API calls to Google Maps from the cloud node will fail. To solve this, GatherPoint AI will adopt a **Microservice Architecture**. 

The local machine will serve as the **Orchestrator** (handling the UI, Agent Loop, and Google Maps API calls), while the AMD GPU will serve as a dedicated **Bare-Metal Compute Engine** (running the LLM, Vector Embeddings, and C++ Spatial Math).

---

## Phase 1: RAG & Memory GPU Microservice

The local machine cannot efficiently run the PyTorch `sentence-transformers` embedding model. We must offload the ChromaDB vector database to the AMD GPU.

### 1. Cloud Execution (`rag_server.py`)
* Create a FastAPI application on the AMD Radeon Cloud instance.
* Load the `sentence-transformers` model onto the GPU.
* Connect the FastAPI app to the local ChromaDB storage directory (`gatherpoint_db`).

### 2. Expose Retrieval Endpoints
Create POST endpoints that accept a text query, generate embeddings on the GPU, and query ChromaDB:
* `POST /retrieve_profiles`
* `POST /retrieve_memory`

### 3. Local Orchestrator Update
* **File:** `src/agent_service.py` & `src/memory_service.py`
* **Action:** Delete local ChromaDB logic. Replace it with `requests.post()` calls that hit the FastAPI endpoints running on the cloud node (exposed via port-forwarding or public IP).

---

## Phase 2: GIS Math GPU Microservice

We must physically separate the Google Maps API fetching (Local) from the heavy grid math and ray-casting (Cloud GPU).

### 1. Build the HIP Kernel
* **File:** `intersect.hip` & `build.sh`
* **Action:** Write the C++ HIP kernel to assign one GPU thread to every grid square for Haversine distance and polygon overlap calculations.
* **Compile:** Run `build.sh` using `hipcc` to generate a Shared Object library (`libintersect.so`).

### 2. Break Down the Python GIS Tools
Split `gistool.py` into distinct operational phases:
* **Local (Data Gathering):** The ReAct agent queries Google Maps to get raw routing data.
* **Local (Formatting):** Python structures these coordinates into flat Numpy float arrays.
* **Cloud (Math):** The heavy intersection math is processed by the C++ kernel.

### 3. Create the GIS FastAPI Wrapper
* **Action:** Add a new endpoint to the FastAPI app on the AMD node (e.g., `POST /calculate_intersection`).
* **The Payload:** The local machine sends the flat Numpy arrays via JSON.
* **The Execution:** The FastAPI script uses Python's `ctypes` to load `libintersect.so` and passes the memory pointers directly into the C++ kernel (achieving zero-copy overhead).
* **The Response:** The GPU computes the optimal geographic center and returns the `{"latitude": X, "longitude": Y}` as a JSON response.

### 4. Final Local Step
* The local orchestrator receives the optimal coordinate from the GPU API.
* The ReAct agent makes one final (local) Google Maps API call to fetch specific venues (cafes, restaurants) around that optimized center.
* The venue data is fed back into the Llama 3 LLM context for the final output.