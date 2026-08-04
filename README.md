# GatherPoint AI 

GatherPoint AI is a highly performant, hardware-optimized AI agent. By analyzing user locations, preferred transit modes, and personal constraints (such as dietary restrictions), the agent autonomously calculates the most geographically fair and semantically appropriate meeting spots for groups.

Built specifically for the AMD ROCm ecosystem, GatherPoint AI moves beyond simple API fetching by executing massive parallel spatial routing entirely on-device.

## Architecture Overview

Our secure split architecture divides the labor perfectly between the local machine and the AMD GPU cloud:

1.  **Frontend (`src/app.py`)**: A custom Streamlit UI managing conversation state, user inputs, and live planning panels.
2.  **Agent Orchestrator (`src/agent_service.py`)**: Handles multi-turn memory, profile retrieval, and prompt generation before querying the local LLM.
3.  **AMD GPU Cloud (Backend)**:
    * **vLLM Instance**: Serves the core LLM over port 8001.
    * **GIS Engine**: Executes the compiled `intersect.hip` C++ kernel for parallelized spatial math.

## Key Technical Achievements (AMD ROCm Optimizations)

We didn't just use the AMD GPU as a black box; we actively optimized the framework and architecture:

* **Bare-Metal Spatial Math (Custom HIP Kernel)**: We bypassed Python's Global Interpreter Lock (GIL) by writing a custom HIP C++ kernel (`intersect.hip`). Instead of a slow Python loop, we assigned one GPU thread to every map grid square, dropping map intersection calculation times from tens of seconds down to ~110ms for 1 million locations.
* **ROCm-Native RAG & Edge Computing**: We built a Hybrid Data Schema using a local ChromaDB instance. By downloading Hugging Face models offline and mapping the PyTorch backend to ROCm, `sentence-transformers` executes matrix multiplications natively on the Radeon GPU without CPU fallback or cloud API latency.
* **VRAM Memory Partitioning**: We actively tuned vLLM's `--gpu-memory-utilization` parameter to safely partition the Radeon GPU's VRAM between the massive LLM KV-cache and our auxiliary PyTorch embedding models, preventing Out-Of-Memory (OOM) crashes.
* **Self-Correcting Agent Loop**: The ReAct agent is bulletproofed against hallucinations with try/except error loops and explicit edge-case bailouts (e.g., gracefully identifying when users are physically too far apart to meet).

## Installation & Setup

### 1. Compile the HIP Kernel (Cloud Node)
If you are setting this up for the first time on the AMD instance, you must compile the C++ kernel into a Shared Object (`.so`) library so Python's `ctypes` can bridge the memory pointers.
Ensure you run the build script using `hipcc` on the cloud node:
```bash
chmod +x build.sh
./build.sh
```

### 2. Local Dependencies
Ensure your local environment has all required packages installed for the frontend UI.
```bash
pip install -r requirements.txt
```

## Sample Run Guide

Because of our secure split architecture, you need to securely tunnel into the cloud instance before starting the frontend application. This bridges your local machine directly to the GPU's LLM and backend services, effectively bypassing cloud firewall restrictions.

### Step 1: Connect Local Laptop to AMD GPU (Port Forwarding)
Open a terminal and run the following SSH command to forward your local port to the remote vLLM server:
```bash
ssh -N -L 8001:127.0.0.1:8001 root@36.150.116.206 -p 31132
```
*Keep this terminal open in the background!*

### Step 2: Start the Frontend UI
Open a new terminal window in your local project directory. Make sure all dependencies are downloaded, then launch the Streamlit chat interface:
```bash
streamlit run src/app.py
```
Navigate to `http://localhost:8501` in your browser. You can now chat with the agent, view the live planning panel, and let GatherPoint AI calculate your optimal meetup spots!

---


## Submission Requirements

### Track 2: Development & Local Deployment of Private AI Agents

1. **Project Specification Document**
   - Application scenarios
   - Agent architecture diagram
   - Introduction to core capabilities
   - Model introduction & local deployment plan
   - Optimization description for inference speed on AMD Radeon GPU
2. **Project Source Code**
   - Complete source code repository
   - README file including environment configuration, startup guide and dependency list
3. **Demo Video**
   - Recommended duration: 3–5 minutes
   - Demonstrate the actual operation process
   - The actual execution performance on an AMD Radeon GPU, from command line/GUI to the final result (fluidity and functional completeness)
4. **Supplementary Materials (Choose One)**
   - PPT / Poster
