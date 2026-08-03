# GatherPoint AI: Deployment Architecture Plan

This document outlines the split-environment architecture to bypass secure cloud firewall restrictions while maximizing AMD GPU compute for the 2026 AMD AI DevMaster Hackathon.

## 1. Local Environment (Laptop)
**Role:** Frontend, Orchestration, and Pre/Post-Processing.
**Capabilities:** Full outbound internet access.

* **UI/CLI:** The main interface where users input their locations (e.g., "University of British Columbia").
* **Pre-Processor (Geocoding):** Hits external APIs (Google Maps/OSM) to convert text addresses into exact `(latitude, longitude)` coordinate pairs.
* **Orchestrator:** Packages the raw coordinates and sends them via HTTP over the SSH tunnel to the Radeon Cloud instance.
* **Post-Processor (Places API):** Receives the optimal geographic centroid back from the GPU, then queries Yelp/Google Places to find actual venues near that centroid.

## 2. AMD GPU Environment (Radeon Cloud Node)
**Role:** Heavy Computation, Spatial Math, and AI Inference.
**Capabilities:** Restricted network (firewall blocked), high-performance compute.

* **API Receiver:** A lightweight local server (e.g., FastAPI/Flask) listening on a forwarded port (e.g., `localhost:8000`) to receive coordinates from the local environment.
* **Agent Core:** The ReAct agent loop, strictly using tools that do *not* require outbound internet.
* **ROCm Spatial Engine:** The `intersect.hip` C++ kernel. Receives raw coordinates, computes the travel-time intersection grid natively on the AMD GPU, and returns the optimal centroid.
* **LLM / Vector DB:** Local vLLM instance and ChromaDB for on-device inference and retrieval.

## 3. The Bridge (SSH Port Forwarding)
* Communication occurs entirely over the established SSH tunnel.
* The GPU environment acts purely as a "calculator" and "reasoning engine" for the local environment.