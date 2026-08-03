# Phase 3 Architecture Plan: ROCm FastAPI Microservice

## The Architectural Reality Check
Because Python's built-in `ctypes` library passes RAM memory pointers directly into the compiled `.hip` library, the Python script executing the C++ code must physically reside on the same machine as the GPU. You cannot pass local RAM pointers over an internet connection. Furthermore, the AMD Radeon Cloud restricts outbound API traffic. 

**The Solution:** A True Split Microservice. We will run a FastAPI server on the GPU node to handle the math, and keep the Google API integration locally.

---

## Step 1: The Cloud FastAPI Server (Math Engine)
**Location:** AMD Radeon Cloud Node
**Objective:** Create a lightweight web server to interface with your bare-metal C++ ROCm code.

* **Action:** Write a FastAPI script (e.g., `gpu_server.py`) that acts as the "Zero-Copy Python Bridge".
* **Integration:** This script will use Python's `ctypes` to load your compiled `libintersect.so` shared library. 
* **Behavior:** It will expose an HTTP `POST` endpoint. When it receives a request containing bounding box coordinates, it maps the arrays directly to C++ memory pointers, bypassing Python's slow spatial loops.
* **Execution:** The GPU stream processors crunch the math, and the FastAPI server returns the optimal meeting coordinate back as a JSON response.

---

## Step 2: The Local GIS Rewrite (`google_api_based_gis_tools.py`)
**Location:** Local Machine
**Objective:** Strip out the slow local math and redirect it to the cloud.

* **Action:** Modify your existing `google_api_based_gis_tools.py`.
* **Integration:** Remove the single-threaded Python `getCommon` loops that iterate over map coordinates.
* **Behavior:** Instead of doing the math locally, the script will format the fetched Google Maps coordinates into a flat array of floats and send an HTTP request to your cloud's FastAPI server. 
* **Result:** The local script handles the API fetching (bypassing the cloud firewall) and simply waits milliseconds for the GPU to return the intersection data.

---

## Step 3: Secure Communication (SSH Tunneling)
**Location:** Local Terminal
**Objective:** Securely connect the local script to the cloud server.

* **Action:** Since we do not want to expose the AMD GPU's FastAPI port to the public internet, we will route the local HTTP requests through an SSH tunnel.
* **Behavior:** Your local `google_api_based_gis_tools.py` will send its `POST` requests to a `localhost` port, which the SSH tunnel will securely forward directly to the FastAPI server running on the Radeon Cloud.