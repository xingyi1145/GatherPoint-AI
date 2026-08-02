# Phase 4 Plan: Custom HIP C++ Kernel for Spatial Grid Intersection

**Primary Goal:** Replace the slow Python-based spatial grid intersection loop in `gistool.py` and `google_api_based_gis_tools.py` with a high-performance, parallelized AMD HIP C++ kernel compiled directly for the Radeon GPU.

## Step 1: Isolate the Spatial Bottleneck
* **Action:** Review `gistool.py` to identify where polygon overlap/grid-sampling checks are performed.
* **Why:** We need to flatten the input polygons (isochrones) into raw coordinate arrays of floats so they can be passed directly into C memory pointers without Python serialization overhead.

## Step 2: Write the HIP Kernel (`intersect.hip`)
* **Action:** Create a custom C++ file using AMD's HIP runtime API.
* **Action:** Assign individual GPU threads to every single coordinate point or grid square simultaneously.
* **Why:** Instead of sequential Python iteration checking points one by one, thousands of hardware cores on the Radeon GPU will compute spatial inclusion checks in parallel.

## Step 3: Python-to-C Integration via `ctypes`
* **Action:** Use Python's built-in `ctypes` library to load the compiled shared object (`.so`) library.
* **Why:** NumPy arrays share a contiguous block of memory with C-arrays under the hood. We can pass memory pointers directly into our HIP library to prevent costly data-copying delays.

## Step 4: The Compilation Script (`build.sh`)
* **Action:** Write a compilation script utilizing AMD's `hipcc` compiler.
* **Action:** Compile `intersect.hip` into a shared plugin library (`libintersect.so`) on the Radeon Cloud node.

## Step 5: Drop-in Replacement & Timing Benchmark
* **Action:** Integrate `libintersect.so` into `google_api_based_gis_tools.py` behind a conditional switch that triggers if cached profiles are loaded from ChromaDB.
* **Action:** Add execution timing benchmarks to prove the speedup of bare-metal GPU compute vs. standard Python CPU loops for the final Hackathon submission documentation.