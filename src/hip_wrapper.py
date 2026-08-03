import ctypes
import numpy as np
import time
import os

# Load the compiled AMD ROCm shared library
lib_path = os.path.join(os.path.dirname(__file__), 'libintersect.so')
intersect_lib = ctypes.CDLL(lib_path)

# Define the C-function argument types
intersect_lib.run_intersection.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int,  # Grid
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int,  # Users
    ctypes.c_float, ctypes.POINTER(ctypes.c_int)                                   # Radius & Output
]

def find_fast_overlap_gpu(grid_lats, grid_lons, user_lats, user_lons, max_radius_km=15.0):
    """
    Passes numpy map arrays directly to the AMD GPU to find valid meeting zones.
    """
    num_grids = len(grid_lats)
    num_users = len(user_lats)
    
    # Ensure inputs are contiguous C-style float32 arrays
    g_lats_c = np.ascontiguousarray(grid_lats, dtype=np.float32)
    g_lons_c = np.ascontiguousarray(grid_lons, dtype=np.float32)
    u_lats_c = np.ascontiguousarray(user_lats, dtype=np.float32)
    u_lons_c = np.ascontiguousarray(user_lons, dtype=np.float32)
    
    # Pre-allocate the output array (0s)
    result_flags = np.zeros(num_grids, dtype=np.int32)
    
    start_time = time.time()
    
    # Call the HIP C++ Kernel
    intersect_lib.run_intersection(
        g_lats_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        g_lons_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        num_grids,
        u_lats_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        u_lons_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        num_users,
        ctypes.c_float(max_radius_km),
        result_flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    
    gpu_time = (time.time() - start_time) * 1000
    print(f"[ROCm] GPU Processed {num_grids} locations in {gpu_time:.2f} ms")
    
    # Extract the valid coordinates where flag == 1
    valid_indices = np.where(result_flags == 1)[0]
    valid_lats = g_lats_c[valid_indices]
    valid_lons = g_lons_c[valid_indices]
    
    return list(zip(valid_lats, valid_lons))

# --- QUICK TEST ---
if __name__ == "__main__":
    # Generate 1 Million fake map grid coordinates to stress test the GPU
    print("Generating 1,000,000 map grid coordinates...")
    test_grid_lats = np.random.uniform(43.0, 44.0, 1000000)
    test_grid_lons = np.random.uniform(-80.0, -79.0, 1000000)
    
    # 3 Friends living in Toronto
    test_user_lats = np.array([43.6452, 43.6704, 43.6532])
    test_user_lons = np.array([-79.3806, -79.3868, -79.3832])
    
    print("Sending to AMD Radeon VRAM...")
    valid_spots = find_fast_overlap_gpu(test_grid_lats, test_grid_lons, test_user_lats, test_user_lons, 5.0)
    print(f"Found {len(valid_spots)} valid meeting zones where all users overlap!")