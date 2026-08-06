import ctypes
import numpy as np
import time
import os

# Load the compiled AMD ROCm shared library
lib_path = os.path.join(os.path.dirname(__file__), 'libintersect.so')
intersect_lib = ctypes.CDLL(lib_path)

# Define the C-function argument and return types.
# libintersect.so exports calculate_center, not run_intersection.
intersect_lib.calculate_center.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
intersect_lib.calculate_center.restype = ctypes.c_int

def calculate_center_gpu(latitudes, longitudes):
    """
    Passes latitude and longitude arrays to the AMD GPU and returns their center.
    """
    latitudes_c = np.ascontiguousarray(latitudes, dtype=np.float32)
    longitudes_c = np.ascontiguousarray(longitudes, dtype=np.float32)

    if latitudes_c.ndim != 1 or longitudes_c.ndim != 1:
        raise ValueError("latitudes and longitudes must be one-dimensional arrays")
    if len(latitudes_c) == 0:
        raise ValueError("at least one coordinate is required")
    if len(latitudes_c) != len(longitudes_c):
        raise ValueError("latitudes and longitudes must have the same length")

    output_latitude = ctypes.c_float()
    output_longitude = ctypes.c_float()
    start_time = time.time()
    status = intersect_lib.calculate_center(
        latitudes_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        longitudes_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        len(latitudes_c),
        ctypes.byref(output_latitude),
        ctypes.byref(output_longitude),
    )

    gpu_time = (time.time() - start_time) * 1000
    if status != 0:
        raise RuntimeError(f"calculate_center failed with HIP error code {status}")

    print(f"[ROCm] GPU processed {len(latitudes_c)} locations in {gpu_time:.2f} ms")
    return output_latitude.value, output_longitude.value

# --- QUICK TEST ---
if __name__ == "__main__":
    test_lats = np.array([43.6452, 43.6704, 43.6532])
    test_lons = np.array([-79.3806, -79.3868, -79.3832])
    latitude, longitude = calculate_center_gpu(test_lats, test_lons)
    print(f"GPU center: ({latitude:.4f}, {longitude:.4f})")