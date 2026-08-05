#include <hip/hip_runtime.h>
#include <math.h>

#define PI 3.14159265358979323846

// 1. GPU DEVICE FUNCTION: Calculates Haversine distance between two coordinates
__device__ float haversine_distance(float lat1, float lon1, float lat2, float lon2) {
    float R = 6371.0; // Earth radius in kilometers
    float dLat = (lat2 - lat1) * PI / 180.0;
    float dLon = (lon2 - lon1) * PI / 180.0;
    
    lat1 = lat1 * PI / 180.0;
    lat2 = lat2 * PI / 180.0;
    
    float a = sin(dLat/2) * sin(dLat/2) +
              sin(dLon/2) * sin(dLon/2) * cos(lat1) * cos(lat2);
    float c = 2 * atan2(sqrt(a), sqrt(1-a));
    return R * c;
}

// 2. GPU GLOBAL KERNEL: Assigns one thread to every map grid point
__global__ void intersect_kernel(
    const float* grid_lats, const float* grid_lons, int num_grids,
    const float* user_lats, const float* user_lons, int num_users,
    float max_radius_km, int* result_flags) 
{
    // Get the unique ID for this GPU thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // --- ADD THIS LINE ---
    // BOUNDS CHECK: Kill extra threads before they read unallocated memory
    if (idx >= num_grids) {
        return;
    }
    // ---------------------

    float g_lat = grid_lats[idx];
    float g_lon = grid_lons[idx];
    
    if (idx < num_grids) {
        bool is_valid = true;
        
        // Check if this grid point is within the radius for ALL users
        for(int i = 0; i < num_users; i++) {
            float dist = haversine_distance(grid_lats[idx], grid_lons[idx], user_lats[i], user_lons[i]);
            if (dist > max_radius_km) {
                is_valid = false;
                break; // Fails condition, stop checking
            }
        }
        
        // Output 1 if it's a valid meeting spot, 0 if not
        result_flags[idx] = is_valid ? 1 : 0;
    }
}

// 3. GPU GLOBAL KERNEL: Atomically accumulates lat/lon sums for a center reduction.
// Bounds-checked per thread; each thread updates only the two shared scalar
// accumulators (never per-thread output slots), so thread count can safely
// exceed the number of scalar outputs (which is always exactly 1 each).
__global__ void reduce_center_kernel(
    const float* lats, const float* lons, int size,
    float* sum_lat, float* sum_lon)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size) {
        return;
    }

    atomicAdd(sum_lat, lats[idx]);
    atomicAdd(sum_lon, lons[idx]);
}

// 4. C-INTERFACE: Allows Python's ctypes to call this library
extern "C" {
    // Returns 0 on success, or a non-zero hipError_t code on failure so the
    // Python/FastAPI caller can detect and report GPU faults instead of
    // silently trusting a corrupted result (or crashing).
    int calculate_center(
        const float* h_lats, const float* h_lons, int size,
        float* out_lat, float* out_lon)
    {
        if (size <= 0) {
            return -1;
        }

        float *d_lats = nullptr, *d_lons = nullptr;
        float *d_sum_lat = nullptr, *d_sum_lon = nullptr;
        hipError_t err;

        err = hipMalloc(&d_lats, size * sizeof(float));
        if (err != hipSuccess) return (int)err;

        err = hipMalloc(&d_lons, size * sizeof(float));
        if (err != hipSuccess) { hipFree(d_lats); return (int)err; }

        // Single-scalar accumulators: every thread must only ever atomicAdd
        // into these, never index into them by thread id.
        err = hipMalloc(&d_sum_lat, sizeof(float));
        if (err != hipSuccess) { hipFree(d_lats); hipFree(d_lons); return (int)err; }

        err = hipMalloc(&d_sum_lon, sizeof(float));
        if (err != hipSuccess) {
            hipFree(d_lats); hipFree(d_lons); hipFree(d_sum_lat);
            return (int)err;
        }

        hipMemset(d_sum_lat, 0, sizeof(float));
        hipMemset(d_sum_lon, 0, sizeof(float));

        hipMemcpy(d_lats, h_lats, size * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_lons, h_lons, size * sizeof(float), hipMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

        hipLaunchKernelGGL(reduce_center_kernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0,
                           d_lats, d_lons, size, d_sum_lat, d_sum_lon);

        // Force host to wait for kernel completion and surface any
        // launch/execution fault here instead of on a later, unrelated call.
        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            hipFree(d_lats); hipFree(d_lons); hipFree(d_sum_lat); hipFree(d_sum_lon);
            return (int)err;
        }

        float sum_lat = 0.0f, sum_lon = 0.0f;
        hipMemcpy(&sum_lat, d_sum_lat, sizeof(float), hipMemcpyDeviceToHost);
        hipMemcpy(&sum_lon, d_sum_lon, sizeof(float), hipMemcpyDeviceToHost);

        hipFree(d_lats);
        hipFree(d_lons);
        hipFree(d_sum_lat);
        hipFree(d_sum_lon);

        *out_lat = sum_lat / size;
        *out_lon = sum_lon / size;

        return 0;
    }

    void run_intersection(
        const float* h_grid_lats, const float* h_grid_lons, int num_grids,
        const float* h_user_lats, const float* h_user_lons, int num_users,
        float max_radius_km, int* h_result_flags) 
    {
        // Pointers for GPU Memory
        float *d_grid_lats, *d_grid_lons, *d_user_lats, *d_user_lons;
        int *d_result_flags;

        // Allocate VRAM on the Radeon GPU
        hipMalloc(&d_grid_lats, num_grids * sizeof(float));
        hipMalloc(&d_grid_lons, num_grids * sizeof(float));
        hipMalloc(&d_user_lats, num_users * sizeof(float));
        hipMalloc(&d_user_lons, num_users * sizeof(float));
        hipMalloc(&d_result_flags, num_grids * sizeof(int));

        // Copy data from RAM (Python) to VRAM (GPU)
        hipMemcpy(d_grid_lats, h_grid_lats, num_grids * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_grid_lons, h_grid_lons, num_grids * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_user_lats, h_user_lats, num_users * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_user_lons, h_user_lons, num_users * sizeof(float), hipMemcpyHostToDevice);

        // Define GPU Thread Blocks (256 threads per block)
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_grids + threadsPerBlock - 1) / threadsPerBlock;

        // Launch the Kernel!
        hipLaunchKernelGGL(intersect_kernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, 
                           d_grid_lats, d_grid_lons, num_grids, 
                           d_user_lats, d_user_lons, num_users, 
                           max_radius_km, d_result_flags);

        // Copy the results back to Python RAM
        hipMemcpy(h_result_flags, d_result_flags, num_grids * sizeof(int), hipMemcpyDeviceToHost);

        // Free VRAM
        hipFree(d_grid_lats); hipFree(d_grid_lons);
        hipFree(d_user_lats); hipFree(d_user_lons);
        hipFree(d_result_flags);
    }
}