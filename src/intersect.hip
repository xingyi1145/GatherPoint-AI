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
    float* sum_lat, float* sum_lon, int* sum_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ float shared_lat[256];
    __shared__ float shared_lon[256];
    __shared__ int shared_count[256];

    if (idx < size) {
        shared_lat[tid] = lats[idx];
        shared_lon[tid] = lons[idx];
        shared_count[tid] = 1;
    } else {
        shared_lat[tid] = 0.0f;
        shared_lon[tid] = 0.0f;
        shared_count[tid] = 0;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_lat[tid] += shared_lat[tid + stride];
            shared_lon[tid] += shared_lon[tid + stride];
            shared_count[tid] += shared_count[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum_lat, shared_lat[0]);
        atomicAdd(sum_lon, shared_lon[0]);
        atomicAdd(sum_count, shared_count[0]);
    }
}

extern "C" int calculate_center(const float* lats, const float* lons, int size, float* out_lat, float* out_lon) {
    // 1. Allocate Device (GPU) Pointers for Inputs AND Outputs
    float *d_lats, *d_lons, *d_sum_lat, *d_sum_lon;
    int *d_sum_count;
    
    hipMalloc(&d_lats, size * sizeof(float));
    hipMalloc(&d_lons, size * sizeof(float));
    hipMalloc(&d_sum_lat, sizeof(float));
    hipMalloc(&d_sum_lon, sizeof(float));
    hipMalloc(&d_sum_count, sizeof(int));

    // 2. Initialize the Output accumulators to 0.0 on the Device
    float zero = 0.0f;
    int zero_count = 0;
    hipMemcpy(d_sum_lat, &zero, sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_sum_lon, &zero, sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_sum_count, &zero_count, sizeof(int), hipMemcpyHostToDevice);

    // 3. Copy Input arrays from Python (Host) to GPU (Device)
    hipMemcpy(d_lats, lats, size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_lons, lons, size * sizeof(float), hipMemcpyHostToDevice);

    // 4. Launch the Kernel (using Copilot's kernel name)
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(reduce_center_kernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, 
                       d_lats, d_lons, size, d_sum_lat, d_sum_lon, d_sum_count);

    // 5. Synchronize to catch errors
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        // Free memory before returning error to prevent leaks
        hipFree(d_lats); hipFree(d_lons); hipFree(d_sum_lat); hipFree(d_sum_lon); hipFree(d_sum_count);
        return (int)err; 
    }

    // 6. Copy the reduced results from the GPU back to Python's ctypes pointers
    float sum_lat = 0.0f;
    float sum_lon = 0.0f;
    int count = 0;

    hipMemcpy(&sum_lat, d_sum_lat, sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(&sum_lon, d_sum_lon, sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(&count, d_sum_count, sizeof(int), hipMemcpyDeviceToHost);

    if (count == 0) {
        *out_lat = 0.0f;
        *out_lon = 0.0f;
    } else {
        *out_lat = sum_lat / count;
        *out_lon = sum_lon / count;
    }

    // 7. Cleanup! (Crucial to prevent VRAM leaks)
    hipFree(d_lats);
    hipFree(d_lons);
    hipFree(d_sum_lat);
    hipFree(d_sum_lon);
    hipFree(d_sum_count);

    return 0; // Success
}