import time
import numpy as np
from math import radians, cos, sin, asin, sqrt

def calculate_center_cpu(grid_lats, grid_lons, user_lats, user_lons, max_radius_km=15.0):
    """
    Standard single-threaded CPU calculation for benchmark comparison against AMD GPU.
    """
    start_time = time.time()
    
    valid_lats = []
    valid_lons = []
    
    # 1. Single-threaded loop over every single grid point
    for i in range(len(grid_lats)):
        g_lat = grid_lats[i]
        g_lon = grid_lons[i]
        
        is_valid = True
        
        # 2. Check distance against every user
        for j in range(len(user_lats)):
            u_lat = user_lats[j]
            u_lon = user_lons[j]
            
            # Haversine distance math in pure Python
            r_glat, r_glon, r_ulat, r_ulon = map(radians, [g_lat, g_lon, u_lat, u_lon])
            dlat = r_ulat - r_glat
            dlon = r_ulon - r_glon
            
            a = sin(dlat/2)**2 + cos(r_glat) * cos(r_ulat) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            distance = 6371.0 * c
            
            # If this grid point is too far from ANY user, it's invalid
            if distance > max_radius_km:
                is_valid = False
                break 
        
        if is_valid:
            valid_lats.append(g_lat)
            valid_lons.append(g_lon)
            
    # 3. Calculate the geographic center (Average)
    count = len(valid_lats)
    if count > 0:
        center_lat = sum(valid_lats) / count
        center_lon = sum(valid_lons) / count
    else:
        center_lat, center_lon = 0.0, 0.0
        
    cpu_time = (time.time() - start_time) * 1000
    print(f"[LOCAL CPU] Processed {len(grid_lats)} grid points in {cpu_time:.2f} ms")
    
    return center_lat, center_lon, count

# --- QUICK BENCHMARK TEST ---
if __name__ == "__main__":
    # Generate 1,000,000 dummy grid points simulating a massive map scan
    print("Generating 1,000,000 map grid coordinates for benchmark...")
    test_grid_lats = np.random.uniform(43.0, 44.0, 1000000)
    test_grid_lons = np.random.uniform(-80.0, -79.0, 1000000)
    
    test_user_lats = np.array([43.65, 43.66, 43.64])
    test_user_lons = np.array([-79.38, -79.39, -79.37])
    
    print("Running Local CPU Math...")
    lat, lon, count = calculate_center_cpu(
        test_grid_lats, test_grid_lons, 
        test_user_lats, test_user_lons, 
        max_radius_km=15.0
    )
    print(f"Result -> Lat: {lat:.4f}, Lon: {lon:.4f}, Valid Points: {count}")