import googlemaps
import requests
import json
import numpy as np
import os
import re
from math import radians, cos, sin, asin, sqrt, degrees, atan2
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "❌ Google API not found!\n"
    )
gmaps = googlemaps.Client(key=API_KEY)

# -------------------- Geometric Utility Functions (API-independent) --------------------

def _haversine(lon1, lat1, lon2, lat2):
    """Calculate the great-circle distance (km) between two points using the Haversine formula."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c


def _polygon_centroid(coords):
    """
    Calculate the centroid of a polygon given a list of lat/lng coordinates (first and last point should match).
    coords: [(lat, lng), ...] or [(lng, lat), ...] — specify format accordingly.
    Uses the Shoelace formula.
    """
    n = len(coords) - 1  # subtract the repeated closing point
    if n < 3:
        raise ValueError("Polygon requires at least 3 vertices")
    
    # Convert to radians
    points_rad = [(radians(lat), radians(lng)) for lat, lng in coords[:n]]
    
    area = 0
    cx = 0
    cy = 0
    
    for i in range(n):
        j = (i + 1) % n
        lat_i, lng_i = points_rad[i]
        lat_j, lng_j = points_rad[j]
        cross = lat_i * lng_j - lat_j * lng_i
        area += cross
        cx += (lat_i + lat_j) * cross
        cy += (lng_i + lng_j) * cross
    
    area /= 2
    if abs(area) < 1e-12:
        # Area too small; fall back to simple average
        avg_lat = sum(lat for lat, _ in coords[:n]) / n
        avg_lng = sum(lng for _, lng in coords[:n]) / n
        return (avg_lat, avg_lng)
    
    cx = cx / (6 * area)
    cy = cy / (6 * area)
    
    # Convert back to degrees
    return (degrees(cx), degrees(cy))


def _point_in_polygon(point, polygon):
    """
    Ray-casting algorithm to check if a point lies inside a polygon.
    point: (lat, lng)
    polygon: [(lat, lng), ...] — first and last point should match
    """
    lat, lng = point
    n = len(polygon) - 1  # actual vertex count
    inside = False
    j = n - 1
    for i in range(n):
        lat_i, lng_i = polygon[i]
        lat_j, lng_j = polygon[j]
        if ((lng_i > lng) != (lng_j > lng)) and \
           (lat < (lat_j - lat_i) * (lng - lng_i) / (lng_j - lng_i) + lat_i):
            inside = not inside
        j = i
    return inside


# -------------------- Isochrone API Calls --------------------

ISOCHRONE_API_URL = "https://isochrones.googleapis.com/v1/isochrones:generate"

# ISOCHRONE_API_KEY reuses the existing API_KEY
API_KEY = 'AIzaSyCnw9D0WUqrs_ZbZkjRGnoJjgnX6XNHOKs'
gmaps = googlemaps.Client(key=API_KEY)


def preProcess(user_enters: list) -> list:
    """
    Preprocessing: convert user-entered address strings into standardized location objects (with coordinates and place_id).
    First calls Places Autocomplete to get an exact match, then uses Geocoding to obtain coordinates.
    Returns a list of dicts, each containing 'formatted_address', 'lat', 'lng', 'place_id'.
    """
    enter_list = []
    for address in user_enters:
        try:
            # Step 1: Use Places Autocomplete to get the best matching place_id
            autocomplete_result = gmaps.places_autocomplete(address)
            if not autocomplete_result:
                print(f"Warning: Address '{address}' could not be matched via Autocomplete")
                return None
            
            # Use the first match's place_id
            best_match = autocomplete_result[0]
            place_id = best_match['place_id']
            
            # Step 2: Use place_id to call Geocoding (or Place Details) for precise coordinates
            place_details = gmaps.place(place_id, fields=['geometry', 'formatted_address'])
            result = place_details.get('result', {})
            if not result:
                print(f"Warning: Could not retrieve details for place_id '{place_id}'")
                return None
            
            lat = result['geometry']['location']['lat']
            lng = result['geometry']['location']['lng']
            
            enter_list.append({
                'formatted_address': result.get('formatted_address', best_match.get('description', address)),
                'lat': lat,
                'lng': lng,
                'place_id': place_id
            })
        except Exception as e:
            print(f"preProcess error: {e}")
            return None
    
    return enter_list

import re

def parse_travel_time(travel_time_str: str) -> int:
    """
    Parse a human-readable travel time string into seconds.
    Supported formats:
        "30m" / "45min" / "45 minutes"  → minutes
        "1h"  / "1.5h" / "1hr"          → hours
        "3600s"                          → seconds (pass-through)
    Returns: int seconds; raises ValueError on failure.
    """
    s = travel_time_str.strip()
    m = re.match(r'^(\d+(?:\.\d+)?)\s*(m|min|minutes?)$', s, re.I)
    if m:
        return int(float(m.group(1)) * 60)
    m = re.match(r'^(\d+(?:\.\d+)?)\s*(h|hr|hours?)$', s, re.I)
    if m:
        return int(float(m.group(1)) * 3600)
    m = re.match(r'^(\d+)\s*s(?:ec(?:onds?)?)?$', s, re.I)
    if m:
        return int(m.group(1))
    raise ValueError(
        f"Cannot parse travel_time: '{travel_time_str}'. "
        f"Supported formats: '30m', '45min', '1h', '1.5h', '3600s'"
    )

def getIsochrone(start_point: dict, method: str,
                 travel_time_seconds: int) -> dict:
    """
    Retrieve isochrone polygon data for a given start point.
    travel_time_seconds: travel time in seconds (call parse_travel_time() beforehand to convert).
    """
    valid_methods = ['DRIVE', 'BICYCLE', 'WALK']
    if method.upper() not in valid_methods:
        raise ValueError(f"Travel mode must be one of {valid_methods}, got: {method}")
    travel_duration = f"{travel_time_seconds}s"
    # Build request payload
    if start_point.get('place_id'):
        payload = {"place": f"places/{start_point['place_id']}"}
    else:
        payload = {
            "location": {
                "latLng": {
                    "latitude": start_point['lat'],
                    "longitude": start_point['lng']
                }
            }
        }
    payload["travelMode"] = method.upper()
    payload["travelDuration"] = travel_duration
    payload["travelDirection"] = "FROM"

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        # Correct field mask: isochrone.geo_json (no routes, no isochronePolyline)
        "X-Goog-FieldMask": "isochrone.geo_json"
    }

    # Send request
    try:
        response = requests.post(ISOCHRONE_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        detail = ""
        try:
            detail = response.text
        except Exception:
            pass
        raise RuntimeError(f"Isochrone API request failed: {e}\nResponse content: {detail}")
    except json.JSONDecodeError:
        raise RuntimeError("Isochrone API returned non-JSON response")

    # Parse response
    isochrone = data.get('isochrone', {})
    geo_json = isochrone.get('geoJson', {})

    # geo_json is a GeoJSON: {"type":"Polygon", "coordinates": [...]}
    coords_container = geo_json.get('coordinates', [])
    if not coords_container:
        raise RuntimeError("No coordinate data found in isochrone API response")

    # Take the first ring
    ring = coords_container[0]

    # ring may be [[lng,lat], ...] or [[[lng,lat],...], ...] (multi-segment)
    # If ring[0] is a list and ring[0][0] is also a list → multi-segment, flatten
    if ring and isinstance(ring[0], list) and isinstance(ring[0][0], list):
        # Multi-segment: merge all segments into one list
        coords_lng_lat = []
        for segment in ring:
            coords_lng_lat.extend(segment)
    else:
        # Standard format: use directly
        coords_lng_lat = ring

    # GeoJSON coordinates are [lng, lat], convert to (lat, lng)
    polygon_coords = [(lat, lng) for lng, lat in coords_lng_lat]
    if polygon_coords[0] != polygon_coords[-1]:
        polygon_coords.append(polygon_coords[0])

    centroid = _polygon_centroid(polygon_coords)

    max_dist = 0
    for lat, lng in polygon_coords[:-1]:
        d = _haversine(centroid[1], centroid[0], lng, lat)
        if d > max_dist:
            max_dist = d

    return {
        'origin': (start_point['lat'], start_point['lng']),
        'place_id': start_point.get('place_id'),
        'method': method.upper(),
        'travel_time_seconds': travel_time_seconds,
        'polygon_coords': polygon_coords,
        'centroid': centroid,
        'radius_km': max_dist
    }


def getCommon(isochrones: list) -> dict:
    """
    Compute the intersection area of all isochrones (using a simple grid sampling method, no shapely dependency).
    Parameters:
        isochrones: list of results from getIsochrone()
    Returns:
        {
            'intersection_centroid': (lat, lng),  # Centroid of the intersection area
            'intersection_radius_km': float,       # Coverage radius of the intersection area
            'intersection_valid': bool,            # Whether a non-empty intersection exists
            'intersection_polygons': [polygon_coords, ...], # List of original polygons (for downstream filtering)
            'combined_methods': list               # Deduplicated list of travel modes
        }
    """
    if not isochrones:
        return {'intersection_valid': False}
    
    # Collect all polygons and travel modes
    polygons = [iso['polygon_coords'] for iso in isochrones]
    methods = [iso['method'] for iso in isochrones]
    
    # ---- Grid sampling to approximate intersection ----
    # 1. Determine the bounding box of all polygons (lat/lng extents)
    all_lats = []
    all_lngs = []
    for poly in polygons:
        for lat, lng in poly:
            all_lats.append(lat)
            all_lngs.append(lng)
    
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lng, max_lng = min(all_lngs), max(all_lngs)
    
    # 2. Sample uniformly within the bounding box, test which points lie inside all polygons
    # Sampling density: ~50x50 grid (adjustable based on precision requirements)
    grid_size = 50
    step_lat = (max_lat - min_lat) / grid_size if grid_size > 0 else 0.001
    step_lng = (max_lng - min_lng) / grid_size if grid_size > 0 else 0.001
    
    intersection_points = []
    
    for i in range(grid_size + 1):
        lat = min_lat + i * step_lat
        for j in range(grid_size + 1):
            lng = min_lng + j * step_lng
            point = (lat, lng)
            
            # Check if the point lies inside all polygons
            in_all = True
            for poly in polygons:
                if not _point_in_polygon(point, poly):
                    in_all = False
                    break
            
            if in_all:
                intersection_points.append(point)
    
    if not intersection_points:
        return {
            'intersection_valid': False,
            'intersection_polygons': polygons,
            'combined_methods': list(set(methods))
        }
    
    # 3. Compute the centroid of the intersection points
    cent_lat = sum(p[0] for p in intersection_points) / len(intersection_points)
    cent_lng = sum(p[1] for p in intersection_points) / len(intersection_points)
    centroid = (cent_lat, cent_lng)
    
    # 4. Compute coverage radius (distance to the farthest intersection point)
    max_dist = 0
    for lat, lng in intersection_points:
        d = _haversine(centroid[1], centroid[0], lng, lat)
        if d > max_dist:
            max_dist = d
    
    return {
        'intersection_centroid': centroid,
        'intersection_radius_km': max_dist,
        'intersection_valid': True,
        'intersection_point_count': len(intersection_points),
        'intersection_polygons': polygons,
        'combined_methods': list(set(methods))
    }


def getSuggestions(isochrones: list, placetype: str,
                   requires: dict = None) -> list:
    """
    Search for POIs matching the given criteria within the common intersection area of all isochrones.
    Parameters:
        isochrones: list of results from getIsochrone()
        placetype: POI type (e.g., 'restaurant', 'cafe', 'park')
        requires: optional dict of additional requirements, may contain:
            - 'keyword': str           # Keyword filter (e.g., "vegan")
            - 'min_rating': float      # Minimum rating
            - 'open_now': bool         # Only return currently open places
    Returns:
        List of POI dicts, each containing name, address, lat, lng, rating, travel_times, etc.
    """
    if requires is None:
        requires = {}
    
    keyword = requires.get('keyword', '')
    min_rating = requires.get('min_rating', 0.0)
    open_now = requires.get('open_now', False)
    
    # 1. Compute the intersection area
    common = getCommon(isochrones)
    if not common['intersection_valid']:
        print("Warning: No common intersection area; cannot recommend places.")
        return []
    
    center_lat, center_lng = common['intersection_centroid']
    radius_m = min(common['intersection_radius_km'] * 1000, 50000)  # convert to meters, max 50km
    polygons = common['intersection_polygons']
    
    # 2. Use Places API to search within the bounding circle of the intersection
    location = (center_lat, center_lng)
    search_radius = max(int(radius_m), 100)  # at least 100m
    
    try:
        response = gmaps.places_nearby(
            location=location,
            radius=search_radius,
            type=placetype,
            keyword=keyword if keyword else None,
            open_now=open_now if open_now else None
        )
    except Exception as e:
        print(f"Places API search failed: {e}")
        return []
    
    candidates = response.get('results', [])
    if not candidates:
        return []
    
    # 3. For each candidate, check if it truly lies within the intersection (precise filtering)
    filtered = []
    for poi in candidates:
        poi_lat = poi['geometry']['location']['lat']
        poi_lng = poi['geometry']['location']['lng']
        poi_point = (poi_lat, poi_lng)
        
        # Check if point lies inside all isochrone polygons
        in_all = True
        for poly in polygons:
            if not _point_in_polygon(poi_point, poly):
                in_all = False
                break
        
        if not in_all:
            continue
        
        # Check rating
        rating = poi.get('rating', 0) or 0
        if rating < min_rating:
            continue
        
        # Calculate travel times from each origin (using Distance Matrix)
        origins = [(iso['origin'][0], iso['origin'][1]) for iso in isochrones]
        methods_for_matrix = [iso['method'] for iso in isochrones]
        
        travel_times = {}
        ROUTES_API_URL = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
        for idx, (origin, method) in enumerate(zip(origins, methods_for_matrix)):
            try:
                payload = {
                    "origins": [{
                        "waypoint": {
                            "location": {
                                "latLng": {
                                    "latitude": origin[0],
                                    "longitude": origin[1]
                                }
                            }
                        }
                    }],
                    "destinations": [{
                        "waypoint": {
                            "location": {
                                "latLng": {
                                    "latitude": poi_lat,
                                    "longitude": poi_lng
                                }
                            }
                        }
                    }],
                    "travelMode": method.upper(),
                    "routingPreference": "TRAFFIC_AWARE" if method.upper() == "DRIVE" else None,
                    "units": "METRIC"
                }
                # Remove None values
                payload = {k: v for k, v in payload.items() if v is not None}
                headers = {
                    "Content-Type": "application/json",
                    "X-Goog-Api-Key": API_KEY,
                    "X-Goog-FieldMask": "originIndex,destinationIndex,duration"
                }
                resp = requests.post(ROUTES_API_URL, json=payload, headers=headers)
                resp.raise_for_status()
                matrix_data = resp.json()
                # matrix_data is an array where each element is a route result
                if isinstance(matrix_data, list) and len(matrix_data) > 0:
                    duration_str = matrix_data[0].get("duration", None)
                    if duration_str:
                        # duration format: "1234s"
                        secs = int(duration_str.rstrip("s"))
                        travel_times[f"person_{idx}_{method}"] = secs / 60.0
                    else:
                        travel_times[f"person_{idx}_{method}"] = float('inf')
                else:
                    travel_times[f"person_{idx}_{method}"] = float('inf')
            except Exception as e:
                print(f"⚠ Distance Matrix call failed (person_{idx}_{method}): {e}")
                travel_times[f"person_{idx}_{method}"] = float('inf')
        
        # Compute average travel time
        valid_times = [v for v in travel_times.values() if v != float('inf')]
        avg_time = np.mean(valid_times) if valid_times else float('inf')
        
        filtered.append({
            'name': poi.get('name', 'Unknown'),
            'address': poi.get('vicinity', ''),
            'lat': poi_lat,
            'lng': poi_lng,
            'place_id': poi.get('place_id', ''),
            'rating': rating,
            'user_ratings_total': poi.get('user_ratings_total', 0),
            'travel_times': travel_times,
            'avg_travel_time': avg_time,
            'types': poi.get('types', [])
        })
    
    return filtered


def sortPlaces(suggestions: list) -> list:
    """
    Sort candidate POIs.
    Rules:
        1. Ascending average travel time (closer is better)
        2. Descending Google rating (higher is better)
        3. Descending number of ratings (more reviews = more reliable)
    """
    def sort_key(poi):
        avg_time = poi.get('avg_travel_time', float('inf'))
        rating = poi.get('rating', 0) or 0
        total = poi.get('user_ratings_total', 0) or 0
        # Shorter time first (ascending), higher rating first (descending), more reviews first (descending)
        return (avg_time, -rating, -total)
    
    return sorted(suggestions, key=sort_key)


def queryAI(suggestions: list) -> str:
    """
    Format the sorted candidate POIs into a structured text block for LLM decision-making.
    Returns a string that can be directly injected into an LLM prompt.
    """
    if not suggestions:
        return ("[No recommendations] No matching places were found within the common intersection "
                "of everyone's isochrones. Suggestion: expand the search time or try a different place type.")
    
    lines = [
        "Below are the recommended meeting places, ranked by isochrone intersection and ratings:",
        "=" * 60
    ]
    
    for i, poi in enumerate(suggestions, 1):
        name = poi.get('name', 'Unknown')
        addr = poi.get('address', 'No address')
        rating = poi.get('rating', 'No rating')
        avg_time = poi.get('avg_travel_time', float('inf'))
        
        # Individual travel time details
        travel_details = []
        for key, value in poi.get('travel_times', {}).items():
            if value != float('inf'):
                # Extract info from key, e.g., "person_0_DRIVE"
                parts = key.split('_')
                if len(parts) >= 3:
                    person_idx = parts[1]
                    method = '_'.join(parts[2:])
                    travel_details.append(f"  Person {person_idx} ({method}): {value:.1f} min")
                else:
                    travel_details.append(f"  {key}: {value:.1f} min")
        
        lines.append(f"\n{i}. {name}")
        lines.append(f"   Address: {addr}")
        lines.append(f"   Rating: {rating} ({poi.get('user_ratings_total', 0)} reviews)")
        lines.append(f"   Average travel time: {avg_time:.1f} min")
        if travel_details:
            lines.append("   Individual travel times:")
            lines.extend(travel_details[:5])  # Show at most 5 individuals
    
    lines.append("\n" + "=" * 60)
    lines.append("Based on the above information, considering convenience, ratings, and user preferences, ")
    lines.append("please select the best meeting place.")
    lines.append("If you need to re-search (e.g., adjust time threshold, place type, or keyword), let me know.")
    
    return "\n".join(lines)


# -------------------- Advanced: Iterative Control --------------------

def adjustTimeThreshold(isochrones: list, new_time: str) -> list:
    """
    Adjust the isochrone time threshold and re-fetch (for iterative backtracking).
    Parameters:
        isochrones: original list of getIsochrone() results
        new_time: new time threshold string, e.g., "45m", "1h"
    Returns: new list of isochrones
    """
    new_seconds = parse_travel_time(new_time)
    new_isochrones = []
    for iso in isochrones:
        start_point = {
            'lat': iso['origin'][0],
            'lng': iso['origin'][1],
            'place_id': iso.get('place_id') 
        }
        new_iso = getIsochrone(start_point, iso['method'], new_seconds)
        new_isochrones.append(new_iso)
    return new_isochrones


def full_pipeline(user_addresses: list, transport_modes: list,
                  place_type: str, travel_time: str = "30m",
                  requirements: dict = None,
                  top_n: int = 5) -> dict:
    """
    Execute the full pipeline in one call: Preprocess → Isochrones → Intersection → POI Search → Sort → AI Text
    Returns a dict with all intermediate results, suitable for flexible use by the Agent.
    """
    # 1. Preprocessing
    travel_time_seconds = parse_travel_time(travel_time)
    processed = preProcess(user_addresses)
    if processed is None:
        return {"error": "Address parsing failed; please check inputs", "stage": "preProcess"}
    
    # 2. Get isochrones
    isochrones = []
    for point, mode in zip(processed, transport_modes):
        try:
            iso = getIsochrone(point, mode, travel_time_seconds)
            isochrones.append(iso)
        except Exception as e:
            return {"error": f"Isochrone retrieval failed: {e}", "stage": "getIsochrone"}
    
    # 3. Compute intersection
    common = getCommon(isochrones)
    if not common['intersection_valid']:
        return {
            "error": "The isochrones of all participants have no overlap; consider increasing the time threshold",
            "stage": "getCommon",
            "isochrones": isochrones,
            "common": common
        }
    
    # 4. Search for POIs
    suggestions = getSuggestions(isochrones, place_type, requirements)
    
    # 5. Sort and take top_N
    sorted_places = sortPlaces(suggestions)[:top_n]
    
    # 6. Generate AI-friendly text
    ai_text = queryAI(sorted_places)
    
    return {
        "stage": "complete",
        "processed_addresses": processed,
        "isochrones": isochrones,
        "common_area": common,
        "suggestions": sorted_places,
        "ai_text": ai_text,
        "suggestions_count": len(sorted_places)
    }


# -------------------- Tests --------------------
if __name__ == '__main__':
    # ──────────────────────────────────────────────────────────────
    # Test scenario: 4 people meeting in downtown / midtown Toronto
    #
    #   A: Union Station (downtown core)  → WALK   30min (~3km)
    #   B: Yonge & Bloor (midtown)        → BICYCLE 30min (~7-8km)
    #   C: Distillery District (east downtown) → WALK   30min
    #   D: Liberty Village (west downtown) → BICYCLE 30min
    #
    #   Their isochrones should have an intersection in the downtown core.
    # ──────────────────────────────────────────────────────────────
    users = [
        "Union Station, 65 Front Street West, Toronto, ON",
        "Yonge and Bloor, Toronto, ON",
        "Distillery District, 55 Mill Street, Toronto, ON",
        "Liberty Village, 171 E Liberty Street, Toronto, ON",
    ]
    modes = ["WALK", "BICYCLE", "WALK", "BICYCLE"]
    result = full_pipeline(
        user_addresses=users,
        transport_modes=modes,
        place_type="restaurant",
        travel_time="30m",
        requirements={"min_rating": 4.0}
    )
    if "error" in result:
        print(f"❌ Pipeline error: {result['error']} (stage: {result['stage']})")
    else:
        print(f"✅ Found {result['suggestions_count']} recommended places\n")
        print(result['ai_text'])
