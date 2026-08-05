import googlemaps
import requests
import json
import numpy as np
import os
import re
import time
from functools import wraps
from math import radians, cos, sin, asin, sqrt, degrees, atan2
from dotenv import load_dotenv

load_dotenv()
# Fallback: allow .env to live one directory above src/ (repo-adjacent layout).
if not os.getenv("GOOGLE_API_KEY"):
    _repo_dotenv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.isfile(_repo_dotenv):
        load_dotenv(_repo_dotenv)
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Google API key not found. Set GOOGLE_API_KEY in your .env file.")
gmaps = googlemaps.Client(key=API_KEY)

def baseline_measurement(label_or_func="[BASELINE MEASUREMENT]", display_name=None):
    """Print a consistent timing line for performance baselining."""

    if callable(label_or_func):
        func = label_or_func

        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                print(f"[BASELINE MEASUREMENT] {func.__name__} took {elapsed_ms:.2f} ms")

        return wrapper

    label = str(label_or_func)

    def decorator(inner_func):
        measured_name = display_name or inner_func.__name__

        @wraps(inner_func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return inner_func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                print(f"{label} {measured_name} took {elapsed_ms:.2f} ms")

        return wrapper

    return decorator

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

@baseline_measurement("[API LATENCY]", "fetching_isochrones")
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


@baseline_measurement("[GIS MATH]", "getCommon")
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


@baseline_measurement("[API LATENCY]", "searching_meeting_places")
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
        # Semantic rerank info (present only after semantic_rerank ran).
        if poi.get("match_score") is not None:
            lines.append(f"   Preference match: {poi['match_score']}/10")
            if poi.get("match_reason"):
                lines.append(f"   Why: {poi['match_reason']}")
    
    lines.append("\n" + "=" * 60)
    lines.append("Based on the above information, considering convenience, ratings, and user preferences, ")
    lines.append("please select the best meeting place.")
    lines.append("If you need to re-search (e.g., adjust time threshold, place type, or keyword), let me know.")
    
    return "\n".join(lines)


# -------------------- Semantic Rerank (LLM preference matching) --------------------

# Cloud vLLM endpoint (SSH-tunneled from local). Configurable via env vars.
VLLM_BASE_URL = os.getenv("GATHERPOINT_VLLM_URL", "http://127.0.0.1:8001/v1").rstrip("/")
VLLM_MODEL = os.getenv("GATHERPOINT_VLLM_MODEL", "gatherpoint-local")

_RERANK_SYSTEM_PROMPT = (
    "You are a meetup planner's semantic ranking engine. "
    "Given a list of candidate venues (each with Google rating and review text) "
    "and the group's preferences, score EVERY venue 1-10 on how well it matches "
    "the preferences (10 = perfect match). Consider dietary needs and transit habits. "
    "Do NOT rank by Google rating alone; review text matters more for preference fit. "
    "Rules:\n"
    "- Evaluate EVERY member's preference separately. In the reason, say exactly "
    "which member(s) match and which do not.\n"
    "- A venue that satisfies only one member gets a moderate score (3-7), not 10.\n"
    "- If a venue has NO reviews, do not assume it is bad: give a neutral score "
    "(3-7) based on its type and rating, and note 'no reviews' in the reason.\n"
    "Return ONLY a JSON object, no markdown, with this exact shape:\n"
    '{"scores": [{"name": str, "match_score": int, "reason": str}]}'
)


def _fetch_reviews(place_id: str, max_reviews: int = 3) -> list:
    """
    Fetch Google review texts for one place via Place Details.
    Reviews are truncated to 200 chars each to keep the LLM prompt small
    (cloud vLLM max_model_len = 4096). Returns [] on failure — never raises.
    """
    if not place_id:
        return []
    try:
        details = gmaps.place(place_id, fields=["review"])
        reviews = details.get("result", {}).get("reviews", []) or []
        texts = [
            str(r.get("text", "")).strip()[:200]
            for r in reviews
            if r.get("text")
        ]
        return texts[:max_reviews]
    except Exception as e:
        print(f"\u26a0 Place Details failed for {place_id}: {e}")
        return []


def _parse_rerank_scores(raw_text: str) -> dict:
    """
    Extract {name: {'match_score': int, 'reason': str}} from LLM output.
    Tolerates markdown fences and surrounding prose. Returns {} on failure.
    """
    text = str(raw_text).strip()
    # Strip markdown code fences.
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        print(f"\u26a0 semantic_rerank: no JSON object in LLM output: {text[:200]}")
        return {}
    try:
        data = json.loads(text[start : end + 1])
    except Exception as e:
        print(f"\u26a0 semantic_rerank: JSON parse failed: {e}")
        return {}
    scores = {}
    for item in data.get("scores", []) or []:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        scores[name] = {
            "match_score": int(item.get("match_score", 5)),
            "reason": str(item.get("reason", "")).strip(),
        }
    return scores


@baseline_measurement("[SEMANTIC RERANK]", "semantic_rerank")
def semantic_rerank(venues: list, preferences: str,
                    max_reviews: int = 3, timeout: int = 180) -> list:
    """
    Re-rank candidate POIs by matching Google review text against the group's
    preferences using the cloud vLLM (AMD Radeon GPU via SSH tunnel).

    Each venue gets 'match_score' (0-10) and 'match_reason' attached, then the
    list is re-sorted by blended score = 0.6*rating + 0.7*match_score.

    Never raises: on any failure (no reviews, LLM down, bad JSON) the input
    order is returned unchanged so the main pipeline stays robust.

    Set GATHERPOINT_RERANK=0 to disable (e.g. when the tunnel is down).
    """
    if os.getenv("GATHERPOINT_RERANK", "1") == "0":
        return venues
    if not venues or not preferences or not preferences.strip():
        return venues

    # 1. Attach review text to every venue (skipped on API failure).
    for poi in venues:
        poi["reviews"] = _fetch_reviews(poi.get("place_id", ""), max_reviews)

    # 2. Build the LLM request.
    candidates = [
        {
            "name": poi.get("name", "Unknown"),
            "address": poi.get("address", ""),
            "rating": float(poi.get("rating", 0) or 0),
            "user_ratings_total": int(poi.get("user_ratings_total", 0) or 0),
            "avg_travel_time": float(poi["avg_travel_time"]) if poi.get("avg_travel_time") is not None else None,
            "reviews": poi.get("reviews", [])[:max_reviews],
        }
        for poi in venues
    ]
    user_prompt = (
        "Group preferences:\n" + preferences.strip() + "\n\n"
        "Candidate venues:\n" + json.dumps(candidates, ensure_ascii=False)
    )
    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": _RERANK_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 400,
    }

    # 3. Call the cloud vLLM (SSH tunnel on 127.0.0.1:8001).
    try:
        resp = requests.post(
            f"{VLLM_BASE_URL}/chat/completions", json=payload, timeout=timeout
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        scores = _parse_rerank_scores(content)
        if not scores:
            print("\u26a0 semantic_rerank: LLM returned no parseable scores; keeping order")
            return venues
    except requests.HTTPError as e:
        body = ""
        try:
            body = resp.text[:500]
        except Exception:
            pass
        print(f"\u26a0 semantic_rerank: LLM HTTP {e.response.status_code if e.response is not None else '?'}: {body}")
        return venues
    except Exception as e:
        print(f"\u26a0 semantic_rerank: LLM call failed ({e}); keeping original order")
        return venues

    # 4. Attach scores and re-sort by blended fit.
    for poi in venues:
        s = scores.get(poi.get("name", ""))
        poi["match_score"] = int(s["match_score"]) if s else 5
        poi["match_reason"] = s["reason"] if s else ""

    def rerank_key(poi):
        avg_time = poi.get("avg_travel_time", float("inf"))
        rating = (poi.get("rating", 0) or 0) / 5.0 * 10.0  # 0-10
        match = poi.get("match_score", 5) or 5              # 0-10
        blended = 0.6 * rating + 0.7 * match
        total = poi.get("user_ratings_total", 0) or 0
        return (avg_time, -blended, -total)

    return sorted(venues, key=rerank_key)


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
                  top_n: int = 5,
                  preferences: str = "") -> dict:
    """
    Execute the full pipeline in one call: Preprocess → Isochrones → Intersection → POI Search → Sort → Semantic Rerank → AI Text
    Returns a dict with all intermediate results, suitable for flexible use by the Agent.

    preferences: optional string describing the group's preferences (e.g.
        "Alice is vegan and takes the subway. Bob hates spicy food and bikes.").
        When provided, top_n candidates are re-ranked by semantic review matching
        via the cloud vLLM. May also be supplied via requirements['preferences'].
    """
    requirements = requirements or {}
    preferences = preferences or str(requirements.get("preferences", "") or "")
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

    # 5b. Semantic rerank: match Google review text against group preferences
    #     using the cloud vLLM (AMD Radeon GPU). Falls back to original order.
    reranked = False
    if preferences.strip():
        _before = [p.get("name") for p in sorted_places]
        sorted_places = semantic_rerank(sorted_places, preferences)
        reranked = [p.get("name") for p in sorted_places] != _before

    # 6. Generate AI-friendly text
    ai_text = queryAI(sorted_places)
    
    return {
        "stage": "complete",
        "processed_addresses": processed,
        "isochrones": isochrones,
        "common_area": common,
        "suggestions": sorted_places,
        "ai_text": ai_text,
        "suggestions_count": len(sorted_places),
        "reranked": reranked
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
