import googlemaps
import requests
import json
import numpy as np
from math import radians, cos, sin, asin, sqrt, degrees, atan2

# -------------------- 几何工具函数（独立于 API） --------------------

def _haversine(lon1, lat1, lon2, lat2):
    """计算两点间的大圆弧距离（km）"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c


def _polygon_centroid(coords):
    """
    计算多边形质心（经纬度坐标列表，首尾闭合）
    coords: [(lat, lng), ...] 或 [(lng, lat), ...]，需指定 format
    """
    # 使用 Shoelace 公式计算多边形质心
    n = len(coords) - 1  # 减去重复的首尾点
    if n < 3:
        raise ValueError("多边形至少需要3个顶点")
    
    # 转换为弧度
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
        # 面积太小，直接取平均
        avg_lat = sum(lat for lat, _ in coords[:n]) / n
        avg_lng = sum(lng for _, lng in coords[:n]) / n
        return (avg_lat, avg_lng)
    
    cx = cx / (6 * area)
    cy = cy / (6 * area)
    
    # 转换回度
    return (degrees(cx), degrees(cy))


def _point_in_polygon(point, polygon):
    """
    射线法判断点是否在多边形内
    point: (lat, lng)
    polygon: [(lat, lng), ...] 首尾闭合
    """
    lat, lng = point
    n = len(polygon) - 1  # 实际顶点数
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


# -------------------- 等时圈 API 调用 --------------------

ISOCHRONE_API_URL = "https://routes.googleapis.com/v1alpha:computeIsochrone"

# ISOCHRONE_API_KEY 复用已有 API_KEY
API_KEY = 'AIzaSyCnw9D0WUqrs_ZbZkjRGnoJjgnX6XNHOKs'
gmaps = googlemaps.Client(key=API_KEY)


def preProcess(user_enters: list) -> list:
    """
    预处理：将用户输入的地址字符串转换为标准地点对象（含坐标和 place_id）
    先调用 Places Autocomplete 获取精确匹配，再用 Geocoding 获取坐标
    返回列表，每个元素为字典，包含 'formatted_address', 'lat', 'lng', 'place_id'
    """
    enter_list = []
    for address in user_enters:
        try:
            # 第一步：用 Places Autocomplete 获取最匹配的 place_id
            autocomplete_result = gmaps.places_autocomplete(address)
            if not autocomplete_result:
                print(f"警告：地址 '{address}' 无法通过 Autocomplete 匹配")
                return None
            
            # 取第一个匹配结果的 place_id
            best_match = autocomplete_result[0]
            place_id = best_match['place_id']
            
            # 第二步：用 place_id 调用 Geocoding（或 Place Details）获取精确坐标
            place_details = gmaps.place(place_id, fields=['geometry', 'formatted_address'])
            result = place_details.get('result', {})
            if not result:
                print(f"警告：无法获取 place_id '{place_id}' 的详细信息")
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
            print(f"preProcess 出错: {e}")
            return None
    
    return enter_list


def getIsochrone(start_point: dict, method: str,
                 travel_time: str = "30m") -> dict:
    """
    调用 Google Routes API computeIsochrone 获取等时圈多边形
    参数：
        start_point: preProcess 返回的地点字典（必须含 'place_id' 或 'lat'/'lng'）
        method:   交通方式，可选 'DRIVE', 'TRANSIT', 'WALK', 'BICYCLE', 'TWO_WHEELER'
        travel_time: 等时圈时间阈值，如 "30m"（30分钟）、"1h"（1小时）
    返回：
        {
            'origin': (lat, lng),            # 起点坐标
            'method': str,                    # 交通方式
            'travel_time': str,               # 时间阈值
            'polygon_coords': [(lat, lng),...], # 等时圈多边形顶点（首尾闭合）
            'centroid': (lat, lng),           # 多边形质心
            'radius_km': float                # 多边形近似半径（到质心的最大距离）
        }
    """
    # 参数校验
    valid_methods = ['DRIVE', 'TRANSIT', 'WALK', 'BICYCLE', 'TWO_WHEELER']
    if method.upper() not in valid_methods:
        raise ValueError(f"交通方式必须为 {valid_methods} 之一，收到: {method}")
    
    # 构造请求体
    origin = {}
    if start_point.get('place_id'):
        origin['placeId'] = start_point['place_id']
    else:
        origin['address'] = f"{start_point['lat']},{start_point['lng']}"
    
    payload = {
        "origin": origin,
        "travelMode": method.upper(),
        "travelTime": travel_time,
        "polylineEncoding": "GEOJSON_LINESTRING"
    }
    
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "routes.isochronePolyline"
    }
    
    try:
        response = requests.post(ISOCHRONE_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"等时圈 API 请求失败: {e}")
    except json.JSONDecodeError:
        raise RuntimeError("等时圈 API 返回非 JSON 响应")
    
    # 解析响应
    routes = data.get('routes', [])
    if not routes:
        raise RuntimeError(f"等时圈 API 返回空结果 ({travel_time} / {method})")
    
    isochrone = routes[0].get('isochronePolyline', {})
    geojson = isochrone.get('geojsonLinestring', {})
    coords_lng_lat = geojson.get('coordinates', [])
    
    if not coords_lng_lat:
        raise RuntimeError("等时圈 API 响应中无坐标数据")
    
    # 转换为 (lat, lng) 格式
    polygon_coords = [(lat, lng) for lng, lat in coords_lng_lat]
    
    # 确保首尾闭合
    if polygon_coords[0] != polygon_coords[-1]:
        polygon_coords.append(polygon_coords[0])
    
    # 计算质心
    centroid = _polygon_centroid(polygon_coords)
    
    # 计算近似半径（到质心的最大距离）
    max_dist = 0
    for lat, lng in polygon_coords[:-1]:  # 不计算重复的最后一个点
        d = _haversine(centroid[1], centroid[0], lng, lat)
        if d > max_dist:
            max_dist = d
    
    return {
        'origin': (start_point['lat'], start_point['lng']),
        'method': method.upper(),
        'travel_time': travel_time,
        'polygon_coords': polygon_coords,
        'centroid': centroid,
        'radius_km': max_dist
    }


def getCommon(isochrones: list) -> dict:
    """
    计算所有等时圈的交集区域（使用简易网格采样法，无需 shapely 依赖）
    参数：
        isochrones: getIsochrone 返回的列表
    返回：
        {
            'intersection_centroid': (lat, lng),  # 交集区域的质心
            'intersection_radius_km': float,       # 交集区域的覆盖半径
            'intersection_valid': bool,            # 是否有非空交集
            'intersection_polygons': [polygon_coords,...], # 原始多边形列表（供后续过滤用）
            'combined_methods': list               # 去重后的交通方式列表
        }
    """
    if not isochrones:
        return {'intersection_valid': False}
    
    # 收集所有多边形和交通方式
    polygons = [iso['polygon_coords'] for iso in isochrones]
    methods = [iso['method'] for iso in isochrones]
    
    # ---- 网格采样法求近似交集 ----
    # 1. 确定所有多边形的外包围盒（经纬度边界）
    all_lats = []
    all_lngs = []
    for poly in polygons:
        for lat, lng in poly:
            all_lats.append(lat)
            all_lngs.append(lng)
    
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lng, max_lng = min(all_lngs), max(all_lngs)
    
    # 2. 在包围盒内均匀采样，判断哪些点在所有多边形内
    # 采样密度：约 50×50 网格（可根据精度需求调整）
    grid_size = 50
    step_lat = (max_lat - min_lat) / grid_size if grid_size > 0 else 0.001
    step_lng = (max_lng - min_lng) / grid_size if grid_size > 0 else 0.001
    
    intersection_points = []
    
    for i in range(grid_size + 1):
        lat = min_lat + i * step_lat
        for j in range(grid_size + 1):
            lng = min_lng + j * step_lng
            point = (lat, lng)
            
            # 检查该点是否在所有多边形内
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
    
    # 3. 计算交集点的质心
    cent_lat = sum(p[0] for p in intersection_points) / len(intersection_points)
    cent_lng = sum(p[1] for p in intersection_points) / len(intersection_points)
    centroid = (cent_lat, cent_lng)
    
    # 4. 计算覆盖半径（到最远交集点的距离）
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
    在所有等时圈的交集区域内搜索符合条件的 POI
    参数：
        isochrones: getIsochrone 返回的列表
        placetype: POI 类型（如 'restaurant', 'cafe', 'park'）
        requires: 额外要求字典，可包含：
            - 'keyword': str           # 关键词过滤（如 "vegan"）
            - 'min_rating': float      # 最低评分
            - 'open_now': bool         # 是否仅当前营业的
    返回：
        POI 列表，每个元素包含 name, address, lat, lng, rating, travel_times 等
    """
    if requires is None:
        requires = {}
    
    keyword = requires.get('keyword', '')
    min_rating = requires.get('min_rating', 0.0)
    open_now = requires.get('open_now', False)
    
    # 1. 计算交集区域
    common = getCommon(isochrones)
    if not common['intersection_valid']:
        print("警告：没有公共交集区域，无法推荐地点")
        return []
    
    center_lat, center_lng = common['intersection_centroid']
    radius_m = min(common['intersection_radius_km'] * 1000, 50000)  # 转米，上限 50km
    polygons = common['intersection_polygons']
    
    # 2. 用 Places API 在交集外接圆内搜索
    location = (center_lat, center_lng)
    search_radius = max(int(radius_m), 100)  # 至少 100m
    
    try:
        response = gmaps.places_nearby(
            location=location,
            radius=search_radius,
            type=placetype,
            keyword=keyword if keyword else None,
            open_now=open_now if open_now else None
        )
    except Exception as e:
        print(f"Places API 搜索失败: {e}")
        return []
    
    candidates = response.get('results', [])
    if not candidates:
        return []
    
    # 3. 对每个候选点，检查是否真正在交集中（精确过滤）
    filtered = []
    for poi in candidates:
        poi_lat = poi['geometry']['location']['lat']
        poi_lng = poi['geometry']['location']['lng']
        poi_point = (poi_lat, poi_lng)
        
        # 检查点是否在所有等时圈多边形内
        in_all = True
        for poly in polygons:
            if not _point_in_polygon(poi_point, poly):
                in_all = False
                break
        
        if not in_all:
            continue
        
        # 检查评分
        rating = poi.get('rating', 0) or 0
        if rating < min_rating:
            continue
        
        # 计算到各起点的旅行时间（使用 Distance Matrix）
        origins = [(iso['origin'][0], iso['origin'][1]) for iso in isochrones]
        methods_for_matrix = [iso['method'] for iso in isochrones]
        
        travel_times = {}
        for idx, (origin, method) in enumerate(zip(origins, methods_for_matrix)):
            try:
                matrix = gmaps.distance_matrix(
                    origins=[origin],
                    destinations=[(poi_lat, poi_lng)],
                    mode=method.lower(),
                    units='metric'
                )
                element = matrix['rows'][0]['elements'][0]
                if element['status'] == 'OK':
                    travel_times[f"person_{idx}_{method}"] = element['duration']['value'] / 60.0  # 分钟
                else:
                    travel_times[f"person_{idx}_{method}"] = float('inf')
            except Exception:
                travel_times[f"person_{idx}_{method}"] = float('inf')
        
        # 计算平均旅行时间
        valid_times = [v for v in travel_times.values() if v != float('inf')]
        avg_time = np.mean(valid_times) if valid_times else float('inf')
        
        filtered.append({
            'name': poi.get('name', '未知'),
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
    对候选 POI 排序
    规则：
        1. 平均旅行时间升序（越近越好）
        2. Google 评分降序（评分越高越好）
        3. 评价数量降序（越多人评价越可靠）
    """
    def sort_key(poi):
        avg_time = poi.get('avg_travel_time', float('inf'))
        rating = poi.get('rating', 0) or 0
        total = poi.get('user_ratings_total', 0) or 0
        # 时间短优先（升序），评分高优先（降序），评价多优先（降序）
        return (avg_time, -rating, -total)
    
    return sorted(suggestions, key=sort_key)


def queryAI(suggestions: list) -> str:
    """
    将排序后的候选 POI 整理成结构化文本，供 LLM 决策
    返回可直接注入 LLM prompt 的字符串
    """
    if not suggestions:
        return "【无推荐结果】在所有人的等时圈交集内没有找到符合类型要求的地点。" + \
                "建议：扩大搜索时间范围，或更换地点类型。"
    
    lines = [
        "以下是综合等时圈交集和评分排序后的推荐会面地点：",
        "=" * 60
    ]
    
    for i, poi in enumerate(suggestions, 1):
        name = poi.get('name', '未知')
        addr = poi.get('address', '无地址')
        rating = poi.get('rating', '无评分')
        avg_time = poi.get('avg_travel_time', float('inf'))
        
        # 各人旅行时间详情
        travel_details = []
        for key, value in poi.get('travel_times', {}).items():
            if value != float('inf'):
                # 从 key 中提取信息，如 "person_0_DRIVE"
                parts = key.split('_')
                if len(parts) >= 3:
                    person_idx = parts[1]
                    method = '_'.join(parts[2:])
                    travel_details.append(f"  参与者{person_idx}（{method}）：{value:.1f} 分钟")
                else:
                    travel_details.append(f"  {key}：{value:.1f} 分钟")
        
        lines.append(f"\n{i}. {name}")
        lines.append(f"   地址：{addr}")
        lines.append(f"   评分：{rating}（{poi.get('user_ratings_total', 0)} 条评价）")
        lines.append(f"   平均到达时间：{avg_time:.1f} 分钟")
        if travel_details:
            lines.append("   各参与者用时：")
            lines.extend(travel_details[:5])  # 最多展示5人
    
    lines.append("\n" + "=" * 60)
    lines.append("请根据以上信息，综合考虑便利性、评分和用户偏好，选择最佳会面地点。")
    lines.append("如需重新搜索（例如调整时间阈值、地点类型或关键词），请告知。")
    
    return "\n".join(lines)


# -------------------- 高级功能：回溯控制 --------------------

def adjustTimeThreshold(isochrones: list, new_time: str) -> list:
    """
    调整等时圈时间阈值并重新获取（用于回溯迭代）
    参数：
        isochrones: 原始 getIsochrone 结果列表
        new_time: 新的时间阈值，如 "45m", "1h"
    返回：新的 isochrones 列表
    """
    new_isochrones = []
    for iso in isochrones:
        # 从原始起点信息重建 start_point 字典
        start_point = {
            'lat': iso['origin'][0],
            'lng': iso['origin'][1],
            'place_id': None  # 如果之前没存 place_id，可以用坐标
        }
        # 尝试从 isochrones 中找回 place_id（如果有存的话）
        # 目前 getIsochrone 返回中未存 place_id，可用坐标代替
        new_iso = getIsochrone(start_point, iso['method'], new_time)
        new_isochrones.append(new_iso)
    return new_isochrones


def full_pipeline(user_addresses: list, transport_modes: list,
                  place_type: str, travel_time: str = "30m",
                  requirements: dict = None) -> dict:
    """
    一键执行完整流程：预处理 → 等时圈 → 求交集 → 搜索 POI → 排序 → AI 文本
    返回包含所有中间结果的字典，供 Agent 灵活使用
    """
    # 1. 预处理
    processed = preProcess(user_addresses)
    if processed is None:
        return {"error": "地址解析失败，请检查输入", "stage": "preProcess"}
    
    # 2. 获取等时圈
    isochrones = []
    for point, mode in zip(processed, transport_modes):
        try:
            iso = getIsochrone(point, mode, travel_time)
            isochrones.append(iso)
        except Exception as e:
            return {"error": f"等时圈获取失败: {e}", "stage": "getIsochrone"}
    
    # 3. 计算交集
    common = getCommon(isochrones)
    if not common['intersection_valid']:
        return {
            "error": "所有人的等时圈没有交集，建议增大时间阈值",
            "stage": "getCommon",
            "isochrones": isochrones,
            "common": common
        }
    
    # 4. 搜索 POI
    suggestions = getSuggestions(isochrones, place_type, requirements)
    
    # 5. 排序
    sorted_places = sortPlaces(suggestions)
    
    # 6. AI 文本
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


# -------------------- 测试 --------------------
if __name__ == '__main__':
    # 示例：两个用户的会面地点规划
    users = [
        "1600 Amphitheatre Parkway, Mountain View, CA",
        "1 Infinite Loop, Cupertino, CA"
    ]
    modes = ["DRIVE", "DRIVE"]
    
    result = full_pipeline(
        user_addresses=users,
        transport_modes=modes,
        place_type="restaurant",
        travel_time="30m",
        requirements={"keyword": "vegan", "min_rating": 3.5}
    )
    
    if "error" in result:
        print(f"❌ 流程出错: {result['error']} (阶段: {result['stage']})")
    else:
        print(f"✅ 找到 {result['suggestions_count']} 个推荐地点\n")
        print(result['ai_text'])