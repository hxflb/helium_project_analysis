# models.py
import requests
import time
from datetime import datetime

AMAP_API_KEY = '5f2b53d631dddd8949b9f265e2fca230'  # 你的真实高德 Key
AMAP_GEOCODING_URL = "https://restapi.amap.com/v3/geocode/geo"
API_KEY = "X1sWRMQi024tNHmZN9TmZhtn"  # 你的真实文心一言 Key
SECRET_KEY = "0QSaN1G95c2pkoaPfIT2rEXcG7wAZ3Cc"  # 你的真实文心一言 Secret Key

# 获取高德地图经纬度
def get_location_from_address(address, city=None, retries=3, delay=1):
    for i in range(retries):
        params = {
            'key': AMAP_API_KEY,
            'address': address
        }
        if city:
            params['city'] = city

        response = requests.get(AMAP_GEOCODING_URL, params=params)
        data = response.json()

        if data['status'] == '1' and data['geocodes']:
            location = data['geocodes'][0]['location']
            lon, lat = location.split(',')
            return float(lat), float(lon)
        else:
            print(f"地址 '{address}' 解析失败 (尝试 {i + 1}/{retries})，返回信息：{data}")
            if i < retries - 1:
                time.sleep(delay)
    return None, None

# 解析时间字符串
def parse_time(time_str):
    try:
        return datetime.strptime(time_str, "%Y年%m月%d日 %H:%M")
    except ValueError:
        return None

# 获取文心一言 access token
def get_access_token():
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    response = requests.post(url, params=params)
    try:
        response.raise_for_status()
        token = response.json().get("access_token")
        if token:
            return token
        else:
            raise ValueError("无法获取 access token，请检查 API Key 和 Secret Key")
    except requests.exceptions.RequestException as e:
        print(f"获取 access token 失败: {e}")
        raise

# 使用文心一言推荐旅行城市
def ask_wenxin(locations):
    url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token={get_access_token()}"
    headers = {'Content-Type': 'application/json'}

    visited_locations = "\n".join([f"{loc['address'].split(' ')[0]} {loc['address'].split(' ')[1]}" for loc in locations])

    # Improved prompt for automatic suggestion
    prompt = "请根据用户去过的地点，推荐下一个旅行城市，只输出一个城市，格式为：省份市，例如：四川省成都市。\n"
    prompt += f"用户去过的地点：\n{visited_locations}"

    payload = {
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()['result']
        recommended_location = result.strip().split('\n')[-1].strip()
        return recommended_location.replace(" ", "")  # Remove spaces

    except Exception as e:
        print(f"Error calling Wenxin API: {e}")
        return f"Error: {e}"
