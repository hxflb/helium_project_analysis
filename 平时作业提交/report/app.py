from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import os
from models import get_location_from_address, parse_time, ask_wenxin  # 从 models.py 导入相关函数
import datetime  # 导入 datetime 模块

app = Flask(__name__)
app.secret_key = os.urandom(24)  # 用于 session 的密钥


# 首页，处理 GET 和 POST 请求
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # 获取前端提交的照片数据
        photos_data = request.get_json()
        if photos_data:
            try:
                photos = photos_data
                # 解析每张照片的时间
                for photo in photos:
                    photo['parsed_time'] = parse_time(photo['time'])
                    if not photo['parsed_time']:
                        return jsonify({"error": f"时间格式错误: {photo['time']}"})

                # 将照片数据存储到 session 中
                session['photos'] = photos
                return jsonify({"status": "success"})
            except (KeyError, ValueError, TypeError) as e:
                return jsonify({"error": f"数据格式错误: {str(e)}"})
        else:
            return jsonify({"error": "缺少照片数据"})
    return render_template("index.html")


# 旅行日志页面，展示照片的地图
@app.route("/travel_journal")
def travel_journal():
    # 从 session 中获取照片数据
    photos = session.get('photos')

    if photos:
        try:
            # 按时间顺序对照片进行排序
            photos.sort(key=lambda x: x['parsed_time'])
            coordinates = []
            for photo in photos:
                lat, lon = get_location_from_address(photo['address'])
                if lat and lon:
                    coordinates.append((lat, lon, photo['address'], photo['time']))
                else:
                    print(f"地址 '{photo['address']}' 解析失败")

            # 如果有有效的坐标，生成地图
            if coordinates:
                map_center = [(coordinates[0][0] + coordinates[-1][0]) / 2,
                              (coordinates[0][1] + coordinates[-1][1]) / 2]
                import folium
                m = folium.Map(location=map_center, zoom_start=5)

                for i, (lat, lon, address, time) in enumerate(coordinates):
                    folium.Marker([lat, lon], popup=f"{address}\n{time}",
                                  icon=folium.Icon(icon='camera', prefix='fa')).add_to(m)

                    # 添加时间标签
                    folium.Marker([lat, lon], popup=f'<div style="width:100px; font-size:10px;">{time}</div>',
                                  icon=folium.DivIcon(
                                      icon_size=(150, 36),
                                      icon_anchor=(7, 20),
                                      html=f'<div style="font-size:10px; color: #000">{i + 1}</div>'
                                  )).add_to(m)

                # 连接所有地点形成路径
                path = [(lat, lon) for lat, lon, _, _ in coordinates]
                folium.PolyLine(path, color="black", weight=2.5, opacity=1).add_to(m)

                # 保存地图为 HTML 文件
                map_filename = f"static/maps/{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.html"
                os.makedirs("static/maps", exist_ok=True)
                m.save(map_filename)

                # 返回渲染的 map.html 页面
                map_path = url_for('static', filename=map_filename[7:])
                return render_template("map.html", map_path=map_path)
            else:
                return jsonify({"error": "所有地址解析失败"})
        except Exception as e:
            return jsonify({"error": str(e)})
    else:
        return redirect(url_for('index'))


# 自定义旅行计划页面
@app.route("/custom_plan", methods=["GET", "POST"])
def custom_plan():
    is_feature_complete = False  # 假设该功能尚未完成

    # 如果功能未完成，直接显示提示信息
    if not is_feature_complete:
        return render_template("custom_plan.html", is_feature_complete=True)

    # 如果功能已完成，处理自定义计划的逻辑
    photos = session.get('photos')
    if not photos:
        return redirect(url_for('index'))

    recommended_location = ask_wenxin(photos)  # 获取推荐的旅行城市

    # 如果推荐成功，则获取该城市的经纬度
    if "Error:" not in recommended_location:
        lat, lng = get_location_from_address(recommended_location)

        if lat and lng:
            import folium
            # 在地图上标记推荐地点
            m = folium.Map(location=[lat, lng], zoom_start=10)
            folium.Marker([lat, lng], popup=recommended_location).add_to(m)

            # 保存定制地图为 HTML 文件
            custom_map_filename = os.path.join("static/maps",
                                               f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}_custom.html")
            m.save(custom_map_filename)

            custom_map_path = url_for('static', filename=custom_map_filename[7:])
            return render_template("custom_plan.html", answer=recommended_location, custom_map_path=custom_map_path)
        else:
            return render_template("custom_plan.html", answer="无法获取推荐地点的经纬度。", custom_map_path=None)
    else:
        return render_template("custom_plan.html", answer=recommended_location, custom_map_path=None)


if __name__ == "__main__":
    app.run(debug=True)
