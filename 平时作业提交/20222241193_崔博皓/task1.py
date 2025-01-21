# 定义一个包含出租车常用字段的列表
fields = ['VendorID', 'lpep_pickup_datetime', 'trip_distance', 'fare_amount', 'total_amount']

# 定义一个字典，包含每个字段的含义
field_meanings = {
    'VendorID': '供应商ID',
    'lpep_pickup_datetime': '接客时间',
    'trip_distance': '行程距离',
    'fare_amount': '车费金额',
    'total_amount': '总费用'
}

# 使用for循环遍历字典，打印每个字段的名称和含义
print("字段名称和含义：")
for field in fields:
    print(f"{field}: {field_meanings[field]}")

# 假设有一个字典包含每个字段的实际数值数据
taxi_data = {
    'VendorID': 1,
    'lpep_pickup_datetime': '2024-11-27 08:30:00',
    'trip_distance': 5.3,  # 公里
    'fare_amount': 15.0,  # 车费金额
    'total_amount': 20.0  # 总费用
}

# 使用条件语句找出总费用最高的字段
max_field = None
max_value = None

for field in fields:
    # 只比较数字类型的字段
    if isinstance(taxi_data[field], (int, float)):
        if max_value is None or taxi_data[field] > max_value:
            max_value = taxi_data[field]
            max_field = field

print(f"\n总费用最高的字段是: {max_field}，值为: {max_value}")