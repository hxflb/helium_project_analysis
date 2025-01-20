# 定义一个包含5个出租车记录的列表
taxi_data = [
    {"VendorID": 1, "lpep_pickup_datetime": "2024-11-25 08:00:00", "trip_distance": 2.5, "fare_amount": 15.0, "total_amount": 18.5},
    {"VendorID": 2, "lpep_pickup_datetime": "2024-11-25 09:15:00", "trip_distance": 3.0, "fare_amount": 20.0, "total_amount": 25.0},
    {"VendorID": 1, "lpep_pickup_datetime": "2024-11-25 10:30:00", "trip_distance": 1.8, "fare_amount": 12.0, "total_amount": 14.5},
    {"VendorID": 3, "lpep_pickup_datetime": "2024-11-25 11:00:00", "trip_distance": 5.0, "fare_amount": 30.0, "total_amount": 35.0},
    {"VendorID": 2, "lpep_pickup_datetime": "2024-11-25 12:45:00", "trip_distance": 7.2, "fare_amount": 40.0, "total_amount": 45.0}
]

# 定义字典，包含每个字段的含义
field_meanings = {
    "VendorID": "供应商ID",
    "lpep_pickup_datetime": "接客时间",
    "trip_distance": "行程距离",
    "fare_amount": "车费金额",
    "total_amount": "总费用"
}

# 使用for循环遍历字典，打印每个字段的名称和含义
print("字段及其含义：")
for field, meaning in field_meanings.items():
    print(f"{field}: {meaning}")

# 使用条件语句找出总费用最高的出租车记录，并打印结果
max_total_amount = 0
max_record = None

for record in taxi_data:
    if record["total_amount"] > max_total_amount:
        max_total_amount = record["total_amount"]
        max_record = record

print("\n总费用最高的记录:")
print(max_record)
