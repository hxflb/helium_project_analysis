import pandas as pd
# 读取CSV文件
df = pd.read_csv('green_tripdata_2016-12.csv')
# 定义包含5个纽约绿牌出租车的常用字段的列表
fields = ['VendorID', 'lpep_pickup_datetime', 'trip_distance', 'fare_amount', 'total_amount']
# 定义一个字典，包含每个字段的含义
fields_description = {
    'VendorID': '供应商ID',
    'lpep_pickup_datetime': '接客时间',
    'trip_distance': '行程距离',
    'fare_amount': '车费金额',
    'total_amount': '总费用'
}
# 使用for循环遍历字典，打印每个字段的名称和含义
for field, description in fields_description.items():
    print(f'字段名称：{field}, 字段含义：{description}')
# 使用条件语句找出总费用最高的字段，并打印结果
max_total_amount = df['total_amount'].max()
max_total_amount_row = df[df['total_amount'] == max_total_amount]
print('\n总费用最高的记录如下：')
print(max_total_amount_row)

