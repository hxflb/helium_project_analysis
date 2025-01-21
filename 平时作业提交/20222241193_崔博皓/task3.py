import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 定义数据清洗函数
def clean_data(df):
    # 处理缺失值
    # 可以选择填充缺失值或者删除含有缺失值的行，下面用填充平均值或中位数来填充
    df['fare_amount'].fillna(df['fare_amount'].mean())  # 用平均值填充 fare_amount
    df['trip_distance'].fillna(df['trip_distance'].median())  # 用中位数填充 trip_distance
    df['total_amount'].fillna(df['total_amount'].mean())  # 用平均值填充 total_amount
    # 删除其他列的缺失值，或者可以进行其他处理
    df.dropna(subset=['VendorID', 'lpep_pickup_datetime', 'lpep_dropoff_datetime'], inplace=True)
    # 处理异常值，假设异常值为负数
    # 对于数值列，如 fare_amount、trip_distance 等，替换负值为 NaN，然后在填充时处理
    numerical_columns = ['fare_amount', 'trip_distance', 'total_amount', 'tip_amount', 'mta_tax', 'tolls_amount']
    for col in numerical_columns:
        df[col] = df[col].apply(lambda x: np.nan if x < 0 else x)  # 负数设为 NaN
    # 再次填充负值替换后的 NaN
    df['fare_amount'].fillna(df['fare_amount'].mean())
    df['trip_distance'].fillna(df['trip_distance'].median())
    df['total_amount'].fillna(df['total_amount'].mean())
    # 可以根据需要进一步清理其他列的异常值
    return df

# 读取xlsx文件
file_path = r'E:\大学课程材料\人工智能基础\上机材料\green_tripdata_2016-12.xlsx'
df = pd.read_excel(file_path)
# 对数据进行清洗
df_cleaned = clean_data(df)

# 计算行程距离和车费金额的平均值、最大值和最小值
trip_distance_stats = df_cleaned['trip_distance'].describe()  # 使用describe来获取统计信息
fare_amount_stats = df_cleaned['fare_amount'].describe()
print("行程距离统计信息：")
print(trip_distance_stats)
print("\n车费金额统计信息：")
print(fare_amount_stats)

# 使用Matplotlib绘制行程距离和车费金额的直方图
plt.figure(figsize=(14, 6))

# 绘制行程距离的直方图
plt.subplot(1, 2, 1)
plt.hist(df_cleaned['trip_distance'], bins=50, color='skyblue', edgecolor='black')
plt.title('dis_distribution')
plt.xlabel('dis (km)')
plt.ylabel('frequency')

# 绘制车费金额的直方图
plt.subplot(1, 2, 2)
plt.hist(df_cleaned['fare_amount'], bins=50, color='lightgreen', edgecolor='black')
plt.title('fare_distribution')
plt.xlabel('fare ($)')
plt.ylabel('frequency')

plt.tight_layout()
plt.show()

# 计算每日行程距离和车费金额的变化趋势

# 转换时间字段为 datetime 类型
df_cleaned['lpep_pickup_datetime'] = pd.to_datetime(df_cleaned['lpep_pickup_datetime'])

# 以日期为单位计算每日的总行程距离和车费金额
df_cleaned['pickup_date'] = df_cleaned['lpep_pickup_datetime'].dt.date
daily_stats = df_cleaned.groupby('pickup_date').agg(
    total_trip_distance=('trip_distance', 'sum'),
    total_fare_amount=('fare_amount', 'sum')
).reset_index()

# 绘制折线图，观察每日行程距离和车费金额的变化趋势
plt.figure(figsize=(14, 6))

# 绘制每日行程距离的变化趋势
plt.subplot(1, 2, 1)
plt.plot(daily_stats['pickup_date'], daily_stats['total_trip_distance'], color='blue', marker='o')
plt.title('dis_trend')
plt.xlabel('date')
plt.ylabel('total_dis(km)')
plt.xticks(rotation=45)

# 绘制每日车费金额的变化趋势
plt.subplot(1, 2, 2)
plt.plot(daily_stats['pickup_date'], daily_stats['total_fare_amount'], color='red', marker='o')
plt.title('fare_trend')
plt.xlabel('date')
plt.ylabel('total_fare($)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()