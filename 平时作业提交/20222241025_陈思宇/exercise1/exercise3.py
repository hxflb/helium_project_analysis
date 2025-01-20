import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei'
# 读取数据
df = pd.read_csv('../data/green_tripdata_2016-12.csv')


# 清洗数据的函数
def clean_data(df):
    # 处理缺失值
    # 1. 对于缺失的数值型数据，可以选择填充为0或均值，或者直接删除缺失值
    df['fare_amount'].fillna(df['fare_amount'].mean(), inplace=True)
    df['trip_distance'].fillna(df['trip_distance'].mean(), inplace=True)
    df['total_amount'].fillna(df['total_amount'].mean(), inplace=True)
    df['tip_amount'].fillna(df['tip_amount'].mean(), inplace=True)
    df['passenger_count'].fillna(df['passenger_count'].mean(), inplace=True)

    # 2. 删除包含关键字段缺失的行（比如VendorID, PULocationID, DOLocationID等）
    df.dropna(subset=['VendorID', 'PULocationID', 'DOLocationID', 'lpep_pickup_datetime', 'lpep_dropoff_datetime', 'store_and_fwd_flag'], inplace=True)

    # 处理异常值
    # 1. 异常值检查：对于行程距离和费用，确保它们的值合理
    df = df[(df['trip_distance'] > 0) & (df['trip_distance'] < 100)]  # 假设行程距离应小于100公里
    df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 500)]  # 假设车费应小于500美元
    df = df[(df['total_amount'] > 0) & (df['total_amount'] < 500)]  # 总费用也应该小于500美元

    # 2. 过滤掉异常的`passenger_count`，比如负数或大于20
    df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 20)]

    df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
    df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])


    return df

# 清洗数据

df = clean_data(df)

# 计算行程距离和车费金额的平均值、最大值、最小值
avg_trip_distance = df['trip_distance'].mean()
max_trip_distance = df['trip_distance'].max()
min_trip_distance = df['trip_distance'].min()

avg_fare_amount = df['fare_amount'].mean()
max_fare_amount = df['fare_amount'].max()
min_fare_amount = df['fare_amount'].min()

# 打印结果
print(f"行程距离：平均值 = {avg_trip_distance}, 最大值 = {max_trip_distance}, 最小值 = {min_trip_distance}")
print(f"车费金额：平均值 = {avg_fare_amount}, 最大值 = {max_fare_amount}, 最小值 = {min_fare_amount}")

# 绘制行程距离的直方图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(df['trip_distance'], bins=50, color='skyblue', edgecolor='black')
plt.title('行程距离的分布')
plt.xlabel('行程距离 (miles)')
plt.ylabel('频数')

# 绘制车费金额的直方图
plt.subplot(1, 2, 2)
plt.hist(df['fare_amount'], bins=50, color='salmon', edgecolor='black')
plt.title('车费金额的分布')
plt.xlabel('车费金额 ($)')
plt.ylabel('频数')

plt.tight_layout()
plt.show()

# 计算每日行程距离和车费金额的变化趋势
df['pickup_date'] = df['lpep_pickup_datetime'].dt.date
daily_trip_distance = df.groupby('pickup_date')['trip_distance'].sum()
daily_fare_amount = df.groupby('pickup_date')['fare_amount'].sum()

# 绘制每日行程距离和车费金额的变化趋势
plt.figure(figsize=(12, 6))

# 行程距离的变化趋势
plt.subplot(1, 2, 1)
plt.plot(daily_trip_distance.index, daily_trip_distance.values, marker='o', color='teal')
plt.title('每日行程距离变化趋势')
plt.xlabel('日期')
plt.ylabel('总行程距离 (miles)')
plt.xticks(rotation=45)

# 车费金额的变化趋势
plt.subplot(1, 2, 2)
plt.plot(daily_fare_amount.index, daily_fare_amount.values, marker='o', color='orange')
plt.title('每日车费金额变化趋势')
plt.xlabel('日期')
plt.ylabel('总车费金额 ($)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
