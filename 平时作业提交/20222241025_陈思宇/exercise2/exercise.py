import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams['font.family'] = 'SimHei'

# 读取数据
df = pd.read_csv('../data/green_tripdata_2016-12.csv')

# 将lpep_pickup_datetime和lpep_dropoff_datetime列转换为datetime格式
df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])

# 处理缺失值
# 1. 对于缺失的数值型数据，可以选择填充为0或均值，或者直接删除缺失值
df['fare_amount'].fillna(0, inplace=True)
df['trip_distance'].fillna(df['trip_distance'].mean(), inplace=True)
df['total_amount'].fillna(0, inplace=True)
df['tip_amount'].fillna(0, inplace=True)

# 2. 删除包含关键字段缺失的行（比如VendorID, PULocationID, DOLocationID等）
df.dropna(subset=['VendorID', 'PULocationID', 'DOLocationID'], inplace=True)

# 处理异常值
# 1. 异常值检查：对于行程距离和费用，确保它们的值合理
df = df[(df['trip_distance'] > 0) & (df['trip_distance'] < 100)]  # 假设行程距离应小于100公里
df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 500)]  # 假设车费应小于500美元
df = df[(df['total_amount'] > 0) & (df['total_amount'] < 500)]  # 总费用也应该小于500美元

# 2. 过滤掉异常的`passenger_count`，比如负数或大于20
df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 20)]

#。 提取每10分钟的流量数据
df['pickup_time'] = df['lpep_pickup_datetime'].dt.floor('10T')  # 向下取整到每10分钟
trip_counts = df.groupby('pickup_time').size()  # 统计每个时间段内的出租车数量

# 展示为折线图
plt.figure(figsize=(12, 6))
trip_counts.plot(kind='line')
plt.title('每10分钟车流量折线图')
plt.xlabel('时间')
plt.ylabel('车流量')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('taxi_trip_counts_10min.png')
plt.show()

# 分割数据为训练集和测试集
train_size = int(len(trip_counts) * 0.8)
train_data = trip_counts[:train_size]
test_data = trip_counts[train_size:]

# 准备训练数据和测试数据
train_data_df = pd.DataFrame({'time': np.arange(len(train_data)), 'trips': train_data.values})
test_data_df = pd.DataFrame({'time': np.arange(len(train_data), len(trip_counts)), 'trips': test_data.values})

# 训练线性回归模型
X_train = train_data_df[['time']]  # 特征是时间
y_train = train_data_df['trips']  # 目标是流量
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# 训练决策树回归模型
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# 在测试集上进行预测
X_test = test_data_df[['time']]
y_test = test_data_df['trips']


# 使用线性回归进行预测
y_pred_linear = linear_model.predict(X_test)

# 使用决策树回归进行预测
y_pred_tree = tree_model.predict(X_test)

# 计算均方误差（MSE）和决定系数（R²）
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)

print(f'线性回归 - 均方误差 (MSE): {mse_linear}')
print(f'线性回归 - 决定系数 (R²): {r2_linear}')
print(f'决策树回归 - 均方误差 (MSE): {mse_tree}')
print(f'决策树回归 - 决定系数 (R²): {r2_tree}')

# 绘制实际值与预测值的对比图
plt.figure(figsize=(12, 6))



# 线性回归预测与实际值对比
plt.plot(test_data_df['time'], y_test, label='实际值', color='blue')
plt.plot(test_data_df['time'], y_pred_linear, label='线性回归预测', color='red', linestyle='dashed')

# 决策树回归预测与实际值对比
plt.plot(test_data_df['time'], y_pred_tree, label='决策树回归预测', color='green', linestyle='dotted')

plt.title('车流量实际值与预测值对比图')
plt.xlabel('时间')
plt.ylabel('车流量')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted_taxi_trips_comparison.png')
plt.show()
