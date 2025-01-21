import pandas as pd
df = pd.read_csv("green_tripdata_2016-12.csv")
df = df.dropna()  # 移除缺失值
df = df[(df['trip_distance'] > 0) & (df['fare_amount'] > 0)]  # 移除异常值

df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['hour'] = df['pickup_datetime'].dt.hour
df['weekday'] = df['pickup_datetime'].dt.weekday

from geopy.distance import geodesic
def calculate_distance(row):
    pickup = (row['pickup_latitude'], row['pickup_longitude'])
    dropoff = (row['dropoff_latitude'], row['dropoff_longitude'])
    return geodesic(pickup, dropoff).km
df['distance'] = df.apply(calculate_distance, axis=1)

df = pd.get_dummies(df, columns=['payment_type', 'vehicle_type'], drop_first=True)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['distance', 'hour', 'fare_amount']] = scaler.fit_transform(df[['distance', 'hour', 'fare_amount']])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(512, activation='relu', input_dim=X_train.shape[1]),  # 输入层
    Dense(512, activation='relu'),  # 隐藏层
    Dense(1)  # 输出层（回归任务）
])
model.compile(optimizer='adam', loss='mse')

from sklearn.model_selection import train_test_split
X = df.drop('fare_amount', axis=1)  # 选择预测特征
y = df['fare_amount']  # 目标值
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

from sklearn.metrics import mean_absolute_error
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
