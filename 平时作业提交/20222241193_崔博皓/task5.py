import os
# 禁用 TensorFlow 的调试日志输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 设置为 '2' 只显示错误信息
# 禁用 oneDNN 优化相关的日志输出
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# 你的模型代码
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

fields = ["VendorID", "lpep_pickup_datetime", "trip_distance", "fare_amount", "total_amount"]
file_path = r'E:\大学课程材料\人工智能基础\上机材料\green_tripdata_2016-12.xlsx'
def load_csv(file_path):
    try:
        df = pd.read_csv(file_path, index_col=False)
        return df
    except FileNotFoundError:
        print(f"错误：文件未找到 - {file_path}")
        return None
    except Exception as e:
        print(f"错误：读取CSV文件时发生异常 - {e}")
        return None

def clean_data(dfs):
    del dfs['ehail_fee']
    dfs = dfs.dropna()
    #df_dropped = df.drop(columns=['ehail_fee'])
    #df_dropped = df.dropna()
    dfs = dfs[dfs['PULocationID'] > 0]
    dfs = dfs[dfs['RatecodeID'] >= 0]
    dfs = dfs[dfs['DOLocationID'] >= 0]
    dfs = dfs[dfs['passenger_count'] >= 0]
    dfs = dfs[dfs['trip_distance'] >= 0]
    dfs = dfs[dfs['fare_amount'] >= 0]
    dfs = dfs[dfs['extra'] >= 0]
    dfs = dfs[dfs['mta_tax'] > 0]
    dfs = dfs[dfs['tip_amount'] >= 0]
    dfs = dfs[dfs['tolls_amount'] >= 0]
    dfs = dfs[dfs['improvement_surcharge'] >= 0]
    dfs = dfs[dfs['payment_type'] >= 0]
    dfs['fare_amount'] = pd.to_numeric(dfs['fare_amount'], errors='coerce')
    dfs = dfs[dfs['fare_amount'] >= 0]
    return dfs


# 主程序
if __name__ == "__main__":
    # 读取数据
    df = pd.read_excel(file_path)
    if df is not None:
        # 数据预处理
        df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])  # 转换为日期时间类型
        df_cleaned = clean_data(df)

        # 提取日期时间信息
        df_cleaned['pickup_hour'] = df_cleaned['lpep_pickup_datetime'].dt.hour
        df_cleaned['pickup_weekday'] = df_cleaned['lpep_pickup_datetime'].dt.weekday

        # One-Hot Encoding
        df_encoded = pd.get_dummies(df_cleaned, columns=['trip_type', 'payment_type'])

    else:
        print("未能加载数据，程序终止。")

    X = df.drop(columns=['total_amount','VendorID','lpep_pickup_datetime','lpep_dropoff_datetime','store_and_fwd_flag','RatecodeID','PULocationID','DOLocationID','mta_tax']).values  # 转换为NumPy数组
    y = df['total_amount'].values  # 转换为NumPy数组

    # 手动划分据集
    # 假设我们想要70%的数据用于训练，15%用于验证，15%用于测试
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    test_size = len(X) - train_size - val_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    # Reshape the input data to (samples, timesteps, features) for Conv1D
    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val_reshaped = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    # 使用TensorFlow的Normalization层来标准化连续特征
    normalizer = Normalization()
    normalizer.adapt(X_train)
    # 构建具有卷积层的模型
    model = Sequential([
        normalizer,
        # 1D卷积层用于处理时序数据（假设数据是时序型的）
        Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dense(1)  # 输出层，回归问题
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    # 训练模型
    history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_val, y_val),
                        callbacks=[early_stopping])
    mae = model.evaluate(X_test, y_test, verbose=0)
    print(f'Mean Absolute Error on test set: {mae}')


    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    predictions = model.predict(X_test)
    plt.scatter(y_test, predictions)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.show()

