import pandas as pd
import numpy as np

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
# 打印清洗后的前10行数据
print(df_cleaned.head(10))
