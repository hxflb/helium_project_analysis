import pandas as pd

# 读取CSV文件
def load_data(file_path):
    # 使用pandas读取CSV文件
    return pd.read_csv(file_path)


# 清理数据的函数
def clean_data(df):
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

    return df


# 主程序
def main():
    # 读取数据
    file_path = '../data/green_tripdata_2016-12.csv'  # 假设文件路径是当前目录
    df = load_data(file_path)

    # 清理数据
    df = clean_data(df)

    # 打印清理后的数据的前10行
    print(df.head(10).to_markdown())


if __name__ == "__main__":
    main()
