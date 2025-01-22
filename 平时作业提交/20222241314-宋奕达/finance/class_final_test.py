import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
import re
import logging
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def predict_product_categories_with_confidence(model_path, vectorizer_path, csv_file_path, label_mapping_path,
                                               threshold=0.7):
    try:
        # 加载模型和TF-IDF向量化器
        model = load(model_path)
        vectorizer = load(vectorizer_path)
        logging.info("模型和向量化器加载成功。")
    except FileNotFoundError:
        logging.error(f"未找到模型或向量化器文件。请检查路径。")
        return

    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file_path, encoding='utf-8-sig', header=None)  # 确保没有将第一行作为列名
        product_list_en = df.iloc[:, 0].tolist()  # 英文产品名称列表
        product_list_cn = df.iloc[:, 1].tolist()  # 中文产品名称列表
        logging.info("CSV文件加载成功。")
    except FileNotFoundError:
        logging.error(f"未找到CSV文件。请检查路径。")
        return
    except Exception as e:
        logging.error(f"读取CSV文件时出错：{e}")
        return

    try:
        # 读取标签映射文件
        label_mapping_df = pd.read_csv(label_mapping_path, encoding='utf-8-sig', header=None)
        # 清理标签并创建映射字典，确保大小写和空格一致
        label_mapping = {re.sub(r'[^a-zA-Z\s]', '', k).lower().strip(): v for k, v in
                         zip(label_mapping_df[0], label_mapping_df[1])}
        logging.info("标签映射文件加载成功。")
    except FileNotFoundError:
        logging.error(f"未找到标签映射文件。请检查路径: {label_mapping_path}")
        return
    except Exception as e:
        logging.error(f"读取标签映射文件时出错: {e}")
        return

    # 文本预处理函数
    def preprocess_text(text):
        # 清理文本，去掉非字母和空格，并转换为小写
        text = re.sub(r'[^a-zA-Z\s]', '', str(text)).lower()
        return text.strip()

    # 修改后的预测函数：根据阈值判断类别是否有效
    def predict_with_confidence(model, vectorizer, input_text, threshold=0.7):
        # 对输入文本进行预处理和特征提取
        input_vector = vectorizer.transform([input_text])

        # 获取每个类别的预测概率
        proba = model.predict_proba(input_vector)

        # 获取最大概率和对应的标签
        max_proba = np.max(proba)
        predicted_label = model.classes_[np.argmax(proba)]

        # 如果最大概率小于阈值，则返回 "Unknown"
        if max_proba < threshold:
            return 'Unknown'  # 如果模型预测不够确信，返回“未知”
        else:
            return predicted_label

    # 对英文产品名称进行预处理
    preprocessed_inputs = [preprocess_text(product_en) for product_en in product_list_en]

    # 将预处理后的文本转换为特征向量
    input_vectors = vectorizer.transform(preprocessed_inputs)

    # 使用训练好的模型进行预测
    predictions = [predict_with_confidence(model, vectorizer, product_en, threshold) for product_en in product_list_en]

    # 输出每个产品的中文类别标签
    found_unknown = False  # 用于标记是否有未知类别
    for idx, (product_cn, prediction) in enumerate(zip(product_list_cn, predictions), 1):
        # 清理预测结果
        cleaned_prediction = re.sub(r'[^a-zA-Z\s]', '', prediction).lower().strip()
        # 查找映射表中的中文标签
        chinese_prediction = label_mapping.get(cleaned_prediction)

        if chinese_prediction:
            print(f"{chinese_prediction} ：{product_cn}")
        else:
            # 如果未找到对应的中文标签，标记未知并跳出循环
            found_unknown = True
            continue  # 跳过这个产品，等待最后输出未知类别

    # 如果有未知类别，另起一行输出“未知”
    if found_unknown:
        print("未知：")
        for product_cn, prediction in zip(product_list_cn, predictions):
            cleaned_prediction = re.sub(r'[^a-zA-Z\s]', '', prediction).lower().strip()
            if cleaned_prediction == 'unknown':
                print(f"  {product_cn}")


# --- 使用示例 ---
model_path = "D:/finance/logistic_regression_model.joblib"  # 模型文件路径
vectorizer_path = "D:/finance/tfidf_vectorizer.joblib"  # 向量化器文件路径
csv_file_path = "new_text.csv"  # 新输入的CSV文件路径
label_mapping_path = "new_lable.csv"  # 标签映射文件路径

# 调用预测函数，阈值设置为 0.7
predict_product_categories_with_confidence(model_path, vectorizer_path, csv_file_path, label_mapping_path,
                                           threshold=0.7)
