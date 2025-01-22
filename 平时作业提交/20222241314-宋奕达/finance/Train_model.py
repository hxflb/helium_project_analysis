import pandas as pd
import re
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression  # 引入逻辑回归
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# 加载数据
df = pd.read_csv('combined_file.csv')
df.columns = df.columns.str.strip()  # 清理列名中的空格

# 文本预处理
df['text'] = df['text'].apply(lambda x: re.sub(r'\W+|\d+', ' ', str(x)).lower())

# 查看类别分布情况
print("Label distribution:")
print(df['label'].value_counts())

# 检查是否有缺失值（NaN 或空值）在标签列中
print("Checking for missing values in labels:")
print(df['label'].isna().sum())  # 检查 NaN 数量

# 如果有 NaN，删除含 NaN 的行
df.dropna(subset=['label'], inplace=True)

# 确保没有空值
print("After removing rows with NaN labels:")
print(df['label'].value_counts())

# 检查每个类别的样本数，确保每个类别至少有 2 个样本
label_counts = df['label'].value_counts()
print("Label counts after removing NaN labels:")
print(label_counts)

# 特征提取：增加 ngram 范围来控制特征空间的大小
vectorizer = TfidfVectorizer(max_features=10000, min_df=2, max_df=0.99, ngram_range=(1, 2))  # 考虑双字词组合
X = vectorizer.fit_transform(df['text'])  # 定义 X 变量，这里会从文本数据中提取特征

# 数据集划分，移除 stratify 参数
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

# 选择模型：可以使用逻辑回归
# Logistic Regression
classifier = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)


# 训练模型
classifier.fit(X_train, y_train)

# 保存模型和向量化器
dump(classifier, 'logistic_regression_model.joblib')  # 或 'random_forest_model.joblib' 取决于选择的模型
dump(vectorizer, 'tfidf_vectorizer.joblib')

# 模型评估
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# 输出分类报告，处理 zero_division
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))  # 处理零除错误

# 使用 StratifiedKFold 交叉验证
skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)  # 降低折数为 5
cross_val_accuracy = cross_val_score(classifier, X_train, y_train, cv=skf, scoring='f1_weighted')  # 使用 f1_weighted 作为评分指标
print(f"Cross-validation Accuracy (Weighted F1): {cross_val_accuracy.mean():.4f}")
