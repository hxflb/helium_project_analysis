from model import GPT2PPLV2 as GPT2PPL
import pandas as pd
import re
from anti_detect import rewrite_text

model = GPT2PPL()

file_path1 = "data/youtube/GBcomments.csv"
file_path2 = "data/youtube/UScomments.csv"
count = 0
correct = 0
error = 0
processed_correct = 0
processed_error = 0


df = pd.read_csv(file_path1, on_bad_lines='skip', encoding='utf-8')
texts = df["comment_text"]
texts = texts.head(1000)
for text in texts:
    if isinstance(text, str) and len(text) > 50:
        text = re.sub(r'[^a-zA-Z0-9.,!?/\s]', '', text)
        if type(text) == str: #and len(text) > 100:
            print(text + "\n")
            count += 1
            predicted_class = model(text, 100, "v1.1")
            processed_text = rewrite_text(text)
            print("processed:   " + processed_text + "\n")
            processed_class = model(text, 100, "v1.1")
            if predicted_class == 1:
                correct += 1
            elif predicted_class == 0:
                error += 1
            if processed_class == 1:
                processed_correct += 1
            elif processed_class == 0:
                processed_error += 1

rate = (correct * 1.0) / (count * 1.0)
processed_rate = (processed_correct * 1.0) / (count * 1.0)
print("count:", count, "\ncorrect:", correct, "\nerror:", error, "\nrate:", rate * 100, "%")
print("count:", count, "\nprocessed_correct:", processed_correct, "\nprocessed_error:", processed_error, "\nprocessed_rate:", processed_rate * 100, "%")


'''
df = pd.read_csv(file_path2, on_bad_lines='skip', encoding='utf-8')
texts = df["comment_text"]
texts = texts.head(1000)
for text in texts:
    if isinstance(text, str):
        text = re.sub(r'[^a-zA-Z0-9.,!?\s]', '', text)
        print(text + "\n")
        if type(text) == str:
            inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                input_ids = inputs['input_ids']
                outputs = model(**inputs)
                # 获取预测结果
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=1).item()
                count += 1
                if predicted_class == 0:
                    correct += 1
                elif predicted_class == 1:
                    error += 1
rate = (correct * 1.0) / (count * 1.0)
print("count:", count, "\ncorrect:", correct, "\nerror:", error, "\nrate:", rate * 100, "%")
'''