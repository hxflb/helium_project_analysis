from transformers import *
import pandas as pd
import re
import torch
from anti_detect import rewrite_text

# 加载模型和tokenizer
model_name = "roberta-base"  # 或者你之前用于训练的模型名称
model_path = "roberta-base"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=2)
saved_path = "D:/Project/py/SemEval2024_QUST-main/AIGC_text_detector-main/results/roberta-base_unfilter_fullCLEAN_save_original_single_0/1_32_5e-05_0.0/sentence_deletion-0.25_1_dual_softmax_dyn_dtrun_0.4_0.2_55/best-model.pt"

# 加载之前保存的模型参数
checkpoint = torch.load(saved_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()


file_path = "data/quora/questions.csv"
count = 0
correct = 0
error = 0
processed_correct = 0
processed_error = 0

df = pd.read_csv(file_path, on_bad_lines='skip', encoding='utf-8')
texts = df["question1"]
texts = texts.head(2000)
for text in texts:
    if isinstance(text, str):
        text = re.sub(r'[^a-zA-Z0-9.,!?\s]', '', text)
        if type(text) == str:
            print(text + "\n")
            inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            processed_text = rewrite_text(text)
            print("processed:   " + processed_text + "\n")
            processed_inputs = tokenizer(processed_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
            processed_inputs = {k: v.to(device) for k, v in processed_inputs.items()}
            with torch.no_grad():
                input_ids = inputs['input_ids']
                outputs = model(**inputs)
                # 获取预测结果
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=1).item()
                count += 1

                processed_outputs = model(**processed_inputs)
                processed_logits = processed_outputs.logits
                processed_class = torch.argmax(processed_logits, dim=1).item()
                if predicted_class == 0:
                    correct += 1
                elif predicted_class == 1:
                    error += 1
                if processed_class == 0:
                    processed_correct += 1
                elif processed_class == 1:
                    processed_error += 1

rate = (correct * 1.0) / (count * 1.0)
processed_rate = (processed_correct * 1.0) / (count * 1.0)
print("count:", count, "\ncorrect:", correct, "\nerror:", error, "\nrate:", rate * 100, "%")
print("count:", count, "\nprocessed_correct:", processed_correct, "\nprocessed_error:", processed_error, "\nprocessed_rate:", processed_rate * 100, "%")