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


file_path = "data/archive (2)/LLM.csv"
human_count = 0
ai_count = 0
human_correct = 0
ai_correct = 0
processed_human_correct = 0
processed_ai_correct = 0

df = pd.read_csv(file_path, on_bad_lines='skip', encoding='utf-8')
df = df.head(2000)
columns = df.index
for index in range(len(df)):
    text = df.iat[index,0]
    if isinstance(text, str):
        text = re.sub(r'[^a-zA-Z0-9.,!?\s]', '', text)
        if type(text) == str:
            if df.iat[index, 1] == 'ai':
                label = 1
                ai_count += 1
            else:
                label = 0
                human_count += 1
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
                if predicted_class == label and label == 0:
                    human_correct += 1
                elif predicted_class == label and label == 1:
                    ai_correct += 1

                processed_outputs = model(**processed_inputs)
                processed_logits = processed_outputs.logits
                processed_class = torch.argmax(processed_logits, dim=1).item()
                if processed_class == label and label == 0:
                    processed_human_correct += 1
                elif processed_class == label and label == 1:
                    processed_ai_correct += 1

correct = human_correct + ai_correct
count = human_count + ai_count
processed_correct = processed_human_correct + processed_ai_correct
rate = (correct * 1.0) / (count * 1.0)
processed_rate = (processed_correct * 1.0) / (count * 1.0)
error = count - correct
processed_error = count - processed_correct
human_error = human_count - human_correct
ai_error = ai_count - ai_correct
processed_human_error = human_count - processed_human_correct
processed_ai_error = ai_count - processed_ai_correct

if human_count != 0:
    human_rate = (human_correct * 1.0) / (human_count * 1.0)
    processed_human_rate = (processed_human_correct * 1.0) / (human_count * 1.0)
else:
    human_rate = 0
    processed_human_rate = 0

if ai_count != 0:
    ai_rate = (ai_correct * 1.0) / (ai_count * 1.0)
    processed_ai_rate = (processed_ai_correct * 1.0) / (ai_count * 1.0)
else:
    ai_rate = 0
    processed_ai_rate = 0
print("count:", count, "\ncorrect:", correct, "\nerror:", error, "\nrate:", rate * 100, "%")
print("count:", count, "\nprocessed_correct:", processed_correct, "\nprocessed_error:", processed_error,
      "\nprocessed_rate:", processed_rate * 100, "%")
print("human_count:", human_count, "\nhuman_correct:", human_correct, "\nhuman_error:", human_error, "\nhuman_rate:",
      human_rate * 100, "%")
print("ai_count:", ai_count, "\nai_correct:", ai_correct, "\nai_error:", ai_error, "\nai_rate:", ai_rate * 100, "%")
print("human_count:", human_count, "\nprocessed_human_correct:", processed_human_correct, "\nprocessed_human_error:",
      processed_human_error, "\nprocessed_human_rate:", processed_human_rate * 100, "%")
print("ai_count:", ai_count, "\nprocessed_ai_correct:", processed_ai_correct,
      "\nprocessed_ai_error:", processed_ai_error, "\nprocessed_ai_rate:", processed_ai_rate * 100, "%")
