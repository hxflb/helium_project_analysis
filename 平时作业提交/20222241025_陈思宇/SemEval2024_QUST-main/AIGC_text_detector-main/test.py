import torch
from train import *
from transformers import *

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

# 定义文本
inputs = tokenizer("You can't just go around assassinating the leaders of countries you don't like ! The international condemnation would be brutal . Even though noone likes Kim Jong - Un , and everyone thinks North Korea is pretty shitty to its citizens , if say the US were to send agents over ( and don't think they aren't capable of it ) and they got caught .... every country , every world leader would be a potential target . Who 's next ... Castro ? Angela Merkel ? Anyways , rumour has it that he 's ultra paranoid about exactly that and travels around in tanks and armoured trains that make Limo 1 look like a tonka toy .", return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}  # 将输入数据也移动到GPU上

# 前向传播
with torch.no_grad():
    outputs = model(**inputs)

# 获取预测结果
logits = outputs.logits
predicted_class = torch.argmax(logits, dim=1).item()

# 输出预测结果
print("Predicted class:", predicted_class)
