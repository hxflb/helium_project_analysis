import os
import json
import torch
import numpy as np
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import random
from collections import defaultdict
import re
from model import CNN
# 设备配置，使用GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置数据路径
train_data_dir = "data/gtFine/train/aachen"
test_data_dir = "data/gtFine/val/frankfurt"

height = 512
width = 1024

# 定义类别与索引的映射
classes = {
    "flat": 0,
    "human": 1,
    "vehicle": 2,
    "construction": 3,
    "object": 4,
    "nature": 5,
    "sky": 6,
    "void": 7
}

# 定义子标签到主类别的映射
sublabel_to_class = {
    "road": "flat",
    "sidewalk": "flat",
    "parking": "flat",
    "rail track": "flat",
    "person": "human",
    "rider": "human",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "on rails": "vehicle",
    "motorcycle": "vehicle",
    "bicycle": "vehicle",
    "caravan": "vehicle",
    "trailer": "vehicle",
    "ego vehicle": "vehicle",
    "building": "construction",
    "wall": "construction",
    "fence": "construction",
    "guard rail": "construction",
    "bridge": "construction",
    "tunnel": "construction",
    "pole": "object",
    "pole group": "object",
    "traffic sign": "object",
    "traffic light": "object",
    "vegetation": "nature",
    "terrain": "nature",
    "sky": "sky",
    "ground": "void",
    "dynamic": "void",
    "static": "void"
}

# 查找数据集中的图像和标签的匹配对
def find_matching_pairs(root_directory):
    matching_pairs = []
    data_files = defaultdict(list)
    label_files = defaultdict(list)
    data_pattern = re.compile(r'.*_instanceIds\.png$')
    label_pattern = re.compile(r'.*_polygons\.json$')

    def traverse_directory(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                traverse_directory(item_path)
            else:
                if data_pattern.match(item):
                    base_name = '_'.join(item.split('_')[:3])
                    data_files[base_name].append(item_path)
                elif label_pattern.match(item):
                    base_name = '_'.join(item.split('_')[:3])
                    label_files[base_name].append(item_path)

    traverse_directory(root_directory)

    for base_name, data_paths in data_files.items():
        if base_name in label_files:
            for data_path in data_paths:
                for label_path in label_files[base_name]:
                    matching_pairs.append((data_path, label_path))
                    break

    return matching_pairs

# 图像预处理
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((width, height))
    image_np = np.array(image) / 255.0
    return image_np

# 自定义数据集类
class SegmentationDataset(Dataset):
    def __init__(self, matching_pairs, sublabel_to_class, classes, width, height, transform=None):
        self.pairs = matching_pairs
        self.sublabel_to_class = sublabel_to_class
        self.classes = classes
        self.width = width
        self.height = height
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, label_path = self.pairs[idx]
        image_np = preprocess_image(image_path)

        with open(label_path, 'r') as f:
            label_data = json.load(f)
            img_height = label_data['imgHeight']
            img_width = label_data['imgWidth']

            mask = np.zeros((self.height, self.width), dtype=np.uint8)

            # 填充mask
            for obj in label_data['objects']:
                label = obj['label']
                if label in self.sublabel_to_class:
                    class_index = self.classes[self.sublabel_to_class[label]]
                    num_classes = len(classes)
                    one_hot_encoding = np.zeros(num_classes)
                    one_hot_encoding[class_index] = 1
                    polygon = np.array(obj['polygon'], dtype=np.int32)
                    scaled_polygon = (polygon * [self.width / img_width, self.height / img_height]).astype(np.int32)
                    cv2.fillPoly(mask, [scaled_polygon], 255)
                    extracted_region = cv2.bitwise_and(image_np, image_np, mask=mask)
                    extracted_region_uint8 = extracted_region.astype(np.uint8)
                    extracted_region_pil = Image.fromarray(extracted_region_uint8)

                    if self.transform:
                        image_tensor = self.transform(extracted_region_pil)

                    return image_tensor, one_hot_encoding

# 数据增强和转换
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.ToTensor()
])

# 数据集准备
matching_pairs_train = find_matching_pairs(train_data_dir)
matching_pairs_test = find_matching_pairs(test_data_dir)

train_dataset = SegmentationDataset(matching_pairs_train, sublabel_to_class, classes, width, height, transform)
test_dataset = SegmentationDataset(matching_pairs_test, sublabel_to_class, classes, width, height, transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


# 定义简单的CNN模型（示例）

# 创建模型和优化器
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练函数
def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {running_loss / 10}")
            running_loss = 0.0

    # 记录每轮训练的loss
    epoch_loss = running_loss / len(train_loader.dataset) if running_loss != 0 else 0
    train_losses.append(epoch_loss)


# 测试函数
def test():
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            print("O:{}".format(output))
            target = target.argmax(dim=1)
            print("T:{}".format(target))
            _, predicted = torch.max(output.data, 1)
            print("P:{}".format(predicted))
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100 * correct / total
    print(f"Test Loss: {test_loss}, Accuracy: {accuracy}%")

    # 记录每轮测试的loss
    test_losses.append(test_loss)
train_losses = []
test_losses = []
# 训练循环
epochs = 30
for epoch in range(1, epochs + 1):
    train(epoch)
    test()

torch.save(model.state_dict(), 'model.pth')

# 生成loss图
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.savefig('loss_graph.png')  # 保存loss图
plt.show()  # 显示loss图