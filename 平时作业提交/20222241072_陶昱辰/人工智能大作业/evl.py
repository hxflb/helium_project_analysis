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

criterion = nn.CrossEntropyLoss()
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_data_dir = "data/gtFine/val"
model = CNN()
model_path = 'model.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
model.to(device)
height = 512
width = 1024

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
                    base_name = '_'.join(item.split('_')[:3])  # 根据文件名结构调整
                    data_files[base_name].append(item_path)
                elif label_pattern.match(item):
                    base_name = '_'.join(item.split('_')[:3])  # 与数据文件匹配
                    label_files[base_name].append(item_path)

    traverse_directory(root_directory)

    for base_name, data_paths in data_files.items():
        if base_name in label_files:
            for data_path in data_paths:
                for label_path in label_files[base_name]:
                    matching_pairs.append((data_path, label_path))
                    break

    return matching_pairs

def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((width, height))
    image_np = np.array(image) / 255.0
    return image_np

class SegmentationDataset(Dataset):
    def __init__(self, matching_pairs, sublabel_to_class, classes, width, height):
        self.pairs = matching_pairs
        self.sublabel_to_class = sublabel_to_class
        self.classes = classes
        self.width = width
        self.height = height

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

                    to_tensor = transforms.ToTensor()
                    image_tensor = to_tensor(extracted_region_pil)

                    return image_tensor, one_hot_encoding

def test():
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            print(output)
            loss = criterion(output, target)
            test_loss += loss.item()
            print(target)
            target = target.argmax(dim=1)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100 * correct / total
    print(f"Test Loss: {test_loss}, Accuracy: {accuracy}%")


matching_pairs_test = find_matching_pairs(test_data_dir)
test_dataset = SegmentationDataset(matching_pairs_test, sublabel_to_class, classes, width, height)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
test()