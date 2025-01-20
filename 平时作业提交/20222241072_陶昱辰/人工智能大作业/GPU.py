from random import random
from model import CNN
import mindspore.dataset as ds
from PIL import Image, ImageOps
import numpy as np
from mindspore.dataset import vision
import mindspore as ms
import mindspore.dataset.transforms.c_transforms as C
import mindspore.nn as nn
import mindspore.ops.functional as F
import mindspore.common.dtype as mstype
from mindspore import Tensor, context, export, load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, LossMonitor, TimeMonitor
from mindspore.train import Model
import json
import os
from mindspore.dataset import GeneratorDataset
import re
import cv2
from mindspore.nn import Adam
from mindspore.train.callback import Callback
import matplotlib.pyplot as plt

epochs = 30
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")  # 设置为GPU模式

# 数据集路径
train_data_dir = "人工智能大作业/data/gtFine/train"
test_data_dir = "人工智能大作业/data/gtFine/val"

import os
import re
from collections import defaultdict

height = 512
width = 1024

# 定义主类别和对应的索引
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
                    # 提取基本文件名（这里假设基本文件名是文件名中前两个下划线之间的部分，根据实际情况调整）
                    base_name = '_'.join(item.split('_')[:3])  # 注意：这里需要根据您的实际文件名格式调整
                    data_files[base_name].append(item_path)
                elif label_pattern.match(item):
                    base_name = '_'.join(item.split('_')[:3])  # 假设与数据文件相同
                    label_files[base_name].append(item_path)

    # 从根目录开始遍历并收集文件
    traverse_directory(root_directory)

    # 匹配数据文件和标签文件
    for base_name, data_paths in data_files.items():
        if base_name in label_files:
            # 假设每个基本文件名只对应一个数据文件和一个标签文件（或多个数据文件对应一个标签文件）
            # 这里取数据文件的第一个路径（如果有多个，可以根据需要选择）和标签文件的路径进行匹配
            for data_path in data_paths:
                for label_path in label_files[base_name]:
                    matching_pairs.append((data_path, label_path))
                    break

    return matching_pairs


def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    resize_op = ImageOps.fit(image, (width, height), centering=(0.5, 0.5))
    image_np = np.array(resize_op)
    image_np = image_np / 255.0
    return image_np


class LossMonitor(Callback):
    def __init__(self, test_dataset=None, train_network=None,
                 loss_fn=nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean'), interval=1):
        super(LossMonitor, self).__init__()
        self.train_network = train_network
        self.test_dataset = test_dataset
        self.interval = interval
        self.loss_fn = loss_fn
        self.train_losses = []
        self.test_losses = []

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        batch_loss = cb_params.net_outputs
        if isinstance(batch_loss, (tuple, list)) and len(batch_loss) == 1:
            batch_loss = batch_loss[0]
        cur_step_in_epoch = cb_params.cur_step_num
        epoch_size = cb_params.batch_num

        # 每个epoch结束时的处理
        if (cur_step_in_epoch + 1) % epoch_size == 0:
            epoch_loss = batch_loss.asnumpy().mean()
            self.train_losses.append(epoch_loss)
            print(f'Epoch {cb_params.cur_epoch_num + 1}, Train Loss: {epoch_loss}')

            # 如果提供了测试数据集，则进行测试
            if self.test_dataset and (cb_params.cur_epoch_num + 1) % self.interval == 0:
                test_loss = self.eval_dataset(self.test_dataset)
                self.test_losses.append(test_loss)
                print(f'Epoch {cb_params.cur_epoch_num + 1}, Test Loss: {test_loss}')

    def eval_dataset(self, dataset):
        model = self.train_network
        dataset_size = dataset.get_dataset_size()
        total_loss = 0.0
        num_correct = 0

        for data in dataset.create_dict_iterator(output_numpy=True):
            inputs, labels = data['data'], data['label']
            inputs = Tensor(inputs, ms.float32).to(ms.device_target("GPU"))  # 确保数据在GPU上
            labels = Tensor(labels, ms.int32).to(ms.device_target("GPU"))  # 确保标签在GPU上
            logits = model(inputs)
            loss = self.loss_fn(logits, labels)
            total_loss += loss.asnumpy().mean()

        average_loss = total_loss / dataset_size
        return average_loss


times = 0


def data_generator(matching_pairs, sublabel_to_class, classes, width, height, data_size=8, shuffle=True):

    if shuffle:
        np.random.shuffle(matching_pairs)

    Train_X, Train_y = [], []

    for image_path, label_path in matching_pairs:
        image_np = preprocess_image(image_path)

        with open(label_path, 'r') as f:
            label_data = json.load(f)
            img_height = label_data['imgHeight']
            img_width = label_data['imgWidth']

            for obj in label_data['objects']:
                label = obj['label']
                if label in sublabel_to_class:
                    class_index = classes[sublabel_to_class[label]]
                    one_hot_encoding = np.zeros(len(classes))
                    one_hot_encoding[class_index] = 1

                    polygon = np.array(obj['polygon'], dtype=np.int32)
                    scaled_polygon = (polygon * [width / img_width, height / img_height]).astype(np.int32)
                    mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.fillPoly(mask, [scaled_polygon], 255)

                    extracted_region = cv2.bitwise_and(image_np, image_np, mask=mask)
                    extracted_region_uint8 = extracted_region.astype(np.uint8)
                    extracted_region_pil = Image.fromarray(extracted_region_uint8)
                    extracted_region_tensor = Tensor(np.array(extracted_region_pil))

                    # 定义数据增强操作
                    transform = [
                        vision.RandomHorizontalFlip(prob=0.5),
                        vision.RandomRotation(degrees=45),
                    ]

                    single_image_dataset = ds.GeneratorDataset(
                        source=[(extracted_region_tensor,)],
                        column_names=["image"],
                        shuffle=False
                    )
                    single_image_dataset = single_image_dataset.map(operations=transform, input_columns=["image"])

                    for item in single_image_dataset.create_dict_iterator():
                        extracted_region_transformed = item["image"]

                    Train_X.append(extracted_region_transformed)
                    Train_y.append(one_hot_encoding)

                    if len(Train_X) >= data_size:
                        batch_X = ms.ops.Concat(0)(Train_X)
                        batch_y = np.concatenate(Train_y, axis=0)
                        yield batch_X, batch_y

                        Train_X, Train_y = [], []  # 清空批次

    # 返回剩余的样本
    if len(Train_X) > 0:
        batch_X = ms.ops.Concat(0)(Train_X)
        batch_y = np.concatenate(Train_y, axis=0)
        yield batch_X, batch_y


# 配置匹配对数据
matching_pairs_train = find_matching_pairs(train_data_dir)
matching_pairs_test = find_matching_pairs(test_data_dir)
print("------------------------文件配对完毕-----------------------------------------")

# 创建CNN模型
net = CNN()

# 设置损失函数和优化器
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
optimizer = Adam(params=net.trainable_params(), learning_rate=0.001)

# 创建模型并将其移动到GPU
model = Model(net, loss_fn, optimizer)
model.set_train()

# 训练过程
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    # 训练数据生成器
    generator_train = data_generator(matching_pairs_train, sublabel_to_class, classes, width, height)
    dataset = ds.GeneratorDataset(generator_train, ["data", "label"])

    # 测试数据生成器
    generator_test = data_generator(matching_pairs_test, sublabel_to_class, classes, width, height)

    # 设置损失监控回调
    loss_monitor = LossMonitor(test_dataset=generator_test, train_network=net, interval=1)

    # 开始训练
    model.train(1, dataset, callbacks=[loss_monitor])
    print(f"Epoch {epoch + 1} completed!")