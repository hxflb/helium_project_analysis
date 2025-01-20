from random import random
from mindspore.dataset import vision
import mindspore.dataset as ds
from PIL import Image, ImageOps
import numpy as np
import mindspore
import mindspore.dataset.vision as C
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
from collections import defaultdict

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
                    print("match {}".format(base_name))
                    break

    return matching_pairs

root_directory_path = 'data/gtFine/val'  # 替换为你的根目录路径
matching_pairs = find_matching_pairs(root_directory_path)
print(len(matching_pairs))
# height = 256
# width = 512
# def preprocess_image(image_path):
#     image = Image.open(image_path).convert('RGB')
#     resize_op = ImageOps.fit(image, (width, height), centering=(0.5, 0.5))
#     image_np = np.array(resize_op)
#     image_np = image_np / 255.0
#     return image_np
# transforms = [
#     C.RandomHorizontalFlip(),  # 随机水平翻转图像
#     C.RandomRotation(45),  # 随机旋转图像，旋转角度范围为-45到45度
#     C.RandomColorAdjust(0.2, 0.2, 0.2, 0.2)  # 随机调整图像亮度、对比度和饱和度
# ]
# image_path ='data/gtFine/train/aachen/aachen_000000_000019_gtFine_color.png'
# image = Image.open(image_path)
# image_np = Image.open(image_path).convert('RGB')
# image_np = preprocess_image(image_path)
# label_path = 'data/gtFine/train/aachen/aachen_000000_000019_gtFine_polygons.json'
# with open(label_path, 'r') as f:
#     label_data = json.load(f)
#     img_height = label_data['imgHeight']
#     img_width = label_data['imgWidth']
#     # 遍历对象列表并绘制多边形
#     for obj in label_data['objects']:
#         label = obj['label']
#         polygon = np.array(obj['polygon'], dtype=np.int32)
#         scaled_polygon = (polygon * [width/img_width, height/img_height]).astype(np.int32)
#         mask = np.zeros((height,width), dtype=np.uint8)
#         cv2.fillPoly(mask, [scaled_polygon], 255)
#         extracted_region = cv2.bitwise_and(image_np, image_np, mask=mask)
#         extracted_region = extracted_region
#         extracted_region_uint8 = (extracted_region *255).astype(np.uint8)
#         # 将提取的区域转换为 PIL 图像
#         extracted_region_pil = Image.fromarray(extracted_region_uint8)
#         extracted_region_tensor = Tensor(np.array(extracted_region_pil))
#         transform = [
#             vision.RandomHorizontalFlip(prob=0.5),  # 水平翻转
#             vision.RandomRotation(degrees=45),  # 旋转
#             vision.RandomColorAdjust(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)  # 颜色调整
#         ]
#         single_image_dataset = ds.GeneratorDataset(
#             source=[(extracted_region_tensor,)],
#             column_names=["image"],
#             shuffle=False
#         )
#
#         single_image_dataset = single_image_dataset.map(operations=transform, input_columns=["image"])
#
#         for item in single_image_dataset.create_dict_iterator():
#             extracted_region_transformed = item["image"]
#         extracted_region_transformed_np = extracted_region_transformed.asnumpy()
#         image_np[mask == 255] = extracted_region_transformed_np[mask == 255]
#         extracted_region2 = cv2.bitwise_and(image_np, image_np, mask=mask)




        # (h, w) = extracted_region.shape[:2]
        #
        # # 设置旋转中心为图像中心
        # center = (w // 2, h // 2)
        #
        # # 设置旋转角度（例如，45度）和缩放比例（例如，1.0表示不缩放）
        # angle = 45
        # scale = 1.0
        #
        # # 计算旋转矩阵
        # M = cv2.getRotationMatrix2D(center, angle, scale)
        #
        # # 执行旋转操作，并处理边界情况（这里使用边界填充）
        # cos = np.abs(M[0, 0])
        # sin = np.abs(M[0, 1])
        # # 计算新图像的尺寸，以适应旋转后的内容
        # new_w = int((h * sin) + (w * cos))
        # new_h = int((h * cos) + (w * sin))
        # rotated_extracted_region = cv2.warpAffine(extracted_region, M, (new_w, new_h), flags=cv2.INTER_CUBIC,
        #                                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        #
        # cv2.imshow('{}'.format(label), extracted_region_transformed_np)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
